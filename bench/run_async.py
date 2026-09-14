"""W2 -- overlapped pipeline.

The question:

    What changes when stages overlap?

W1 established that overlap cannot make inference faster; it can only remove
idle time, so the gain available is bounded by stage balance. This measures
what is actually collected, what it costs in latency, and whether the output is
still correct.

Structure, following DX-APP's async runner:

    read_worker -> [read_q] -> preprocess_worker --submit--> (ORT callback)
                                                                  |
                        main <- [result_q] <- postprocess_worker <-+ [output_q]

Submission and collection are separate, which is what makes this asynchronous
rather than merely threaded: `_preprocess_worker` calls `session.run_async` and
moves straight to the next frame; the completion callback, running on an ONNX
Runtime thread, hands the outputs to a different worker. A single worker that
submitted and then waited would block on collection and overlap nothing.

Two things W1's harness was built to support and this run finally needs:

* **Service time is kept apart from wait time.** Queue waits are the
  consequence of overlap, not work that overlap can hide, so they are recorded
  as separate stages and excluded from the overlap ceiling.
* **Order is checked, not assumed.** Async completions do not preserve
  submission order. Counting frames would not notice; `OrderChecker` does.

    python -m bench.run_async --model models/resnet50_224.onnx --backend cpu \
        --source clips/bbb_720_5MB.mp4 --inflight 4 --repeat 3 --tag w2
"""
from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from pathlib import Path

from bench import env_fingerprint, noise
from bench.backends import OnnxBackend, available
from bench.harness import Recorder
from bench.pipeline import postprocess, preprocess
from bench.reorder import OrderChecker, ReorderBuffer
from bench.sources import make_source

SERVICE = ("read", "preprocess", "inference", "postprocess")
WAITS = ("q_read", "q_inflight", "q_output", "q_reorder")

_SENTINEL = object()


class _Stop(Exception):
    pass


def parse_args():
    p = argparse.ArgumentParser(description="W2 overlapped inference pipeline")
    p.add_argument("--model", required=True)
    p.add_argument("--backend", default="cpu", help="cpu | cuda | openvino")
    p.add_argument("--ov-device", default=None)
    p.add_argument("--source", default=None)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--intra-threads", type=int, default=None)
    p.add_argument("--inflight", type=int, default=4,
                   help="max inference requests in flight; the sweep variable")
    p.add_argument("--queue-size", type=int, default=4,
                   help="bound on each inter-stage queue (DX-APP uses 4)")
    p.add_argument("--reorder", choices=("on", "off"), default="on",
                   help="'off' publishes completion order, to show the defect")
    p.add_argument("--reorder-cap", type=int, default=None,
                   help="force-flush threshold; defaults to 2x --inflight")
    p.add_argument("--stall-every", type=int, default=0,
                   help="delay postprocess of every Nth frame, to force "
                        "out-of-order completion deliberately")
    p.add_argument("--stall-ms", type=float, default=40.0)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--cooldown", type=float, default=0.0)
    p.add_argument("--max-busy-pct", type=float, default=25.0)
    p.add_argument("--no-idle-gate", action="store_true")
    p.add_argument("--allow-fallback", action="store_true")
    p.add_argument("--tag", default="w2")
    p.add_argument("--out-dir", default="results")
    return p.parse_args()


class AsyncPipeline:
    """One overlapped pass over the source."""

    def __init__(self, backend: OnnxBackend, args, height: int, width: int):
        self.backend, self.args = backend, args
        self.height, self.width = height, width

        self.read_q: queue.Queue = queue.Queue(maxsize=args.queue_size)
        self.output_q: queue.Queue = queue.Queue(maxsize=args.queue_size)
        self.result_q: queue.Queue = queue.Queue(maxsize=args.queue_size)

        # The in-flight limit is the backpressure knob. Acquired before
        # submitting, released in the completion callback, so the pipeline can
        # never have more than `inflight` requests outstanding.
        self.slots = threading.Semaphore(args.inflight)
        self.error: list[tuple[str, str]] = []
        self.submitted = 0

    # -- workers ---------------------------------------------------------
    def _read_worker(self, source) -> None:
        try:
            frames = source.frames()
            idx = 0
            while True:
                # Time the pull, not the loop body: the generator decodes
                # lazily, so timing anything after the frame is in hand
                # measures nothing and reports read as 0.00 ms.
                t0 = time.perf_counter()
                try:
                    frame = next(frames)
                except StopIteration:
                    break
                t1 = time.perf_counter()
                rec = {"idx": idx, "t_start": t0, "read": t1 - t0,
                       "frame": frame, "t_enq_read": t1}
                idx += 1
                self.read_q.put(rec)
        except Exception as exc:                         # noqa: BLE001
            self.error.append(("read_worker", repr(exc)))
        finally:
            self.read_q.put(_SENTINEL)

    def _preprocess_worker(self) -> None:
        try:
            while True:
                rec = self.read_q.get()
                if rec is _SENTINEL:
                    break
                rec["q_read"] = time.perf_counter() - rec["t_enq_read"]

                t = time.perf_counter()
                tensor = preprocess(rec.pop("frame"), self.height, self.width)
                rec["preprocess"] = time.perf_counter() - t

                # Waiting for a slot is queueing, not work: record it apart.
                t = time.perf_counter()
                self.slots.acquire()
                rec["q_inflight"] = time.perf_counter() - t

                rec["t_submit"] = time.perf_counter()
                self.submitted += 1
                self.backend.session.run_async(
                    [self.backend.output_name],
                    {self.backend.input_name: tensor},
                    self._on_complete, rec)
        except Exception as exc:                         # noqa: BLE001
            self.error.append(("preprocess_worker", repr(exc)))
        finally:
            # Drain the in-flight requests before signalling end of stream.
            # Submission returns immediately, so at this point several frames
            # are still inside the runtime; pushing the sentinel now lets it
            # overtake them and their completions are discarded by the
            # downstream worker, which has already shut down.
            #
            # The failure mode is the one this project keeps meeting: the count
            # comes out almost right (196 of 200) and nothing looks wrong.
            # Acquiring every permit means every release has happened, which
            # means every completion callback has run.
            for _ in range(self.args.inflight):
                self.slots.acquire()
            self.output_q.put(_SENTINEL)

    def _on_complete(self, results, rec, err) -> None:
        """Runs on an ONNX Runtime thread, not one of ours."""
        try:
            if err:
                self.error.append(("inference", str(err)))
                rec["logits"] = None
            else:
                rec["logits"] = results[0]
            now = time.perf_counter()
            rec["inference"] = now - rec["t_submit"]
            rec["t_enq_output"] = now
            self.output_q.put(rec)
        finally:
            self.slots.release()

    def _postprocess_worker(self) -> None:
        try:
            while True:
                rec = self.output_q.get()
                if rec is _SENTINEL:
                    break
                rec["q_output"] = time.perf_counter() - rec["t_enq_output"]

                t = time.perf_counter()
                if rec["logits"] is not None:
                    postprocess(rec.pop("logits"), self.args.top_k)
                if (self.args.stall_every
                        and rec["idx"] % self.args.stall_every == 0):
                    # Deliberately make one frame heavier so a lighter
                    # successor overtakes it. Without an asymmetry like this,
                    # completions on a uniform workload rarely reorder and the
                    # correctness question never gets exercised.
                    time.sleep(self.args.stall_ms / 1000.0)
                rec["postprocess"] = time.perf_counter() - t
                rec["t_done_post"] = time.perf_counter()
                self.result_q.put(rec)
        except Exception as exc:                         # noqa: BLE001
            self.error.append(("postprocess_worker", repr(exc)))
        finally:
            self.result_q.put(_SENTINEL)

    # -- driver ----------------------------------------------------------
    def run(self, source, rec_out: Recorder) -> dict:
        cap = self.args.reorder_cap or max(2, 2 * self.args.inflight)
        buf = ReorderBuffer(cap=cap)
        checker = OrderChecker()
        use_reorder = self.args.reorder == "on"

        threads = [
            threading.Thread(target=self._read_worker, args=(source,),
                             name="read", daemon=True),
            threading.Thread(target=self._preprocess_worker,
                             name="preprocess", daemon=True),
            threading.Thread(target=self._postprocess_worker,
                             name="postprocess", daemon=True),
        ]
        for t in threads:
            t.start()

        t_first = time.perf_counter()
        while True:
            rec = self.result_q.get()
            if rec is _SENTINEL:
                break
            if use_reorder:
                for idx, r in buf.push(rec["idx"], rec):
                    self._emit(r, checker, rec_out)
            else:
                self._emit(rec, checker, rec_out)
        if use_reorder:
            for idx, r in buf.drain():
                self._emit(r, checker, rec_out)

        for t in threads:
            t.join(timeout=10)

        return {
            "wall_s": time.perf_counter() - t_first,
            "submitted": self.submitted,
            "reorder": buf.stats() if use_reorder else {"enabled": False},
            "order": checker.stats(),
            "errors": self.error,
        }

    def _emit(self, rec: dict, checker: OrderChecker, rec_out: Recorder) -> None:
        checker.observe(rec["idx"])
        now = time.perf_counter()
        rec["q_reorder"] = now - rec["t_done_post"]
        rec_out.add_frame(
            {k: rec.get(k, 0.0) for k in SERVICE + WAITS},
            now - rec["t_start"],
        )


def measure_once(backend: OnnxBackend, args, height: int, width: int):
    source = make_source(args.source, args.frames)
    rec = Recorder(warmup=args.warmup, stages=SERVICE, wait_stages=WAITS,
                   pipelined=True)
    pipe = AsyncPipeline(backend, args, height, width)
    info = pipe.run(source, rec)
    return rec, source.describe(), info


def main() -> int:
    args = parse_args()

    print(f"available backends: {available()}")
    backend = OnnxBackend(args.model, args.backend, args.intra_threads,
                          args.ov_device)
    if backend.fell_back and not args.allow_fallback:
        print(f"!! requested '{args.backend}' but running on "
              f"{backend.actual_providers}; aborting. Use --allow-fallback to "
              f"measure the fallback deliberately.")
        return 2
    if args.intra_threads is not None and args.intra_threads < 2:
        print("!! run_async completions are delivered by ORT's intra-op pool; "
              "with fewer than 2 threads there may be none spare and the "
              "pipeline can serialise")
    height, width = backend.hw
    print(f"model input {height}x{width}  providers={backend.actual_providers}")
    print(f"inflight={args.inflight}  queue_size={args.queue_size}  "
          f"reorder={args.reorder}")

    gate = {"checked": False, "reason": "disabled"}
    if not args.no_idle_gate:
        print("idle gate:")
        gate = noise.idle_gate(max_busy_pct=args.max_busy_pct)

    dev = f"/{backend.ov_device}" if backend.ov_device else ""
    results, sources, infos = [], [], []
    for i in range(max(1, args.repeat)):
        clock_before = noise.clock()
        rec, src, info = measure_once(backend, args, height, width)
        info["cpu_freq_mhz"] = {"before": clock_before, "after": noise.clock()}
        title = (f"{Path(args.model).name} | {args.backend}{dev} | ASYNC "
                 f"inflight={args.inflight} | run {i + 1}/{args.repeat}")
        rec.print_summary(title)
        o, ro = info["order"], info["reorder"]
        print(f" order         emitted {o['emitted']}  violations "
              f"{o['order_violations']}  "
              f"{'IN SUBMISSION ORDER' if o['in_submission_order'] else 'SHUFFLED'}")
        if ro.get("enabled") is not False:
            print(f" reorder       max depth {ro['max_depth']}  "
                  f"out-of-order arrivals {ro['out_of_order_arrivals']}  "
                  f"max displacement {ro['max_displacement']}  "
                  f"forced flushes {ro['forced_flushes']}")
        if info["errors"]:
            print(f" !! errors: {info['errors'][:3]}")

        r = rec.result()
        r.update({k: info[k] for k in
                  ("submitted", "reorder", "order", "errors", "cpu_freq_mhz")})
        results.append(r)
        sources.append(src)
        infos.append(info)
        if i < args.repeat - 1:
            noise.cooldown(args.cooldown)

    meta = {
        "tag": args.tag, "mode": "async", "args": vars(args),
        "backend": backend.describe(), "sources": sources,
        "model_input_hw": [height, width], "idle_gate": gate,
        "env": env_fingerprint.collect(),
    }
    dev_slug = f"-{backend.ov_device.lower()}" if backend.ov_device else ""
    src_slug = Path(args.source).stem if args.source else "synthetic"
    out = (Path(args.out_dir) /
           f"{args.tag}__async{args.inflight}__{args.backend}{dev_slug}"
           f"__{Path(args.model).stem}__{src_slug}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "runs": results}, indent=2),
                   encoding="utf-8")
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
