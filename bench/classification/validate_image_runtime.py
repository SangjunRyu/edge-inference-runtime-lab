"""Validate order, tensor ownership and ingress policies on real ResNet50 images."""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import statistics
import threading
import time

import numpy as np

from bench.classification.run_async import AsyncPipeline, SERVICE, WAITS
from bench.classification.verify_equivalence import compare
from bench.measurement.env_fingerprint import collect
from bench.measurement.harness import Recorder
from bench.runtime.backends import OnnxBackend
from bench.runtime.sources import ImageListSource


class Ingress:
    """Pace image identifiers and make overflow decisions before JPEG decode."""

    def __init__(self, capacity, policy):
        self.capacity, self.policy = capacity, policy
        self.pending = deque()
        self.dropped = []
        self.offer_lateness_ms = []
        self.peak = 0
        self.blocked_ms = []
        self.closed = False
        self.ready = threading.Condition()

    def offer(self, image_id, scheduled):
        self.offer_lateness_ms.append((time.perf_counter() - scheduled) * 1000)
        with self.ready:
            if self.closed:
                return False
            if len(self.pending) == self.capacity:
                if self.policy == "block":
                    started = time.perf_counter()
                    self.ready.wait_for(lambda: self.closed or len(self.pending) < self.capacity)
                    self.blocked_ms.append((time.perf_counter() - started) * 1000)
                elif self.policy == "drop-newest":
                    self.dropped.append(image_id)
                    return False
                else:
                    self.dropped.append(self.pending.popleft()[0])
            if self.closed:
                return False
            self.pending.append((image_id, scheduled))
            self.peak = max(self.peak, len(self.pending))
            self.ready.notify_all()
            return True

    def take(self):
        with self.ready:
            self.ready.wait_for(lambda: self.pending or self.closed)
            if not self.pending:
                return None
            entry = self.pending.popleft()
            self.ready.notify_all()
            return entry

    def close(self):
        with self.ready:
            self.closed = True
            self.ready.notify_all()


class InputPool:
    """Own reusable CPU input tensors until the corresponding ORT callback."""

    def __init__(self, capacity, shape, audit):
        self.buffers = [np.empty(shape, np.float32) for _ in range(capacity)]
        self.free = list(reversed(range(capacity)))
        self.owned = {}
        self.lock = threading.Lock()
        self.audit = audit
        self.acquired = self.released = self.peak = self.integrity_failures = 0

    @staticmethod
    def fingerprint(tensor):
        return hashlib.sha256(memoryview(tensor).cast("B")).digest()

    def stage(self, rec, tensor):
        with self.lock:
            slot = self.free.pop()
            if slot in self.owned:
                raise RuntimeError("Input tensor reused before its callback")
            self.owned[slot] = rec["idx"]
            self.acquired += 1
            self.peak = max(self.peak, len(self.owned))
            rec["input_slot"] = slot
        buffer = self.buffers[slot]
        np.copyto(buffer, tensor)
        if self.audit:
            rec["input_fingerprint"] = self.fingerprint(buffer)
        return buffer

    def release(self, rec):
        slot = rec.get("input_slot")
        if slot is None:
            return
        intact = not self.audit or self.fingerprint(self.buffers[slot]) == rec.get(
            "input_fingerprint"
        )
        with self.lock:
            if self.owned.get(slot) != rec["idx"]:
                raise RuntimeError("Input tensor released by a non-owner")
            del self.owned[slot]
            self.free.append(slot)
            self.released += 1
            self.integrity_failures += not intact

    def stats(self):
        return {
            "capacity": len(self.buffers), "peak": self.peak,
            "acquired": self.acquired, "released": self.released,
            "outstanding": len(self.owned), "fingerprint_audit": self.audit,
            "integrity_failures": self.integrity_failures,
            "tensor_bytes": self.buffers[0].nbytes,
        }


class ValidationPipeline(AsyncPipeline):
    def __init__(self, backend, args, *, rate=None, policy="block"):
        super().__init__(backend, args, *backend.hw, collector=[])
        self.rate = rate
        self.ingress = Ingress(args.queue_size, policy) if rate else None
        self.pool = InputPool(args.inflight, (1, 3, *backend.hw), audit=rate is None)
        self.emitted = []
        self.started = None

    def _stage_input(self, rec, tensor):
        return self.pool.stage(rec, tensor)

    def _release_input(self, rec):
        self.pool.release(rec)

    def _produce(self, count):
        self.started = time.perf_counter()
        try:
            for image_id in range(count):
                scheduled = self.started + image_id / self.rate
                delay = scheduled - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                self.ingress.offer(image_id, scheduled)
                if self.ingress.closed:
                    break
        except Exception as exc:
            self.error.append(("ingress_producer", repr(exc)))
        finally:
            self.ingress.close()

    def _read_records(self, source):
        if self.ingress is None:
            yield from super()._read_records(source)
            return
        producer = threading.Thread(target=self._produce, args=(source.n,), daemon=True)
        producer.start()
        try:
            sequence = 0
            while (job := self.ingress.take()) is not None:
                image_id, scheduled = job
                started = time.perf_counter()
                frame = source.read_at(image_id)
                end = time.perf_counter()
                # Reorder expects dense indices. Intentional drops leave gaps only
                # in the external image IDs used for validation.
                yield {"idx": sequence, "image_id": image_id, "scheduled": scheduled,
                       "t_start": started, "frame": frame, "read": end - started,
                       "t_enq_read": end}
                sequence += 1
        finally:
            self.ingress.close()
            producer.join()

    def _emit(self, rec, checker, rec_out):
        if self.ingress is not None:
            self.emitted.append((rec["idx"], rec["image_id"],
                                 time.perf_counter() - rec["scheduled"]))
        super()._emit(rec, checker, rec_out)


def percentile(values, q):
    return float(np.percentile(values, q)) if values else None


def config(args, reorder):
    return argparse.Namespace(
        duration_s=0, top_k=5, inflight=args.inflight, queue_size=args.queue_size,
        reorder=reorder, reorder_cap=None, stall_every=0, stall_ms=0,
        stall_timeout_s=30,
    )


def run_one(backend, source, reference, args, kind, repeat, rate, policy, reorder):
    pipeline = ValidationPipeline(backend, config(args, reorder), rate=rate, policy=policy)
    recorder = Recorder(warmup=0, stages=SERVICE, wait_stages=WAITS, pipelined=True)
    info = pipeline.run(source, recorder)
    backend._assert_providers_intact()
    if pipeline.ingress is None:
        ids = [seq for seq, _ in pipeline.collector]
        predictions = dict(pipeline.collector)
        dropped = []
        ingress = None
    else:
        ids = [image_id for _, image_id, _ in pipeline.emitted]
        external_ids = {seq: image_id for seq, image_id, _ in pipeline.emitted}
        predictions = {external_ids[seq]: top for seq, top in pipeline.collector}
        dropped = pipeline.ingress.dropped
        ages_ms = [age * 1000 for _, _, age in pipeline.emitted]
        ingress = {
            "peak": pipeline.ingress.peak, "blocked_offers": len(pipeline.ingress.blocked_ms),
            "blocked_ms": sum(pipeline.ingress.blocked_ms),
            "offer_lateness_p95_ms": percentile(pipeline.ingress.offer_lateness_ms, 95),
            "age_p50_ms": percentile(ages_ms, 50), "age_p95_ms": percentile(ages_ms, 95),
            "age_ms": ages_ms,
            "elapsed_s": time.perf_counter() - pipeline.started,
        }
    parity = compare({i: reference[i] for i in ids}, predictions, 1e-4)
    pool = pipeline.pool.stats()
    accounted = sorted(ids + dropped) == list(range(source.n))
    correct_order = ids == sorted(ids)
    normal_order = correct_order and not info["order"]["order_violations"]
    healthy = (
        accounted and len(ids) == info["submitted"] == pool["acquired"] == pool["released"]
        and len(ids) == len(predictions) == recorder.counted and bool(ids) and pool["outstanding"] == 0
        and pool["integrity_failures"] == 0 and parity["identical"]
        and not info["errors"] and not info["stalls"]
        and (policy != "block" or not dropped)
        and (reorder == "off" or (normal_order and not info["reorder"]["forced_flushes"]))
    )
    return {
        "kind": kind, "repeat": repeat, "target_images_per_s": rate, "policy": policy,
        "reorder": reorder, "offered": source.n, "emitted": len(ids), "emitted_ids": ids,
        "dropped": len(dropped), "dropped_ids": dropped, "accounted": accounted,
        "output_order_violations": info["order"]["order_violations"],
        "top5": parity, "pool": pool, "ingress": ingress, "pipeline": info,
        "metrics": recorder.result(), "healthy": healthy,
    }


def make_summary(report):
    rows = report["runs"]
    out = [
        "# Imagenette runtime policy validation", "",
        "Age starts at the scheduled arrival, including upstream waiting in Block.",
        "Ingress holds only image IDs. All emitted images were decoded and inferred with ResNet50.",
        "Order violations count departures from the next expected dense submission ID.",
        "Input pool holds the reusable CPU FP32 tensor through the ORT callback.",
        "The lifetime test hashes staged tensors before submission and at callback.",
        "The paced policy runs retain the same pool without per-image hash overhead.", "",
        "| Reorder | Emitted | Order violations | Pool peak / capacity | Acquire / release | Outstanding | Top-5 parity |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        if row["kind"] != "ordering":
            continue
        pool = row["pool"]
        out.append(f"| {row['reorder']} #{row['repeat']} | {row['emitted']} | "
                   f"{row['output_order_violations']} | {pool['peak']} / {pool['capacity']} | "
                   f"{pool['acquired']} / {pool['released']} | {pool['outstanding']} | "
                   f"{row['top5']['identical']} |")
    out.extend(["", "| Target images/s | Policy | Emitted mean | Dropped mean | Age p95 mean (ms) | Producer block mean (ms) |",
                "|---:|---|---:|---:|---:|---:|"])
    for target in sorted({r["target_images_per_s"] for r in rows if r["ingress"]}):
        for policy in ("block", "drop-oldest", "drop-newest"):
            matching = [r for r in rows if r["target_images_per_s"] == target and r["policy"] == policy]
            if matching:
                summary = [statistics.mean(values) for values in zip(*[
                    (r["emitted"], r["dropped"], r["ingress"]["age_p95_ms"],
                     r["ingress"]["blocked_ms"]) for r in matching
                ])]
                out.append(f"| {target:g} | {policy} | {summary[0]:.1f} | {summary[1]:.1f} | "
                           f"{summary[2]:.2f} | {summary[3]:.2f} |")
    return "\n".join(out) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="models/resnet50_224.onnx")
    p.add_argument("--source", default="data/imagenette-320-subset/manifest.json")
    p.add_argument("--reference", required=True)
    p.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--frames", type=int, default=500)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--inflight", type=int, default=16)
    p.add_argument("--queue-size", type=int, default=4)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--rates", type=float, nargs="+", default=[100, 800])
    p.add_argument("--out-dir")
    args = p.parse_args()
    if min(args.frames, args.warmup, args.inflight, args.queue_size, args.repeat) < 1:
        p.error("frames, warmup, inflight, queue-size and repeat must be positive")
    if args.warmup > args.frames or min(args.rates) <= 0:
        p.error("warmup must not exceed frames and target rates must be positive")
    if len(set(args.rates)) != len(args.rates) or not all(np.isfinite(args.rates)):
        p.error("target rates must be distinct and finite")

    source = ImageListSource(args.source, args.frames)
    reference_path = Path(args.reference)
    reference = {int(i): top for i, top in json.loads(reference_path.read_text(encoding="utf-8"))}
    meta = json.loads(reference_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    model_hash = hashlib.sha256(Path(args.model).read_bytes()).hexdigest()
    if model_hash != meta["model_sha256"] or source.manifest_sha != meta["source"]["manifest_sha256"]:
        p.error("reference model or manifest differs; regenerate the reference")
    if not set(range(source.n)) <= reference.keys():
        p.error("reference does not cover all requested image IDs")

    backend = OnnxBackend(args.model, args.backend, 4)
    if backend.fell_back:
        raise RuntimeError(f"Requested {args.backend}; got {backend.actual_providers}")
    warm = ImageListSource(args.source, args.warmup)
    warmer = ValidationPipeline(backend, config(args, "on"))
    warm_rec = Recorder(warmup=0, stages=SERVICE, wait_stages=WAITS, pipelined=True)
    warm_info = warmer.run(warm, warm_rec)
    if warm_info["errors"] or warm_info["stalls"] or warm_rec.counted != args.warmup:
        raise RuntimeError(f"Warmup failed: {warm_info}")

    out = Path(args.out_dir or ("results/image-runtime-validation/" +
                               datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")))
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "args": vars(args), "source": source.describe(), "model_sha256": model_hash,
        "backend": backend.describe(), "env": collect(), "runs": [],
    }
    for repeat in range(1, args.repeat + 1):
        jobs = [("ordering", None, "block", reorder) for reorder in ("off", "on")]
        jobs.extend(("ingress", rate, policy, "on") for rate in args.rates
                    for policy in ("block", "drop-oldest", "drop-newest"))
        random.Random(20261122 + repeat).shuffle(jobs)
        for kind, rate, policy, reorder in jobs:
            row = run_one(backend, source, reference, args, kind, repeat, rate, policy, reorder)
            report["runs"].append(row)
            report["passed"] = all(r["healthy"] for r in report["runs"])
            (out / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"{kind} repeat={repeat} rate={rate} policy={policy} reorder={reorder}: "
                  f"{row['emitted']} emitted, {row['dropped']} dropped, "
                  f"{row['output_order_violations']} order violations", flush=True)
            if not row["healthy"]:
                raise RuntimeError(f"Validation failed; see {out / 'results.json'}")
    (out / "SUMMARY.md").write_text(make_summary(report), encoding="utf-8")
    print(f"Completed: {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
