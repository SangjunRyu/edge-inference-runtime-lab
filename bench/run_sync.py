"""W1 -- sequential pipeline.

The question this run answers, and the only one:

    Where does one frame spend its time?

No threads, no queues. Establishes the baseline that every later number is
compared against, and fixes the stage boundaries so those comparisons mean
something.

Because the instrument is a laptop, a single run is not evidence. Use
``--repeat`` so the result carries its own run-to-run spread: a difference
smaller than that spread is not a finding.

    python -m bench.run_sync --model models/resnet50_224.onnx --backend cpu \
        --source clips/bbb_360_2MB.mp4 --frames 300 --warmup 150 \
        --repeat 3 --cooldown 20 --tag w1-360
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from bench import env_fingerprint, noise
from bench.backends import OnnxBackend, available
from bench.harness import Recorder, Timer
from bench.pipeline import postprocess, preprocess
from bench.reorder import OrderChecker
from bench.sources import make_source


def parse_args():
    p = argparse.ArgumentParser(description="W1 sequential inference pipeline")
    p.add_argument("--model", required=True, help="path to .onnx")
    p.add_argument("--backend", default="cpu", help="cpu | cuda | openvino")
    p.add_argument("--ov-device", default=None,
                   help="openvino backend only: CPU | GPU | NPU")
    p.add_argument("--source", default=None,
                   help="video/image path; omitted = synthetic frames")
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--duration-s", type=float, default=0.0,
                   help="measure for this long instead of for a frame count")
    p.add_argument("--warmup-s", type=float, default=0.0,
                   help="discard this many seconds instead of --warmup frames")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--intra-threads", type=int, default=None,
                   help="ORT intra_op threads; pin it for comparable runs")
    p.add_argument("--repeat", type=int, default=1,
                   help="independent runs; >=3 to get a usable spread")
    p.add_argument("--cooldown", type=float, default=0.0,
                   help="seconds to pause between repeats so heat does not carry over")
    p.add_argument("--max-busy-pct", type=float, default=15.0,
                   help="refuse to start until system CPU is below this")
    p.add_argument("--no-idle-gate", action="store_true")
    p.add_argument("--allow-fallback", action="store_true",
                   help="proceed even if the requested provider is unavailable")
    p.add_argument("--tag", default="run")
    p.add_argument("--out-dir", default="results")
    return p.parse_args()


def measure_once(backend, args, height: int, width: int):
    """One complete pass. The source is rebuilt per run so repeats are independent.

    Order is checked here too. A sequential pipeline cannot emit out of order,
    so this must report zero -- which is what makes the same check meaningful
    when the async pipeline reports zero.
    """
    source = make_source(args.source, args.frames)
    rec = Recorder(warmup=args.warmup, warmup_s=getattr(args, 'warmup_s', 0.0))
    checker = OrderChecker()
    idx = 0
    frames = source.frames()
    # Both modes must stop on the same rule. Giving the async path a deadline
    # and leaving this one on a frame count would make every paired comparison
    # measure the two for different lengths of time.
    deadline = (time.perf_counter() + getattr(args, "duration_s", 0.0)
                if getattr(args, "duration_s", 0.0) else None)
    while True:
        if deadline is not None and time.perf_counter() >= deadline:
            break
        with Timer() as t_e2e:
            # Read is timed by timing the pull from the source. The source
            # generator decodes lazily, so the decode happens inside this
            # block and nowhere else.
            with Timer() as t_read:
                try:
                    frame = next(frames)
                except StopIteration:
                    break
            with Timer() as t_pre:
                tensor = preprocess(frame, height, width)
            with Timer() as t_inf:
                logits = backend.run(tensor)
            with Timer() as t_post:
                _ = postprocess(logits, args.top_k)
            checker.observe(idx)
            idx += 1
        rec.add_frame(
            {"read": t_read.elapsed, "preprocess": t_pre.elapsed,
             "inference": t_inf.elapsed, "postprocess": t_post.elapsed},
            t_e2e.elapsed,
        )
    return rec, source.describe(), checker.stats()


def print_repeat_summary(results: list[dict]) -> dict:
    """Between-run spread. The number that decides what counts as a finding."""
    tp = noise.spread([r["throughput_fps"] for r in results])
    e2e = noise.spread([r["e2e_ms"]["p50"] for r in results])
    ceil = noise.spread([r["overlap_ceiling_x"] for r in results])
    w = 68
    print("\n" + "=" * w)
    print(f" ACROSS {tp['n']} RUNS")
    print("-" * w)
    print(f" {'metric':<18}{'mean':>10}{'min':>10}{'max':>10}{'CV':>10}")
    for name, s in (("throughput FPS", tp), ("E2E p50 ms", e2e),
                    ("overlap ceiling", ceil)):
        print(f" {name:<18}{s['mean']:>10.2f}{s['min']:>10.2f}"
              f"{s['max']:>10.2f}{100 * s['cv']:>9.1f}%")
    print("-" * w)
    cv = tp["cv"]
    print(f" A change smaller than {100 * cv:.1f}% throughput is inside this"
          f" machine's run-to-run noise.")
    if ceil["mean"] > 1.0 and cv > (ceil["mean"] - 1.0):
        print(f" <-- CHECK: noise ({100 * cv:.1f}%) exceeds the best possible"
              f" overlap gain ({100 * (ceil['mean'] - 1):.0f}%)")
    print("=" * w)
    return {"throughput_fps": tp, "e2e_p50_ms": e2e, "overlap_ceiling_x": ceil}


def main() -> int:
    args = parse_args()

    print(f"available backends: {available()}")
    backend = OnnxBackend(args.model, args.backend, args.intra_threads,
                          args.ov_device)
    if backend.fell_back:
        msg = (f"requested '{args.backend}'"
               f"{'/' + args.ov_device if args.ov_device else ''} but the "
               f"session is running on {backend.actual_providers}")
        if args.allow_fallback:
            print(f"!! {msg} -- continuing because --allow-fallback was given")
        else:
            # Refuse rather than warn. A CPU result stored under a GPU or NPU
            # tag is worse than no result: it is wrong and it looks fine.
            print(f"!! {msg}\n"
                  f"   Aborting so a CPU measurement is not saved under the "
                  f"'{args.tag}' tag. Fix the provider, or pass "
                  f"--allow-fallback if you meant to measure the fallback.")
            return 2
    height, width = backend.hw
    print(f"model input {height}x{width}  providers={backend.actual_providers}")

    gate = {"checked": False, "reason": "disabled"}
    if not args.no_idle_gate:
        print("idle gate:")
        gate = noise.idle_gate(max_busy_pct=args.max_busy_pct)

    dev = f"/{backend.ov_device}" if backend.ov_device else ""
    runs, results, sources = [], [], []
    for i in range(max(1, args.repeat)):
        clock_before = noise.clock()
        rec, src, order = measure_once(backend, args, height, width)
        clock_after = noise.clock()
        title = (f"{Path(args.model).name} | {args.backend}{dev} "
                 f"-> {backend.actual_providers[0]} | source={src['kind']}"
                 f" | run {i + 1}/{args.repeat}")
        rec.print_summary(title)
        if clock_before and clock_after:
            print(f" cpu clock     {clock_before:.0f} -> {clock_after:.0f} MHz")
        r = rec.result()
        r["cpu_freq_mhz"] = {"before": clock_before, "after": clock_after}
        r["order"] = order
        print(f" order         emitted {order['emitted']}  violations "
              f"{order['order_violations']}")
        results.append(r)
        runs.append(rec)
        sources.append(src)
        if i < args.repeat - 1:
            noise.cooldown(args.cooldown)

    across = print_repeat_summary(results) if len(results) > 1 else None

    meta = {
        "tag": args.tag,
        "args": vars(args),
        "backend": backend.describe(),
        "sources": sources,
        "model_input_hw": [height, width],
        "idle_gate": gate,
        "env": env_fingerprint.collect(),
    }
    # The source belongs in the filename. Without it a backend sweep and a
    # resolution sweep collide on the same path and silently overwrite each
    # other -- the two axes this project exists to separate.
    dev_slug = "-" + backend.ov_device.lower() if backend.ov_device else ""
    src_slug = Path(args.source).stem if args.source else "synthetic"
    out = (Path(args.out_dir) /
           f"{args.tag}__{args.backend}{dev_slug}__{Path(args.model).stem}"
           f"__{src_slug}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    import json
    out.write_text(json.dumps(
        {"meta": meta, "runs": results, "across_runs": across}, indent=2),
        encoding="utf-8")
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
