"""In-flight depth sweep.

W2 showed that overlapping pays when inference is off the host. This asks how
far to push it: each extra request in flight buys throughput until the slowest
stage saturates, after which it buys nothing and still costs latency, because
the extra frames are queueing rather than being worked on.

The depth is the only variable. Queue sizes stay at DX-APP's 4 for every point,
so the curve reflects in-flight requests and not buffering. Depth 1 is the
baseline: the async machinery with no concurrency, which should land near the
sequential pipeline.

Two numbers decide where the knee is:

* **marginal throughput** per doubling, compared against the run-to-run spread
  at that depth. A gain smaller than the noise is not a gain.
* **latency paid** for it. Past saturation, E2E grows roughly linearly with
  depth while throughput is flat -- queueing, by Little's Law.

Any depth that emits frames out of submission order is reported as failed and
excluded from the curve rather than plotted with an asterisk.

    python -m bench.run_sweep --model models/resnet50_224.onnx \\
        --backend openvino --ov-device NPU --source clips/bbb_1080_10MB.mp4 \\
        --depths 1 2 4 8 16 --repeat 3 --tag sweep-npu
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

from bench import env_fingerprint, noise, run_async
from bench.backends import OnnxBackend, available


def parse_args():
    p = argparse.ArgumentParser(description="in-flight depth sweep")
    p.add_argument("--model", required=True)
    p.add_argument("--backend", default="cpu")
    p.add_argument("--ov-device", default=None)
    p.add_argument("--source", default=None)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--duration-s", type=float, default=0.0,
                   help="measure each point for this long; equalises wall-clock "
                        "and thermal exposure across depths")
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--intra-threads", type=int, default=None)
    p.add_argument("--depths", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    p.add_argument("--queue-size", type=int, default=4,
                   help="held fixed across the sweep so depth is the only variable")
    p.add_argument("--reorder", choices=("on", "off"), default="on")
    p.add_argument("--reorder-cap", type=int, default=None)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--cooldown", type=float, default=15.0)
    p.add_argument("--between-depths", type=float, default=20.0)
    p.add_argument("--max-busy-pct", type=float, default=25.0)
    p.add_argument("--no-idle-gate", action="store_true")
    p.add_argument("--allow-fallback", action="store_true")
    p.add_argument("--stall-timeout-s", type=float, default=20.0,
                   help="passed to run_async: give up on an outstanding "
                        "request after this long")
    p.add_argument("--in-process", action="store_true",
                   help="run every depth in this process (default: one "
                        "subprocess per depth)")
    p.add_argument("--depth-timeout-s", type=float, default=0.0,
                   help="kill a depth that exceeds this; 0 = computed")
    p.add_argument("--tag", default="sweep")
    p.add_argument("--out-dir", default="results")
    return p.parse_args()


def _child_args(args, depth: int) -> Namespace:
    return Namespace(
        source=args.source, frames=args.frames, warmup=args.warmup,
        duration_s=args.duration_s,
        top_k=args.top_k, queue_size=args.queue_size, inflight=depth,
        reorder=args.reorder, reorder_cap=args.reorder_cap,
        stall_every=0, stall_ms=0.0,
    )


def _depth_budget(args) -> float:
    """How long a single depth may take before it is killed.

    A stalled request cannot be cancelled -- ONNX Runtime has no API for it --
    so the watchdog inside run_async only stops *waiting*; the request stays in
    the session. Isolating each depth in its own process is what actually
    contains that, and this budget is the outer bound: if a depth exceeds it,
    the process is killed and the depth is dropped rather than left to poison
    the rest of the sweep.
    """
    if args.depth_timeout_s > 0:
        return args.depth_timeout_s
    per_run = (args.duration_s or 12.0) + args.stall_timeout_s + 25.0
    return args.repeat * per_run + (args.repeat - 1) * args.cooldown + 60.0


def _run_depth_isolated(args, depth: int, height: int, width: int,
                        tmpdir: str) -> tuple[list[dict], str | None]:
    """Run one depth in a fresh process. Returns (runs, failure reason)."""
    tag = f"_sweepdepth{depth}"
    cmd = [sys.executable, "-u", "-m", "bench.run_async",
           "--model", args.model, "--backend", args.backend,
           "--frames", str(args.frames), "--warmup", str(args.warmup),
           "--top-k", str(args.top_k), "--inflight", str(depth),
           "--queue-size", str(args.queue_size), "--reorder", args.reorder,
           "--repeat", str(args.repeat), "--cooldown", str(args.cooldown),
           "--stall-timeout-s", str(args.stall_timeout_s),
           "--no-idle-gate", "--tag", tag, "--out-dir", tmpdir]
    if args.source:
        cmd += ["--source", args.source]
    if args.ov_device:
        cmd += ["--ov-device", args.ov_device]
    if args.reorder_cap is not None:
        cmd += ["--reorder-cap", str(args.reorder_cap)]
    if args.intra_threads is not None:
        cmd += ["--intra-threads", str(args.intra_threads)]
    if args.duration_s:
        cmd += ["--duration-s", str(args.duration_s)]
    if args.allow_fallback:
        cmd += ["--allow-fallback"]

    budget = _depth_budget(args)
    try:
        rc = subprocess.run(cmd, timeout=budget).returncode
    except subprocess.TimeoutExpired:
        return [], f"killed after {budget:.0f}s"
    if rc != 0:
        return [], f"exited {rc}"

    hits = sorted(Path(tmpdir).glob(f"{tag}__*.json"))
    if not hits:
        return [], "produced no result file"
    d = json.loads(hits[0].read_text(encoding="utf-8"))
    return d.get("runs", []), None


def main() -> int:
    args = parse_args()

    print(f"available backends: {available()}")
    backend = OnnxBackend(args.model, args.backend, args.intra_threads,
                          args.ov_device)
    if backend.fell_back and not args.allow_fallback:
        print(f"!! requested '{args.backend}' but running on "
              f"{backend.actual_providers}; aborting.")
        return 2
    height, width = backend.hw
    dev = f"/{backend.ov_device}" if backend.ov_device else ""
    print(f"{Path(args.model).name} | {args.backend}{dev} | "
          f"queue_size={args.queue_size} fixed | depths {args.depths}")

    gate = {"checked": False, "reason": "disabled"}
    if not args.no_idle_gate:
        print("idle gate:")
        gate = noise.idle_gate(max_busy_pct=args.max_busy_pct)

    isolated = not args.in_process
    if isolated:
        print(f"one subprocess per depth, budget {_depth_budget(args):.0f}s each")
    tmpdir = tempfile.mkdtemp(prefix="sweep_")

    points, killed = [], []
    for di, depth in enumerate(args.depths):
        print(f"\n--- inflight {depth} ---")
        if isolated:
            runs, why = _run_depth_isolated(args, depth, height, width, tmpdir)
            if why:
                print(f"  !! depth {depth} dropped: {why}")
                killed.append([depth, why])
                if di < len(args.depths) - 1:
                    noise.cooldown(args.between_depths, verbose=False)
                continue
        else:
            runs = []
            for i in range(max(1, args.repeat)):
                rec, src, info = run_async.measure_once(
                    backend, _child_args(args, depth), height, width)
                info["source"] = src
                r = rec.result()
                r.update({k: info[k] for k in
                          ("order", "reorder", "errors", "stalls", "source")})
                runs.append(r)
                print(f"  run {i + 1}: {r['throughput_fps']:7.2f} FPS   "
                      f"E2E p50 {r['e2e_ms']['p50']:7.2f} ms   "
                      f"resident {r['concurrency_littles_law']:5.2f}   "
                      f"violations {r['order']['order_violations']}")
                if i < args.repeat - 1:
                    noise.cooldown(args.cooldown, verbose=False)

        # A run that produced no counted frames is not a slow run either: at a
        # low duration the warm-up can swallow the whole budget, leaving a
        # throughput of 0.0 that would otherwise be averaged in as if real.
        empty = [r for r in runs if not r.get("frames_counted")]
        # A stalled run is not a slow run. Averaging one in produces a point
        # that describes neither the stall nor the steady state: the Arc iGPU
        # sweep reported 144 FPS at depth 8 as the mean of 0.2, 149.7 and
        # 283.4, which is a number no configuration ever achieved.
        stalled = [r for r in runs if r.get("stalls")]
        clean = [r for r in runs
                 if not r.get("stalls") and r.get("frames_counted")]
        if empty:
            print(f"  !! {len(empty)} run(s) counted no frames -- warm-up "
                  f"({args.warmup}) may exceed what --duration-s allows")
        if not clean:
            print(f"  !! depth {depth} has no usable run; dropping")
            killed.append([depth, "no clean run"])
            if di < len(args.depths) - 1:
                noise.cooldown(args.between_depths, verbose=False)
            continue

        tp = [r["throughput_fps"] for r in clean]
        e50 = [r["e2e_ms"]["p50"] for r in clean]
        e95 = [r["e2e_ms"]["p95"] for r in clean]
        viol = sum(r["order"]["order_violations"] for r in runs)
        points.append({
            "inflight": depth,
            "throughput": noise.spread(tp),
            "e2e_p50": noise.spread(e50),
            "e2e_p95": noise.spread(e95),
            "resident": statistics.median(
                r["concurrency_littles_law"] for r in clean),
            "q_inflight_p50": statistics.median(
                r["stages_ms"]["q_inflight"]["p50"] for r in clean),
            "inference_p50": statistics.median(
                r["stages_ms"]["inference"]["p50"] for r in clean),
            "inference_max_ms": max(
                r["stages_ms"]["inference"]["max"] for r in runs),
            "order_violations": viol,
            "stalled_runs": len(stalled),
            "clean_runs": len(clean),
            "source": runs[0].get("source"),
            "seconds_measured": statistics.median(
                r["frames_counted"] / r["throughput_fps"] for r in runs
                if r["throughput_fps"]),
            "runs": runs,
        })
        if di < len(args.depths) - 1:
            noise.cooldown(args.between_depths, verbose=False)

    ok = [p for p in points if not p["order_violations"] and not p["stalled_runs"]]
    bad = [p for p in points if p["order_violations"] or p["stalled_runs"]]

    w = 88
    print("\n" + "=" * w)
    print(f" IN-FLIGHT SWEEP -- {args.backend}{dev}, "
          f"{Path(args.source).stem if args.source else 'synthetic'}")
    print("-" * w)
    print(f" {'depth':>6}{'FPS':>9}{'CV':>7}{'x vs d=1':>10}"
          f"{'E2E p50':>10}{'E2E p95':>10}{'q_infl':>9}{'infer':>8}{'resident':>10}")
    base = ok[0]["throughput"]["mean"] if ok else 0.0
    for p in points:
        flag = ""
        if p["order_violations"]:
            flag += "  ORDER VIOLATIONS"
        if p["stalled_runs"]:
            flag += (f"  {p['stalled_runs']}/{p['stalled_runs'] + p['clean_runs']}"
                     f" RUNS STALLED (worst inference "
                     f"{p['inference_max_ms'] / 1000:.0f}s)")
        rel = (p["throughput"]["mean"] / base) if base else 0.0
        print(f" {p['inflight']:>6}{p['throughput']['mean']:>9.1f}"
              f"{100 * p['throughput']['cv']:>6.1f}%{rel:>10.2f}"
              f"{p['e2e_p50']['mean']:>10.1f}{p['e2e_p95']['mean']:>10.1f}"
              f"{p['q_inflight_p50']:>9.2f}{p['inference_p50']:>8.2f}"
              f"{p['resident']:>10.2f}{flag}")
    print("-" * w)

    # The knee: the last doubling whose throughput gain is distinguishable from
    # zero. Past it the pipeline is buying latency and nothing else.
    #
    # Comparing the gain against the raw CV is wrong twice over: the CV
    # describes one run, not the uncertainty in the mean of several, and taking
    # the worse of the two depths lets one noisy point veto its neighbour. The
    # test is the difference between two means against their combined standard
    # error, at 2 sigma. On the 360p sweep the CV rule called 8 -> 16 a gain at
    # +3.2% against 3.2% noise and reported saturation at 16, when the curve
    # plainly flattens at 4.
    def _rel_sem(sp: dict) -> float:
        n = max(1, sp.get("n", 1))
        return (sp.get("cv", 0.0) / (n ** 0.5)) if n > 1 else sp.get("cv", 0.0)

    knee = None
    for prev, cur in zip(ok, ok[1:]):
        gain = cur["throughput"]["mean"] / prev["throughput"]["mean"] - 1.0
        sem = (_rel_sem(cur["throughput"]) ** 2
               + _rel_sem(prev["throughput"]) ** 2) ** 0.5
        bar = max(2 * sem, 0.02)
        lat = cur["e2e_p50"]["mean"] / prev["e2e_p50"]["mean"] - 1.0
        verdict = "gain" if gain > bar else "NOISE"
        print(f" {prev['inflight']:>3} -> {cur['inflight']:<3}"
              f"  throughput {100 * gain:+6.1f}%  (needs >{100 * bar:.1f}%)"
              f"   latency {100 * lat:+6.1f}%   {verdict}")
        if gain > bar:
            knee = cur["inflight"]
    print("-" * w)
    if knee is None:
        print(" No depth increase produced a throughput gain above its own noise.")
    else:
        kp = next(p for p in ok if p["inflight"] == knee)
        deepest = ok[-1]
        print(f" Throughput saturates at inflight={knee}"
              f" ({kp['throughput']['mean']:.1f} FPS).")
        if deepest["inflight"] != knee:
            dl = deepest["e2e_p50"]["mean"] / kp["e2e_p50"]["mean"]
            dt = deepest["throughput"]["mean"] / kp["throughput"]["mean"]
            print(f" Going on to {deepest['inflight']} buys {100 * (dt - 1):+.1f}%"
                  f" throughput for {dl:.2f}x the latency.")
    if killed:
        print(f" Depths killed or failed outright: {killed}")
    if bad:
        print(f" Excluded from the curve: "
              f"{[p['inflight'] for p in bad]}"
              f"  (order violations or stalled runs)")
    print("=" * w)

    meta = {"tag": args.tag, "mode": "sweep", "args": vars(args),
            "backend": backend.describe(), "idle_gate": gate,
            "env": env_fingerprint.collect()}
    dev_slug = f"-{backend.ov_device.lower()}" if backend.ov_device else ""
    src_slug = Path(args.source).stem if args.source else "synthetic"
    out = (Path(args.out_dir) /
           f"{args.tag}__sweep__{args.backend}{dev_slug}"
           f"__{Path(args.model).stem}__{src_slug}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    meta["isolated_per_depth"] = isolated
    meta["depth_budget_s"] = _depth_budget(args)
    out.write_text(json.dumps({"meta": meta, "points": points, "knee": knee,
                               "killed_depths": killed}, indent=2),
                   encoding="utf-8")
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
