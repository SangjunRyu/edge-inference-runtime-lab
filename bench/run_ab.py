"""Paired sync vs async comparison.

The claim this project exists to support is "overlapping the pipeline makes it
faster". Two numbers taken an hour apart cannot support it: this machine's
throughput moves 10-30% between runs on its own, which is larger than the
effect. So sync and async are run **alternately, in one process, sharing one
inference session**, and the result reported is the ratio within each pair.
Drift that would corrupt two absolute numbers largely cancels in a ratio taken
seconds apart.

Two rules are enforced rather than hoped for:

* **Order is verified on every run, both modes.** An async pipeline that emits
  frames out of submission order is not a faster pipeline, it is a different
  and wrong one, so a throughput figure from a run with order violations is
  refused rather than reported with a caveat.
* **The pair is the unit.** Each sync run is compared with the async run that
  followed it, and the spread of those per-pair ratios is reported. If the
  ratios disagree with each other, the machine is not steady enough to support
  the claim, whatever the mean says.

    python -m bench.run_ab --model models/resnet50_224.onnx --backend cpu \\
        --source clips/bbb_360_2MB.mp4 --inflight 4 --pairs 5 --tag ab
"""
from __future__ import annotations

import argparse
import json
import statistics
from argparse import Namespace
from pathlib import Path

from bench import env_fingerprint, noise, run_async, run_sync
from bench.backends import OnnxBackend, available


def parse_args():
    p = argparse.ArgumentParser(description="paired sync vs async comparison")
    p.add_argument("--model", required=True)
    p.add_argument("--backend", default="cpu")
    p.add_argument("--ov-device", default=None)
    p.add_argument("--source", default=None)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--intra-threads", type=int, default=None)
    p.add_argument("--inflight", type=int, default=4)
    p.add_argument("--queue-size", type=int, default=4)
    p.add_argument("--reorder", choices=("on", "off"), default="on")
    p.add_argument("--reorder-cap", type=int, default=None)
    p.add_argument("--stall-every", type=int, default=0)
    p.add_argument("--stall-ms", type=float, default=40.0)
    p.add_argument("--pairs", type=int, default=5,
                   help="sync+async pairs; >=3 to get a usable spread")
    p.add_argument("--cooldown", type=float, default=20.0,
                   help="pause between every run, so neither mode inherits the "
                        "other's heat")
    p.add_argument("--max-busy-pct", type=float, default=25.0)
    p.add_argument("--no-idle-gate", action="store_true")
    p.add_argument("--allow-fallback", action="store_true")
    p.add_argument("--tag", default="ab")
    p.add_argument("--out-dir", default="results")
    return p.parse_args()


def _child_args(args) -> Namespace:
    """The subset of the arguments the two measure_once functions read."""
    return Namespace(
        source=args.source, frames=args.frames, warmup=args.warmup,
        top_k=args.top_k, queue_size=args.queue_size, inflight=args.inflight,
        reorder=args.reorder, reorder_cap=args.reorder_cap,
        stall_every=args.stall_every, stall_ms=args.stall_ms,
    )


def main() -> int:
    args = parse_args()
    ca = _child_args(args)

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
          f"inflight={args.inflight} reorder={args.reorder}")

    gate = {"checked": False, "reason": "disabled"}
    if not args.no_idle_gate:
        print("idle gate:")
        gate = noise.idle_gate(max_busy_pct=args.max_busy_pct)

    pairs = []
    for i in range(max(1, args.pairs)):
        print(f"\n--- pair {i + 1}/{args.pairs} ---")

        s_rec, s_src, s_order = run_sync.measure_once(backend, ca, height, width)
        s = s_rec.result()
        s["order"] = s_order
        print(f"  sync   {s['throughput_fps']:7.2f} FPS   "
              f"E2E p50 {s['e2e_ms']['p50']:6.2f} ms   "
              f"order violations {s_order['order_violations']}")
        noise.cooldown(args.cooldown, verbose=False)

        a_rec, a_src, a_info = run_async.measure_once(backend, ca, height, width)
        a = a_rec.result()
        a.update({k: a_info[k] for k in ("order", "reorder", "errors")})
        print(f"  async  {a['throughput_fps']:7.2f} FPS   "
              f"E2E p50 {a['e2e_ms']['p50']:6.2f} ms   "
              f"order violations {a['order']['order_violations']}")
        noise.cooldown(args.cooldown, verbose=False)

        pairs.append({
            "pair": i + 1, "sync": s, "async": a,
            "throughput_ratio": (a["throughput_fps"] / s["throughput_fps"]
                                 if s["throughput_fps"] else 0.0),
            "e2e_p50_ratio": (a["e2e_ms"]["p50"] / s["e2e_ms"]["p50"]
                              if s["e2e_ms"]["p50"] else 0.0),
            "clock_mhz": noise.clock(),
        })

    bad = [p for p in pairs
           if p["async"]["order"]["order_violations"]
           or p["sync"]["order"]["order_violations"]]
    tp = [p["throughput_ratio"] for p in pairs]
    lat = [p["e2e_p50_ratio"] for p in pairs]
    tp_s, lat_s = noise.spread(tp), noise.spread(lat)

    w = 70
    print("\n" + "=" * w)
    print(f" {args.pairs} PAIRS, same session, alternating")
    print("-" * w)
    print(f" {'pair':<6}{'sync FPS':>10}{'async FPS':>11}{'x':>8}"
          f"{'sync p50':>10}{'async p50':>11}{'x':>8}")
    for p in pairs:
        print(f" {p['pair']:<6}{p['sync']['throughput_fps']:>10.2f}"
              f"{p['async']['throughput_fps']:>11.2f}"
              f"{p['throughput_ratio']:>8.2f}"
              f"{p['sync']['e2e_ms']['p50']:>10.2f}"
              f"{p['async']['e2e_ms']['p50']:>11.2f}"
              f"{p['e2e_p50_ratio']:>8.2f}")
    print("-" * w)
    print(f" throughput  x{tp_s['mean']:.2f}  "
          f"(min {tp_s['min']:.2f}, max {tp_s['max']:.2f}, CV "
          f"{100 * tp_s['cv']:.1f}%)")
    print(f" E2E p50     x{lat_s['mean']:.2f}  "
          f"(min {lat_s['min']:.2f}, max {lat_s['max']:.2f}, CV "
          f"{100 * lat_s['cv']:.1f}%)")
    print("-" * w)

    if bad:
        print(f" REFUSED: {len(bad)} of {len(pairs)} pairs emitted frames out "
              f"of submission order.")
        print(" A pipeline that reorders its output is not a faster pipeline,")
        print(" so no speedup is reported for this configuration. Raise")
        print(" --reorder-cap, or leave --reorder on.")
    elif tp_s["n"] > 1 and tp_s["cv"] > 0.10:
        print(f" WEAK: the per-pair ratios themselves vary by "
              f"{100 * tp_s['cv']:.0f}%. The machine was not steady enough for")
        print(" this comparison; quote the range, not the mean.")
    else:
        med = statistics.median(tp)
        print(f" Async is {med:.2f}x the throughput of sync, with every frame")
        print(f" emitted in submission order, measured back to back in one")
        print(f" session. Latency per frame is {statistics.median(lat):.2f}x.")
    print("=" * w)

    meta = {"tag": args.tag, "mode": "ab", "args": vars(args),
            "backend": backend.describe(), "idle_gate": gate,
            "env": env_fingerprint.collect()}
    dev_slug = f"-{backend.ov_device.lower()}" if backend.ov_device else ""
    src_slug = Path(args.source).stem if args.source else "synthetic"
    out = (Path(args.out_dir) /
           f"{args.tag}__ab{args.inflight}__{args.backend}{dev_slug}"
           f"__{Path(args.model).stem}__{src_slug}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"meta": meta, "pairs": pairs,
         "throughput_ratio": tp_s, "e2e_p50_ratio": lat_s,
         "order_clean": not bad}, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
