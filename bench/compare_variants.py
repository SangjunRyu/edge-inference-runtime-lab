"""Paired comparison of two C++ build configurations.

Phase 3 measures optimisations, and an optimisation measured against a number
from ten minutes ago is not measured at all. Running four variants back to back
on this machine produced `--reuse-tensor` apparently making *preprocessing*
2.3x faster, which it cannot, and inference swinging 1.9x between variants,
which the flags do not touch. That was drift, not a result.

So the two configurations alternate within one session and the ratio inside
each pair is what gets reported -- the same discipline `run_ab` applies to
sync-versus-async and `compare_impl` to C++-versus-Python.

The stage that neither variant changes is the control. If `--fused` appears to
alter `inference`, or `--reuse-tensor` to alter `read`, the pairing has not
cancelled the drift and nothing else in the run is trustworthy.

    python -m bench.compare_variants --cpp cpp/build/run_sync_cpp.exe \\
        --model models/resnet50_224.onnx --device NPU \\
        --source clips/bbb_1080_10MB.mp4 --variant --fused --pairs 4
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

from bench import env_fingerprint, noise

STAGES = ("read", "preprocess", "inference", "postprocess")


def parse_args():
    p = argparse.ArgumentParser(description="paired C++ variant comparison")
    p.add_argument("--cpp", default="cpp/build/run_sync_cpp.exe",
                   help="binary for the A side")
    p.add_argument("--cpp-b", default=None,
                   help="binary for the B side; defaults to --cpp. Giving a "
                        "different one pairs two programs rather than two flag "
                        "sets -- which is how the sequential and overlapped "
                        "runners get compared without the drift that made "
                        "their separate runs incomparable.")
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="CPU")
    p.add_argument("--source", default=None)
    p.add_argument("--frames", type=int, default=40000)
    p.add_argument("--duration-s", type=float, default=8.0)
    p.add_argument("--warmup-s", type=float, default=2.0)
    # Taken as one string, not nargs: argparse stops a nargs list at the first
    # token beginning with a dash, which is every flag worth testing.
    p.add_argument("--baseline", default="",
                   help='extra flags for the A side, e.g. --baseline ""')
    p.add_argument("--variant", required=True,
                   help='the flags under test as one string, '
                        'e.g. --variant "--fused"')
    p.add_argument("--control-stage", default=None,
                   help="stage neither side should change; default is picked "
                        "from the variant flags. Pass 'none' when no stage "
                        "qualifies -- comparing the sequential and overlapped "
                        "runners changes every host stage by construction, and "
                        "a control that the comparison itself moves reports "
                        "drift that is not there.")
    p.add_argument("--pairs", type=int, default=4)
    p.add_argument("--cooldown", type=float, default=15.0)
    p.add_argument("--tag", default="variant")
    p.add_argument("--out-dir", default="results")
    return p.parse_args()


def run_cpp(args, extra: list[str], tmpdir: str, exe_path: str = None) -> dict:
    tag = "_var"
    exe = str(Path(exe_path or args.cpp).resolve())
    cmd = [exe, "--model", str(Path(args.model).resolve()),
           "--device", args.device, "--frames", str(args.frames),
           "--duration-s", str(args.duration_s), "--warmup-s", str(args.warmup_s),
           "--tag", tag, "--out-dir", tmpdir] + list(extra)
    if args.source:
        cmd += ["--source", str(Path(args.source).resolve())]

    env = dict(os.environ)
    env["PATH"] = os.pathsep.join(
        [str(Path(exe).parent), r"C:\msys64\ucrt64\bin", env.get("PATH", "")])
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"run failed ({r.returncode}):\n{r.stdout[-1500:]}\n"
                           f"{r.stderr[-1500:]}")
    hits = sorted(Path(tmpdir).glob(f"{tag}__*.json"))
    if not hits:
        raise RuntimeError("no result file")
    d = json.loads(hits[0].read_text(encoding="utf-8"))
    hits[0].unlink()
    return d["runs"][0]


def main() -> int:
    args = parse_args()
    if not Path(args.cpp).exists():
        print(f"C++ binary not found: {args.cpp}")
        return 1

    args.baseline = args.baseline.split()
    args.variant = args.variant.split()
    if not args.variant and not args.cpp_b:
        print("--variant must name at least one flag, or --cpp-b a binary")
        return 1

    control = args.control_stage
    if control is None:
        # A flag that touches preprocessing must leave inference alone, and one
        # that touches the inference call must leave decoding alone.
        control = "inference" if "--fused" in args.variant else "read"

    print(f"{Path(args.model).name} | {args.device} | {args.pairs} pairs")
    print(f"  A: {Path(args.cpp).name} {' '.join(args.baseline)}")
    print(f"  B: {Path(args.cpp_b or args.cpp).name} {' '.join(args.variant)}")
    print(f"  control stage: {control} (neither side should change it)")

    tmpdir = tempfile.mkdtemp(prefix="var_")
    pairs = []
    for i in range(max(1, args.pairs)):
        print(f"\n--- pair {i + 1}/{args.pairs} ---")
        a = run_cpp(args, args.baseline, tmpdir, args.cpp)
        print(f"  A  {a['throughput_fps']:7.2f} FPS  E2E p50 {a['e2e_ms']['p50']:6.3f}"
              f"  drift {a['drift']['ratio']:.2f}x")
        noise.cooldown(args.cooldown, verbose=False)
        b = run_cpp(args, args.variant, tmpdir, args.cpp_b or args.cpp)
        print(f"  B  {b['throughput_fps']:7.2f} FPS  E2E p50 {b['e2e_ms']['p50']:6.3f}"
              f"  drift {b['drift']['ratio']:.2f}x")
        noise.cooldown(args.cooldown, verbose=False)
        pairs.append({"pair": i + 1, "a": a, "b": b})

    def drifted(r):
        d = r["drift"]["ratio"]
        return d is not None and (d > 1.05 or d < 0.95)

    usable = [q for q in pairs if not drifted(q["a"]) and not drifted(q["b"])]
    skipped = len(pairs) - len(usable)
    src = usable or pairs

    w = 72
    print("\n" + "=" * w)
    print(f" {len(pairs)} PAIRS, alternating"
          + (f"  ({skipped} excluded for drift)" if skipped else ""))
    print("-" * w)
    print(f" {'stage':<14}{'A ms':>10}{'B ms':>10}{'B/A':>9}{'per-pair CV':>14}")

    ratios = {}
    for s in STAGES:
        per_pair = [q["b"]["stages_ms"][s]["p50"] / q["a"]["stages_ms"][s]["p50"]
                    for q in src if q["a"]["stages_ms"][s]["p50"] > 0]
        sp = noise.spread(per_pair)
        av = statistics.median(q["a"]["stages_ms"][s]["p50"] for q in src)
        bv = statistics.median(q["b"]["stages_ms"][s]["p50"] for q in src)
        ratios[s] = sp
        print(f" {s:<14}{av:>10.3f}{bv:>10.3f}{sp['mean']:>9.2f}"
              f"{100 * sp['cv']:>13.1f}%")

    tp = noise.spread([q["b"]["throughput_fps"] / q["a"]["throughput_fps"]
                       for q in src if q["a"]["throughput_fps"]])
    print("-" * w)
    if len(src) < 2:
        print(f" throughput    B/A x{tp['mean']:.2f}  from ONE usable pair of "
              f"{len(pairs)}: an observation, not a measurement")
    else:
        print(f" throughput    B/A x{tp['mean']:.2f}  "
              f"(min {tp['min']:.2f}, max {tp['max']:.2f}, "
              f"CV {100 * tp['cv']:.1f}%)")

    # The control: a flag that should not touch this stage had better not.
    if control == "none":
        print(" No control stage: this comparison changes every host stage by")
        print(" construction, so nothing here is untouched by the effect being")
        print(" measured. Read the ratios as the design difference plus"
              " whatever")
        print(" drift the pairing did not cancel, and weigh them by the"
              " per-pair CV.")
    c = ratios.get(control)
    if c and control != "none":
        drift_left = abs(c["mean"] - 1.0)
        print(f" control       {control} B/A x{c['mean']:.2f}"
              f"  <- neither side changes this")
        if drift_left > 0.10:
            print(f" <-- CHECK: the control stage moved {100 * drift_left:.0f}%."
                  f" The pairing did not cancel the drift,")
            print("     so the other rows are describing the machine, not the flag.")
        else:
            print(" The control held, so the rows above are the flag rather than"
                  " the machine.")
    print("=" * w)

    out = (Path(args.out_dir) /
           f"{args.tag}__variant__{args.device.lower()}"
           f"__{Path(args.model).stem}"
           f"__{Path(args.source).stem if args.source else 'synthetic'}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"meta": {"tag": args.tag, "mode": "variant", "args": vars(args),
                  "control_stage": control, "pairs_excluded_for_drift": skipped,
                  "env": env_fingerprint.collect()},
         "pairs": pairs, "stage_ratio_b_over_a": ratios,
         "throughput_ratio": tp}, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
