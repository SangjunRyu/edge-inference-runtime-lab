"""Paired comparison of two C++ build configurations.

Phase 3 measures optimisations, and an optimisation measured against a number
from ten minutes ago is not measured at all. Running four variants back to back
on this machine produced `--reuse-tensor` apparently making *preprocessing*
2.3x faster, which it cannot, and inference swinging 1.9x between variants,
which the flags do not touch. That was drift, not a result.

So the two configurations alternate within one session and the ratio inside
each pair is what gets reported -- the same discipline `run_ab` applies to
sync-versus-async and `compare_impl` to C++-versus-Python.

The stage that neither variant changes is the control, and what it has to hold
is its **spread across pairs**, not its level. A control detects the machine
changing between the two halves of a pair; a level that reproduces to a few
percent is a constant, and a constant is a property of the two configurations
rather than evidence of drift. Judging the level was the mistake the sequential
comparison made -- it declared a control held at 0.92 whose pairs ran 0.72 to
1.01, and it took six counterbalanced pairs to find out that 0.86 was simply
what two different runtimes cost.

The order inside a pair alternates, for the same reason: whatever warms up
between the halves of a pair otherwise lands on the same side every time, and a
paired design cannot cancel a bias it applies uniformly.

Both corrections arrived after phase 3 was measured, so `--reanalyse` re-reads a
saved run under the current criteria. The pairs are stored in full; correcting a
criterion therefore costs no machine time.

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

from argparse import Namespace

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
    p.add_argument("--model", default=None)
    p.add_argument("--device", default="CPU")
    p.add_argument("--source", default=None)
    p.add_argument("--frames", type=int, default=40000)
    p.add_argument("--duration-s", type=float, default=8.0)
    p.add_argument("--warmup-s", type=float, default=2.0)
    # Taken as one string, not nargs: argparse stops a nargs list at the first
    # token beginning with a dash, which is every flag worth testing.
    p.add_argument("--baseline", default="",
                   help='extra flags for the A side, e.g. --baseline ""')
    p.add_argument("--variant", default=None,
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
    p.add_argument("--no-alternate-order", dest="alternate_order",
                   action="store_false",
                   help="run A first in every pair. The default alternates, "
                        "because whatever warms up inside a pair otherwise "
                        "lands entirely on the second half and it is always "
                        "the same half")
    p.add_argument("--reanalyse", default=None, metavar="RESULT.json",
                   help="re-report a saved run under the current criteria "
                        "instead of measuring")
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
    if args.reanalyse:
        saved = json.loads(Path(args.reanalyse).read_text(encoding="utf-8"))
        sa = dict(saved["meta"]["args"])
        sa["reanalyse"] = args.reanalyse
        sa.setdefault("alternate_order", False)
        control = saved["meta"].get("control_stage") or "none"
        print(f"re-reading {args.reanalyse}: {len(saved['pairs'])} pairs,"
              f" control `{control}`, no machine time spent")
        return report(Namespace(**sa), saved["pairs"], control, reanalysed=True)
    if args.model is None or args.variant is None:
        print("--model and --variant are required unless --reanalyse is given")
        return 1
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

    def show(label, r):
        print(f"  {label}  {r['throughput_fps']:7.2f} FPS  "
              f"E2E p50 {r['e2e_ms']['p50']:6.3f}  "
              f"drift {r['drift']['ratio']:.2f}x")
        return r

    pairs = []
    for i in range(max(1, args.pairs)):
        # The order inside a pair alternates. With A always first, anything
        # that warms up between the halves is charged to B every time, and a
        # paired design cannot cancel a bias it applies uniformly.
        a_first = (i % 2 == 0) or not args.alternate_order
        print(f"\n--- pair {i + 1}/{args.pairs} "
              f"({'A first' if a_first else 'B first'}) ---")
        halves = [("a", lambda: show("A", run_cpp(args, args.baseline, tmpdir,
                                                  args.cpp))),
                  ("b", lambda: show("B", run_cpp(args, args.variant, tmpdir,
                                                  args.cpp_b or args.cpp)))]
        if not a_first:
            halves.reverse()
        got = {}
        for name, fn in halves:
            got[name] = fn()
            noise.cooldown(args.cooldown, verbose=False)
        pairs.append({"pair": i + 1, "a_first": a_first,
                      "a": got["a"], "b": got["b"]})

    return report(args, pairs, control)


def report(args, pairs, control, reanalysed: bool = False) -> int:
    """Everything after the measuring, so a saved run can be re-read."""

    def drifted(r):
        d = r["drift"]["ratio"]
        return d is not None and (d > 1.05 or d < 0.95)

    usable = [q for q in pairs if not drifted(q["a"]) and not drifted(q["b"])]
    skipped = len(pairs) - len(usable)
    # Falling back to every pair when none is admissible is not the same
    # thing as every pair being admissible, and the header has to say which.
    src = usable or pairs
    fell_back = not usable

    w = 78
    orders = {q.get("a_first", True) for q in pairs}
    order_note = ("order counterbalanced" if len(orders) > 1
                  else "A first in every pair")
    print("\n" + "=" * w)
    print(f" {len(pairs)} PAIRS, alternating, {order_note}"
          + ("  (NO pair held steady within itself; reporting all"
                          f" {len(pairs)} anyway)" if fell_back
                             else f"  ({skipped} excluded for drift)" if skipped
                             else ""))
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

    def tp_of(group):
        return noise.spread([q["b"]["throughput_fps"] / q["a"]["throughput_fps"]
                             for q in group if q["a"]["throughput_fps"]])

    tp = tp_of(src)
    print("-" * w)
    if len(src) < 2:
        print(f" throughput    B/A x{tp['mean']:.2f}  from ONE pair:"
              f" an observation, not a measurement")
    else:
        print(f" throughput    B/A x{tp['mean']:.2f}  "
              f"(min {tp['min']:.2f}, max {tp['max']:.2f}, "
              f"CV {100 * tp['cv']:.1f}%)"
              + ("" if fell_back else f"   {len(src)} steady pair(s)"))
    if skipped and not fell_back:
        # What admissibility removed, so its cost is on the page rather than in
        # a decision nobody can see.
        t2 = tp_of(pairs)
        extra = ""
        if control != "none":
            c2 = noise.spread(
                [q["b"]["stages_ms"][control]["p50"]
                 / q["a"]["stages_ms"][control]["p50"] for q in pairs
                 if q["a"]["stages_ms"][control]["p50"] > 0])
            extra = f"; control x{c2['mean']:.2f} CV {100 * c2['cv']:.0f}%"
        print(f" all {len(pairs)} pairs   B/A x{t2['mean']:.2f}  "
              f"(min {t2['min']:.2f}, max {t2['max']:.2f}, "
              f"CV {100 * t2['cv']:.1f}%)")
        print(f"               including the {skipped} excluded for drift{extra}")

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
        print(f" control       {control} B/A x{c['mean']:.2f}  "
              f"(min {c['min']:.2f}, max {c['max']:.2f},"
              f" CV {100 * c['cv']:.1f}%)")
        # Judged by spread, not level. A control detects the machine *changing*
        # between the halves of a pair, so its spread is the thing that has to
        # hold; a level that reproduces across pairs is a constant difference
        # between the two configurations and is information, not a fault. The
        # sequential comparison learned this the expensive way, declaring a
        # control held at 0.92 whose pairs ran 0.72 to 1.01.
        if len(src) < 2:
            print(" One pair, so the control has no spread to judge and this"
                  " line proves")
            print(" nothing. Run more pairs.")
        elif c["cv"] > 0.10:
            print(f" <-- CHECK: the control moved {100 * c['cv']:.0f}% between"
                  f" pairs. The machine changed")
            print("     between the two halves, so the rows above carry that"
                  " movement as")
            print("     well as the flag. Quote the ranges, not the means.")
        else:
            print(f" The control holds to {100 * c['cv']:.0f}% across pairs, so"
                  f" the pairing cancelled the")
            print(" machine's movement and the rows above are the flag.")
            if abs(c["mean"] - 1.0) > 0.10:
                print(f" Its level is {c['mean']:.2f}, not 1.00. A constant"
                      f" offset on a stage neither")
                print(" side touches is worth explaining, but it is not drift"
                      " -- what would")
                print(" invalidate the rows is the control moving.")
    print("=" * w)

    if reanalysed:
        print("\nre-read only; the saved file is left as it was measured")
        return 0

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
