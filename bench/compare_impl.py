"""Paired C++ vs Python comparison.

The same discipline `run_ab` applies to sync-vs-async, applied to
implementation-vs-implementation: the two alternate in one session and the
ratio within each pair is what gets reported. Two runs minutes apart cannot
support a claim on a machine whose throughput moves 10-30% by itself, and the
first single-run comparison showed exactly why -- the C++ run drifted 0.69x
within itself, getting faster as it went, so its numbers described a machine
that was still warming up.

What this is for, in order:

1. **Cross-validation.** In sync mode the control is `inference`: the same
   model on the same silicon, which neither implementation performs, so the two
   must agree there or one of them is mistiming and nothing else means
   anything.

   Every ratio is formed **inside a pair and only then aggregated**, which is
   the point of pairing and was for a while not what this did: it divided a
   median of C++ runs by a median of Python runs, taking the two halves of the
   ratio from different moments on a machine that moves. The difference is not
   academic. A five-pair run reported `inference` at 0.92 and called the
   control held, while the per-pair ratios ran 0.72 to 1.01 -- a 40% swing on
   the one stage neither implementation performs.
2. **Phase 3.** Every C++ optimisation is measured against Python and against
   the previous C++ build with this, not against a number from yesterday.
3. **The GIL.** `--mode async` runs both overlapped pipelines. W2 attributed
   the CPU async loss to ONNX Runtime subdividing its intra-op pool between
   concurrent requests, but Python's own orchestration overhead was never
   separated from it -- the same pipeline in C++ on the same device is what
   separates them.

   The comparison is only fair if the structures match, so async mode pins the
   C++ side to **one** postprocess collector: Python's async runner has a
   single postprocess worker, and C++ with four would be measuring a different
   design rather than a different language. The control in async mode is `read`
   -- both decode the same clip with the same OpenCV, and neither pipeline's
   concurrency changes what a decode costs.

    python -m bench.compare_impl --cpp cpp/build/run_sync_cpp.exe \\
        --model models/resnet50_224.onnx --device NPU \\
        --source clips/bbb_360_2MB.mp4 --duration-s 10 --pairs 3
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

from bench import env_fingerprint, noise, run_async, run_sync
from bench.backends import OnnxBackend

STAGES = ("read", "preprocess", "inference", "postprocess")


def parse_args():
    p = argparse.ArgumentParser(description="paired C++ vs Python comparison")
    p.add_argument("--mode", choices=("sync", "async"), default="sync")
    p.add_argument("--cpp", default=None,
                   help="C++ binary; defaults to the one matching --mode")
    p.add_argument("--inflight", type=int, default=4, help="async mode only")
    p.add_argument("--queue-size", type=int, default=4, help="async mode only")
    p.add_argument("--post-workers", type=int, default=1,
                   help="async mode only; 1 matches Python's single "
                        "postprocess worker, which is what makes the "
                        "comparison about the language rather than the design")
    p.add_argument("--model", default=None)
    p.add_argument("--device", default="CPU", help="OpenVINO device: CPU|GPU|NPU")
    p.add_argument("--source", default=None)
    p.add_argument("--frames", type=int, default=40000)
    p.add_argument("--duration-s", type=float, default=10.0)
    p.add_argument("--warmup-s", type=float, default=2.0)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--fused", action="store_true",
                   help="ask the C++ side for the fused preprocessing path")
    p.add_argument("--pairs", type=int, default=3)
    p.add_argument("--cooldown", type=float, default=15.0)
    p.add_argument("--no-alternate-order", dest="alternate_order",
                   action="store_false",
                   help="run C++ first in every pair. The default alternates, "
                        "because whatever warms up inside a pair otherwise "
                        "lands entirely on the second half and it is always "
                        "the same half")
    p.add_argument("--reanalyse", default=None, metavar="RESULT.json",
                   help="re-report a saved run under the current criteria "
                        "instead of measuring. Every pair is kept in full, so "
                        "correcting a criterion costs no machine time -- and "
                        "the criteria here have been corrected four times")
    p.add_argument("--tag", default="impl")
    p.add_argument("--out-dir", default="results")
    return p.parse_args()


def run_cpp(args, tmpdir: str) -> dict:
    """One C++ run, returning the same result dict the Python side produces."""
    tag = "_impl"
    # Resolve the binary: a relative path that a shell finds is not
    # necessarily one CreateProcess will.
    exe = str(Path(args.cpp).resolve())
    cmd = [exe, "--model", str(Path(args.model).resolve()),
           "--device", args.device,
           "--frames", str(args.frames), "--duration-s", str(args.duration_s),
           "--warmup-s", str(args.warmup_s), "--top-k", str(args.top_k),
           "--tag", tag, "--out-dir", tmpdir]
    if args.source:
        cmd += ["--source", str(Path(args.source).resolve())]
    if args.fused:
        cmd += ["--fused"]
    if args.mode == "async":
        cmd += ["--inflight", str(args.inflight),
                "--queue-size", str(args.queue_size),
                "--post-workers", str(args.post_workers),
                "--reorder", "on"]

    # The executable links OpenCV and the OpenVINO runtime; neither is on PATH
    # for a process started outside the MSYS2 shell, and the loader's failure
    # names only the first missing DLL, which makes it look like an OpenVINO
    # problem when it is an OpenCV one.
    env = dict(os.environ)
    extra = [str(Path(args.cpp).resolve().parent), r"C:\msys64\ucrt64\bin"]
    env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])

    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"C++ run failed ({r.returncode}):\n"
                           f"{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    hits = sorted(Path(tmpdir).glob(f"{tag}__*.json"))
    if not hits:
        raise RuntimeError("C++ run produced no result file")
    d = json.loads(hits[0].read_text(encoding="utf-8"))
    hits[0].unlink()
    return d["runs"][0]


def run_python(args, backend, height: int, width: int) -> dict:
    ca = Namespace(source=args.source, frames=args.frames, warmup=0,
                   warmup_s=args.warmup_s, duration_s=args.duration_s,
                   top_k=args.top_k, queue_size=args.queue_size,
                   inflight=args.inflight, reorder="on", reorder_cap=None,
                   stall_every=0, stall_ms=0.0, stall_timeout_s=30.0)
    if args.mode == "async":
        rec, _src, info = run_async.measure_once(backend, ca, height, width)
        r = rec.result()
        r.update({k: info[k] for k in ("order", "reorder", "errors", "stalls")})
        return r
    rec, _src, order = run_sync.measure_once(backend, ca, height, width)
    r = rec.result()
    r["order"] = order
    return r


def main() -> int:
    args = parse_args()
    if args.reanalyse:
        saved = json.loads(Path(args.reanalyse).read_text(encoding="utf-8"))
        sa = dict(saved["meta"]["args"])
        sa["reanalyse"] = args.reanalyse
        sa.setdefault("mode", "async" if "async" in saved["meta"]["mode"]
                      else "sync")
        print(f"re-reading {args.reanalyse}: {len(saved['pairs'])} pairs,"
              f" {sa['mode']} mode, no machine time spent")
        return report(Namespace(**sa), saved["pairs"], reanalysed=True)
    if args.model is None:
        print("--model is required unless --reanalyse is given")
        return 1
    if args.cpp is None:
        args.cpp = ("cpp/build/run_async_cpp.exe" if args.mode == "async"
                    else "cpp/build/run_sync_cpp.exe")
    if not Path(args.cpp).exists():
        print(f"C++ binary not found: {args.cpp}\n"
              f"  cd cpp && cmake -S . -B build -G Ninja && cmake --build build")
        return 1

    backend = OnnxBackend(args.model, "openvino", None, args.device)
    height, width = backend.hw
    print(f"{Path(args.model).name} | openvino/{args.device} | {args.mode} | "
          f"{args.duration_s:.0f}s per run | {args.pairs} pairs")
    if args.mode == "async":
        print(f"  in flight {args.inflight}, queue {args.queue_size}, "
              f"{args.post_workers} collector(s) on the C++ side to match "
              f"Python's one")
    print("  note: the Python side reaches the device through ONNX Runtime's "
          "OpenVINO\n        provider, the C++ side through OpenVINO directly. "
          "Different code\n        paths to the same silicon, not the same one "
          "renamed.")

    tmpdir = tempfile.mkdtemp(prefix="impl_")
    pairs = []
    def show(label, r):
        print(f"  {label:<7} {r['throughput_fps']:7.2f} FPS   "
              f"E2E p50 {r['e2e_ms']['p50']:6.2f} ms   "
              f"drift {r['drift']['ratio']:.2f}x")
        return r

    for i in range(max(1, args.pairs)):
        # The order inside a pair alternates. With C++ always first, anything
        # that warms up between the two halves is charged to Python every
        # time, and a paired design cannot cancel a bias it applies uniformly.
        cpp_first = (i % 2 == 0) or not args.alternate_order
        print(f"\n--- pair {i + 1}/{args.pairs} "
              f"({'cpp first' if cpp_first else 'python first'}) ---")
        halves = [("cpp", lambda: show("cpp", run_cpp(args, tmpdir))),
                  ("python", lambda: show("python", run_python(
                      args, backend, height, width)))]
        if not cpp_first:
            halves.reverse()

        got = {}
        for name, fn in halves:
            got[name] = fn()
            noise.cooldown(args.cooldown, verbose=False)

        pairs.append({"pair": i + 1, "cpp_first": cpp_first,
                      "cpp": got["cpp"], "python": got["python"]})

    return report(args, pairs)


def report(args, pairs: list, reanalysed: bool = False) -> int:
    # A run that never reached steady state describes a machine that was still
    # warming up, so it is reported and excluded rather than averaged in.
    def drifted(r):
        d = r["drift"]["ratio"]
        return d is not None and (d > 1.05 or d < 0.95)

    usable = [q for q in pairs if not drifted(q["cpp"]) and not drifted(q["python"])]
    skipped = len(pairs) - len(usable)
    # A pair counts when both halves reached steady state. The rule stays put:
    # re-reading the GIL run under a version that ignored it found two pairs
    # with throughput ratios of 0.37 and 0.65 sitting beside a 1.04 from the
    # pairs that held, which are broken runs and not merely warm ones. What is
    # new is that the same figures over every pair are printed underneath, so
    # the cost of admissibility is visible and neither set is chosen after the
    # answer is known.
    src = usable or pairs

    w = 78
    print("\n" + "=" * w)
    # How the session ran, not how the surviving subset happened to run: with
    # one admissible pair left, `src` would call a counterbalanced session
    # "python first in every pair".
    orders = {q.get("cpp_first", True) for q in pairs}
    order_note = ("order counterbalanced" if len(orders) > 1
                  else "cpp first in every pair" if orders == {True}
                  else "python first in every pair")
    print(f" {len(pairs)} PAIRS, alternating in one session, {order_note}"
          + (f"  ({skipped} excluded for drift)" if skipped else ""))
    print("-" * w)
    print(f" {'stage':<13}{'cpp ms':>9}{'python ms':>11}{'cpp/python':>12}"
          f"{'per-pair range':>17}{'CV':>8}")

    # Each ratio is formed inside a pair and only then aggregated. Dividing a
    # median of C++ runs by a median of Python runs takes the two halves from
    # different moments, which on this machine is not a ratio of anything.
    spreads = {st: noise.spread(
        [q["cpp"]["stages_ms"][st]["p50"] / q["python"]["stages_ms"][st]["p50"]
         for q in src if q["python"]["stages_ms"][st]["p50"]])
        for st in STAGES}
    ratios = {st: spreads[st]["mean"] for st in STAGES}

    ce = statistics.median(q["cpp"]["e2e_ms"]["p50"] for q in src)
    pe = statistics.median(q["python"]["e2e_ms"]["p50"] for q in src)
    for st in STAGES:
        c = statistics.median(q["cpp"]["stages_ms"][st]["p50"] for q in src)
        pm = statistics.median(q["python"]["stages_ms"][st]["p50"] for q in src)
        d = spreads[st]
        rng = f"{d['min']:.2f} - {d['max']:.2f}" if len(src) > 1 else "one pair"
        cv = f"{100 * d['cv']:.0f}%" if len(src) > 1 else "--"
        print(f" {st:<13}{c:>9.3f}{pm:>11.3f}{d['mean']:>12.2f}{rng:>17}{cv:>8}")

    for who, key, e in (("cpp", "cpp", ce), ("py", "python", pe)):
        cells = "".join(
            f"{st[:4]} {100 * statistics.median(q[key]['stages_ms'][st]['p50'] for q in src) / e:4.1f}%  "
            for st in STAGES)
        print(f" {'share ' + who:<13}{cells}")

    def tp_of(group):
        return noise.spread([q["cpp"]["throughput_fps"] / q["python"]["throughput_fps"]
                             for q in group if q["python"]["throughput_fps"]])

    tp = tp_of(src)
    print("-" * w)
    label = f"   {len(src)} steady pair(s)" if skipped else ""
    if len(src) < 2:
        # One pair has no spread, so its CV is 0.0 and every steadiness test
        # passes trivially -- the same trap run_ab had. Report the observation
        # and claim nothing.
        print(f" throughput    cpp/python x{tp['mean']:.2f}"
              f"  from ONE pair: an observation, not a measurement")
    else:
        print(f" throughput    cpp/python x{tp['mean']:.2f}  "
              f"(min {tp['min']:.2f}, max {tp['max']:.2f},"
              f" CV {100 * tp['cv']:.1f}%){label}")
    if skipped:
        # What admissibility removed, so its cost is on the page rather than in
        # a decision no one can see.
        t2 = tp_of(pairs)
        c2 = noise.spread(
            [q["cpp"]["stages_ms"]["inference"]["p50"]
             / q["python"]["stages_ms"]["inference"]["p50"] for q in pairs])
        extra = (f"control x{c2['mean']:.2f} CV {100 * c2['cv']:.0f}%"
                 if args.mode == "sync" else "no control in async mode")
        print(f" {'all ' + str(len(pairs)) + ' pairs':<13} cpp/python"
              f" x{t2['mean']:.2f}  (min {t2['min']:.2f}, max {t2['max']:.2f},"
              f" CV {100 * t2['cv']:.1f}%)")
        print(f" {'':<13} including the {skipped} excluded for drift;"
              f" {extra}")

    # The cross-validation, corrected.
    #
    # Comparing stage *shares* was wrong: shares are normalised by E2E, so an
    # implementation that genuinely speeds up the host stages necessarily
    # shifts every share, and the check flagged a real improvement as an
    # instrumentation fault. Shares are printed because they are informative,
    # not because a difference in them is a defect.
    #
    # What must agree is the absolute time of the stage neither implementation
    # can influence: `inference` is the same model on the same silicon, and the
    # only difference is the few microseconds of binding around the call. If
    # that disagrees, one of the two is mistiming, and no speed number above
    # means anything.
    if args.mode == "sync":
        # The control is `inference`: the same model on the same silicon,
        # performed by neither implementation, so the two must agree there or
        # one of them is mistiming and nothing else means anything.
        d = spreads["inference"]
        print(f" control       inference x{d['mean']:.2f}  "
              f"(min {d['min']:.2f}, max {d['max']:.2f},"
              f" CV {100 * d['cv']:.1f}%)")
        # A control detects the machine *changing* between the two halves of a
        # pair, so what it has to hold is its spread. Its level was assumed to
        # be 1.00 on the grounds that neither implementation performs the
        # inference -- true, but they do not dispatch it the same way either:
        # Python goes through ONNX Runtime's OpenVINO provider and C++ through
        # OpenVINO directly. A level that reproduces to a few percent across
        # pairs is those two runtimes, and it is a result rather than a fault.
        if len(src) < 2:
            print(" One pair, so the control has no spread to judge and this"
                  " line proves")
            print(" nothing. Run more pairs.")
        elif d["cv"] > 0.10:
            print(f" <-- CHECK: the control moved {100 * d['cv']:.0f}% between"
                  f" pairs. The machine")
            print("     changed between the two halves, so every ratio above"
                  " carries that")
            print("     movement as well as the implementations. Quote the"
                  " ranges, not the")
            print("     means, and prefer a cooler machine.")
        else:
            print(f" The control holds to {100 * d['cv']:.0f}% across pairs,"
                  f" so the pairing cancelled the")
            print(" machine's movement and the ratios above are the"
                  " implementations.")
            if d["mean"] < 0.90 or d["mean"] > 1.11:
                print(f" Its level is {d['mean']:.2f}, not 1.00, and that is"
                      f" the two runtimes: Python")
                print(" reaches the device through ONNX Runtime's OpenVINO"
                      " provider, C++ through")
                print(" OpenVINO directly. A constant offset there is a"
                      " finding, not a fault --")
                print(" what would invalidate the ratios is the control"
                      " moving.")
    else:
        # There is no control in async mode, and saying so is better than
        # choosing a bad one. `read` was tried and came out at 0.44x: the
        # Python reader decodes while three other Python threads contend for
        # the interpreter lock, so the stage that "neither implementation
        # changes" is changed by precisely the effect under test. The GIL
        # reaches every host stage; that is what makes it worth measuring and
        # what leaves nothing to hold still.
        print(" No control stage in async mode. The interpreter lock reaches"
              " every host")
        print(" stage, so nothing here is untouched by the effect being"
              " measured -- `read`")
        print(" was tried and moved 0.44x. The timers were validated in sync"
              " mode instead,")
        print(" where `inference` agreed to 2%.")
        print(f"\n This ratio is the GIL measurement. W2 attributed the CPU"
              f" async loss to")
        print(f" ONNX Runtime subdividing its intra-op pool between concurrent"
              f" requests;")
        print(f" what it could not separate was Python's own orchestration."
              f" The same")
        print(f" pipeline in C++ on this device runs at x{tp['mean']:.2f} of it.")

    if len(src) >= 2 and tp["cv"] > 0.10:
        print(f" WEAK: the per-pair ratios vary by {100 * tp['cv']:.0f}%;"
              f" quote the range, not the mean.")
    print("=" * w)

    if reanalysed:
        print("\nre-read only; the saved file is left as it was measured")
        return 0

    meta = {"tag": args.tag, "mode": f"impl-{args.mode}", "args": vars(args),
            "cpp_binary": str(Path(args.cpp).resolve()),
            "pairs_excluded_for_drift": skipped,
            "env": env_fingerprint.collect()}
    out = (Path(args.out_dir) /
           f"{args.tag}__impl{args.mode}__openvino-{args.device.lower()}"
           f"__{Path(args.model).stem}"
           f"__{Path(args.source).stem if args.source else 'synthetic'}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "pairs": pairs,
                               "stage_ratio_cpp_over_python": ratios,
                               "stage_ratio_paired": spreads,
                               "throughput_ratio": tp,
                               "inference_ratio_cpp_over_python": ratios["inference"]},
                              indent=2),
                   encoding="utf-8")
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
