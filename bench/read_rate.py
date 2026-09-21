"""Does `read` get faster when it is called more often, and why?

W2 measured `read` at 9.75 ms in the sequential pipeline and 5.05 ms in the
overlapped one, for identical work. Overlap cannot shorten a frame's own service
time, so that number has sat in `findings-w2.md` unattributed, and with it the
0.83x end-to-end latency figure that leans on it. `handoff/open-threads.md` calls
this the highest-value portable thread in the project for exactly that reason:
the test either attributes the number or retracts it.

The standing hypothesis is decoder pipelining. FFmpeg decodes H.264 with several
threads, and those threads overlap across *calls*: one `read()` sends a packet
and waits for a frame, so a caller that returns immediately keeps the workers
busy, while a caller that waits 19 ms between calls pays a full single-frame
latency every time.

That hypothesis makes a prediction this tool tests by **turning the proposed
cause off**, which is stronger than measuring the effect again:

    many decoder threads   back-to-back reads are much cheaper than spaced ones
    one decoder thread     the difference largely disappears

A second factor separates pipelining from cache locality. The gap between calls
can be spent idle or spent evicting cache. If only the gap's *length* matters it
is the decoder; if filling it with work makes things worse, some of it is
locality.

    python -m bench.read_rate --compare-threads 16,1
    python -m bench.read_rate --source clips/bbb_360_2MB.mp4 --gaps 0,5,10,19,30
    python -m bench.read_rate --ffmpeg-threads 1 --json

The thread count reaches FFmpeg through `OPENCV_FFMPEG_CAPTURE_OPTIONS`, which
must be set before OpenCV loads, so `--compare-threads` runs each setting in its
own process. `CAP_PROP_N_THREADS` reports 16 whatever is requested and cannot be
used to confirm the setting; the effect on decode time can, and is printed.

No accelerator and no model. This measures a video decoder and a clock.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


def pct(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = p * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] * (1.0 - (idx - lo)) + s[hi] * (idx - lo)


def cv(vals: list[float]) -> float:
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return 0.0
    m = statistics.fmean(vals)
    return (statistics.stdev(vals) / m) if m else 0.0


def pin(cpus: list[int]) -> list[int] | None:
    """Hold the process on chosen logical CPUs.

    The 255H is a hybrid part, and W1 finding 4 already attributed the two-level
    CPU timings in this project to an unpinned thread pool landing on different
    core types. The first version of this experiment reproduced exactly that: a
    control of pure integer work, no memory traffic and no library threads,
    moved 24% and 49% between the halves of a pair. That is not a decoder
    effect and it is not the code under test -- it is the scheduler, and protocol
    rule 4 says to pin whatever is being compared.

    Pinning is therefore part of the experiment rather than a convenience, and
    the unpinned run is kept because the contrast between the two is itself the
    evidence that the machine, not the decoder, produced the earlier spread."""
    try:
        import psutil
        proc = psutil.Process()
        proc.cpu_affinity(cpus)
        return proc.cpu_affinity()
    except Exception:
        return None


def run_one(args) -> dict:
    """Everything that touches OpenCV. Imported here so the FFmpeg options set
    in `main` are already in the environment when the library loads."""
    import cv2
    import numpy as np

    # Big enough that walking it evicts a decoded frame's working set, and slow
    # enough that timing it is not dominated by the clock's own resolution.
    scratch = np.zeros(16 * 1024 * 1024, dtype=np.uint8)
    scratch_b = np.empty_like(scratch)

    def _spin(iters: int) -> int:
        x = 12345
        for _ in range(iters):
            x = (x * 1103515245 + 12345) & 0xFFFFFFFF
        return x

    def control_work() -> float:
        """A fixed cost neither condition changes.

        Pure integer work in the interpreter: a deterministic instruction count,
        no memory traffic and no library threads.

        Two earlier versions of this control failed, and both failures are worth
        keeping because they are about instruments rather than about decoding.

        The first copied 64 MB and moved 35-42% between pairs. A control must not
        compete with the thing under test for the resource under test; that one
        measured the experiment.

        The second was this loop, timed immediately after a condition that spends
        most of its time asleep. It moved 23-35%, and the reason is that the core
        had dropped to a low power state during the sleeping condition and the
        control measured the climb back out. **A control taken next to an idle
        condition measures the CPU's power state, not the machine's speed.** So
        it now spins first to bring the core back up, then times, then takes a
        median of three so a single descheduling does not decide the reading."""
        _spin(120000)                     # wake the core, untimed
        samples = []
        for _ in range(3):
            t0 = time.perf_counter()
            _spin(200000)
            samples.append((time.perf_counter() - t0) * 1000.0)
        return statistics.median(samples)

    def burn_busy(seconds: float) -> None:
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            np.copyto(scratch_b[: 1 << 20], scratch[: 1 << 20])

    def burn_idle(seconds: float) -> None:
        end = time.perf_counter() + seconds
        while True:
            left = end - time.perf_counter()
            if left <= 0:
                return
            if left > 0.002:
                time.sleep(left - 0.001)

    class Decoder:
        """Reopens at end of stream rather than seeking. Seeking back to frame 0
        is neither free nor the same work; reopening is, and the restart count
        is reported so a reader can see how much was one pass."""

        def __init__(self, path: str):
            self.path = path
            self.restarts = 0
            self.cap = self._open()

        def _open(self):
            cap = cv2.VideoCapture(self.path)
            if not cap.isOpened():
                raise SystemExit(f"cannot open {self.path}")
            return cap

        def read(self) -> bool:
            ok, _ = self.cap.read()
            if ok:
                return True
            self.cap.release()
            self.cap = self._open()
            self.restarts += 1
            ok, _ = self.cap.read()
            return ok

        def close(self) -> None:
            self.cap.release()

    def measure(dec: "Decoder", gap_s: float, busy: bool) -> dict:
        burn = burn_busy if busy else burn_idle
        for _ in range(args.warmup):
            dec.read()
            if gap_s > 0:
                burn(gap_s)
        times: list[float] = []
        for _ in range(args.frames):
            t0 = time.perf_counter()
            dec.read()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
            if gap_s > 0:
                burn(gap_s)
        return {"p50": pct(times, 0.50), "p95": pct(times, 0.95),
                "max": max(times), "mean": statistics.fmean(times),
                "frames": len(times)}

    pinned = None
    if args.pin:
        pinned = pin([int(c) for c in args.pin.split(",") if c.strip()])

    gaps = [float(g) for g in args.gaps.split(",") if g.strip()]
    dec = Decoder(args.source)

    # Repeat the sweep and rotate the gap order each time. One pass through the
    # gaps charges any drift during the pass to whichever gap came last.
    cells: dict = {g: {"idle": [], "busy": []} for g in gaps}
    for rep in range(args.repeat):
        order = gaps[rep % len(gaps):] + gaps[: rep % len(gaps)]
        for g in order:
            cells[g]["idle"].append(measure(dec, g / 1000.0, busy=False)["p50"])
            cells[g]["busy"].append(
                measure(dec, g / 1000.0, busy=True)["p50"] if g > 0
                else cells[g]["idle"][-1])
    sweep = []
    for g in gaps:
        i, b = cells[g]["idle"], cells[g]["busy"]
        sweep.append({"gap_ms": g,
                      "idle": {"p50": statistics.median(i), "cv": cv(i), "runs": i},
                      "busy": {"p50": statistics.median(b), "cv": cv(b), "runs": b}})

    # Paired: back-to-back against the cadence W2's sequential pipeline had.
    # Order alternates inside the pairs, because a paired design cannot cancel a
    # bias it applies uniformly.
    ratios, controls, spaced, b2b = [], [], [], []
    gap_s = args.gap_ms / 1000.0
    for i in range(args.pairs):
        # The control is timed inside each half, so its within-pair ratio says
        # whether the machine changed between the two halves being compared.
        # A control taken once between them cannot say that.
        # The gap is filled with work by default, because that is what the
        # sequential pipeline actually does with it: preprocess and inference
        # sit between two reads, not sleep. Comparing against an idle gap
        # measures the CPU's power state as much as the decoder's behaviour.
        busy_gap = args.paired_gap_kind == "busy"
        if i % 2 == 0:
            ca = control_work(); a = measure(dec, gap_s, busy_gap)
            cb = control_work(); b = measure(dec, 0.0, False)
        else:
            cb = control_work(); b = measure(dec, 0.0, False)
            ca = control_work(); a = measure(dec, gap_s, busy_gap)
        spaced.append(a["p50"])
        b2b.append(b["p50"])
        controls.append(cb / ca if ca else 1.0)
        if a["p50"] > 0:
            ratios.append(b["p50"] / a["p50"])

    reported = -1
    try:
        reported = int(dec.cap.get(cv2.CAP_PROP_N_THREADS))
    except Exception:
        pass
    restarts = dec.restarts
    dec.close()

    return {
        "source": args.source,
        "frames": args.frames,
        "warmup": args.warmup,
        "ffmpeg_threads_requested": args.ffmpeg_threads,
        "pinned_to": pinned,
        "cap_prop_n_threads_reported": reported,
        "restarts": restarts,
        "sweep": sweep,
        "paired": {
            "gap_ms": args.gap_ms,
            "gap_kind": args.paired_gap_kind,
            "pairs": args.pairs,
            "ratio_b2b_over_spaced": statistics.median(ratios) if ratios else 0.0,
            "ratio_cv": cv(ratios),
            "control_cv": cv(controls),
            "control_ratio": statistics.median(controls) if controls else 1.0,
            "spaced_p50": statistics.median(spaced),
            "b2b_p50": statistics.median(b2b),
        },
    }


def print_one(r: dict) -> None:
    t = r["ffmpeg_threads_requested"]
    print(f"decoder threads requested {t if t else 'default'}"
          f"   pinned to {r.get('pinned_to') or 'nothing'}"
          f"   restarts {r['restarts']}")
    print(f"  {'gap ms':>7}  {'idle p50':>9} {'CV':>6}  {'busy p50':>9} {'CV':>6}")
    for row in r["sweep"]:
        print(f"  {row['gap_ms']:7.0f}  {row['idle']['p50']:9.2f} {row['idle']['cv'] * 100:5.1f}%"
              f"  {row['busy']['p50']:9.2f} {row['busy']['cv'] * 100:5.1f}%")
    base = r["sweep"][0]["idle"]["p50"]
    spaced_only = [x["idle"]["p50"] for x in r["sweep"] if x["gap_ms"] > 0]
    if spaced_only:
        lo, hi = min(spaced_only), max(spaced_only)
        print(f"  back-to-back {base:.2f} ms, spaced {lo:.2f}-{hi:.2f} ms")
    p = r["paired"]
    print(f"  paired at a {p['gap_ms']:.0f} ms {p.get('gap_kind','idle')} gap,"
          f" {p['pairs']} pairs:"
          f" b2b {p['b2b_p50']:.2f} ms, spaced {p['spaced_p50']:.2f} ms,"
          f" ratio {p['ratio_b2b_over_spaced']:.3f}"
          f" (ratio CV {p['ratio_cv'] * 100:.1f}%, control CV {p['control_cv'] * 100:.1f}%)")
    print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="bench.read_rate",
        description="Does read get faster when called more often, and why?")
    ap.add_argument("--source", default="clips/bbb_1080_10MB.mp4")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--gaps", default="0,5,10,19,30")
    ap.add_argument("--gap-ms", type=float, default=19.0,
                    help="the gap for the paired test, matching W2's sequential cadence")
    ap.add_argument("--pairs", type=int, default=5)
    ap.add_argument("--repeat", type=int, default=3, help="sweep repeats, gap order rotated")
    ap.add_argument("--pin", help="logical CPUs to hold the process on, e.g. 0 or 0,1")
    ap.add_argument("--paired-gap-kind", choices=["busy", "idle"], default="busy",
                    help="what fills the gap in the paired test (default busy, "
                         "matching what a sequential pipeline puts there)")
    ap.add_argument("--ffmpeg-threads", type=int, default=0,
                    help="0 leaves FFmpeg's default; 1 turns the hypothesis off")
    ap.add_argument("--compare-threads",
                    help="run each setting in its own process and compare, e.g. 16,1")
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    ap.add_argument("--out", help="write the result as JSON")
    args = ap.parse_args(argv)

    if args.compare_threads:
        settings = [int(t) for t in args.compare_threads.split(",") if t.strip()]
        results = []
        for nt in settings:
            cmd = [sys.executable, "-m", "bench.read_rate",
                   "--source", args.source, "--frames", str(args.frames),
                   "--warmup", str(args.warmup), "--gaps", args.gaps,
                   "--gap-ms", str(args.gap_ms), "--pairs", str(args.pairs),
                   "--repeat", str(args.repeat),
                   "--paired-gap-kind", args.paired_gap_kind,
                   ] + (["--pin", args.pin] if args.pin else []) + [
                   "--ffmpeg-threads", str(nt), "--json"]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print(proc.stdout)
                print(proc.stderr, file=sys.stderr)
                return proc.returncode
            results.append(json.loads(proc.stdout))

        print(f"source   {args.source}")
        print(f"frames   {args.frames} counted after {args.warmup} warm-up, per condition\n")
        for r in results:
            print_one(r)

        print("Does the effect follow the decoder's thread count?\n")
        print(f"  {'threads':>8} {'b2b p50':>9} {'spaced p50':>11} {'saving':>8} {'ctrl CV':>9}")
        for r in results:
            p = r["paired"]
            saving = 1.0 - p["ratio_b2b_over_spaced"]
            print(f"  {r['ffmpeg_threads_requested'] or 'default':>8}"
                  f" {p['b2b_p50']:9.2f} {p['spaced_p50']:11.2f}"
                  f" {saving * 100:7.0f}% {p['control_cv'] * 100:8.1f}%")

        worst_ctrl = max(r["paired"]["control_cv"] for r in results)
        savings = [1.0 - r["paired"]["ratio_b2b_over_spaced"] for r in results]
        print()
        if worst_ctrl > 0.15:
            print("verdict  UNMEASURABLE. A control moved more than 15% between pairs, so")
            print("         this window describes the laptop rather than the decoder.")
        elif min(savings) < -0.10:
            print("verdict  The effect runs the OTHER WAY. Calling back to back makes read")
            print("         more expensive, not less, so the pipelining hypothesis in")
            print("         findings-w2.md is contradicted by its own prediction and the")
            print("         attribution of W2's falling read time must be withdrawn.")
        elif max(savings) <= 0.10:
            print("verdict  No effect worth attributing at this call rate. W2's falling read")
            print("         time is NOT explained by call cadence, and the hypothesis in")
            print("         findings-w2.md should be retracted rather than repeated.")
        elif savings[0] > 0.10 and savings[-1] < savings[0] / 2:
            print("verdict  The effect is real and follows the decoder's thread count. Turning")
            print("         threading down removes most of it, which is what pipelining")
            print("         predicted. W2's falling read time belongs to decode, not overlap.")
        else:
            print("verdict  The effect is real but does NOT follow the thread count, so decoder")
            print("         pipelining is not the whole explanation. Report the effect and")
            print("         leave the mechanism open.")

        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"\nwrote {args.out}")
        return 0

    # A single setting. The option has to be in the environment before OpenCV
    # loads, which is why the comparison above uses separate processes.
    if args.ffmpeg_threads:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"threads;{args.ffmpeg_threads}"
    result = run_one(args)
    if args.json:
        print(json.dumps(result))
    else:
        print(f"source   {args.source}")
        print(f"frames   {args.frames} counted after {args.warmup} warm-up, per condition\n")
        print_one(result)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
