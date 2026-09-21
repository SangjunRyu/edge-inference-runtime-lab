"""Run every reproduction and print one table.

Each reproduction has a broken mode that is expected to fail and a fixed mode
that is expected to pass, and exits 0 when it behaved as its mode predicts. So
there are three outcomes worth distinguishing, not two:

  ok            broken reproduced the defect and fixed did not
  NOT REPRO     the broken mode came out clean -- the condition is no longer
                being exercised here. Not the same as "the code is safe", and
                not a reason to drop the fix
  REGRESSION    the fixed mode failed. This is the one to act on

Two of the five C++ reproductions currently sit at NOT REPRO on this machine,
and the table says so rather than rounding it to a pass.

    python repro/run_all.py                     # everything runnable here
    python repro/run_all.py --device NPU        # the OpenVINO ones on the NPU
    python repro/run_all.py --skip-openvino     # portable subset only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUILD = REPO / "cpp" / "build" / "repro"
MODEL = REPO / "models" / "resnet50_224.onnx"

# name, needs OpenVINO, one-line summary
CPP = [
    ("01_dangling_callback", True,
     "a retained pointer to a dead callback descriptor"),
    ("02_callback_blocks_queue", True,
     "backpressure applied to the runtime's own thread"),
    ("03_request_busy", True,
     "reading a completed request that has not left running state"),
    ("04_sentinel_overtakes", False,
     "end of stream announced while work was in flight"),
    ("05_reorder_off", False,
     "right frame count, wrong frame order"),
    ("06_inflight_overcount", False,
     "a concurrency peak above the pool that holds it"),
    ("07_bound_wrong_metric", False,
     "a correct bound asserted about the wrong quantity"),
    ("08_reorder_window_overflow", False,
     "a ring sized from the wrong bound, losing entries and hanging"),
    ("09_constructor_cannot_refuse", False,
     "an absent device opens fine, and the disabled guard could not have helped"),
]
STATS = [
    ("01_unpaired_ratio", "a paired comparison that formed ratios across pairs"),
    ("02_order_bias", "A ran first in every pair"),
    ("03_single_pair", "one pair passes every test that measures spread"),
]


def run(cmd: list[str], timeout: int) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return 124, "timed out"
    except FileNotFoundError:
        return 127, "not built"


def classify(broken_rc: int, fixed_rc: int) -> str:
    if fixed_rc != 0:
        return "REGRESSION"
    if broken_rc != 0:
        return "NOT REPRO"
    return "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="CPU", help="OpenVINO device for 01-03")
    ap.add_argument("--skip-openvino", action="store_true",
                    help="only the reproductions that need no runtime at all")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    py = sys.executable
    rows = []

    for name, needs_ov, what in CPP:
        exe = BUILD / f"repro_{name}.exe"
        if needs_ov and a.skip_openvino:
            rows.append((f"cpp/{name}", "skipped", what))
            continue
        if not exe.exists():
            rows.append((f"cpp/{name}", "not built", what))
            continue
        base = [str(exe), "--timeout-ms", "6000"]
        if needs_ov:
            base += ["--model", str(MODEL), "--device", a.device]
        rcs = {}
        for mode in ("broken", "fixed"):
            rc, out = run(base + ["--mode", mode], a.timeout)
            rcs[mode] = rc
            if a.verbose:
                print(f"--- {name} {mode} ---\n{out}")
        rows.append((f"cpp/{name}", classify(rcs["broken"], rcs["fixed"]), what))

    for name, what in STATS:
        script = REPO / "repro" / "stats" / f"{name}.py"
        rcs = {}
        for mode in ("broken", "fixed"):
            rc, out = run([py, str(script), "--mode", mode], a.timeout)
            rcs[mode] = rc
            if a.verbose:
                print(f"--- stats/{name} {mode} ---\n{out}")
        rows.append((f"stats/{name}", classify(rcs["broken"], rcs["fixed"]), what))

    w = 78
    print("=" * w)
    print(f" {'reproduction':<32}{'status':<13}what it is")
    print("-" * w)
    for name, status, what in rows:
        print(f" {name:<32}{status:<13}{what}")
    print("=" * w)

    bad = [n for n, s, _ in rows if s == "REGRESSION"]
    quiet = [n for n, s, _ in rows if s == "NOT REPRO"]
    if quiet:
        print(f"\n{len(quiet)} reproduction(s) no longer exercise their"
              f" condition: {', '.join(quiet)}.")
        print("Each one's header says what it depends on. A quiet broken mode")
        print("is a fact about this machine, not a licence to remove the fix.")
    if bad:
        print(f"\nREGRESSION in {', '.join(bad)} -- a fixed path failed.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
