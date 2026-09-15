"""W2/W4: one surviving pair passes every steadiness test, because it has no
spread to fail.

SYMPTOM   `run_ab` reported a confident speedup from a single usable pair at a
          coefficient of variation of 0.0%. Every quality gate passed. The same
          trap was later found in `compare_impl` and `compare_variants`.

CAUSE     The gates are all statements about spread -- CV below a bar, min and
          max within a band, a knee that clears the noise. A sample of one has
          a CV of exactly zero, so every one of them passes trivially. The
          instrument was most confident precisely where it had the least data.

FIX       Treat n < 2 as a separate case and refuse the claim. The tools now
          print "an observation, not a measurement" and report nothing else.

WHY IT MATTERS BEYOND ITSELF
          A quality gate that cannot fail is worse than no gate, because it
          launders an unsupported number into a supported-looking one. This is
          the reason the admissibility rule prints how many pairs it removed:
          a run that started with six pairs and ends with one should not read
          the same as a run that had one all along.

NEEDS NOTHING.

    python repro/stats/03_single_pair.py --mode broken
    python repro/stats/03_single_pair.py --mode fixed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_machine import Machine, spread, verdict   # noqa: E402

TRUE_RATIO = 1.28
CV_BAR = 0.10


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=("broken", "fixed"), default="fixed")
    p.add_argument("--pairs", type=int, default=6)
    p.add_argument("--seed", type=int, default=3)
    a = p.parse_args()

    # A wildly unsteady machine: most pairs fail the steady-state rule and are
    # excluded, which is what actually happened in the sixth hour of measuring.
    # The seed is chosen so that exactly ONE pair survives, because that is the
    # scenario under test -- not because the numbers are friendlier there. Pass
    # --seed to see the rule behave normally at other survivor counts.
    m = Machine(session_drift=0.8, noise=0.12, bimodal_p=0.5, bimodal_factor=1.6,
                seed=a.seed)

    pairs = []
    for _ in range(a.pairs):
        s_a, s_b = m.run_state(), m.run_state()
        m.advance(1.0 / a.pairs)
        # "Steady" here stands in for the drift rule in the real tools.
        steady = abs(s_a - s_b) / max(s_a, s_b) < 0.05
        pairs.append({"ratio": TRUE_RATIO * s_b / s_a, "steady": steady})

    usable = [q for q in pairs if q["steady"]]
    print(f"{a.pairs} pairs on a very unsteady machine;"
          f" {len(usable)} met the steady-state rule")

    if not usable:
        print("nothing admissible, so there is nothing to claim either way")
        return verdict(a.mode, False, "no admissible pairs")

    d = spread([q["ratio"] for q in usable])
    shown = ", ".join(f"{q['ratio']:.3f}" for q in usable)
    print(f"  admissible ratios  {shown}")
    print(f"  mean               {d['mean']:.3f}")
    print(f"  CV                 {100 * d['cv']:.1f}%   (bar {100 * CV_BAR:.0f}%)")
    print(f"  true               {TRUE_RATIO:.3f}")

    if a.mode == "broken":
        # Every gate is a statement about spread, so all of them pass.
        gates = {
            "CV below bar": d["cv"] <= CV_BAR,
            "range within 10%": (d["max"] - d["min"]) / d["mean"] <= 0.10,
            "min above 1.0": d["min"] > 1.0,
        }
        for name, ok in gates.items():
            print(f"    {name:<20} {'PASS' if ok else 'fail'}")
        claimed = all(gates.values())
        print(f"\n  claim: {'x%.2f, reported as measured' % d['mean'] if claimed else 'withheld'}")
        err = abs(d["mean"] - TRUE_RATIO) / TRUE_RATIO
        # The defect is the claim, not the error. A single pair may land close
        # to the truth by luck -- this one lands within 1% -- and it is still
        # unsupported, because nothing in the run could have told you so.
        reproduced = claimed and len(usable) < 2
        detail = (f"claimed x{d['mean']:.2f} from {len(usable)} pair(s) with"
                  f" every gate passing; it happens to be {100 * err:.0f}%"
                  f" from the truth, which nothing in the run could know")
    else:
        if len(usable) < 2:
            print("\n  n < 2: no spread exists, so every gate would pass"
                  " trivially.")
            print("  claim: withheld -- an observation, not a measurement")
            reproduced = False
            detail = f"{len(usable)} pair(s): the claim is withheld"
        else:
            print("\n  n >= 2: the gates mean something and are applied.")
            reproduced = False
            detail = f"{len(usable)} pairs: the gates are applied normally"

    return verdict(a.mode, reproduced, detail)


if __name__ == "__main__":
    sys.exit(main())
