"""W4: the paired comparison did not form its ratios inside the pairs.

SYMPTOM   `compare_impl` reported the control stage -- `inference`, the same
          model on the same silicon, performed by neither implementation -- at
          0.92 and said the control held. The per-pair ratios ran 0.72 to 1.01,
          a 40% swing, and the run was unusable.

CAUSE     The ratio was `median(all C++ runs) / median(all Python runs)`. The
          two halves came from different moments on a machine that moves, so
          the quotient is not a ratio of anything -- and, worse, it is a single
          number. A single number has no spread, so the check that was supposed
          to detect a moving machine had nothing to look at and passed by
          default.

FIX       Form the ratio inside each pair, then aggregate the ratios. The
          spread that falls out IS the check.

WHY IT MATTERS BEYOND ITSELF
          The defect is invisible to any check applied after aggregation,
          which is exactly where the check was. Averaging first destroys the
          evidence that the average should not be trusted. The bias in the
          central value is the small part; the missing spread is the whole
          failure.

THE MACHINE MODEL matters here, so it is stated rather than tuned: W1 found
this laptop's cells bimodal rather than noisy -- per-run inference switched
between two levels, consistent with a hybrid P/E-core part and an unpinned
thread pool. So a run's state is drawn once per run and is independent of when
the run happened. A ratio taken within one moment divides that state out; one
taken across two moments cannot.

NEEDS NOTHING. Ground truth is known by construction, so "the estimator is
wrong" is a fact here rather than a suspicion.

    python repro/stats/01_unpaired_ratio.py --mode broken
    python repro/stats/01_unpaired_ratio.py --mode fixed
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_machine import Machine, spread, verdict   # noqa: E402

# Ground truth. C++ is genuinely faster at the host stages and does not perform
# the inference at all, so the control's true ratio is exactly 1.00.
TRUE = {"read": 0.70, "preprocess": 0.66, "inference": 1.00, "postprocess": 0.09}
COST_PY = {"read": 0.62, "preprocess": 1.60, "inference": 3.75, "postprocess": 0.17}

CONTROL = "inference"
CV_BAR = 0.10     # a control that moves more than this means the pairing failed


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=("broken", "fixed"), default="fixed")
    p.add_argument("--pairs", type=int, default=6)
    a = p.parse_args()

    m = Machine(session_drift=0.35, noise=0.03, bimodal_p=0.45, bimodal_factor=1.45)

    pairs, states = [], []
    for _ in range(a.pairs):
        # Both halves of a pair are measured close together; the session moves
        # on between pairs. Each run has ONE state, shared by its stages.
        s_cpp = m.run_state()
        m.advance(0.02)
        s_py = m.run_state()
        m.advance(1.0 / a.pairs)
        states.append((s_cpp, s_py))
        pairs.append({
            "cpp": {k: COST_PY[k] * TRUE[k] * s_cpp for k in TRUE},
            "python": {k: COST_PY[k] * s_py for k in TRUE},
        })

    # What actually happened, known because we drew it.
    truth = spread([sc / sp for sc, sp in states])
    moved = truth["cv"] > CV_BAR

    print(f"{a.pairs} pairs on a bimodal machine (45% of runs land on a level"
          f" 1.45x slower)")
    print(f"ground truth: the machine's own state ratio moved"
          f" {100 * truth['cv']:.1f}% between pairs"
          f"  ({truth['min']:.2f} - {truth['max']:.2f})")
    print()

    print(f"{'stage':<13}{'true':>8}{'estimated':>11}{'error':>9}{'spread':>22}")
    for st in TRUE:
        per_pair = [q["cpp"][st] / q["python"][st] for q in pairs]
        if a.mode == "broken":
            est = (statistics.median(q["cpp"][st] for q in pairs)
                   / statistics.median(q["python"][st] for q in pairs))
            shown = "none -- one number"
        else:
            d = spread(per_pair)
            est = d["mean"]
            shown = (f"{d['min']:.2f} - {d['max']:.2f}, CV"
                     f" {100 * d['cv']:.0f}%")
        err = abs(est - TRUE[st]) / TRUE[st]
        print(f"{st:<13}{TRUE[st]:>8.2f}{est:>11.2f}{100 * err:>8.1f}%{shown:>22}")

    print()
    if a.mode == "broken":
        agg = (statistics.median(q["cpp"][CONTROL] for q in pairs)
               / statistics.median(q["python"][CONTROL] for q in pairs))
        fired = False
        print(f"control check: `{CONTROL}` came out at {agg:.2f}. There is one"
              f" number, so there")
        print(f"is no spread to test and the check PASSES -- on a machine that"
              f" moved"
              f" {100 * truth['cv']:.0f}%.")
        print("The ratios above are quoted as though the pairing had cancelled"
              " the drift.")
    else:
        d = spread([q["cpp"][CONTROL] / q["python"][CONTROL] for q in pairs])
        fired = d["cv"] > CV_BAR
        print(f"control check: `{CONTROL}` CV {100 * d['cv']:.0f}%"
              f" against a {100 * CV_BAR:.0f}% bar -- "
              f"{'FIRES' if fired else 'passes'}")
        if fired:
            print("The run is reported as unusable, which is what it is.")

    detail = (f"the machine moved {100 * truth['cv']:.0f}% and the control check"
              f" {'did not fire' if not fired else 'fired'}")
    # The defect is not a wrong average. It is a moved machine going undetected.
    return verdict(a.mode, moved and not fired, detail)


if __name__ == "__main__":
    sys.exit(main())
