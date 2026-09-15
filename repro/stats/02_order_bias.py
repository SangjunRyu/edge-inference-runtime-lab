"""W4: A ran first in every pair, so a paired design could not cancel it.

SYMPTOM   The per-pair control ratio slid 1.01, 0.90, 0.86, 0.72, 0.83 down one
          session -- a monotone shape that no implementation can produce.

CAUSE     Every pair ran C++ first and Python second. Whatever warms up between
          the two halves of a pair was therefore charged to Python in every
          pair, and a paired design cannot cancel a bias it applies uniformly.
          Pairing removes what differs between pairs; it is blind to what is
          the same in all of them.

FIX       Alternate the order: A->B, B->A, A->B, B->A. The within-pair effect
          then lands on each side equally often and cancels in the mean.

WHAT HAPPENED WHEN IT WAS TESTED
          Nothing. Six counterbalanced pairs on the real machine gave 1.274
          cpp-first against 1.289 python-first -- 1.2% apart on a 4.6% spread.
          The slide was session-long drift, not order. The fix stayed because
          the design was indefensible either way, but it bought nothing, and
          that is the result rather than a disappointment.

          Which makes this reproduction the useful half: on real hardware the
          bias was absent and the experiment could not distinguish "no bias"
          from "no sensitivity". Here the bias is dialled in by hand, so the
          question "would the counterbalanced design have detected it?" has an
          answer. If it would not, counterbalancing was theatre.

NEEDS NOTHING.

    python repro/stats/02_order_bias.py --mode broken
    python repro/stats/02_order_bias.py --mode fixed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_machine import Machine, spread, verdict   # noqa: E402

TRUE_RATIO = 1.28        # A is genuinely this much faster than B
PAIR_DRIFT = 0.15        # the machine is 15% slower by the second half of a pair


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=("broken", "fixed"), default="fixed")
    p.add_argument("--pairs", type=int, default=8)
    a = p.parse_args()

    m = Machine(session_drift=0.20, noise=0.02, pair_drift=PAIR_DRIFT)

    ratios, orders = [], []
    for i in range(a.pairs):
        a_first = True if a.mode == "broken" else (i % 2 == 0)
        # The second half of a pair is measured on a machine that has warmed by
        # PAIR_DRIFT. Which side that is depends entirely on the order.
        first = m.run(1.0)
        second = m.run(1.0, second_half=True)
        m.advance(1.0 / a.pairs)

        t_a, t_b = (first, second) if a_first else (second, first)
        t_a /= TRUE_RATIO        # A really is faster
        ratios.append(t_b / t_a)  # throughput ratio A/B, as the tool reports it
        orders.append(a_first)

    d = spread(ratios)
    bias = (d["mean"] - TRUE_RATIO) / TRUE_RATIO

    print(f"{a.pairs} pairs, the machine {100 * PAIR_DRIFT:.0f}% slower by the"
          f" second half of each pair")
    print(f"order: {'A first in every pair' if a.mode == 'broken' else 'counterbalanced'}")
    print()
    print(f"  true ratio        {TRUE_RATIO:.3f}")
    print(f"  measured          {d['mean']:.3f}  ({d['min']:.2f} - {d['max']:.2f},"
          f" CV {100 * d['cv']:.1f}%)")
    print(f"  bias              {100 * bias:+.1f}%")

    if a.mode == "fixed":
        af = [r for r, o in zip(ratios, orders) if o]
        bf = [r for r, o in zip(ratios, orders) if not o]
        print(f"  A-first subset    {spread(af)['mean']:.3f}")
        print(f"  B-first subset    {spread(bf)['mean']:.3f}")
        print("\nThe two subsets straddle the truth and their mean recovers it."
              "\nComparing them is also the test for whether order mattered at"
              " all --\non the real machine they came out 1.2% apart, which is"
              " how the\nhypothesis was rejected rather than assumed.")
    else:
        print("\nEvery pair charges the warming to the same side, so the bias"
              " survives\naveraging. The spread stays small, which makes the"
              " wrong answer look\nlike a precise one.")

    detail = f"bias {100 * bias:+.1f}% against a true ratio of {TRUE_RATIO:.2f}"
    return verdict(a.mode, abs(bias) > 0.05, detail)


if __name__ == "__main__":
    sys.exit(main())
