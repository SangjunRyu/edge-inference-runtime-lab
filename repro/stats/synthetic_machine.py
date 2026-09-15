"""A laptop that drifts, with ground truth.

The measurement defects in this repository are harder to reproduce than the
threading ones, because on real hardware there is nothing to check an answer
against: if the instrument says 1.47x and the truth is 1.28x, nothing on the
machine will say so.

So the machine is simulated instead. Its speed follows a known curve and the
two implementations differ by a known constant. Then an estimator either
recovers that constant or it does not, and "or it does not" is a fact rather
than an opinion.

Nothing here touches an accelerator, ONNX Runtime or OpenVINO. It runs
anywhere, which matters because these are the defects most likely to come back
in a rewrite.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class Machine:
    """A machine whose speed moves while you measure it.

    ``warm`` is the slow-down accumulated over the session: 0.0 at the start,
    ``session_drift`` at the end. ``noise`` is the per-run jitter. Neither is
    meant to model this particular laptop accurately -- what matters is that
    the drift is monotone and the jitter is not, because that is the pairing
    problem in one sentence.
    """

    session_drift: float = 0.60     # 60% slower by the end of the session
    pair_drift: float = 0.0         # extra slow-down between the halves of a pair
    noise: float = 0.03             # per-run coefficient of variation
    # W1 found this laptop's CPU cells to be bimodal rather than noisy: per-run
    # inference switched between two levels instead of drifting, consistent
    # with a hybrid P/E-core part and an unpinned thread pool. A run lands on
    # the slow level with probability `bimodal_p` and is `bimodal_factor` times
    # slower there. It is the single most important property to model, because
    # a run's state is then independent of when it happened -- which is exactly
    # what an estimator that mixes moments cannot cope with.
    bimodal_p: float = 0.0
    bimodal_factor: float = 1.45
    seed: int = 20260915

    _rng: random.Random = field(init=False, repr=False)
    _t: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    def advance(self, fraction: float) -> None:
        """Move `fraction` of the way through the session."""
        self._t = min(1.0, self._t + fraction)

    def run_state(self):
        """Draw one run's state: every stage of a run shares it.

        Drawing the state once per RUN rather than once per stage is the whole
        point. The machine is slow for a whole run or fast for a whole run, so
        a ratio formed within one moment divides the state out and a ratio
        formed across two moments does not.
        """
        slow = 1.0 + self.session_drift * self._t
        if self._rng.random() < self.bimodal_p:
            slow *= self.bimodal_factor
        return slow * self._rng.gauss(1.0, self.noise)

    def run(self, true_cost_ms: float, *, second_half: bool = False) -> float:
        """One measured stage time, in ms.

        The machine multiplies every true cost by the same factor, which is
        what makes a ratio taken *at one moment* recoverable and a ratio taken
        across two moments not.
        """
        slow = 1.0 + self.session_drift * self._t
        if second_half:
            slow *= 1.0 + self.pair_drift
        if self._rng.random() < self.bimodal_p:
            slow *= self.bimodal_factor
        return true_cost_ms * slow * self._rng.gauss(1.0, self.noise)


def spread(values):
    """mean/min/max/CV, matching bench.noise.spread closely enough to compare."""
    n = len(values)
    if n == 0:
        return {"n": 0}
    mean = sum(values) / n
    if n == 1:
        return {"n": 1, "mean": mean, "min": mean, "max": mean, "cv": 0.0}
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    sd = math.sqrt(var)
    return {"n": n, "mean": mean, "min": min(values), "max": max(values),
            "cv": sd / mean if mean else 0.0}


def verdict(mode: str, reproduced: bool, detail: str) -> int:
    """The same contract the C++ reproductions use."""
    as_predicted = reproduced if mode == "broken" else not reproduced
    print()
    print(f"{mode:<8} {'DEFECT REPRODUCED' if reproduced else 'behaved correctly'}")
    print(f"         {detail}")
    if as_predicted:
        print(f"RESULT   as predicted for --mode {mode}")
        return 0
    if mode == "broken":
        print("RESULT   NOT REPRODUCED. The defect did not show itself here.")
    else:
        print("RESULT   REGRESSION. The corrected estimator failed.")
    return 1
