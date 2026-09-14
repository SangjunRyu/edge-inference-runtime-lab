"""Measurement harness.

Design rules, fixed before any model runs:

1. Stages are timed separately, and ``Read`` is its own stage. DX-APP reports
   Read / Preprocess / Inference / Postprocess as separate rows; matching that
   makes the numbers comparable instead of merely similar.
2. ``service time`` (how long a stage was busy) is never mixed with
   ``wait time`` (how long an item sat in a queue). W1 has no queues, so only
   service time exists -- but the field names are fixed now so W2 does not have
   to rename anything.
3. Warm-up frames are discarded, never averaged in.
4. Every stage reports p50/p95/max, not just the mean. A mean hides the
   thermal drift and the scheduling outliers that are the whole point.
5. ``E2E - sum(stages)`` is reported as ``Unaccounted``. That gap is a
   measurement, not an error: it is where copies, allocation and runtime
   overhead live.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

STAGES = ("read", "preprocess", "inference", "postprocess")


class Timer:
    """Context manager returning elapsed seconds via ``.elapsed``."""

    __slots__ = ("_t0", "elapsed")

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - self._t0


@dataclass
class Recorder:
    """Collects per-frame stage timings and produces the summary."""

    warmup: int = 20
    stages: tuple[str, ...] = STAGES
    # Stages that are time spent waiting, not time spent working. Kept apart
    # from `stages` because only service time can be overlapped: a queue wait
    # is the *consequence* of overlap, so folding it into the ceiling would let
    # a pipeline look better the more it queues.
    wait_stages: tuple[str, ...] = ()
    # True when frames overlap in the pipeline. Throughput is then deliberately
    # not 1/E2E -- that divergence is the result, not an instrumentation fault
    # -- so the sequential cross-check is replaced by Little's Law.
    pipelined: bool = False
    _service: dict[str, list[float]] = field(default_factory=dict, repr=False)
    _e2e: list[float] = field(default_factory=list, repr=False)
    _wall: list[float] = field(default_factory=list, repr=False)
    _n_seen: int = 0

    def __post_init__(self) -> None:
        self._service = {s: [] for s in self.all_stages}
        self._t_first = None

    @property
    def all_stages(self) -> tuple[str, ...]:
        return tuple(self.stages) + tuple(self.wait_stages)

    def add_frame(self, service: dict[str, float], e2e: float) -> None:
        """Record one frame. ``service`` and ``e2e`` are in seconds."""
        self._n_seen += 1
        if self._n_seen <= self.warmup:
            return
        now = time.perf_counter()
        if self._t_first is None:
            self._t_first = now
        for s in self.all_stages:
            self._service[s].append(service.get(s, 0.0))
        self._e2e.append(e2e)
        self._wall.append(now - self._t_first)

    # -- results -----------------------------------------------------------
    @property
    def counted(self) -> int:
        return len(self._e2e)

    @staticmethod
    def _stats(vals: list[float]) -> dict[str, float]:
        if not vals:
            return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
        a = np.asarray(vals, dtype=np.float64) * 1000.0  # ms
        # 10% trimmed mean: drops the top and bottom 5% before averaging, so a
        # single OS hiccup cannot move the headline. Reported alongside the
        # plain mean, never instead of it -- the gap between the two is itself
        # a measure of how contaminated the run was.
        lo, hi = np.percentile(a, 5), np.percentile(a, 95)
        kept = a[(a >= lo) & (a <= hi)]
        return {
            "mean": float(a.mean()),
            "mean_trim10": float(kept.mean()) if kept.size else float(a.mean()),
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "max": float(a.max()),
        }

    def result(self) -> dict:
        per_stage = {s: self._stats(self._service[s]) for s in self.all_stages}
        e2e = self._stats(self._e2e)
        stage_sum_mean = sum(per_stage[s]["mean"] for s in self.all_stages)
        service_sum_mean = sum(per_stage[s]["mean"] for s in self.stages)
        wait_sum_mean = sum(per_stage[s]["mean"] for s in self.wait_stages)
        elapsed = (self._wall[-1] - self._wall[0]) if len(self._wall) > 1 else 0.0
        throughput = (self.counted - 1) / elapsed if elapsed > 0 else 0.0
        # Cross-check. In a sequential pipeline the observed rate must match the
        # rate implied by mean E2E. A gap means wall-clock time is being spent
        # outside the timed region -- the run-level twin of `unaccounted`.
        implied = 1000.0 / e2e["mean"] if e2e["mean"] > 0 else 0.0
        # Little's Law: average frames resident = throughput x residence time.
        # In a pipelined run this should land near the configured in-flight
        # limit plus the queue slots; far from it means frames are held
        # somewhere unaccounted for.
        concurrency = throughput * e2e["mean"] / 1000.0
        # Best throughput a perfectly overlapped pipeline could reach: every
        # stage hidden behind the slowest one. Decides whether building an
        # async pipeline is worth it at all -- and whether the measurement
        # noise is small enough to see the gain if it is.
        # Service stages only: the ceiling asks what the pipeline could reach
        # if every stage hid behind the slowest one, and queue waits are not
        # work that can be hidden.
        slowest = max((per_stage[s]["mean"] for s in self.stages), default=0.0)
        ceiling = service_sum_mean / slowest if slowest > 0 else 0.0
        bottleneck = max(self.stages, key=lambda s: per_stage[s]["mean"]) \
            if slowest > 0 else None
        # Spread of the bottleneck stage. A gain smaller than this is not
        # detectable without many more samples.
        bn = per_stage[bottleneck] if bottleneck else {"p50": 0.0, "p95": 0.0}
        noise = (bn["p95"] / bn["p50"]) if bn["p50"] > 0 else 0.0
        return {
            "frames_counted": self.counted,
            "frames_warmup_discarded": min(self._n_seen, self.warmup),
            "stages_ms": per_stage,
            "e2e_ms": e2e,
            "stage_sum_mean_ms": stage_sum_mean,
            "service_sum_mean_ms": service_sum_mean,
            "wait_sum_mean_ms": wait_sum_mean,
            "wait_stages": list(self.wait_stages),
            "unaccounted_mean_ms": e2e["mean"] - stage_sum_mean,
            "throughput_fps": throughput,
            "throughput_implied_by_e2e_fps": implied,
            "concurrency_littles_law": concurrency,
            "pipelined": self.pipelined,
            "bottleneck_stage": bottleneck,
            "overlap_ceiling_x": ceiling,
            "bottleneck_p95_over_p50": noise,
            "drift": self._drift(),
        }

    def _drift(self) -> dict:
        """First-decile vs last-decile E2E mean. Surfaces thermal throttling.

        A laptop under sustained load slows down. If the last 10% of frames are
        materially slower than the first 10%, the run is not in steady state and
        the averages above are describing two different machines.
        """
        n = len(self._e2e)
        if n < 20:
            return {"first_decile_ms": None, "last_decile_ms": None, "ratio": None}
        k = max(1, n // 10)
        a = np.asarray(self._e2e[:k]).mean() * 1000
        b = np.asarray(self._e2e[-k:]).mean() * 1000
        return {"first_decile_ms": float(a), "last_decile_ms": float(b),
                "ratio": float(b / a) if a > 0 else None}

    # -- output ------------------------------------------------------------
    def print_summary(self, title: str = "") -> None:
        r = self.result()
        w = 68
        print()
        print("=" * w)
        if title:
            print(f" {title}")
            print("-" * w)
        print(f" frames {r['frames_counted']}"
              f"  (warm-up discarded: {r['frames_warmup_discarded']})")
        print("-" * w)
        print(f" {'stage':<14}{'mean':>9}{'p50':>9}{'p95':>9}{'max':>9}{'share':>9}")
        e2e_mean = r["e2e_ms"]["mean"] or 1.0
        for s in self.stages:
            st = r["stages_ms"][s]
            print(f" {s:<14}{st['mean']:>8.2f} {st['p50']:>8.2f} "
                  f"{st['p95']:>8.2f} {st['max']:>8.2f} "
                  f"{100 * st['mean'] / e2e_mean:>8.1f}%")
        if self.wait_stages:
            print(f" {'-- waiting':<14}")
            for s in self.wait_stages:
                st = r["stages_ms"][s]
                print(f" {s:<14}{st['mean']:>8.2f} {st['p50']:>8.2f} "
                      f"{st['p95']:>8.2f} {st['max']:>8.2f} "
                      f"{100 * st['mean'] / e2e_mean:>8.1f}%")
        print("-" * w)
        if self.wait_stages:
            print(f" {'sum(service)':<14}{r['service_sum_mean_ms']:>8.2f}")
            print(f" {'sum(waiting)':<14}{r['wait_sum_mean_ms']:>8.2f}")
        print(f" {'sum(stages)':<14}{r['stage_sum_mean_ms']:>8.2f}")
        e = r["e2e_ms"]
        print(f" {'E2E':<14}{e['mean']:>8.2f} {e['p50']:>8.2f} "
              f"{e['p95']:>8.2f} {e['max']:>8.2f}")
        print(f" {'unaccounted':<14}{r['unaccounted_mean_ms']:>8.2f}"
              f"{'':>27}{100 * r['unaccounted_mean_ms'] / e2e_mean:>8.1f}%")
        print("-" * w)
        obs, imp = r["throughput_fps"], r["throughput_implied_by_e2e_fps"]
        if self.pipelined:
            print(f" throughput    {obs:>8.2f} FPS")
            print(f" latency x tput{r['concurrency_littles_law']:>8.2f} frames"
                  f" resident on average (Little's Law)")
            print(f" {'':14}  E2E p50 {r['e2e_ms']['p50']:.1f} ms would be"
                  f" {imp:.1f} FPS if run sequentially:"
                  f" overlap is worth {obs / imp:.2f}x")
        else:
            print(f" throughput    {obs:>8.2f} FPS   (implied by E2E: {imp:.2f})")
            if imp > 0 and obs > 0 and not 0.95 <= obs / imp <= 1.05:
                print(f" {'':14}  <-- CHECK: observed/implied = {obs / imp:.2f};"
                      f" time is being spent outside the timed stages")
        bn, ceil = r["bottleneck_stage"], r["overlap_ceiling_x"]
        noise = r["bottleneck_p95_over_p50"]
        if self.pipelined:
            # The ceiling answers "is overlap worth building?" from a
            # sequential baseline. Once overlapped, the service times it is
            # computed from already include contention between concurrent
            # requests, so it no longer bounds anything.
            print(f" bottleneck    {bn}   (service p95/p50 = {noise:.2f}x)")
            return
        print(f" bottleneck    {bn}   overlap ceiling {ceil:.2f}x"
              f"   (p95/p50 = {noise:.2f}x)")
        if ceil > 1.0 and noise - 1.0 > (ceil - 1.0):
            print(f" {'':14}  <-- CHECK: noise band ({100 * (noise - 1):.0f}%)"
                  f" exceeds the best possible gain ({100 * (ceil - 1):.0f}%);"
                  f" an async result here would not be measurable")
        d = r["drift"]
        if d["ratio"] is not None:
            flag = "  <-- CHECK: not steady state" if d["ratio"] > 1.05 else ""
            print(f" drift         first {d['first_decile_ms']:.2f} ms"
                  f" -> last {d['last_decile_ms']:.2f} ms"
                  f"  ({d['ratio']:.2f}x){flag}")
        print("=" * w)

    def save(self, path: str | Path, meta: dict) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"meta": meta, "result": self.result()}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path
