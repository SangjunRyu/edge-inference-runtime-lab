"""Noise control for measurements taken on a laptop.

A consumer laptop is a hostile measurement instrument: background processes
steal cores, turbo decays within seconds, and thermal throttling changes the
machine mid-run. None of that can be eliminated here, so it is instead made
visible and attributable.

Three mechanisms:

* ``idle_gate`` refuses to start while the machine is already busy, so a run is
  not silently competing with a browser or an indexer.
* ``clock`` samples CPU frequency at the start and end of a run, so a slowdown
  can be attributed to the clock dropping rather than left as a mystery.
* Repetition (``--repeat`` in the runners) is the only honest way to know
  whether a single run's number is reproducible. One run cannot report its own
  reliability.
"""
from __future__ import annotations

import time


def _psutil():
    try:
        import psutil
        return psutil
    except Exception:
        return None


def clock() -> float | None:
    """Current CPU frequency in MHz, or None if unavailable."""
    ps = _psutil()
    if ps is None:
        return None
    try:
        f = ps.cpu_freq()
        return round(f.current, 1) if f else None
    except Exception:
        return None


def busy_percent(sample_s: float = 1.0) -> float | None:
    """System-wide CPU utilisation over ``sample_s`` seconds."""
    ps = _psutil()
    if ps is None:
        return None
    try:
        ps.cpu_percent(interval=None)      # prime the counter
        return float(ps.cpu_percent(interval=sample_s))
    except Exception:
        return None


def idle_gate(max_busy_pct: float = 15.0, attempts: int = 6,
              sample_s: float = 1.0, verbose: bool = True) -> dict:
    """Wait for the machine to go quiet before measuring.

    Returns a record of what was observed. Never raises and never blocks
    forever: after ``attempts`` samples it gives up and reports the machine was
    busy, so the run still happens but the result carries the evidence.
    """
    ps = _psutil()
    if ps is None:
        return {"checked": False, "reason": "psutil not installed"}

    seen: list[float] = []
    for i in range(attempts):
        busy = busy_percent(sample_s)
        if busy is None:
            return {"checked": False, "reason": "cpu_percent unavailable"}
        seen.append(busy)
        if busy <= max_busy_pct:
            if verbose and i:
                print(f"   machine settled at {busy:.0f}% after {i + 1} samples")
            return {"checked": True, "settled": True, "busy_pct": busy,
                    "samples": seen, "threshold_pct": max_busy_pct}
        if verbose:
            print(f"   waiting for idle: CPU at {busy:.0f}% "
                  f"(want <= {max_busy_pct:.0f}%), sample {i + 1}/{attempts}")
    if verbose:
        print(f"!! machine still busy ({seen[-1]:.0f}%); measuring anyway -- "
              f"treat this result as contaminated")
    return {"checked": True, "settled": False, "busy_pct": seen[-1],
            "samples": seen, "threshold_pct": max_busy_pct}


def cooldown(seconds: float, verbose: bool = True) -> None:
    """Pause between repeated runs so heat does not accumulate across them."""
    if seconds <= 0:
        return
    if verbose:
        print(f"   cooling down {seconds:.0f}s")
    time.sleep(seconds)


def spread(values: list[float]) -> dict:
    """Between-run spread of one headline metric.

    ``cv`` (coefficient of variation) is the number that matters: a measured
    improvement smaller than the run-to-run CV is not a result.
    """
    if not values:
        return {"n": 0}
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        return {"n": 1, "mean": mean, "min": mean, "max": mean,
                "stdev": 0.0, "cv": 0.0}
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    sd = var ** 0.5
    return {"n": n, "mean": mean, "min": min(values), "max": max(values),
            "stdev": sd, "cv": (sd / mean) if mean else 0.0}
