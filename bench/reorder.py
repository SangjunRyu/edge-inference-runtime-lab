"""Put asynchronously completed frames back into submission order.

Async completions do not preserve submission order: postprocessing runs off the
completion callback, so a heavier frame can finish after a lighter successor.
Frame *count* stays correct, which is what makes it dangerous -- the output is
wrong and nothing about it looks wrong.

This is a reimplementation of the idea in DEEPX DX-APP's
`src/cpp_example/common/utility/frame_reorder.hpp`, not a translation of it.
The design tension is the interesting part and is kept: strict ordering against
bounded memory and guaranteed progress. Holding out for a frame that never
arrives would stall the consumer forever, so past `cap` buffered items the
lowest available index is released even though it is out of order. Correctness
is traded for liveness deliberately, and every such release is counted so the
trade shows up in the results instead of hiding.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ReorderBuffer:
    """Emit items in submission order, with a bounded wait for gaps.

    Items are `(index, payload)`. Indices are expected to start at 0 and be
    contiguous.
    """

    cap: int = 8
    _buf: dict[int, object] = field(default_factory=dict, repr=False)
    _next: int = 0
    forced_flushes: int = 0
    max_depth: int = 0
    out_of_order_arrivals: int = 0
    max_displacement: int = 0
    _arrivals: int = 0

    def push(self, index: int, payload: object) -> list[tuple[int, object]]:
        """Buffer one completion; return whatever is now emittable, in order."""
        # Arrivals are counted in the order they land. If the nth arrival is
        # not index n, this completion overtook one of its predecessors.
        if index != self._arrivals:
            self.out_of_order_arrivals += 1
            self.max_displacement = max(self.max_displacement,
                                        abs(index - self._arrivals))
        self._arrivals += 1

        self._buf[index] = payload
        self.max_depth = max(self.max_depth, len(self._buf))

        out: list[tuple[int, object]] = []
        while self._buf:
            lowest = min(self._buf)
            if lowest != self._next and len(self._buf) <= self.cap:
                break
            if lowest != self._next:
                # The gap never filled and the buffer is over cap. Release the
                # lowest item anyway so the consumer cannot stall forever.
                self.forced_flushes += 1
            out.append((lowest, self._buf.pop(lowest)))
            self._next = lowest + 1
        return out

    def drain(self) -> list[tuple[int, object]]:
        """Release everything still buffered, in order."""
        out = sorted(self._buf.items())
        self._buf.clear()
        if out:
            self._next = out[-1][0] + 1
        return out

    def stats(self) -> dict:
        return {
            "cap": self.cap,
            "max_depth": self.max_depth,
            "forced_flushes": self.forced_flushes,
            "out_of_order_arrivals": self.out_of_order_arrivals,
            "max_displacement": self.max_displacement,
            "arrivals": self._arrivals,
        }


class OrderChecker:
    """Verifies an emitted sequence is 0, 1, 2, ... with nothing lost.

    Counting frames is not enough: a shuffled stream has the right count. This
    is the check that would have caught the defect DEEPX's reorder buffer
    exists to prevent.
    """

    def __init__(self) -> None:
        self.expected = 0
        self.inversions = 0
        self.seen = 0
        self.first_inversion_at: int | None = None

    def observe(self, index: int) -> None:
        self.seen += 1
        if index != self.expected:
            self.inversions += 1
            if self.first_inversion_at is None:
                self.first_inversion_at = self.seen
        self.expected = index + 1

    def stats(self) -> dict:
        return {
            "emitted": self.seen,
            "order_violations": self.inversions,
            "first_violation_at_frame": self.first_inversion_at,
            "in_submission_order": self.inversions == 0,
        }
