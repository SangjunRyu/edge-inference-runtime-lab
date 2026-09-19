// Counting every allocation the process makes, and saying so.
//
// W5's first claim is "zero allocations in steady state". A claim like that is
// worth nothing without a stated fence, because a global counter sees whatever
// happens to be linked in: OpenCV's allocator, OpenVINO's internals, the
// driver. The plan's rule is that the boundary has to be printed.
//
// This target draws the fence by *linking nothing else*. The ownership harness
// has no OpenVINO and no OpenCV in it, so every allocation the counter sees is
// one this repository's own code asked for, and the honest claim and the
// measured number are the same number. That is also why the harness runs on the
// MX150 unchanged.
//
// Counting is armed explicitly rather than from process start: thread creation,
// the pools themselves and the first frames all allocate by design, and a
// steady-state claim that included them would be measuring start-up.

#ifndef EIRL_ALLOC_COUNTER_HPP
#define EIRL_ALLOC_COUNTER_HPP

#include <cstddef>
#include <cstdint>

namespace eirl::alloc {

struct Counts {
    std::uint64_t allocations;
    std::uint64_t deallocations;
    std::uint64_t bytes;
};

/// Start counting. Counting is process-wide and thread-safe.
void arm();

/// Stop counting. Safe to call when not armed.
void disarm();

/// Zero the counters. Does not change whether counting is armed.
void reset();

Counts read();

}  // namespace eirl::alloc

#endif  // EIRL_ALLOC_COUNTER_HPP
