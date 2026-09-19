// W5-2: a reorder ring sized from the wrong bound corrupts its own accounting
// and then hangs.
//
// SYMPTOM   `w5_scheduler --sweep` stopped producing output and sat there with
//           a live thread and a few megabytes resident. No crash, no error, no
//           partial table -- the kind of failure that looks like a slow machine
//           until you check the clock.
//
// CAUSE     Two mistakes, and the second only mattered because of the first.
//
//           The ring was sized from min(buffers, structural), the ceiling that
//           bounds frames HOLDING A BUFFER. Completions arrive at the reorder
//           *after* their buffers have gone back, so the indices outstanding
//           there are bounded by pipeline occupancy plus the emit path, which
//           is larger. The ring was too small.
//
//           Then, on an arrival past the window, the code flushed exactly one
//           entry and wrote anyway. One flush is not enough when the gap is
//           wide: `index % cap` still aliased an occupied slot, so the write
//           destroyed a pending entry while `++held_` counted it as new. From
//           there the counter and the flags disagreed, and `drain()` --
//           `while (held_ > 0) flush_lowest();` -- had no reason to terminate,
//           because the entry it was waiting for no longer existed.
//
// FIX       Two, matching the two mistakes. Size the ring from the bound that
//           applies to it, and flush until the index actually fits instead of
//           once. `drain()` became a bounded walk over the window rather than a
//           loop that trusts a counter.
//
// WHY IT MATTERS BEYOND ITSELF
//           This is the third time in W5 that a correct bound was attached to
//           the wrong quantity, and the first time the result was not merely a
//           wrong number. 06 mis-measured a real bound. 07 asserted a real
//           bound about a quantity it did not cover. Here the same error sized
//           a data structure, and a capacity argument that is off by a little
//           produces silent corruption rather than a slightly small buffer.
//
//           The rule the three share: a capacity is only valid for the lifetime
//           it was derived from. Write the lifetime beside the formula, or the
//           formula will outlive it.
//
// NEEDS NO ACCELERATOR. Single-threaded and deterministic -- the arrival order
// below is fixed, so this reproduces identically on every machine. The broken
// path is carried here rather than being reachable in the library, and its
// drain is bounded so that a reproduction reports non-termination instead of
// demonstrating it.

#include <cstdint>
#include <cstdio>
#include <vector>

#include "repro.hpp"
#include "ring_reorder.hpp"

namespace {

/// The reorder buffer as it was written: one flush per overflow, a write that
/// does not check whether the slot is occupied, and a drain that trusts the
/// counter. Kept minimal -- only the parts the defect runs through.
class BrokenRingReorder {
public:
    explicit BrokenRingReorder(std::size_t cap)
        : slots_(cap), filled_(cap, 0), cap_(cap) {}

    template <typename Sink>
    void push(std::uint64_t index, std::uint64_t payload, Sink&& sink) {
        if (index < next_) return;
        if (index - next_ >= cap_) {
            ++overflows_;
            flush_lowest(sink);          // once, and only once
        }
        const std::size_t at = static_cast<std::size_t>(index % cap_);
        slots_[at] = payload;            // may land on an occupied slot
        filled_[at] = 1;
        ++held_;                         // counted regardless
        while (filled_[static_cast<std::size_t>(next_ % cap_)]) {
            const std::size_t k = static_cast<std::size_t>(next_ % cap_);
            filled_[k] = 0;
            --held_;
            sink(next_, slots_[k]);
            ++next_;
        }
    }

    /// Returns false if it had to give up, which the original could not do.
    template <typename Sink>
    bool drain(Sink&& sink, std::size_t budget) {
        while (held_ > 0) {
            if (budget-- == 0) return false;
            flush_lowest(sink);
        }
        return true;
    }

    std::uint64_t overflows() const { return overflows_; }
    std::size_t held() const { return held_; }

private:
    template <typename Sink>
    void flush_lowest(Sink& sink) {
        for (std::uint64_t i = next_; i < next_ + cap_; ++i) {
            const std::size_t k = static_cast<std::size_t>(i % cap_);
            if (filled_[k]) {
                filled_[k] = 0;
                --held_;
                sink(i, slots_[k]);
                next_ = i + 1;
                return;
            }
        }
        next_ += 1;   // nothing found, and held_ is left untouched
    }

    std::vector<std::uint64_t> slots_;
    std::vector<std::uint8_t> filled_;
    std::size_t cap_, held_ = 0;
    std::uint64_t next_ = 0, overflows_ = 0;
};

/// Late by more than the broken window, which is what a pipeline produces when
/// several frames finish postprocessing while an early one is still queued.
const std::vector<std::uint64_t> kArrivals = {
    0, 1, 3, 5, 7, 9, 11, 2, 13, 4, 6, 8, 10, 12, 14, 15,
};

struct Outcome {
    std::size_t emitted = 0, lost = 0, duplicated = 0;
    bool terminated = true;
};

Outcome tally(const std::vector<std::uint64_t>& emitted, bool terminated) {
    Outcome o;
    o.emitted = emitted.size();
    o.terminated = terminated;
    std::vector<int> seen(kArrivals.size() + 1, 0);
    for (std::uint64_t i : emitted) {
        if (i < seen.size() && ++seen[static_cast<std::size_t>(i)] > 1) ++o.duplicated;
    }
    for (std::uint64_t idx : kArrivals) {
        if (idx < seen.size() && seen[static_cast<std::size_t>(idx)] == 0) ++o.lost;
    }
    return o;
}

}  // namespace

int main(int argc, char** argv) {
    repro::Opts o = repro::parse(
        argc, argv, "08_reorder_window_overflow -- a ring sized from the wrong bound");

    // Both halves of the mistake, and both halves of the fix.
    //
    // broken: the window the ceiling would have chosen for a real sweep row --
    //         four buffers -- with the write path that flushes only once.
    // fixed:  the window the structural bound gives for the same row, with the
    //         write path that flushes until the index fits.
    //
    // Sizing alone is not the fix and neither is the flush loop. A correctly
    // written ring that is still too small force-flushes and drops late
    // arrivals, which is the documented liveness trade and still loses frames.
    const std::size_t cap = (o.mode == "broken") ? 4 : 16;

    std::vector<std::uint64_t> emitted;
    emitted.reserve(kArrivals.size() * 2);
    auto sink = [&](std::uint64_t i, std::uint64_t) { emitted.push_back(i); };

    Outcome result;
    std::uint64_t overflows = 0;

    if (o.mode == "broken") {
        BrokenRingReorder ring(cap);
        for (std::uint64_t idx : kArrivals) ring.push(idx, idx, sink);
        const bool done = ring.drain(sink, 4 * kArrivals.size());
        overflows = ring.overflows();
        result = tally(emitted, done);
        if (!done) {
            std::printf("drain did not terminate within its budget: held_ says %zu,"
                        " nothing is left in the window\n", ring.held());
        }
    } else {
        eirl::RingReorder<std::uint64_t> ring(cap);
        for (std::uint64_t idx : kArrivals) ring.push(idx, idx, sink);
        ring.drain(sink);
        overflows = ring.overflows();
        result = tally(emitted, true);
    }

    std::printf("ring capacity          %zu\n", cap);
    std::printf("arrivals               %zu\n", kArrivals.size());
    std::printf("window overflows       %llu\n", static_cast<unsigned long long>(overflows));
    std::printf("emitted                %zu of %zu\n", result.emitted, kArrivals.size());
    std::printf("lost                   %zu%s\n", result.lost,
                result.lost ? "   <-- overwritten inside the ring" : "");
    std::printf("duplicated             %zu\n", result.duplicated);
    std::printf("drain terminated       %s\n", result.terminated ? "yes" : "NO");

    const bool reproduced = result.lost > 0 || result.duplicated > 0 || !result.terminated;

    char detail[256];
    std::snprintf(detail, sizeof detail,
                  "capacity %zu, %llu overflows, %zu lost, %zu duplicated, drain %s",
                  cap, static_cast<unsigned long long>(overflows),
                  result.lost, result.duplicated,
                  result.terminated ? "terminated" : "DID NOT terminate");
    return repro::verdict(o, reproduced, detail);
}
