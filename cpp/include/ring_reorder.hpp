// Restoring submission order without allocating, because the displacement is
// already bounded by something the pipeline decided.
//
// `ReorderBuffer` in reorder.hpp is built on `std::map`, which allocates per
// completion. That is the right trade for the OpenVINO runners, where the
// allocation is lost in the noise beside an inference. It is the wrong trade
// here, because W5's claim is that the pipeline allocates nothing in steady
// state and a map would answer that question with the container's behaviour.
//
// The replacement is possible because the pipeline already decides how far
// apart two completions can be: every frame that has not finished occupies a
// place in the structure, so a ring sized from that bound can hold every gap
// that can physically exist, with no dynamic memory.
//
// Getting that derivation wrong is the interesting part, and it was wrong once.
// The first version sized the ring from the in-flight ceiling, min(buffers,
// structure), on the reasoning that an unemitted frame still holds a lease. It
// does not: a completion reaches the reorder *after* its buffers have gone
// back. The ring was therefore too small, arrivals aliased slots that were
// still occupied, and the resulting disagreement between the held counter and
// the flags turned `drain()` into an infinite loop. The caller now sizes it
// from the structural bound plus the emit path; see `scheduler.cpp`, and
// `repro/cpp/08_reorder_window_overflow` for the failure itself.
//
// Overflow is still possible in principle, so it is counted and reported rather
// than hidden, and the lowest index is released out of order so the consumer
// keeps making progress -- the same liveness trade `ReorderBuffer` makes.

#ifndef EIRL_RING_REORDER_HPP
#define EIRL_RING_REORDER_HPP

#include <cstdint>
#include <utility>
#include <vector>

namespace eirl {

template <typename T>
class RingReorder {
public:
    explicit RingReorder(std::size_t capacity)
        : slots_(capacity ? capacity : 1), filled_(capacity ? capacity : 1, 0),
          cap_(capacity ? capacity : 1) {}

    /// Accept one completion. Returns nothing; emittable items go to `sink`.
    template <typename Sink>
    void push(std::uint64_t index, T payload, Sink&& sink) {
        if (index != arrivals_) {
            ++out_of_order_;
            const std::uint64_t d = index > arrivals_ ? index - arrivals_ : arrivals_ - index;
            if (d > max_displacement_) max_displacement_ = d;
        }
        ++arrivals_;

        if (index < next_) {
            // Already forced out; releasing it again would duplicate.
            ++dropped_late_;
            return;
        }
        // Flush until the index fits inside the window. One flush is not
        // enough: a single arrival can be many indices ahead, and stopping
        // early leaves `index % cap_` aliasing a slot that is still occupied.
        // Overwriting it there was a real defect -- it destroyed a pending
        // entry while still incrementing `held_`, so the counter outlived the
        // flags and `drain()` never terminated. See
        // repro/cpp/08_reorder_window_overflow.
        while (index - next_ >= cap_) {
            ++overflows_;
            flush_lowest(sink);
        }
        const std::size_t at = static_cast<std::size_t>(index % cap_);
        if (filled_[at]) {
            // Only reachable if the same index arrived twice; releasing the
            // older payload is better than losing track of it.
            ++duplicates_;
            filled_[at] = 0;
            --held_;
            sink(index, std::move(slots_[at]));
        }
        slots_[at] = std::move(payload);
        filled_[at] = 1;
        if (++held_ > max_held_) max_held_ = held_;

        while (filled_[static_cast<std::size_t>(next_ % cap_)]) {
            const std::size_t k = static_cast<std::size_t>(next_ % cap_);
            filled_[k] = 0;
            --held_;
            sink(next_, std::move(slots_[k]));
            ++next_;
        }
    }

    /// Release anything still held, lowest index first.
    ///
    /// Bounded by construction rather than by the held counter. A loop that
    /// trusts `held_` to reach zero is a loop that hangs the moment the counter
    /// and the flags disagree, and they did.
    template <typename Sink>
    void drain(Sink&& sink) {
        const std::uint64_t base = next_;
        for (std::uint64_t i = base; i < base + cap_; ++i) {
            const std::size_t k = static_cast<std::size_t>(i % cap_);
            if (filled_[k]) {
                filled_[k] = 0;
                if (held_) --held_;
                sink(i, std::move(slots_[k]));
            }
        }
        next_ = base + cap_;
        held_ = 0;
    }

    std::uint64_t out_of_order() const { return out_of_order_; }
    std::uint64_t max_displacement() const { return max_displacement_; }
    std::uint64_t overflows() const { return overflows_; }
    std::uint64_t dropped_late() const { return dropped_late_; }
    std::uint64_t duplicates() const { return duplicates_; }
    std::size_t max_held() const { return max_held_; }
    std::size_t capacity() const { return cap_; }

private:
    template <typename Sink>
    void flush_lowest(Sink& sink) {
        for (std::uint64_t i = next_; i < next_ + cap_; ++i) {
            const std::size_t k = static_cast<std::size_t>(i % cap_);
            if (filled_[k]) {
                filled_[k] = 0;
                --held_;
                sink(i, std::move(slots_[k]));
                next_ = i + 1;
                return;
            }
        }
        next_ += 1;  // nothing held in range; step past the gap
    }

    std::vector<T> slots_;
    std::vector<std::uint8_t> filled_;
    std::size_t cap_;
    std::size_t held_ = 0, max_held_ = 0;
    std::uint64_t next_ = 0, arrivals_ = 0;
    std::uint64_t out_of_order_ = 0, max_displacement_ = 0;
    std::uint64_t overflows_ = 0, dropped_late_ = 0, duplicates_ = 0;
};

}  // namespace eirl

#endif  // EIRL_RING_REORDER_HPP
