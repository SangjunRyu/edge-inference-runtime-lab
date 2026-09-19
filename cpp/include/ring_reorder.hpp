// Restoring submission order without allocating, because the displacement is
// already bounded by something the pipeline decided.
//
// `ReorderBuffer` in reorder.hpp is built on `std::map`, which allocates per
// completion. That is the right trade for the OpenVINO runners, where the
// allocation is lost in the noise beside an inference. It is the wrong trade
// here, because W5's claim is that the pipeline allocates nothing in steady
// state and a map would answer that question with the container's behaviour.
//
// The replacement is possible because of a fact W5-1 established rather than a
// guess about workloads: **a frame that has not been emitted still holds a
// lease**, so the number of indices simultaneously outstanding cannot exceed
// the in-flight ceiling. A ring of that size can therefore hold every gap that
// can physically exist, and needs no dynamic memory to do it.
//
// If the ring ever overflows, the bound was wrong, and that is worth far more
// than a silent resize: the capacity is derived from the ownership rules, so an
// overflow means one of those rules is not holding. Overflow is counted and
// reported, never hidden, and the lowest index is released out of order so the
// consumer still makes progress -- the same liveness trade `ReorderBuffer`
// makes and counts.

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
        if (index - next_ >= cap_) {
            // The bound the ring was sized from did not hold. Make progress and
            // say so rather than growing quietly.
            ++overflows_;
            flush_lowest(sink);
        }
        const std::size_t at = static_cast<std::size_t>(index % cap_);
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
    template <typename Sink>
    void drain(Sink&& sink) {
        while (held_ > 0) flush_lowest(sink);
    }

    std::uint64_t out_of_order() const { return out_of_order_; }
    std::uint64_t max_displacement() const { return max_displacement_; }
    std::uint64_t overflows() const { return overflows_; }
    std::uint64_t dropped_late() const { return dropped_late_; }
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
    std::uint64_t overflows_ = 0, dropped_late_ = 0;
};

}  // namespace eirl

#endif  // EIRL_RING_REORDER_HPP
