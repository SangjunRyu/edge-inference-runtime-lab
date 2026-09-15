// Reordering and order checking -- the C++ side of bench/reorder.py.
//
// Async completions do not preserve submission order: postprocessing runs off
// the completion callback, so a heavier frame can finish after a lighter
// successor. Frame *count* stays correct, which is what makes it dangerous --
// the output is wrong and nothing about it looks wrong.
//
// Reimplements the idea in DEEPX DX-APP's frame_reorder.hpp rather than
// translating it. The design tension is the interesting part and is kept:
// strict ordering against bounded memory and guaranteed progress. Holding out
// for a frame that never arrives would stall the consumer forever, so past
// `cap` buffered items the lowest available index is released even though it is
// out of order. Correctness is traded for liveness deliberately, and every such
// release is counted so the trade shows up in the results instead of hiding.
//
// The Python measured a maximum displacement of 13 against a cap of 2x the
// in-flight limit, which fired the force-flush path and left real order
// violations behind -- displacement exceeds the in-flight limit because a gap
// must also survive the output queue and the postprocess worker. Hence the
// default of 4 x (inflight + queue_size).

#ifndef EIRL_REORDER_HPP
#define EIRL_REORDER_HPP

#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace eirl {

template <typename T>
class ReorderBuffer {
public:
    explicit ReorderBuffer(size_t cap) : cap_(cap) {}

    /// Buffer one completion; returns whatever is now emittable, in order.
    std::vector<std::pair<uint64_t, T>> push(uint64_t index, T payload) {
        // Arrivals are counted in the order they land. If the nth arrival is
        // not index n, this completion overtook one of its predecessors.
        if (index != arrivals_) {
            ++out_of_order_;
            const uint64_t d = index > arrivals_ ? index - arrivals_ : arrivals_ - index;
            if (d > max_displacement_) max_displacement_ = d;
        }
        ++arrivals_;

        buf_.emplace(index, std::move(payload));
        if (buf_.size() > max_depth_) max_depth_ = buf_.size();

        std::vector<std::pair<uint64_t, T>> out;
        while (!buf_.empty()) {
            auto it = buf_.begin();
            if (it->first != next_ && buf_.size() <= cap_) break;
            if (it->first != next_) ++forced_flushes_;
            next_ = it->first + 1;
            out.emplace_back(it->first, std::move(it->second));
            buf_.erase(it);
        }
        return out;
    }

    /// Release everything still buffered, in order.
    std::vector<std::pair<uint64_t, T>> drain() {
        std::vector<std::pair<uint64_t, T>> out;
        for (auto& kv : buf_) out.emplace_back(kv.first, std::move(kv.second));
        if (!out.empty()) next_ = out.back().first + 1;
        buf_.clear();
        return out;
    }

    nlohmann::json stats() const {
        return {{"cap", cap_}, {"max_depth", max_depth_},
                {"forced_flushes", forced_flushes_},
                {"out_of_order_arrivals", out_of_order_},
                {"max_displacement", max_displacement_},
                {"arrivals", arrivals_}};
    }

private:
    std::map<uint64_t, T> buf_;
    size_t cap_, max_depth_ = 0, forced_flushes_ = 0, out_of_order_ = 0;
    uint64_t next_ = 0, arrivals_ = 0, max_displacement_ = 0;
};

/// Verifies an emitted sequence is 0, 1, 2, ... with nothing lost.
///
/// Counting frames is not enough: a shuffled stream has the right count. This
/// is the check that would have caught the defect DEEPX's reorder buffer
/// exists to prevent.
class OrderChecker {
public:
    void observe(uint64_t index) {
        ++seen_;
        if (index != expected_) {
            ++inversions_;
            if (!first_) first_ = seen_;
        }
        expected_ = index + 1;
    }

    nlohmann::json stats() const {
        return {{"emitted", seen_}, {"order_violations", inversions_},
                {"first_violation_at_frame",
                 first_ ? nlohmann::json(*first_) : nlohmann::json(nullptr)},
                {"in_submission_order", inversions_ == 0}};
    }

    uint64_t violations() const { return inversions_; }
    uint64_t emitted() const { return seen_; }

private:
    uint64_t expected_ = 0, seen_ = 0, inversions_ = 0;
    std::optional<uint64_t> first_;
};

}  // namespace eirl

#endif  // EIRL_REORDER_HPP
