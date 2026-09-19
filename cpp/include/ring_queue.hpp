// A bounded queue that allocates once.
//
// `SafeQueue` in safe_queue.hpp is built on `std::queue`, which is a `deque`,
// which allocates and frees blocks as it grows and shrinks. That is invisible
// and harmless when the question is throughput, and fatal when the question is
// "does this pipeline allocate in steady state", because the answer would be
// "yes, in the queue" for reasons that have nothing to do with the design being
// measured.
//
// So the storage is a vector sized at construction and never touched again.
// Items move in and out of fixed cells; a popped cell is left holding a
// moved-from value, which for `FrameContext` means an empty lease and therefore
// nothing to release.
//
// Semantics are otherwise `SafeQueue`'s, including the one that matters: an
// empty optional means closed *and drained*, never merely closed, so a consumer
// cannot stop while work remains. W2 lost four frames to getting that wrong.

#ifndef EIRL_RING_QUEUE_HPP
#define EIRL_RING_QUEUE_HPP

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

namespace eirl {

template <typename T>
class RingQueue {
public:
    explicit RingQueue(std::size_t capacity)
        : buf_(capacity ? capacity : 1), cap_(capacity ? capacity : 1) {}

    /// Blocks while full. False only once closed.
    bool push(T item) {
        std::unique_lock<std::mutex> lk(m_);
        not_full_.wait(lk, [&] { return n_ < cap_ || closed_; });
        if (closed_) return false;
        buf_[tail_] = std::move(item);
        tail_ = (tail_ + 1) % cap_;
        ++n_;
        lk.unlock();
        not_empty_.notify_one();
        return true;
    }

    /// Blocks while empty. Empty optional means closed and drained.
    std::optional<T> pop() {
        std::unique_lock<std::mutex> lk(m_);
        not_empty_.wait(lk, [&] { return n_ > 0 || closed_; });
        if (n_ == 0) return std::nullopt;
        T item = std::move(buf_[head_]);
        buf_[head_] = T{};
        head_ = (head_ + 1) % cap_;
        --n_;
        lk.unlock();
        not_full_.notify_one();
        return item;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lk(m_);
            closed_ = true;
        }
        not_empty_.notify_all();
        not_full_.notify_all();
    }

    std::size_t size() const {
        std::lock_guard<std::mutex> lk(m_);
        return n_;
    }

    std::size_t capacity() const { return cap_; }

private:
    mutable std::mutex m_;
    std::condition_variable not_empty_, not_full_;
    std::vector<T> buf_;
    std::size_t cap_;
    std::size_t head_ = 0, tail_ = 0, n_ = 0;
    bool closed_ = false;
};

}  // namespace eirl

#endif  // EIRL_RING_QUEUE_HPP
