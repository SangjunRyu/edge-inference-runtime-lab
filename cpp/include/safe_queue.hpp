// Bounded queue -- the C++ side of SafeQueue in bench/async_runner.py's model.
//
// Bounded on purpose: an unbounded queue turns backpressure into memory growth
// and hides the very stall the measurement is looking for. DX-APP uses 4.
//
// Shutdown is the part that is easy to get subtly wrong, and the Python side
// got it wrong once already: the end-of-stream marker must not overtake work
// that is still in flight. Here the marker is a distinct `std::optional`
// emptiness rather than a sentinel value, so no payload can be mistaken for
// it.

#ifndef EIRL_SAFE_QUEUE_HPP
#define EIRL_SAFE_QUEUE_HPP

#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>

namespace eirl {

template <typename T>
class SafeQueue {
public:
    explicit SafeQueue(size_t capacity) : cap_(capacity ? capacity : 1) {}

    /// Blocks while full. False only once the queue has been closed.
    bool push(T item) {
        std::unique_lock<std::mutex> lk(m_);
        not_full_.wait(lk, [&] { return q_.size() < cap_ || closed_; });
        if (closed_) return false;
        q_.push(std::move(item));
        lk.unlock();
        not_empty_.notify_one();
        return true;
    }

    /// Blocks while empty. Empty optional means closed *and* drained -- the
    /// consumer therefore never stops while work remains.
    std::optional<T> pop() {
        std::unique_lock<std::mutex> lk(m_);
        not_empty_.wait(lk, [&] { return !q_.empty() || closed_; });
        if (q_.empty()) return std::nullopt;
        T item = std::move(q_.front());
        q_.pop();
        lk.unlock();
        not_full_.notify_one();
        return item;
    }

    /// Stop accepting; waiters wake. Items already queued still drain.
    void close() {
        {
            std::lock_guard<std::mutex> lk(m_);
            closed_ = true;
        }
        not_empty_.notify_all();
        not_full_.notify_all();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lk(m_);
        return q_.size();
    }

private:
    mutable std::mutex m_;
    std::condition_variable not_empty_, not_full_;
    std::queue<T> q_;
    size_t cap_;
    bool closed_ = false;
};

}  // namespace eirl

#endif  // EIRL_SAFE_QUEUE_HPP
