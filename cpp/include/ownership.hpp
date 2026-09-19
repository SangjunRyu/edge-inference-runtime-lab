// Buffers a pipeline owns, and the rules that make ownership checkable.
//
// W4 already had a fixed slot pool, but ownership in it was a convention: a
// worker took an index off a free queue and was trusted to push it back. A
// buffer that never came back was invisible, because nothing was counting and
// nothing could have complained.
//
// W5-1 makes the same structure say what it means:
//
//   * the pool is sized once, at start-up, and never grows;
//   * a lease is move-only, so two owners is not a state the type system
//     permits;
//   * a lease returns itself when it dies, so the common path cannot leak;
//   * the pool knows how many are out, so shutdown is a check rather than a
//     hope;
//   * input and output leases are different types, so they cannot be swapped.
//
// The last one is the DX-RT anchor. `FixedSizeBuffer(size, buffer_count,
// allocType, direction)` in dx_rt (lib/include/dxrt/fixed_size_buffer.h:37-43)
// makes the same two decisions a shipping NPU runtime makes: a fixed count
// decided up front, and a direction the buffer carries rather than one the
// caller remembers. Here the direction is a template parameter, so mixing them
// is a compile error rather than a run-time check -- see
// src/negative_direction.cpp, which exists to fail to build.
//
// What this is not: a custom allocator. The pool hands out slices of one block
// it allocated at construction. There is no free-list arithmetic on the hot
// path beyond a stack pop, and no attempt to be faster than malloc -- only to
// not call it.

#ifndef EIRL_OWNERSHIP_HPP
#define EIRL_OWNERSHIP_HPP

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace eirl {

/// Which way the data flows. Not interchangeable, on purpose.
enum class Direction { Input, Output };

constexpr const char* name(Direction d) {
    return d == Direction::Input ? "input" : "output";
}

template <Direction D>
class BufferPool;

/// A move-only handle to one buffer, returned to its pool when it dies.
template <Direction D>
class BufferLease {
public:
    BufferLease() = default;

    BufferLease(const BufferLease&) = delete;
    BufferLease& operator=(const BufferLease&) = delete;

    BufferLease(BufferLease&& o) noexcept
        : pool_(o.pool_), data_(o.data_), size_(o.size_), index_(o.index_) {
        o.pool_ = nullptr;
        o.data_ = nullptr;
        o.size_ = 0;
    }

    BufferLease& operator=(BufferLease&& o) noexcept {
        if (this != &o) {
            release();
            pool_ = o.pool_;
            data_ = o.data_;
            size_ = o.size_;
            index_ = o.index_;
            o.pool_ = nullptr;
            o.data_ = nullptr;
            o.size_ = 0;
        }
        return *this;
    }

    ~BufferLease() { release(); }

    explicit operator bool() const { return pool_ != nullptr; }

    std::uint8_t* data() const { return data_; }
    std::size_t size() const { return size_; }
    std::uint32_t index() const { return index_; }

    /// Give the buffer back early. Idempotent.
    ///
    /// Note what this makes possible and why the harness has an injection for
    /// it: releasing while something downstream is still reading the buffer is
    /// legal C++ and silently wrong. It is the same contract `repro/cpp/01`
    /// proved for OpenVINO's callback descriptor -- the reader looks at
    /// completion time, not at hand-off time.
    void release() {
        if (pool_) {
            pool_->release(index_);
            pool_ = nullptr;
            data_ = nullptr;
            size_ = 0;
        }
    }

    /// Drop the lease *without* returning it. Only the leak injection calls
    /// this; it exists so the shutdown check can be shown to bite.
    void disown() {
        pool_ = nullptr;
        data_ = nullptr;
        size_ = 0;
    }

private:
    friend class BufferPool<D>;

    BufferLease(BufferPool<D>* p, std::uint32_t i, std::uint8_t* d, std::size_t n)
        : pool_(p), data_(d), size_(n), index_(i) {}

    BufferPool<D>* pool_ = nullptr;
    std::uint8_t* data_ = nullptr;
    std::size_t size_ = 0;
    std::uint32_t index_ = 0;
};

/// A fixed set of equally sized buffers, allocated once.
template <Direction D>
class BufferPool {
public:
    using Lease = BufferLease<D>;

    BufferPool(std::uint32_t count, std::size_t bytes_each)
        : storage_(static_cast<std::size_t>(count) * bytes_each),
          free_(count),
          bytes_each_(bytes_each),
          count_(count) {
        for (std::uint32_t i = 0; i < count; ++i) free_[i] = count - 1 - i;
        free_n_ = count;
    }

    BufferPool(const BufferPool&) = delete;
    BufferPool& operator=(const BufferPool&) = delete;

    /// Blocks until a buffer is free. An empty lease means the pool closed.
    Lease acquire() {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&] { return free_n_ > 0 || closed_; });
        if (free_n_ == 0) return Lease{};
        std::uint32_t i = take_locked();
        return Lease{this, i, slot(i), bytes_each_};
    }

    /// Never blocks. Empty when nothing is free.
    Lease try_acquire() {
        std::lock_guard<std::mutex> lk(m_);
        if (free_n_ == 0) return Lease{};
        std::uint32_t i = take_locked();
        return Lease{this, i, slot(i), bytes_each_};
    }

    /// Wake everyone waiting; outstanding leases still return normally.
    void close() {
        {
            std::lock_guard<std::mutex> lk(m_);
            closed_ = true;
        }
        cv_.notify_all();
    }

    std::uint32_t capacity() const { return count_; }
    std::size_t bytes_each() const { return bytes_each_; }

    std::uint32_t outstanding() const {
        std::lock_guard<std::mutex> lk(m_);
        return count_ - static_cast<std::uint32_t>(free_n_);
    }

    std::uint64_t acquires() const { return acquires_.load(std::memory_order_relaxed); }
    std::uint64_t releases() const { return releases_.load(std::memory_order_relaxed); }

    /// The high-water mark of buffers handed out at once.
    ///
    /// Recorded inside the pool, under the pool's own lock, at the instant the
    /// buffer leaves the free stack. That matters more than it looks: a counter
    /// kept outside the pool and bracketed around the *frame* instead of around
    /// the *lease* can report a peak above the pool's capacity, because the
    /// window between giving a buffer back and decrementing the counter is a
    /// window in which someone else can take it. See
    /// `repro/cpp/06_inflight_overcount`.
    std::uint32_t peak_outstanding() const { return peak_.load(std::memory_order_relaxed); }

    /// The shutdown invariant: everything handed out came back, exactly once.
    bool all_returned() const { return outstanding() == 0 && acquires() == releases(); }

    std::uint8_t* slot(std::uint32_t i) { return storage_.data() + static_cast<std::size_t>(i) * bytes_each_; }

private:
    friend class BufferLease<D>;

    void release(std::uint32_t i) {
        {
            std::lock_guard<std::mutex> lk(m_);
            // A double return would corrupt the free stack silently; refusing
            // it keeps the counters meaningful.
            for (std::size_t k = 0; k < free_n_; ++k) {
                if (free_[k] == i) {
                    double_releases_.fetch_add(1, std::memory_order_relaxed);
                    return;
                }
            }
            free_[free_n_++] = i;
        }
        releases_.fetch_add(1, std::memory_order_relaxed);
        cv_.notify_one();
    }

public:
    std::uint64_t double_releases() const {
        return double_releases_.load(std::memory_order_relaxed);
    }

private:
    mutable std::mutex m_;
    std::condition_variable cv_;
    std::vector<std::uint8_t> storage_;
    std::vector<std::uint32_t> free_;
    std::size_t free_n_ = 0;
    std::size_t bytes_each_ = 0;
    std::uint32_t count_ = 0;
    bool closed_ = false;
    std::atomic<std::uint64_t> acquires_{0};
    std::atomic<std::uint64_t> releases_{0};
    std::atomic<std::uint64_t> double_releases_{0};
    std::atomic<std::uint32_t> peak_{0};

    /// Caller holds the lock. Takes one buffer and records the high-water mark
    /// at the same instant, so the two can never disagree.
    std::uint32_t take_locked() {
        std::uint32_t i = free_[--free_n_];
        acquires_.fetch_add(1, std::memory_order_relaxed);
        const auto out = static_cast<std::uint32_t>(count_ - free_n_);
        if (out > peak_.load(std::memory_order_relaxed)) {
            peak_.store(out, std::memory_order_relaxed);
        }
        return i;
    }
};

using InputLease = BufferLease<Direction::Input>;
using OutputLease = BufferLease<Direction::Output>;
using InputPool = BufferPool<Direction::Input>;
using OutputPool = BufferPool<Direction::Output>;

/// Everything one frame owns while it is in the pipeline.
///
/// Move-only, because it holds leases. That is the property worth having: a
/// frame cannot be in two queues at once, and handing it on is visibly a
/// transfer rather than a copy.
struct FrameContext {
    std::uint64_t seq = 0;
    InputLease in;
    OutputLease out;

    FrameContext() = default;
    FrameContext(FrameContext&&) noexcept = default;
    FrameContext& operator=(FrameContext&&) noexcept = default;
    FrameContext(const FrameContext&) = delete;
    FrameContext& operator=(const FrameContext&) = delete;
};

}  // namespace eirl

#endif  // EIRL_OWNERSHIP_HPP
