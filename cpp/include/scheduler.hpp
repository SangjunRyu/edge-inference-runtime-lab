// W5-2 -- the pipeline as stages, workers and bounded queues.
//
// W4's runners each hard-coded one shape. This is the same shape with the
// numbers pulled out: how many workers run each stage, how deep each queue is,
// and how many buffers exist. The stages themselves are supplied by the caller,
// so the structure can be exercised with synthetic work on a machine with no
// accelerator and with real work on one that has it.
//
// The ownership rules from W5-1 are kept, and one of them decides the whole
// design: the input lease is pinned from submit until the stage reading it has
// finished. That is why a frame occupies a buffer for its whole life here, and
// it is what makes the in-flight ceiling a property of the pool rather than of
// the queues.
//
// DX-RT anchor. `lib/device_pool/` separates choosing a device
// (`device_pool.cpp`, `device_dispatcher.cpp`) from submitting to it
// (`std_device_task_layer.cpp`, `acc_device_task_layer.cpp`), with
// `nfh_layer.cpp` converting formats between them. The question that separation
// raises, and that this header exists to answer with a number, is which knob
// actually bounds concurrency. See `ceiling()`.
//
// Not built here: work stealing, priorities, or a dynamic worker pool. Bounded
// queues, fixed workers and explicit per-frame state are enough for every
// question W5 asks, and the interesting problems are all in the ownership.

#ifndef EIRL_SCHEDULER_HPP
#define EIRL_SCHEDULER_HPP

#include <atomic>
#include <cstdint>
#include <functional>
#include <thread>
#include <vector>

#include "ownership.hpp"
#include "ring_queue.hpp"
#include "ring_reorder.hpp"

namespace eirl {

struct SchedulerConfig {
    std::uint64_t frames = 1000;
    std::uint32_t buffers = 4;
    std::size_t bytes = 150528;
    std::size_t queue_capacity = 2;
    int pre_workers = 1;
    int infer_workers = 2;
    int post_workers = 1;
    std::uint64_t warmup = 64;
    std::uint64_t interrupt_at = 0;  // 0 = run to the end
};

/// The largest number of frames that can be in the pipeline at once.
///
/// A frame holds an input lease from the moment the reader takes one until
/// postprocessing gives it back, so it cannot exceed the pool. It also cannot
/// exceed the number of places there are to stand: one frame being filled by
/// the reader, whatever fits in the three queues, and one per worker at each
/// stage.
///
/// Both halves matter. The first says more queue depth cannot buy more
/// concurrency; the second says more buffers cannot either, once the structure
/// runs out of room. Which one binds is the answer to the anchor question, and
/// it is arithmetic rather than a measurement.
inline std::uint32_t ceiling(const SchedulerConfig& c) {
    const std::uint64_t places =
        1                                                    // the reader's frame
        + 3ull * c.queue_capacity                            // the three queues
        + static_cast<std::uint64_t>(c.pre_workers)
        + static_cast<std::uint64_t>(c.infer_workers)
        + static_cast<std::uint64_t>(c.post_workers);
    const std::uint64_t n = places < c.buffers ? places : c.buffers;
    return static_cast<std::uint32_t>(n);
}

struct Stages {
    /// Produce frame `seq` into `dst`.
    std::function<void(std::uint64_t, std::uint8_t*, std::size_t)> read;
    /// Transform the input buffer in place.
    std::function<void(std::uint8_t*, std::size_t)> preprocess;
    /// Read the pinned input, write the output. Models a device that
    /// dereferences the pointer it was handed, at completion time.
    std::function<void(const std::uint8_t*, std::uint8_t*, std::size_t)> infer;
    /// Check the output. False counts as a value mismatch.
    std::function<bool(std::uint64_t, const std::uint8_t*, std::size_t)> postprocess;
};

struct SchedulerStats {
    std::uint64_t accepted = 0, completed = 0, emitted = 0;
    std::uint64_t mismatched = 0, duplicated = 0;
    std::uint64_t out_of_order = 0, max_displacement = 0;
    std::uint64_t reorder_overflows = 0;
    std::size_t reorder_max_held = 0;
    std::uint32_t peak_inflight = 0;
    std::uint32_t peak_inflight_frame_bracketed = 0;
    std::uint32_t predicted_ceiling = 0;
    std::size_t peak_q_pre = 0, peak_q_infer = 0, peak_q_post = 0;
    std::uint32_t in_outstanding = 0, out_outstanding = 0;
    std::uint64_t in_acquires = 0, in_releases = 0;
    std::uint64_t out_acquires = 0, out_releases = 0;
    std::uint64_t double_releases = 0;
    double seconds = 0.0;
};

/// Runs one pass of the pipeline and returns what it counted.
///
/// `on_warm` fires once, on the reader thread, when the warm-up frames have
/// been accepted. The ownership harness uses it to arm allocation counting;
/// anything that must not be included in a steady-state claim belongs before
/// it.
SchedulerStats run_pipeline(const SchedulerConfig& cfg,
                            const Stages& stages,
                            const std::function<void()>& on_warm = {});

}  // namespace eirl

#endif  // EIRL_SCHEDULER_HPP
