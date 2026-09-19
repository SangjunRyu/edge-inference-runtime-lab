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
// finished.
//
// DX-RT anchor. `lib/device_pool/` separates choosing a device
// (`device_pool.cpp`, `device_dispatcher.cpp`) from submitting to it
// (`std_device_task_layer.cpp`, `acc_device_task_layer.cpp`), with
// `nfh_layer.cpp` converting formats between them. The question that separation
// raises is which knob actually bounds concurrency. There are two answers and
// the smaller one wins; see below.
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
    std::uint32_t buffers = 4;          // input pool
    std::uint32_t out_buffers = 0;      // output pool; 0 means "same as buffers"
    std::size_t bytes = 150528;
    std::size_t out_bytes = 0;          // 0 means "same as bytes"

    // Per-queue capacities. They are separate because the general form of the
    // structural bound sums them; a single number would hide that and invite
    // the constant 3 to be read as a property of runtimes rather than of this
    // topology.
    std::size_t q_pre = 2, q_infer = 2, q_post = 2;

    // How many frames the source itself can hold while preparing one.
    std::size_t source_slots = 1;

    int pre_workers = 1;
    int infer_workers = 2;
    int post_workers = 1;
    std::uint64_t warmup = 64;
    std::uint64_t interrupt_at = 0;  // 0 = run to the end
};

/// How many frames the *structure* has room for.
///
///     C_structural = S + sum(Qi) + sum(Wj)
///
/// One frame being prepared by the source, whatever fits in each bounded
/// queue, and one per worker at each stage. The 3 in "three queues" is a fact
/// about this topology and nothing more: a pipeline with a different number of
/// queues, or with different capacities per queue, changes the sum and not the
/// form.
inline std::uint32_t structural_capacity(const SchedulerConfig& c) {
    const std::uint64_t places =
        c.source_slots + c.q_pre + c.q_infer + c.q_post
        + static_cast<std::uint64_t>(c.pre_workers)
        + static_cast<std::uint64_t>(c.infer_workers)
        + static_cast<std::uint64_t>(c.post_workers);
    return static_cast<std::uint32_t>(places);
}

/// How many frames the *resources* will fund.
///
/// This is the input pool, and only because of the lifetime model this
/// scheduler uses: a frame holds its input lease from the reader to the end of
/// postprocessing, so one input buffer is spoken for by every frame in the
/// pipeline. The output pool is NOT part of this bound -- an output lease is
/// held only from the start of inference to the end of postprocessing, so it
/// constrains a segment of the path rather than the whole of it.
///
/// If the lifetime model changed so that every in-flight frame held both, the
/// bound would become min(B_in, B_out, C_structural). The formula is a property
/// of the lifetimes, not a law.
inline std::uint32_t resource_capacity(const SchedulerConfig& c) { return c.buffers; }

/// The bound a frame count must respect: the smaller of the two.
inline std::uint32_t ceiling(const SchedulerConfig& c) {
    const std::uint32_t s = structural_capacity(c);
    const std::uint32_t r = resource_capacity(c);
    return s < r ? s : r;
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

    // Three different quantities, deliberately not one. Conflating them is how
    // the first version of this scheduler reported nine frames in flight with
    // eight buffers to hold them; see repro/cpp/06_inflight_overcount.
    std::uint32_t peak_pipeline_inflight = 0;   // admitted, not yet finished
    std::uint32_t peak_in_leases = 0;           // input buffers checked out
    std::uint32_t peak_out_leases = 0;          // output buffers checked out
    std::size_t peak_q_pre = 0, peak_q_infer = 0, peak_q_post = 0;

    std::uint32_t structural_capacity = 0;
    std::uint32_t resource_capacity = 0;

    std::uint32_t in_outstanding = 0, out_outstanding = 0;
    std::uint64_t in_acquires = 0, in_releases = 0;
    std::uint64_t out_acquires = 0, out_releases = 0;
    std::uint64_t double_releases = 0;
    double seconds = 0.0;
};

/// Runs one pass of the pipeline and returns what it counted.
///
/// `on_warm` fires once, on the reader thread, when the warm-up frames have
/// been accepted. Anything that must not be included in a steady-state claim
/// belongs before it.
SchedulerStats run_pipeline(const SchedulerConfig& cfg,
                            const Stages& stages,
                            const std::function<void()>& on_warm = {});

}  // namespace eirl

#endif  // EIRL_SCHEDULER_HPP
