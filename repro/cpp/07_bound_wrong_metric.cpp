// W5-2: a correct bound, asserted about the wrong quantity.
//
// SYMPTOM   `w5_scheduler` reported peak pipeline in-flight 9 against a
//           ceiling of 8, and failed. Unlike 06, the instrument was now
//           right: the counter was linearized under one lock and 9 was a
//           value it genuinely held. The pool's own peak was exactly 8.
//
// CAUSE     The ceiling min(resource, structural) had been written down as a
//           bound on "frames in flight", and it is not one. It bounds frames
//           *holding a buffer*. A frame keeps its place in a postprocess
//           worker for a moment after giving its leases back:
//
//               ctx.out.release();
//               ctx.in.release();     <-- holds no buffer from here
//               ...                   <-- but is still inside the pipeline
//               gauge.leave();
//
//           During that moment the pool can fund a new frame while the old one
//           is still standing in a worker. Pipeline occupancy therefore runs
//           above lease occupancy by the size of that tail, and only the
//           structural bound applies to it.
//
// FIX       Two invariants over two quantities:
//
//               leases   <= min(resource, structural)   resource invariant
//               in-flight <= structural                 scheduler invariant
//
//           Neither is the other, and the one that failed was never true.
//
// WHY IT MATTERS BEYOND ITSELF
//           06 was an instrument that mis-measured a real bound. This is the
//           other half: an instrument that measures correctly, checked against
//           a bound that does not apply to what it measures. Both print the
//           same thing -- a number above a limit -- and both first look like
//           the design is broken.
//
//           The general form is that a capacity argument is only valid for the
//           lifetime it was derived from. min(B, structural) follows from "a
//           frame holds an input lease for its whole life", and a frame that
//           has released early, or has not acquired yet, is outside the
//           derivation. Change the lifetime model and the formula changes with
//           it. That is why the scheduler's header states the lifetime the
//           bound came from rather than stating the bound alone.
//
// NEEDS NO ACCELERATOR.

#include <cstdint>
#include <cstdio>

#include "repro.hpp"
#include "scheduler.hpp"

namespace {

constexpr std::uint8_t kPre = 0x33;
constexpr std::uint8_t kInfer = 0x5A;

inline std::uint8_t pattern(std::uint64_t seq, std::size_t i) {
    return static_cast<std::uint8_t>(seq * 31u + i * 7u);
}

eirl::Stages synthetic_stages() {
    eirl::Stages s;
    s.read = [](std::uint64_t seq, std::uint8_t* dst, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) dst[i] = pattern(seq, i);
    };
    s.preprocess = [](std::uint8_t* buf, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) buf[i] = static_cast<std::uint8_t>(buf[i] ^ kPre);
    };
    s.infer = [](const std::uint8_t* src, std::uint8_t* dst, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) dst[i] = static_cast<std::uint8_t>(src[i] ^ kInfer);
    };
    s.postprocess = [](std::uint64_t, const std::uint8_t*, std::size_t) { return true; };
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    repro::Opts o = repro::parse(
        argc, argv, "07_bound_wrong_metric -- the ceiling bounds leases, not frames");

    eirl::SchedulerConfig cfg;
    cfg.frames = o.n > 64 ? static_cast<std::uint64_t>(o.n) : 4000;
    cfg.bytes = 65536;
    // The pool must be the binding side, or the tail has no room to show: with
    // the structure binding instead, every place is already counted and
    // pipeline occupancy cannot pass the ceiling.
    cfg.buffers = 4;
    cfg.q_pre = cfg.q_infer = cfg.q_post = 2;
    cfg.pre_workers = 1;
    cfg.infer_workers = 2;
    cfg.post_workers = 2;
    cfg.warmup = cfg.frames / 8;

    auto stages = synthetic_stages();
    eirl::SchedulerStats st = eirl::run_pipeline(cfg, stages);

    const std::uint32_t structural = st.structural_capacity;
    const std::uint32_t resource = st.resource_capacity;
    const std::uint32_t bound = structural < resource ? structural : resource;

    std::printf("structural capacity    %u = %zu + (%zu+%zu+%zu) + (%d+%d+%d)\n",
                structural, cfg.source_slots, cfg.q_pre, cfg.q_infer, cfg.q_post,
                cfg.pre_workers, cfg.infer_workers, cfg.post_workers);
    std::printf("resource capacity      %u   (the input pool)\n", resource);
    std::printf("ceiling                %u   = min of the two\n", bound);
    std::printf("peak input leases      %u\n", st.peak_in_leases);
    std::printf("peak pipeline frames   %u%s\n", st.peak_pipeline_inflight,
                st.peak_pipeline_inflight > bound ? "   <-- above the ceiling" : "");
    std::printf("emitted                %llu of %llu, %llu wrong\n",
                static_cast<unsigned long long>(st.emitted),
                static_cast<unsigned long long>(cfg.frames),
                static_cast<unsigned long long>(st.mismatched));

    bool reproduced;
    char detail[256];
    if (o.mode == "broken") {
        // The claim as first written: the ceiling bounds frames in flight.
        reproduced = st.peak_pipeline_inflight > bound;
        std::snprintf(detail, sizeof detail,
                      "pipeline peak %u against a ceiling of %u, with leases at %u",
                      st.peak_pipeline_inflight, bound, st.peak_in_leases);
    } else {
        // The claim as corrected: two bounds over two quantities.
        reproduced = !(st.peak_in_leases <= bound
                       && st.peak_pipeline_inflight <= structural);
        std::snprintf(detail, sizeof detail,
                      "leases %u <= ceiling %u, and pipeline %u <= structural %u",
                      st.peak_in_leases, bound, st.peak_pipeline_inflight, structural);
    }
    return repro::verdict(o, reproduced, detail);
}
