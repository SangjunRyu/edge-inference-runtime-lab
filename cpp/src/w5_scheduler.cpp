// W5-2 -- the generic scheduler, and the two bounds that decide concurrency.
//
// The plan's question for this phase was where the in-flight ceiling comes
// from: a queue depth, a worker count, or a buffer count. DX-RT raises it by
// separating device selection from submission (`lib/device_pool/`), which
// leaves three plausible answers and no obvious one.
//
// It has two, and the smaller wins.
//
//     C_resource   = the input pool, because a frame holds its input lease for
//                    its whole life in this lifetime model
//     C_structural = S + sum(Qi) + sum(Wj), because a frame has to be standing
//                    somewhere
//
// Neither is a law. The first depends on which leases a frame holds and for how
// long; the second on the topology. Both are stated as what they are.
//
// Three quantities are reported separately and never added together: frames in
// the pipeline, input buffers checked out, output buffers checked out. They
// have different bounds and different meanings, and collapsing them into one
// "in-flight" number is what produced a peak of nine against eight buffers on
// the first run of this program. See repro/cpp/06_inflight_overcount.
//
//   w5_scheduler --frames 2000 --buffers 8 --queue-capacity 2
//   w5_scheduler --sweep
//
// Links no OpenVINO and no OpenCV, for the reason given in w5_ownership.cpp:
// that link line is the fence around the zero-allocation claim.

#include <atomic>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "alloc_counter.hpp"
#include "scheduler.hpp"

namespace {

using namespace eirl;

constexpr std::uint8_t kPre = 0x33;
constexpr std::uint8_t kInfer = 0x5A;

inline std::uint8_t pattern(std::uint64_t seq, std::size_t i) {
    return static_cast<std::uint8_t>(seq * 31u + i * 7u);
}

Stages synthetic_stages() {
    Stages s;
    s.read = [](std::uint64_t seq, std::uint8_t* dst, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) dst[i] = pattern(seq, i);
    };
    s.preprocess = [](std::uint8_t* buf, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) buf[i] = static_cast<std::uint8_t>(buf[i] ^ kPre);
    };
    s.infer = [](const std::uint8_t* src, std::uint8_t* dst, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) dst[i] = static_cast<std::uint8_t>(src[i] ^ kInfer);
    };
    s.postprocess = [](std::uint64_t seq, const std::uint8_t* out, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) {
            const auto want = static_cast<std::uint8_t>(pattern(seq, i) ^ kPre ^ kInfer);
            if (out[i] != want) return false;
        }
        return true;
    };
    return s;
}

struct Check {
    std::string what;
    bool ok;
    std::string detail;
};

std::string n_of(std::uint64_t a, std::uint64_t b, std::uint64_t c) {
    return std::to_string(a) + " / " + std::to_string(b) + " / " + std::to_string(c);
}

int single(SchedulerConfig cfg, bool json) {
    auto stages = synthetic_stages();
    SchedulerStats st = run_pipeline(cfg, stages, [] {
        eirl::alloc::reset();
        eirl::alloc::arm();
    });
    eirl::alloc::disarm();
    const auto counts = eirl::alloc::read();

    const std::uint32_t out_cap = cfg.out_buffers ? cfg.out_buffers : cfg.buffers;
    const std::uint32_t bound = ceiling(cfg);

    std::vector<Check> checks = {
        {"every input lease returned", st.in_outstanding == 0 && st.in_acquires == st.in_releases,
         std::to_string(st.in_outstanding) + " outstanding, " + std::to_string(st.in_acquires)
             + " acquired, " + std::to_string(st.in_releases) + " released"},
        {"every output lease returned", st.out_outstanding == 0 && st.out_acquires == st.out_releases,
         std::to_string(st.out_outstanding) + " outstanding, " + std::to_string(st.out_acquires)
             + " acquired, " + std::to_string(st.out_releases) + " released"},
        {"no buffer returned twice", st.double_releases == 0,
         std::to_string(st.double_releases) + " double returns"},
        {"accepted == completed == emitted", st.accepted == st.completed && st.completed == st.emitted,
         n_of(st.accepted, st.completed, st.emitted)},
        {"no frame processed twice", st.duplicated == 0,
         std::to_string(st.duplicated) + " duplicated"},
        {"every frame carries its own values", st.mismatched == 0,
         std::to_string(st.mismatched) + " of " + std::to_string(st.emitted) + " wrong"},
        // The resource invariant: buffers checked out cannot exceed the pool.
        {"input leases within the pool", st.peak_in_leases <= cfg.buffers,
         "peak " + std::to_string(st.peak_in_leases) + " of " + std::to_string(cfg.buffers)},
        {"output leases within the pool", st.peak_out_leases <= out_cap,
         "peak " + std::to_string(st.peak_out_leases) + " of " + std::to_string(out_cap)},
        // min(resource, structural) bounds LEASE occupancy, not pipeline
        // occupancy. A frame that holds a buffer occupies a place, so both
        // bounds apply to it; a frame in the pipeline need not hold one.
        {"input leases within the ceiling", st.peak_in_leases <= bound,
         "peak " + std::to_string(st.peak_in_leases) + " of " + std::to_string(bound)
             + " = min(resource " + std::to_string(st.resource_capacity) + ", structural "
             + std::to_string(st.structural_capacity) + ")"},
        // The scheduler invariant. Only the structural bound applies here, and
        // this check can genuinely fail -- it is not true by construction.
        {"pipeline in-flight within the structural capacity",
         st.peak_pipeline_inflight <= st.structural_capacity,
         "peak " + std::to_string(st.peak_pipeline_inflight) + " of "
             + std::to_string(st.structural_capacity) + "; "
             + std::to_string(st.peak_pipeline_inflight > st.peak_in_leases
                              ? st.peak_pipeline_inflight - st.peak_in_leases : 0)
             + " more than the lease peak, which is the post-release tail"},
        {"reorder ring never overflowed", st.reorder_overflows == 0,
         std::to_string(st.reorder_overflows) + " overflows, held at most "
             + std::to_string(st.reorder_max_held) + " of " + std::to_string(bound + 1)},
        {"zero allocations in steady state", counts.allocations == 0,
         std::to_string(counts.allocations) + " allocations, " + std::to_string(counts.bytes) + " bytes"},
    };

    bool all_ok = true;
    for (const auto& c : checks) all_ok = all_ok && c.ok;

    if (json) {
        std::cout << "{\n  \"frames\": " << cfg.frames
                  << ",\n  \"buffers\": " << cfg.buffers
                  << ",\n  \"queues\": [" << cfg.q_pre << ", " << cfg.q_infer << ", " << cfg.q_post << "]"
                  << ",\n  \"workers\": [" << cfg.pre_workers << ", " << cfg.infer_workers
                  << ", " << cfg.post_workers << "]"
                  << ",\n  \"accepted\": " << st.accepted
                  << ",\n  \"emitted\": " << st.emitted
                  << ",\n  \"structural_capacity\": " << st.structural_capacity
                  << ",\n  \"resource_capacity\": " << st.resource_capacity
                  << ",\n  \"ceiling\": " << bound
                  << ",\n  \"peak_pipeline_inflight\": " << st.peak_pipeline_inflight
                  << ",\n  \"peak_in_leases\": " << st.peak_in_leases
                  << ",\n  \"peak_out_leases\": " << st.peak_out_leases
                  << ",\n  \"out_of_order_arrivals\": " << st.out_of_order
                  << ",\n  \"max_displacement\": " << st.max_displacement
                  << ",\n  \"reorder_max_held\": " << st.reorder_max_held
                  << ",\n  \"steady_state_allocations\": " << counts.allocations
                  << ",\n  \"seconds\": " << st.seconds
                  << ",\n  \"all_checks_pass\": " << (all_ok ? "true" : "false") << "\n}\n";
        return all_ok ? 0 : 1;
    }

    std::cout << "w5_scheduler   frames=" << cfg.frames
              << "  buffers=" << cfg.buffers << "(in)/" << out_cap << "(out)"
              << "  queues=" << cfg.q_pre << "/" << cfg.q_infer << "/" << cfg.q_post
              << "  workers=" << cfg.pre_workers << "/" << cfg.infer_workers << "/"
              << cfg.post_workers << "  bytes=" << cfg.bytes << "\n";
    std::cout << "bounds         structural " << st.structural_capacity
              << " = " << cfg.source_slots << " + (" << cfg.q_pre << "+" << cfg.q_infer
              << "+" << cfg.q_post << ") + (" << cfg.pre_workers << "+" << cfg.infer_workers
              << "+" << cfg.post_workers << ")\n";
    std::cout << "               resource   " << st.resource_capacity
              << " = the input pool, which every in-flight frame holds one of\n";
    std::cout << "               ceiling    " << bound << "\n";
    std::cout << "observed       pipeline in-flight " << st.peak_pipeline_inflight
              << ",  input leases " << st.peak_in_leases
              << ",  output leases " << st.peak_out_leases << "\n";
    std::cout << "shutdown       accepted " << st.accepted << "  completed " << st.completed
              << "  emitted " << st.emitted << "\n";
    std::cout << "order          " << st.out_of_order << " arrivals out of order, max displacement "
              << st.max_displacement << ", reorder held at most " << st.reorder_max_held << "\n";
    std::cout << "queues         peak depth " << st.peak_q_pre << "/" << st.peak_q_infer
              << "/" << st.peak_q_post << "\n\n";
    for (const auto& c : checks) {
        std::cout << "  " << (c.ok ? "ok  " : "FAIL") << "  " << c.what << "\n            "
                  << c.detail << "\n";
    }
    std::cout << "\nverdict        " << (all_ok ? "all checks pass" : "a check failed") << "\n";
    std::cout << "elapsed        " << st.seconds << " s\n";
    return all_ok ? 0 : 1;
}

int sweep(std::uint64_t frames, std::size_t bytes) {
    std::cout << "Two bounds, and the smaller wins.\n"
                 "  structural = source + sum(queue capacities) + sum(worker counts)\n"
                 "  resource   = the input pool\n\n";
    std::cout << "  buf  queues   pre/inf/post   struct  resrc  ceil   leases  inflight  binds\n";

    struct Row { std::uint32_t buffers; std::size_t qp, qi, qo; int pre, inf, post; };
    const std::vector<Row> rows = {
        // Hold the structure still and raise the buffers.
        {1, 2, 2, 2, 1, 2, 1}, {2, 2, 2, 2, 1, 2, 1}, {4, 2, 2, 2, 1, 2, 1},
        {8, 2, 2, 2, 1, 2, 1}, {16, 2, 2, 2, 1, 2, 1}, {32, 2, 2, 2, 1, 2, 1},
        // Hold the buffers still and raise the queues, together and singly.
        {32, 1, 1, 1, 1, 2, 1}, {32, 4, 4, 4, 1, 2, 1}, {32, 8, 8, 8, 1, 2, 1},
        {32, 8, 1, 1, 1, 2, 1}, {32, 1, 8, 1, 1, 2, 1},
        // Hold both still and raise the workers.
        {32, 1, 1, 1, 1, 1, 1}, {32, 1, 1, 1, 2, 4, 2}, {32, 1, 1, 1, 4, 8, 4},
        // A pool too small for the structure, and a structure too small for the pool.
        {3, 8, 8, 8, 2, 4, 2}, {64, 1, 1, 1, 1, 1, 1},
    };

    bool invariants_held = true;
    int saturated = 0;
    for (const Row& r : rows) {
        SchedulerConfig cfg;
        cfg.frames = frames;
        cfg.bytes = bytes;
        cfg.buffers = r.buffers;
        cfg.q_pre = r.qp; cfg.q_infer = r.qi; cfg.q_post = r.qo;
        cfg.pre_workers = r.pre;
        cfg.infer_workers = r.inf;
        cfg.post_workers = r.post;
        cfg.warmup = frames / 8;

        auto stages = synthetic_stages();
        SchedulerStats st = run_pipeline(cfg, stages);

        const std::uint32_t bound = ceiling(cfg);
        const char* binds = (st.structural_capacity < st.resource_capacity) ? "structure"
                          : (st.structural_capacity > st.resource_capacity) ? "buffers" : "both";
        // Two invariants, applied to the quantities they actually bound.
        const bool ok = st.peak_pipeline_inflight <= st.structural_capacity
                        && st.peak_in_leases <= bound;
        invariants_held = invariants_held && ok;
        if (st.peak_in_leases == bound) ++saturated;

        printf("  %3u  %2zu/%2zu/%-2zu  %2d/%2d/%-4d  %6u %6u %5u  %8u %7u   %-9s%s\n",
               r.buffers, r.qp, r.qi, r.qo, r.pre, r.inf, r.post,
               st.structural_capacity, st.resource_capacity, bound,
               st.peak_in_leases, st.peak_pipeline_inflight, binds,
               ok ? "" : "   INVARIANT BROKEN");
    }

    std::cout << "\n" << saturated << " of " << rows.size()
              << " configurations reached the ceiling; "
              << (invariants_held ? "none exceeded it" : "AT LEAST ONE EXCEEDED IT") << ".\n";
    std::cout << "\nThe ceiling column bounds LEASES, not pipeline occupancy: a frame keeps"
                 " its place in a worker for a moment after giving its buffers back, so"
                 " the inflight column can sit above it, bounded by the structural"
                 " column alone.\n";
    std::cout << "A ceiling is an upper bound, not a target. Reaching it depends on this run's\n"
                 "scheduling and on the workload being long enough; not exceeding it is the\n"
                 "property of the design, and is the only half asserted.\n";
    return invariants_held ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    SchedulerConfig cfg;
    bool json = false, do_sweep = false;
    try {
        for (int i = 1; i < argc; ++i) {
            std::string k = argv[i];
            auto val = [&]() -> std::string {
                if (i + 1 >= argc) throw std::runtime_error("missing value for " + k);
                return argv[++i];
            };
            if (k == "--frames") cfg.frames = std::stoull(val());
            else if (k == "--buffers") cfg.buffers = static_cast<std::uint32_t>(std::stoul(val()));
            else if (k == "--out-buffers") cfg.out_buffers = static_cast<std::uint32_t>(std::stoul(val()));
            else if (k == "--bytes") cfg.bytes = static_cast<std::size_t>(std::stoull(val()));
            else if (k == "--queue-capacity") {
                const auto q = static_cast<std::size_t>(std::stoull(val()));
                cfg.q_pre = cfg.q_infer = cfg.q_post = q;
            }
            else if (k == "--queue-pre") cfg.q_pre = static_cast<std::size_t>(std::stoull(val()));
            else if (k == "--queue-infer") cfg.q_infer = static_cast<std::size_t>(std::stoull(val()));
            else if (k == "--queue-post") cfg.q_post = static_cast<std::size_t>(std::stoull(val()));
            else if (k == "--source-slots") cfg.source_slots = static_cast<std::size_t>(std::stoull(val()));
            else if (k == "--pre-workers") cfg.pre_workers = std::stoi(val());
            else if (k == "--infer-workers") cfg.infer_workers = std::stoi(val());
            else if (k == "--post-workers") cfg.post_workers = std::stoi(val());
            else if (k == "--warmup") cfg.warmup = std::stoull(val());
            else if (k == "--interrupt-at") cfg.interrupt_at = std::stoull(val());
            else if (k == "--sweep") do_sweep = true;
            else if (k == "--json") json = true;
            else if (k == "-h" || k == "--help") {
                std::cout <<
                    "w5_scheduler -- staged pipeline with bounded queues (no accelerator)\n\n"
                    "  --frames N           frames to submit (default 1000)\n"
                    "  --buffers K          input pool size (default 4)\n"
                    "  --out-buffers K      output pool size (default: same as --buffers)\n"
                    "  --bytes B            bytes per buffer (default 150528)\n"
                    "  --queue-capacity Q   set all three queue depths (default 2)\n"
                    "  --queue-pre/-infer/-post Q   set one queue's depth\n"
                    "  --source-slots N     frames the source may hold (default 1)\n"
                    "  --pre-workers N      preprocess workers (default 1)\n"
                    "  --infer-workers N    infer workers (default 2)\n"
                    "  --post-workers N     postprocess workers (default 1)\n"
                    "  --interrupt-at N     stop accepting after N frames, then drain\n"
                    "  --sweep              vary each knob and report what bounds concurrency\n"
                    "  --json               machine-readable output\n";
                return 0;
            } else throw std::runtime_error("unknown argument " + k);
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
    if (cfg.warmup >= cfg.frames) cfg.warmup = cfg.frames / 4;

    return do_sweep ? sweep(cfg.frames, cfg.bytes) : single(cfg, json);
}
