// W5-2 -- the generic scheduler, and the knob that actually bounds concurrency.
//
// The plan's question for this phase was where the in-flight ceiling comes
// from: a queue capacity, a worker count, or a buffer count. DX-RT raises it by
// separating device selection from submission (`lib/device_pool/`), which
// leaves three plausible answers and no obvious one.
//
// It is not a measurement. A frame holds its input lease from the moment the
// reader takes one until postprocessing gives it back, so concurrency is
// bounded by the pool; and a frame has to be somewhere, so it is also bounded
// by the number of places to stand. The ceiling is the smaller of the two, and
// `--sweep` checks that arithmetic against what the pipeline actually does.
//
// Both halves are worth stating because each kills a plausible intuition:
// deeper queues cannot buy concurrency the pool will not fund, and more buffers
// cannot buy concurrency the structure has no room for.
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
        {"peak in-flight within the computed ceiling", st.peak_inflight <= st.predicted_ceiling,
         std::to_string(st.peak_inflight) + " observed, ceiling " + std::to_string(st.predicted_ceiling)},
        {"reorder ring never overflowed", st.reorder_overflows == 0,
         std::to_string(st.reorder_overflows) + " overflows, held at most "
             + std::to_string(st.reorder_max_held) + " of " + std::to_string(st.predicted_ceiling + 1)},
        {"zero allocations in steady state", counts.allocations == 0,
         std::to_string(counts.allocations) + " allocations, " + std::to_string(counts.bytes) + " bytes"},
    };

    bool all_ok = true;
    for (const auto& c : checks) all_ok = all_ok && c.ok;

    if (json) {
        std::cout << "{\n  \"frames\": " << cfg.frames
                  << ",\n  \"buffers\": " << cfg.buffers
                  << ",\n  \"queue_capacity\": " << cfg.queue_capacity
                  << ",\n  \"pre_workers\": " << cfg.pre_workers
                  << ",\n  \"infer_workers\": " << cfg.infer_workers
                  << ",\n  \"post_workers\": " << cfg.post_workers
                  << ",\n  \"accepted\": " << st.accepted
                  << ",\n  \"emitted\": " << st.emitted
                  << ",\n  \"predicted_ceiling\": " << st.predicted_ceiling
                  << ",\n  \"peak_inflight\": " << st.peak_inflight
                  << ",\n  \"out_of_order_arrivals\": " << st.out_of_order
                  << ",\n  \"max_displacement\": " << st.max_displacement
                  << ",\n  \"reorder_max_held\": " << st.reorder_max_held
                  << ",\n  \"steady_state_allocations\": " << counts.allocations
                  << ",\n  \"seconds\": " << st.seconds
                  << ",\n  \"all_checks_pass\": " << (all_ok ? "true" : "false") << "\n}\n";
        return all_ok ? 0 : 1;
    }

    std::cout << "w5_scheduler   frames=" << cfg.frames
              << "  buffers=" << cfg.buffers
              << "  queue=" << cfg.queue_capacity
              << "  workers pre/infer/post=" << cfg.pre_workers << "/"
              << cfg.infer_workers << "/" << cfg.post_workers
              << "  bytes=" << cfg.bytes << "\n";
    std::cout << "ceiling        " << st.predicted_ceiling
              << " = min(buffers " << cfg.buffers << ", places "
              << (1 + 3 * cfg.queue_capacity + static_cast<std::size_t>(cfg.pre_workers)
                  + static_cast<std::size_t>(cfg.infer_workers)
                  + static_cast<std::size_t>(cfg.post_workers))
              << ")   observed peak " << st.peak_inflight << "\n";
    std::cout << "shutdown       accepted " << st.accepted << "  completed " << st.completed
              << "  emitted " << st.emitted << "\n";
    std::cout << "order          " << st.out_of_order << " arrivals out of order, max displacement "
              << st.max_displacement << ", reorder held at most " << st.reorder_max_held << "\n";
    std::cout << "queues         peak depth pre " << st.peak_q_pre << ", infer " << st.peak_q_infer
              << ", post " << st.peak_q_post << " (capacity " << cfg.queue_capacity << ")\n\n";
    for (const auto& c : checks) {
        std::cout << "  " << (c.ok ? "ok  " : "FAIL") << "  " << c.what << "\n            "
                  << c.detail << "\n";
    }
    std::cout << "\nverdict        " << (all_ok ? "all checks pass" : "a check failed") << "\n";
    std::cout << "elapsed        " << st.seconds << " s\n";
    return all_ok ? 0 : 1;
}

int sweep(std::uint64_t frames, std::size_t bytes) {
    std::cout << "Which knob bounds concurrency? Ceiling is min(buffers, places),\n"
                 "where places = 1 + 3*queue + pre + infer + post.\n\n";
    std::cout << "  buf  queue  pre/inf/post   places  ceiling   peak   binds\n";

    struct Row { std::uint32_t buffers; std::size_t q; int pre, inf, post; };
    const std::vector<Row> rows = {
        // Hold the structure still and raise the buffers.
        {1, 2, 1, 2, 1}, {2, 2, 1, 2, 1}, {4, 2, 1, 2, 1}, {8, 2, 1, 2, 1},
        {16, 2, 1, 2, 1}, {32, 2, 1, 2, 1},
        // Hold the buffers still and raise the queues.
        {32, 1, 1, 2, 1}, {32, 4, 1, 2, 1}, {32, 8, 1, 2, 1},
        // Hold both still and raise the workers.
        {32, 1, 1, 1, 1}, {32, 1, 2, 4, 2}, {32, 1, 4, 8, 4},
        // A pool too small for the structure, and a structure too small for the pool.
        {3, 8, 2, 4, 2}, {64, 1, 1, 1, 1},
    };

    bool invariant_held = true;
    int saturated = 0;
    for (const Row& r : rows) {
        SchedulerConfig cfg;
        cfg.frames = frames;
        cfg.bytes = bytes;
        cfg.buffers = r.buffers;
        cfg.queue_capacity = r.q;
        cfg.pre_workers = r.pre;
        cfg.infer_workers = r.inf;
        cfg.post_workers = r.post;
        cfg.warmup = frames / 8;

        auto stages = synthetic_stages();
        SchedulerStats st = run_pipeline(cfg, stages);

        const std::size_t places = 1 + 3 * r.q + static_cast<std::size_t>(r.pre + r.inf + r.post);
        const char* binds = (places < r.buffers) ? "structure" : "buffers";
        if (places == r.buffers) binds = "both";
        const bool ok = st.peak_inflight <= st.predicted_ceiling;
        invariant_held = invariant_held && ok;
        if (st.peak_inflight == st.predicted_ceiling) ++saturated;

        printf("  %3u  %5zu  %2d/%2d/%-4d  %7zu  %7u  %5u   %-9s%s\n",
               r.buffers, r.q, r.pre, r.inf, r.post, places,
               st.predicted_ceiling, st.peak_inflight, binds,
               ok ? "" : "   INVARIANT BROKEN");
    }

    std::cout << "\n" << saturated << " of " << rows.size()
              << " configurations reached the ceiling; "
              << (invariant_held ? "none exceeded it" : "AT LEAST ONE EXCEEDED IT") << ".\n";
    std::cout << "Reaching it is a property of this run. Not exceeding it is a property of\n"
                 "the design, and is the half worth asserting.\n";
    return invariant_held ? 0 : 1;
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
            else if (k == "--bytes") cfg.bytes = static_cast<std::size_t>(std::stoull(val()));
            else if (k == "--queue-capacity") cfg.queue_capacity = static_cast<std::size_t>(std::stoull(val()));
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
                    "  --buffers K          buffers in each pool (default 4)\n"
                    "  --bytes B            bytes per buffer (default 150528)\n"
                    "  --queue-capacity Q   depth of each of the three queues (default 2)\n"
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
