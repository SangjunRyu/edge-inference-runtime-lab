// W5-1 -- what a bounded pipeline owns, checked rather than described.
//
// This harness has no OpenVINO and no OpenCV in it. That is not a
// simplification, it is the measurement: with nothing else linked, every
// allocation the counter sees was asked for by this repository's own code, so
// the claim "zero allocations in steady state" and the number printed below are
// the same claim. W5's plan requires the fence to be printed; here the fence is
// the link line.
//
// It also means this runs unchanged on the MX150, which has no accelerator.
//
// Four checks, and each one has an injection that makes it fail. A check that
// has never failed has not been shown to work -- the same reason repro/ carries
// `--mode broken` alongside `--mode fixed`, and the reason stats/03 exists.
//
//   --inject none            every check passes
//   --inject leak            an output lease is dropped without returning
//   --inject alloc           one small allocation per frame in steady state
//   --inject early-release   the input lease goes back before the read of it
//                            finishes -- the silent one
//
// The last injection is the point of the phase. Releasing early is legal C++,
// costs nothing, and produces the right number of frames carrying another
// frame's values. `repro/cpp/01` proved the contract it violates: a runtime
// handed a pointer reads it at completion time, not at hand-off time. So the
// rule the pool exists to enforce is not "the lease travels with the frame" but
// "the input lease is pinned to its slot until the work reading it is done".

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "alloc_counter.hpp"
#include "ownership.hpp"
#include "ring_queue.hpp"

namespace {

using namespace eirl;

constexpr std::uint8_t kXor = 0x5A;

/// Somewhere for the alloc injection's pointer to escape to, so the compiler
/// may not delete the allocation it is meant to expose. Never read back.
std::atomic<std::uintptr_t> g_escape{0};

enum class Inject { None, Leak, Alloc, EarlyRelease };

struct Args {
    std::uint64_t frames = 1000;
    std::uint32_t inflight = 4;
    int workers = 2;
    std::size_t bytes = 150528;  // the densenet121 input, so the size is not arbitrary
    std::uint64_t warmup = 64;
    std::uint64_t interrupt_at = 0;  // 0 = run to the end
    Inject inject = Inject::None;
    bool json = false;
};

const char* inject_name(Inject i) {
    switch (i) {
        case Inject::Leak: return "leak";
        case Inject::Alloc: return "alloc";
        case Inject::EarlyRelease: return "early-release";
        default: return "none";
    }
}

/// Which check each injection is predicted to break. The run is correct when
/// exactly this one fails, which is what makes an injection a test of the
/// check rather than a demonstration of a bug.
const char* predicted_failure(Inject i) {
    switch (i) {
        case Inject::Leak: return "every output lease returned";
        case Inject::Alloc: return "zero allocations in steady state";
        case Inject::EarlyRelease: return "every frame carries its own values";
        default: return nullptr;
    }
}

inline std::uint8_t pattern(std::uint64_t seq, std::size_t i) {
    return static_cast<std::uint8_t>(seq * 31u + i * 7u);
}

struct Tally {
    std::atomic<std::uint64_t> accepted{0};
    std::atomic<std::uint64_t> completed{0};
    std::atomic<std::uint64_t> emitted{0};
    std::atomic<std::uint64_t> mismatched{0};
    std::atomic<std::uint64_t> duplicated{0};
    std::atomic<std::uint64_t> leaked_on_purpose{0};
};

struct Check {
    std::string what;
    bool ok;
    std::string detail;
};

bool parse(int argc, char** argv, Args& a) {
    auto next = [&](int& i) -> std::string {
        if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + argv[i]);
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k == "--frames") a.frames = std::stoull(next(i));
        else if (k == "--inflight") a.inflight = static_cast<std::uint32_t>(std::stoul(next(i)));
        else if (k == "--workers") a.workers = std::stoi(next(i));
        else if (k == "--bytes") a.bytes = static_cast<std::size_t>(std::stoull(next(i)));
        else if (k == "--warmup") a.warmup = std::stoull(next(i));
        else if (k == "--interrupt-at") a.interrupt_at = std::stoull(next(i));
        else if (k == "--json") a.json = true;
        else if (k == "--inject") {
            std::string v = next(i);
            if (v == "none") a.inject = Inject::None;
            else if (v == "leak") a.inject = Inject::Leak;
            else if (v == "alloc") a.inject = Inject::Alloc;
            else if (v == "early-release") a.inject = Inject::EarlyRelease;
            else throw std::runtime_error("unknown --inject " + v);
        } else if (k == "-h" || k == "--help") {
            std::cout <<
                "w5_ownership -- bounded pipeline ownership checks (no accelerator)\n\n"
                "  --frames N          frames to submit (default 1000)\n"
                "  --inflight K        buffers in each pool (default 4)\n"
                "  --workers W         worker threads (default 2)\n"
                "  --bytes B           bytes per buffer (default 150528)\n"
                "  --warmup N          frames before allocation counting arms (default 64)\n"
                "  --interrupt-at N    stop accepting after N frames, then drain\n"
                "  --inject WHAT       none | leak | alloc | early-release\n"
                "  --json              machine-readable output\n";
            return false;
        } else {
            throw std::runtime_error("unknown argument " + k);
        }
    }
    if (a.inflight < 1) throw std::runtime_error("--inflight must be at least 1");
    if (a.workers < 1) throw std::runtime_error("--workers must be at least 1");
    if (a.warmup >= a.frames) a.warmup = a.frames / 4;
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    try {
        if (!parse(argc, argv, args)) return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }

    InputPool in_pool(args.inflight, args.bytes);
    OutputPool out_pool(args.inflight, args.bytes);
    RingQueue<FrameContext> queue(args.inflight);
    Tally tally;

    // Allocated before counting arms, for the same reason the pools are: this
    // is start-up, and a steady-state claim that included start-up would be
    // measuring the wrong thing.
    auto seen = std::make_unique<std::atomic<std::uint8_t>[]>(args.frames);
    for (std::uint64_t i = 0; i < args.frames; ++i) seen[i].store(0);

    std::atomic<bool> armed{false};
    const std::uint64_t accept_limit =
        args.interrupt_at ? std::min(args.interrupt_at, args.frames) : args.frames;

    auto t0 = std::chrono::steady_clock::now();

    std::thread reader([&] {
        for (std::uint64_t seq = 0; seq < accept_limit; ++seq) {
            InputLease lease = in_pool.acquire();
            if (!lease) break;  // pool closed
            std::uint8_t* p = lease.data();
            for (std::size_t i = 0; i < args.bytes; ++i) p[i] = pattern(seq, i);

            FrameContext ctx;
            ctx.seq = seq;
            ctx.in = std::move(lease);
            if (!queue.push(std::move(ctx))) break;
            tally.accepted.fetch_add(1, std::memory_order_relaxed);

            if (seq + 1 == args.warmup) {
                eirl::alloc::reset();
                eirl::alloc::arm();
                armed.store(true, std::memory_order_release);
            }
        }
        queue.close();  // stop accepting; queued work still drains
    });

    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(args.workers));
    for (int w = 0; w < args.workers; ++w) {
        workers.emplace_back([&] {
            for (;;) {
                auto got = queue.pop();
                if (!got) break;
                FrameContext ctx = std::move(*got);

                OutputLease out = out_pool.acquire();
                if (!out) break;

                // Submit: the reader of this buffer is handed a pointer now and
                // dereferences it later. That is the contract repro/cpp/01
                // established, reproduced here without a runtime to hold it.
                const std::uint8_t* submitted = ctx.in.data();
                const std::size_t n = ctx.in.size();

                if (args.inject == Inject::EarlyRelease) {
                    ctx.in.release();  // the defect: the slot is free while still in use
                }
                std::this_thread::yield();  // identical in both modes

                std::uint8_t* dst = out.data();
                for (std::size_t i = 0; i < n; ++i) dst[i] = static_cast<std::uint8_t>(submitted[i] ^ kXor);
                tally.completed.fetch_add(1, std::memory_order_relaxed);

                bool ok = true;
                for (std::size_t i = 0; i < n; ++i) {
                    if (dst[i] != static_cast<std::uint8_t>(pattern(ctx.seq, i) ^ kXor)) { ok = false; break; }
                }
                if (!ok) tally.mismatched.fetch_add(1, std::memory_order_relaxed);

                if (seen[ctx.seq].fetch_add(1, std::memory_order_relaxed) != 0) {
                    tally.duplicated.fetch_add(1, std::memory_order_relaxed);
                }

                if (args.inject == Inject::Alloc && armed.load(std::memory_order_acquire)) {
                    // Small, plausible, and exactly the kind of thing that
                    // creeps into a hot path unnoticed.
                    //
                    // The pointer has to escape or there is nothing to count.
                    // The first version of this injection stored it in a
                    // `volatile` local and the counter stayed at zero: since
                    // C++14 a compiler may delete a new/delete pair outright
                    // when the allocation cannot be observed, and `volatile` on
                    // the *variable* does not make the allocation observable.
                    // Publishing the address does. Worth knowing before
                    // trusting any allocation count taken at -O2.
                    auto* scratch = new std::uint8_t[64];
                    g_escape.store(reinterpret_cast<std::uintptr_t>(scratch),
                                   std::memory_order_relaxed);
                    scratch[0] = static_cast<std::uint8_t>(ctx.seq);
                    g_escape.fetch_add(scratch[0], std::memory_order_relaxed);
                    delete[] scratch;
                }

                if (args.inject == Inject::Leak && armed.load(std::memory_order_acquire)
                    && tally.leaked_on_purpose.load(std::memory_order_relaxed) == 0) {
                    tally.leaked_on_purpose.fetch_add(1, std::memory_order_relaxed);
                    out.disown();  // dropped without returning
                }

                tally.emitted.fetch_add(1, std::memory_order_relaxed);
                // ctx and out die here: leases return themselves.
            }
        });
    }

    reader.join();
    for (auto& t : workers) t.join();
    eirl::alloc::disarm();
    const auto counts = eirl::alloc::read();
    auto t1 = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(t1 - t0).count();

    in_pool.close();
    out_pool.close();

    const std::uint64_t accepted = tally.accepted.load();
    const std::uint64_t completed = tally.completed.load();
    const std::uint64_t emitted = tally.emitted.load();

    std::vector<Check> checks;
    checks.push_back({"every input lease returned", in_pool.all_returned(),
                      std::to_string(in_pool.outstanding()) + " outstanding, "
                      + std::to_string(in_pool.acquires()) + " acquired, "
                      + std::to_string(in_pool.releases()) + " released"});
    checks.push_back({"every output lease returned", out_pool.all_returned(),
                      std::to_string(out_pool.outstanding()) + " outstanding, "
                      + std::to_string(out_pool.acquires()) + " acquired, "
                      + std::to_string(out_pool.releases()) + " released"});
    checks.push_back({"no buffer returned twice",
                      in_pool.double_releases() == 0 && out_pool.double_releases() == 0,
                      std::to_string(in_pool.double_releases() + out_pool.double_releases())
                      + " double returns"});
    checks.push_back({"accepted == completed == emitted",
                      accepted == completed && completed == emitted,
                      std::to_string(accepted) + " / " + std::to_string(completed)
                      + " / " + std::to_string(emitted)});
    checks.push_back({"no frame processed twice", tally.duplicated.load() == 0,
                      std::to_string(tally.duplicated.load()) + " duplicated"});
    checks.push_back({"every frame carries its own values", tally.mismatched.load() == 0,
                      std::to_string(tally.mismatched.load()) + " of "
                      + std::to_string(emitted) + " wrong"});
    checks.push_back({"zero allocations in steady state", counts.allocations == 0,
                      std::to_string(counts.allocations) + " allocations, "
                      + std::to_string(counts.bytes) + " bytes, after frame "
                      + std::to_string(args.warmup)});

    const char* predicted = predicted_failure(args.inject);
    bool as_predicted = true;
    for (const auto& c : checks) {
        const bool should_fail = predicted && c.what == predicted;
        if (c.ok == should_fail) as_predicted = false;
    }

    if (args.json) {
        std::cout << "{\n  \"inject\": \"" << inject_name(args.inject) << "\",\n"
                  << "  \"frames\": " << args.frames << ",\n"
                  << "  \"inflight\": " << args.inflight << ",\n"
                  << "  \"workers\": " << args.workers << ",\n"
                  << "  \"bytes_per_buffer\": " << args.bytes << ",\n"
                  << "  \"accepted\": " << accepted << ",\n"
                  << "  \"completed\": " << completed << ",\n"
                  << "  \"emitted\": " << emitted << ",\n"
                  << "  \"steady_state_allocations\": " << counts.allocations << ",\n"
                  << "  \"seconds\": " << seconds << ",\n"
                  << "  \"checks\": [\n";
        for (std::size_t i = 0; i < checks.size(); ++i) {
            std::cout << "    {\"what\": \"" << checks[i].what << "\", \"ok\": "
                      << (checks[i].ok ? "true" : "false") << ", \"detail\": \""
                      << checks[i].detail << "\"}" << (i + 1 < checks.size() ? "," : "") << "\n";
        }
        std::cout << "  ],\n  \"as_predicted\": " << (as_predicted ? "true" : "false") << "\n}\n";
    } else {
        std::cout << "w5_ownership   inject=" << inject_name(args.inject)
                  << "  frames=" << args.frames
                  << "  inflight=" << args.inflight
                  << "  workers=" << args.workers
                  << "  bytes=" << args.bytes << "\n";
        if (args.interrupt_at)
            std::cout << "               interrupted after " << accept_limit << " frames\n";
        std::cout << "pools          input " << in_pool.capacity() << " x " << in_pool.bytes_each()
                  << " B,  output " << out_pool.capacity() << " x " << out_pool.bytes_each()
                  << " B,  allocated once at start-up\n";
        std::cout << "shutdown       submitted " << args.frames
                  << "  accepted " << accepted
                  << "  completed " << completed
                  << "  emitted " << emitted << "\n";
        std::cout << "\n";
        for (const auto& c : checks) {
            std::cout << "  " << (c.ok ? "ok  " : "FAIL") << "  "
                      << c.what << "\n            " << c.detail << "\n";
        }
        std::cout << "\n";
        if (predicted) {
            std::cout << "predicted      injection '" << inject_name(args.inject)
                      << "' breaks: " << predicted << "\n";
            std::cout << "verdict        " << (as_predicted ? "as predicted" : "NOT as predicted")
                      << "\n";
        } else {
            std::cout << "verdict        " << (as_predicted ? "all checks pass" : "a check failed")
                      << "\n";
        }
        std::cout << "elapsed        " << seconds << " s\n";
    }

    return as_predicted ? 0 : 1;
}
