// W5-4 -- what a bounded pipeline does when more work arrives than it can hold.
//
// The plan wrote down, before this was built, why the obvious version does not
// work: with a video file read as fast as the pipeline consumes it, `Block` and
// `DropOldest` are indistinguishable. Under Block the reader simply waits and
// the pipeline paces itself; there is no external clock, so nothing is ever
// late and a drop policy has nothing to drop.
//
// **Freshness only exists when the source has its own clock.** So the source
// here is paced on the wall, at `--source-fps`, and does not wait for anything
// except under `block`, where waiting is the policy. That is what a camera is,
// and every interesting difference below follows from the source refusing to
// slow down.
//
// The measurement that matters is not latency through the pipeline. It is
// **age at output**: how old a frame is when it comes out, measured from the
// moment it was due, not from the moment the pipeline got round to it. A policy
// that quietly captures late has good pipeline latency and stale output, and
// the first number hides the second.
//
//   w5_backpressure --compare
//   w5_backpressure --policy drop-oldest --source-fps 60 --service-us 25000
//
// Throughput is measured, so it is reported with a spread across repeats. Drops
// and peak residency are counted and do not depend on the machine.
//
// Links no OpenVINO and no OpenCV.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "ownership.hpp"

namespace {

using namespace eirl;
using Clock = std::chrono::steady_clock;
using Tp = Clock::time_point;

enum class Policy { Unbounded, Block, DropOldest, DropNewest };

const char* policy_name(Policy p) {
    switch (p) {
        case Policy::Unbounded: return "unbounded";
        case Policy::Block: return "block";
        case Policy::DropOldest: return "drop-oldest";
        case Policy::DropNewest: return "drop-newest";
    }
    return "?";
}

struct Item {
    std::uint64_t seq = 0;
    Tp due{};
    InputLease lease;
};

/// Wait until `tp` without giving the scheduler a chance to overshoot by a
/// whole timer tick. Sleep for the bulk, spin for the last millisecond: Windows
/// timers are coarse enough that `sleep_for` alone turns a 60 FPS source into
/// something else, which would make the source the experiment.
void wait_until(Tp tp) {
    for (;;) {
        const auto now = Clock::now();
        if (now >= tp) return;
        const auto left = tp - now;
        if (left > std::chrono::milliseconds(2)) {
            std::this_thread::sleep_for(left - std::chrono::milliseconds(1));
        } else {
            std::this_thread::yield();
        }
    }
}

/// The admission point. One queue, four policies, and the differences between
/// them live here rather than being spread through the source.
class PolicyQueue {
public:
    PolicyQueue(Policy p, std::size_t capacity) : policy_(p), cap_(capacity) {}

    /// Returns false only when the queue is closed. Drops are counted, not
    /// reported as failures: a dropped frame is an outcome of the policy.
    bool offer(Item item) {
        std::unique_lock<std::mutex> lk(m_);
        if (closed_) return false;

        if (policy_ == Policy::Block) {
            not_full_.wait(lk, [&] { return q_.size() < cap_ || closed_; });
            if (closed_) return false;
        } else if (policy_ != Policy::Unbounded && q_.size() >= cap_) {
            if (policy_ == Policy::DropNewest) {
                ++dropped_;
                return true;   // the lease goes back as `item` dies
            }
            // DropOldest: the freshest frame is the one worth keeping.
            q_.pop_front();
            ++dropped_;
        }

        q_.push_back(std::move(item));
        if (q_.size() > peak_) peak_ = q_.size();
        lk.unlock();
        not_empty_.notify_one();
        return true;
    }

    bool take(Item& out) {
        std::unique_lock<std::mutex> lk(m_);
        not_empty_.wait(lk, [&] { return !q_.empty() || closed_; });
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop_front();
        lk.unlock();
        not_full_.notify_one();
        return true;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lk(m_);
            closed_ = true;
        }
        not_empty_.notify_all();
        not_full_.notify_all();
    }

    std::uint64_t dropped() const {
        std::lock_guard<std::mutex> lk(m_);
        return dropped_;
    }
    std::size_t peak() const {
        std::lock_guard<std::mutex> lk(m_);
        return peak_;
    }

private:
    mutable std::mutex m_;
    std::condition_variable not_empty_, not_full_;
    std::deque<Item> q_;
    Policy policy_;
    std::size_t cap_;
    std::size_t peak_ = 0;
    std::uint64_t dropped_ = 0;
    bool closed_ = false;
};

struct Result {
    Policy policy = Policy::Block;
    std::uint64_t offered = 0, accepted = 0, completed = 0, dropped = 0;
    std::size_t peak_resident = 0;
    double throughput = 0.0;
    double age_p50 = 0.0, age_p95 = 0.0, age_max = 0.0;
    double source_late_p95 = 0.0;
    double seconds = 0.0;
};

struct Config {
    Policy policy = Policy::Block;
    double source_fps = 60.0;
    long service_us = 25000;      // one worker: 40 frames per second of capacity
    std::uint64_t frames = 300;
    std::uint32_t buffers = 8;
    std::size_t capacity = 4;
    std::size_t bytes = 157696;   // the densenet121 input, padded
};

double pct(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double idx = p * static_cast<double>(v.size() - 1);
    const auto lo = static_cast<std::size_t>(idx);
    const auto hi = std::min(lo + 1, v.size() - 1);
    const double f = idx - static_cast<double>(lo);
    return v[lo] * (1.0 - f) + v[hi] * f;
}

double cv(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    const double m = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
    if (m == 0.0) return 0.0;
    double s = 0.0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / static_cast<double>(v.size() - 1)) / m;
}

Result run_once(const Config& cfg) {
    InputPool pool(cfg.buffers, cfg.bytes);
    PolicyQueue queue(cfg.policy, cfg.capacity);

    std::atomic<std::uint64_t> accepted{0}, completed{0};
    std::vector<double> ages;
    std::vector<double> lateness;
    std::mutex ages_m;

    const auto t0 = Clock::now();
    const auto period = std::chrono::duration<double>(1.0 / cfg.source_fps);

    std::thread worker([&] {
        Item item;
        while (queue.take(item)) {
            // Synthetic service time, spun rather than slept so the rate is the
            // rate asked for rather than the timer's granularity.
            const auto until = Clock::now() + std::chrono::microseconds(cfg.service_us);
            volatile std::uint64_t sink = 0;
            while (Clock::now() < until) {
                sink += item.lease.data()[0];
            }
            (void)sink;
            const double age = std::chrono::duration<double, std::milli>(
                                   Clock::now() - item.due).count();
            {
                std::lock_guard<std::mutex> lk(ages_m);
                ages.push_back(age);
            }
            completed.fetch_add(1, std::memory_order_relaxed);
            item.lease.release();
        }
    });

    for (std::uint64_t seq = 0; seq < cfg.frames; ++seq) {
        const Tp due = t0 + std::chrono::duration_cast<Clock::duration>(period * static_cast<double>(seq));
        wait_until(due);
        // How late the source itself is. Under `block` this grows without
        // bound, and it is the cost the policy hides from a pipeline-latency
        // measurement.
        lateness.push_back(std::chrono::duration<double, std::milli>(Clock::now() - due).count());

        InputLease lease = (cfg.policy == Policy::Block) ? pool.acquire() : pool.try_acquire();
        if (!lease) {
            // No buffer and the policy will not wait: the frame never enters.
            continue;
        }
        lease.data()[0] = static_cast<std::uint8_t>(seq);
        Item item{seq, due, std::move(lease)};
        if (!queue.offer(std::move(item))) break;
        accepted.fetch_add(1, std::memory_order_relaxed);
    }
    queue.close();
    worker.join();
    const auto t1 = Clock::now();

    Result r;
    r.policy = cfg.policy;
    r.offered = cfg.frames;
    r.accepted = accepted.load();
    r.completed = completed.load();
    r.dropped = queue.dropped() + (r.offered - r.accepted);
    r.peak_resident = queue.peak();
    r.seconds = std::chrono::duration<double>(t1 - t0).count();
    r.throughput = r.seconds > 0 ? static_cast<double>(r.completed) / r.seconds : 0.0;
    r.age_p50 = pct(ages, 0.50);
    r.age_p95 = pct(ages, 0.95);
    r.age_max = ages.empty() ? 0.0 : *std::max_element(ages.begin(), ages.end());
    r.source_late_p95 = pct(lateness, 0.95);
    return r;
}

void print_one(const Result& r) {
    printf("policy         %s\n", policy_name(r.policy));
    printf("offered        %llu frames\n", static_cast<unsigned long long>(r.offered));
    printf("completed      %llu   dropped %llu\n",
           static_cast<unsigned long long>(r.completed),
           static_cast<unsigned long long>(r.dropped));
    printf("throughput     %.2f FPS over %.2f s\n", r.throughput, r.seconds);
    printf("age at output  p50 %.1f ms   p95 %.1f ms   max %.1f ms\n",
           r.age_p50, r.age_p95, r.age_max);
    printf("source late    p95 %.1f ms\n", r.source_late_p95);
    printf("peak resident  %zu frames\n", r.peak_resident);
}

int compare(Config cfg, int repeats) {
    const std::vector<Policy> policies = {
        Policy::Unbounded, Policy::Block, Policy::DropOldest, Policy::DropNewest};

    std::vector<std::vector<Result>> runs(policies.size());
    for (int r = 0; r < repeats; ++r) {
        // Rotate the order every repeat: a machine that drifts through the
        // session must not charge the drift to whichever policy went last.
        for (std::size_t k = 0; k < policies.size(); ++k) {
            const std::size_t idx = (k + static_cast<std::size_t>(r)) % policies.size();
            Config c = cfg;
            c.policy = policies[idx];
            runs[idx].push_back(run_once(c));
        }
    }

    printf("source %.0f FPS, one worker at %ld us per frame (%.0f FPS of capacity),"
           " queue %zu, buffers %u\n",
           cfg.source_fps, cfg.service_us, 1e6 / static_cast<double>(cfg.service_us),
           cfg.capacity, cfg.buffers);
    printf("%d repeats, order rotated. Age is measured from when a frame was DUE.\n\n",
           repeats);
    printf("  %-12s %9s %6s %9s %9s %9s %9s\n",
           "policy", "FPS", "CV", "p95 age", "max age", "dropped", "resident");

    std::vector<double> baseline;
    for (std::size_t i = 0; i < policies.size(); ++i) {
        std::vector<double> fps, p95;
        std::uint64_t dropped = 0;
        std::size_t resident = 0;
        double maxage = 0;
        for (const auto& r : runs[i]) {
            fps.push_back(r.throughput);
            p95.push_back(r.age_p95);
            dropped += r.dropped;
            resident = std::max(resident, r.peak_resident);
            maxage = std::max(maxage, r.age_max);
        }
        std::sort(fps.begin(), fps.end());
        std::sort(p95.begin(), p95.end());
        const double med_fps = fps[fps.size() / 2];
        if (i == 0) baseline = fps;
        printf("  %-12s %9.2f %5.1f%% %9.1f %9.1f %9llu %9zu\n",
               policy_name(policies[i]), med_fps, cv(fps) * 100,
               p95[p95.size() / 2], maxage,
               static_cast<unsigned long long>(dropped / static_cast<std::uint64_t>(repeats)),
               resident);
    }

    // The claim the experiment was set up to test.
    double lo = 1e9, hi = 0;
    for (std::size_t i = 0; i < policies.size(); ++i) {
        std::vector<double> fps;
        for (const auto& r : runs[i]) fps.push_back(r.throughput);
        std::sort(fps.begin(), fps.end());
        const double m = fps[fps.size() / 2];
        lo = std::min(lo, m);
        hi = std::max(hi, m);
    }
    const double spread = lo > 0 ? (hi - lo) / lo : 0.0;
    printf("\nthroughput spread across policies: %.1f%%\n", spread * 100);
    if (spread < 0.05) {
        printf("All four sustain the same rate. The queue is not what limits it -- the\n"
               "worker is -- so a deeper queue buys nothing but older frames and more\n"
               "resident memory. That is the result this experiment exists to produce.\n");
    } else {
        printf("The policies do NOT sustain the same rate here, which contradicts the\n"
               "expected result and is worth explaining before it is quoted.\n");
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Config cfg;
    bool do_compare = false;
    int repeats = 3;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string k = argv[i];
            auto val = [&]() -> std::string {
                if (i + 1 >= argc) throw std::runtime_error("missing value for " + k);
                return argv[++i];
            };
            if (k == "--policy") {
                const std::string v = val();
                if (v == "unbounded") cfg.policy = Policy::Unbounded;
                else if (v == "block") cfg.policy = Policy::Block;
                else if (v == "drop-oldest") cfg.policy = Policy::DropOldest;
                else if (v == "drop-newest") cfg.policy = Policy::DropNewest;
                else throw std::runtime_error("unknown --policy " + v);
            }
            else if (k == "--source-fps") cfg.source_fps = std::stod(val());
            else if (k == "--service-us") cfg.service_us = std::stol(val());
            else if (k == "--frames") cfg.frames = std::stoull(val());
            else if (k == "--buffers") cfg.buffers = static_cast<std::uint32_t>(std::stoul(val()));
            else if (k == "--queue-capacity") cfg.capacity = static_cast<std::size_t>(std::stoull(val()));
            else if (k == "--repeat") repeats = std::stoi(val());
            else if (k == "--compare") do_compare = true;
            else if (k == "-h" || k == "--help") {
                std::cout <<
                    "w5_backpressure -- admission policies against a source with its own clock\n\n"
                    "  --compare            run all four policies and compare\n"
                    "  --policy P           unbounded | block | drop-oldest | drop-newest\n"
                    "  --source-fps N       the source's rate (default 60)\n"
                    "  --service-us N       one frame's service time (default 25000)\n"
                    "  --frames N           frames offered (default 300)\n"
                    "  --queue-capacity N   admission queue depth (default 4)\n"
                    "  --buffers N          buffer pool size (default 8)\n"
                    "  --repeat N           repeats for --compare (default 3)\n";
                return 0;
            } else throw std::runtime_error("unknown argument " + k);
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }

    if (do_compare) return compare(cfg, repeats);
    print_one(run_once(cfg));
    return 0;
}
