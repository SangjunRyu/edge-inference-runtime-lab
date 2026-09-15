// Measurement harness -- the C++ side of bench/harness.py.
//
// The rules are the Python side's, unchanged, because the point of this port is
// that the two implementations can be compared:
//
//   1. Stages are timed separately and `read` is its own stage.
//   2. Service time is never mixed with wait time. Only service time can be
//      overlapped; a queue wait is the consequence of overlap, so folding it
//      into the ceiling would let a pipeline look better the more it queues.
//   3. Warm-up frames are discarded, never averaged in.
//   4. p50/p95/max always accompany the mean.
//   5. `E2E - sum(stages)` is published as `unaccounted` rather than absorbed.
//
// It emits the **same JSON schema** as the Python recorder. That is the single
// decision that makes the comparison cost nothing: bench/summarize.py,
// bench/compare_backends.py and the rest read C++ output unmodified.

#ifndef EIRL_HARNESS_HPP
#define EIRL_HARNESS_HPP

#include <chrono>
#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace eirl {

using Clock = std::chrono::steady_clock;

inline double now_s() {
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

/// Scoped stopwatch. `elapsed()` is seconds, matching the Python Timer.
class Timer {
public:
    Timer() : t0_(Clock::now()) {}
    void reset() { t0_ = Clock::now(); }
    double elapsed() const {
        return std::chrono::duration<double>(Clock::now() - t0_).count();
    }

private:
    Clock::time_point t0_;
};

/// mean / trimmed mean / p50 / p95 / max, in milliseconds.
struct Stats {
    double mean = 0.0;
    double mean_trim10 = 0.0;
    double p50 = 0.0;
    double p95 = 0.0;
    double max = 0.0;
};

/// Linear-interpolated percentile, matching numpy.percentile's default so the
/// two implementations' numbers are comparable rather than merely similar.
double percentile(std::vector<double> sorted_vals, double q);

Stats compute_stats(const std::vector<double>& seconds);

struct Drift {
    bool valid = false;
    double first_decile_ms = 0.0;
    double last_decile_ms = 0.0;
    double ratio = 0.0;
};

class Recorder {
public:
    /// `stages` are service stages; `wait_stages` are time spent waiting.
    /// `pipelined` switches the sequential cross-check for Little's Law,
    /// because once frames overlap throughput deliberately is not 1/E2E.
    Recorder(std::vector<std::string> stages,
             std::vector<std::string> wait_stages = {},
             int warmup_frames = 20,
             double warmup_seconds = 0.0,
             bool pipelined = false);

    /// One frame. Values are seconds. Unknown stage names are ignored.
    void add_frame(const std::map<std::string, double>& service, double e2e);

    int counted() const { return static_cast<int>(e2e_.size()); }
    int discarded() const { return discarded_; }

    nlohmann::json result() const;
    void print_summary(const std::string& title) const;

private:
    std::vector<std::string> stages_, wait_stages_, all_stages_;
    int warmup_frames_;
    double warmup_seconds_;
    bool pipelined_;

    std::map<std::string, std::vector<double>> service_;
    std::vector<double> e2e_, wall_;
    int seen_ = 0, discarded_ = 0;
    double t_first_ = -1.0, t_warm_ = -1.0;

    Drift drift() const;
};

}  // namespace eirl

#endif  // EIRL_HARNESS_HPP
