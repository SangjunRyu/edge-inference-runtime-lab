#include "harness.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>

namespace eirl {

double percentile(std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    if (v.size() == 1) return v[0];
    // numpy.percentile's default: position (n-1)*q/100, linear interpolation
    // between the neighbouring order statistics. Reimplemented rather than
    // approximated so a C++ p95 and a Python p95 mean the same thing.
    const double pos = (static_cast<double>(v.size()) - 1.0) * (q / 100.0);
    const auto lo = static_cast<size_t>(std::floor(pos));
    const auto hi = static_cast<size_t>(std::ceil(pos));
    if (lo == hi) return v[lo];
    const double frac = pos - static_cast<double>(lo);
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

Stats compute_stats(const std::vector<double>& seconds) {
    Stats s;
    if (seconds.empty()) return s;

    std::vector<double> ms;
    ms.reserve(seconds.size());
    for (double v : seconds) ms.push_back(v * 1000.0);

    s.mean = std::accumulate(ms.begin(), ms.end(), 0.0) / static_cast<double>(ms.size());
    s.p50 = percentile(ms, 50.0);
    s.p95 = percentile(ms, 95.0);
    s.max = *std::max_element(ms.begin(), ms.end());

    // 10% trimmed mean: drop the outer 5% each side, so one OS hiccup cannot
    // move the headline. Reported beside the plain mean, never instead of it --
    // the gap between them measures how contaminated the run was.
    const double lo = percentile(ms, 5.0), hi = percentile(ms, 95.0);
    double sum = 0.0;
    int n = 0;
    for (double v : ms) {
        if (v >= lo && v <= hi) { sum += v; ++n; }
    }
    s.mean_trim10 = n ? sum / n : s.mean;
    return s;
}

Recorder::Recorder(std::vector<std::string> stages,
                   std::vector<std::string> wait_stages,
                   int warmup_frames, double warmup_seconds, bool pipelined)
    : stages_(std::move(stages)),
      wait_stages_(std::move(wait_stages)),
      warmup_frames_(warmup_frames),
      warmup_seconds_(warmup_seconds),
      pipelined_(pipelined) {
    all_stages_ = stages_;
    all_stages_.insert(all_stages_.end(), wait_stages_.begin(), wait_stages_.end());
    for (const auto& s : all_stages_) service_[s] = {};
}

void Recorder::add_frame(const std::map<std::string, double>& service, double e2e) {
    ++seen_;
    const double now = now_s();

    // Time-based warm-up when asked for. A frame count cannot adapt: a slow
    // configuration can need longer to reach it than the whole measurement
    // budget allows, and then the run counts nothing and looks stalled when it
    // was only slow.
    if (warmup_seconds_ > 0.0) {
        if (t_warm_ < 0.0) t_warm_ = now;
        if (now - t_warm_ < warmup_seconds_) { ++discarded_; return; }
    } else if (seen_ <= warmup_frames_) {
        ++discarded_;
        return;
    }

    if (t_first_ < 0.0) t_first_ = now;
    for (const auto& s : all_stages_) {
        auto it = service.find(s);
        service_[s].push_back(it == service.end() ? 0.0 : it->second);
    }
    e2e_.push_back(e2e);
    wall_.push_back(now - t_first_);
}

Drift Recorder::drift() const {
    Drift d;
    const size_t n = e2e_.size();
    if (n < 20) return d;
    const size_t k = std::max<size_t>(1, n / 10);
    const double a = std::accumulate(e2e_.begin(), e2e_.begin() + k, 0.0) / k * 1000.0;
    const double b = std::accumulate(e2e_.end() - k, e2e_.end(), 0.0) / k * 1000.0;
    d.valid = true;
    d.first_decile_ms = a;
    d.last_decile_ms = b;
    d.ratio = a > 0.0 ? b / a : 0.0;
    return d;
}

nlohmann::json Recorder::result() const {
    nlohmann::json per_stage = nlohmann::json::object();
    double stage_sum = 0.0, service_sum = 0.0, wait_sum = 0.0;
    std::string bottleneck;
    double slowest = 0.0;

    for (const auto& s : all_stages_) {
        const Stats st = compute_stats(service_.at(s));
        per_stage[s] = {{"mean", st.mean}, {"mean_trim10", st.mean_trim10},
                        {"p50", st.p50}, {"p95", st.p95}, {"max", st.max}};
        stage_sum += st.mean;
    }
    for (const auto& s : stages_) {
        const double m = per_stage[s]["mean"].get<double>();
        service_sum += m;
        if (m > slowest) { slowest = m; bottleneck = s; }
    }
    for (const auto& s : wait_stages_) wait_sum += per_stage[s]["mean"].get<double>();

    const Stats e2e = compute_stats(e2e_);
    const double elapsed = wall_.size() > 1 ? wall_.back() - wall_.front() : 0.0;
    const double throughput = elapsed > 0.0 ? (counted() - 1) / elapsed : 0.0;
    const double implied = e2e.mean > 0.0 ? 1000.0 / e2e.mean : 0.0;
    const double concurrency = throughput * e2e.mean / 1000.0;

    // Service stages only: the ceiling asks what the pipeline could reach if
    // every stage hid behind the slowest one, and queue waits are not work
    // that can be hidden.
    const double ceiling = slowest > 0.0 ? service_sum / slowest : 0.0;
    double noise = 0.0;
    if (!bottleneck.empty()) {
        const double p50 = per_stage[bottleneck]["p50"].get<double>();
        if (p50 > 0.0) noise = per_stage[bottleneck]["p95"].get<double>() / p50;
    }

    const Drift d = drift();
    nlohmann::json drift_j;
    if (d.valid) {
        drift_j = {{"first_decile_ms", d.first_decile_ms},
                   {"last_decile_ms", d.last_decile_ms}, {"ratio", d.ratio}};
    } else {
        drift_j = {{"first_decile_ms", nullptr}, {"last_decile_ms", nullptr},
                   {"ratio", nullptr}};
    }

    return {
        {"frames_counted", counted()},
        {"frames_warmup_discarded", discarded_},
        {"warmup_seconds", warmup_seconds_},
        {"stages_ms", per_stage},
        {"e2e_ms", {{"mean", e2e.mean}, {"mean_trim10", e2e.mean_trim10},
                    {"p50", e2e.p50}, {"p95", e2e.p95}, {"max", e2e.max}}},
        {"stage_sum_mean_ms", stage_sum},
        {"service_sum_mean_ms", service_sum},
        {"wait_sum_mean_ms", wait_sum},
        {"wait_stages", wait_stages_},
        {"unaccounted_mean_ms", e2e.mean - stage_sum},
        {"throughput_fps", throughput},
        {"throughput_implied_by_e2e_fps", implied},
        {"concurrency_littles_law", concurrency},
        {"pipelined", pipelined_},
        {"bottleneck_stage", bottleneck.empty() ? nlohmann::json(nullptr)
                                                : nlohmann::json(bottleneck)},
        {"overlap_ceiling_x", ceiling},
        {"bottleneck_p95_over_p50", noise},
        {"drift", drift_j},
    };
}

void Recorder::print_summary(const std::string& title) const {
    const nlohmann::json r = result();
    const int w = 68;
    const std::string bar(w, '='), dash(w, '-');

    std::printf("\n%s\n", bar.c_str());
    if (!title.empty()) std::printf(" %s\n%s\n", title.c_str(), dash.c_str());
    std::printf(" frames %d  (warm-up discarded: %d)\n%s\n",
                r["frames_counted"].get<int>(),
                r["frames_warmup_discarded"].get<int>(), dash.c_str());
    std::printf(" %-14s%9s%9s%9s%9s%9s\n", "stage", "mean", "p50", "p95", "max", "share");

    const double e2e_mean = std::max(1e-12, r["e2e_ms"]["mean"].get<double>());
    auto row = [&](const std::string& s) {
        const auto& st = r["stages_ms"][s];
        std::printf(" %-14s%8.2f %8.2f %8.2f %8.2f %8.1f%%\n", s.c_str(),
                    st["mean"].get<double>(), st["p50"].get<double>(),
                    st["p95"].get<double>(), st["max"].get<double>(),
                    100.0 * st["mean"].get<double>() / e2e_mean);
    };
    for (const auto& s : stages_) row(s);
    if (!wait_stages_.empty()) {
        std::printf(" %-14s\n", "-- waiting");
        for (const auto& s : wait_stages_) row(s);
    }
    std::printf("%s\n", dash.c_str());
    if (!wait_stages_.empty()) {
        std::printf(" %-14s%8.2f\n", "sum(service)", r["service_sum_mean_ms"].get<double>());
        std::printf(" %-14s%8.2f\n", "sum(waiting)", r["wait_sum_mean_ms"].get<double>());
    }
    std::printf(" %-14s%8.2f\n", "sum(stages)", r["stage_sum_mean_ms"].get<double>());
    const auto& e = r["e2e_ms"];
    std::printf(" %-14s%8.2f %8.2f %8.2f %8.2f\n", "E2E", e["mean"].get<double>(),
                e["p50"].get<double>(), e["p95"].get<double>(), e["max"].get<double>());
    std::printf(" %-14s%8.2f%27s%8.1f%%\n", "unaccounted",
                r["unaccounted_mean_ms"].get<double>(), "",
                100.0 * r["unaccounted_mean_ms"].get<double>() / e2e_mean);
    std::printf("%s\n", dash.c_str());

    const double obs = r["throughput_fps"].get<double>();
    const double imp = r["throughput_implied_by_e2e_fps"].get<double>();
    if (pipelined_) {
        std::printf(" throughput    %8.2f FPS\n", obs);
        std::printf(" latency x tput%8.2f frames resident on average (Little's Law)\n",
                    r["concurrency_littles_law"].get<double>());
    } else {
        std::printf(" throughput    %8.2f FPS   (implied by E2E: %.2f)\n", obs, imp);
        if (imp > 0.0 && obs > 0.0 && (obs / imp < 0.95 || obs / imp > 1.05)) {
            std::printf(" %14s  <-- CHECK: observed/implied = %.2f; time is being"
                        " spent outside the timed stages\n", "", obs / imp);
        }
    }

    if (!r["bottleneck_stage"].is_null()) {
        const std::string bn = r["bottleneck_stage"].get<std::string>();
        const double ceil = r["overlap_ceiling_x"].get<double>();
        const double noise = r["bottleneck_p95_over_p50"].get<double>();
        if (pipelined_) {
            std::printf(" bottleneck    %s   (service p95/p50 = %.2fx)\n", bn.c_str(), noise);
        } else {
            std::printf(" bottleneck    %s   overlap ceiling %.2fx   (p95/p50 = %.2fx)\n",
                        bn.c_str(), ceil, noise);
            if (ceil > 1.0 && (noise - 1.0) > (ceil - 1.0)) {
                std::printf(" %14s  <-- CHECK: noise band (%.0f%%) exceeds the best"
                            " possible gain (%.0f%%); an async result here would"
                            " not be measurable\n", "", 100.0 * (noise - 1.0),
                            100.0 * (ceil - 1.0));
            }
        }
    }
    if (!r["drift"]["ratio"].is_null()) {
        const double ratio = r["drift"]["ratio"].get<double>();
        std::printf(" drift         first %.2f ms -> last %.2f ms  (%.2fx)%s\n",
                    r["drift"]["first_decile_ms"].get<double>(),
                    r["drift"]["last_decile_ms"].get<double>(), ratio,
                    ratio > 1.05 ? "  <-- CHECK: not steady state" : "");
    }
    std::printf("%s\n", bar.c_str());
}

}  // namespace eirl
