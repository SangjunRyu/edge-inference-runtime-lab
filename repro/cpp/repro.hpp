// Shared scaffolding for the reproductions.
//
// Every program here follows the same contract, because a reproduction that
// needs a human to interpret it is a story, not a test:
//
//   * `--mode broken` runs the code as it was written when the defect was
//     found, and is EXPECTED to fail.
//   * `--mode fixed` runs the correction, and is expected to pass.
//   * exit status 0 means "the program behaved as that mode predicts", so
//     `run_all.py` can check both directions. A broken mode that passes is
//     itself a finding: it means the defect no longer reproduces here, which
//     is worth knowing before trusting the fix.
//
// Nothing blocks forever. Every wait is bounded and a timeout is reported as a
// result rather than left for someone to notice.

#ifndef EIRL_REPRO_HPP
#define EIRL_REPRO_HPP

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

namespace repro {

struct Opts {
    std::string mode = "fixed";
    std::string model;
    std::string device = "CPU";
    std::string collect = "worker";   // 03 only: where the output is read
    int n = 4;
    long timeout_ms = 8000;
};

inline Opts parse(int argc, char** argv, const char* what) {
    Opts o;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", k.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (k == "--mode") o.mode = val();
        else if (k == "--model") o.model = val();
        else if (k == "--device") o.device = val();
        else if (k == "--collect") o.collect = val();
        else if (k == "--n") o.n = std::stoi(val());
        else if (k == "--timeout-ms") o.timeout_ms = std::stol(val());
        else if (k == "-h" || k == "--help") {
            std::printf("%s\n  --mode broken|fixed  --model FILE  --device CPU|GPU|NPU"
                        "  --n N  --timeout-ms MS\n", what);
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", k.c_str());
            std::exit(2);
        }
    }
    if (o.mode != "broken" && o.mode != "fixed" && o.mode != "probe") {
        std::fprintf(stderr, "--mode must be broken, fixed or probe\n");
        std::exit(2);
    }
    return o;
}

/// The verdict line every program ends with, and the process status.
///
/// `reproduced` means the defect showed itself. In `broken` mode that is the
/// expected outcome and the run succeeds; in `fixed` mode it is a regression.
inline int verdict(const Opts& o, bool reproduced, const std::string& detail) {
    const bool as_predicted = (o.mode == "broken") ? reproduced : !reproduced;
    std::printf("\n%-8s %s\n", o.mode.c_str(),
                reproduced ? "DEFECT REPRODUCED" : "behaved correctly");
    std::printf("         %s\n", detail.c_str());
    if (as_predicted) {
        std::printf("RESULT   as predicted for --mode %s\n", o.mode.c_str());
        return 0;
    }
    if (o.mode == "broken") {
        std::printf("RESULT   NOT REPRODUCED. The defect did not show itself here.\n"
                    "         Worth knowing before trusting the fix -- see this\n"
                    "         program's header for what it depends on.\n");
    } else {
        std::printf("RESULT   REGRESSION. The fixed path failed.\n");
    }
    return 1;
}

class Deadline {
public:
    explicit Deadline(long ms)
        : end_(std::chrono::steady_clock::now() + std::chrono::milliseconds(ms)) {}
    bool expired() const { return std::chrono::steady_clock::now() >= end_; }
    long left_ms() const {
        auto d = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_ - std::chrono::steady_clock::now()).count();
        return d > 0 ? static_cast<long>(d) : 0;
    }

private:
    std::chrono::steady_clock::time_point end_;
};

}  // namespace repro

#endif  // EIRL_REPRO_HPP
