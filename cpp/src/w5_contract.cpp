// W5-3 -- the model file decides the buffers, and says no before anything runs.
//
// Up to here the pipeline's buffer size was a literal. 150528 is the right
// number for densenet121's input if you forget that the device reads a *padded*
// tensor, and the padded one is 157696. A pool built from the logical size is
// 7168 bytes short of what the hardware will read, and nothing about it looks
// wrong: the allocation succeeds, the frames flow, the counts match.
//
// So the contract comes from the compiled model:
//
//     python -m tools.dxnn_inspect model.dxnn --contract contracts/model.json
//     cpp/build/w5_contract.exe --contract contracts/model.json --npu-memory 512M
//
// Three properties, each with an injection that breaks it:
//
//   1. pools are sized from the contract, never from a literal;
//   2. a buffer that does not match is refused BEFORE any work is submitted,
//      because afterwards the only evidence is a wrong answer that looks right;
//   3. the buffer count comes from the model's own cost model and the device's
//      memory -- W5-2's resource bound, arriving from the file.
//
// Injections:
//   --inject hardcoded-size   size the pools from 150528, the way the code did
//   --inject bad-allocated    a contract whose allocation is below its logical size
//   --inject bad-padding      a contract that disagrees with its own layout rule
//   --inject overflow         a device too small for even one buffer
//
// Links no OpenVINO and no OpenCV. Runs on a machine with no accelerator.

#include <atomic>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "contract.hpp"
#include "scheduler.hpp"

namespace {

using namespace eirl;

/// The number that used to be written into the pipeline by hand: densenet121's
/// input as a dense tensor, 224 x 224 x 3. Correct arithmetic, wrong buffer.
constexpr std::size_t kHardCodedBytes = 150528;

enum class Inject { None, HardcodedSize, BadAllocated, BadPadding, Overflow };

const char* inject_name(Inject i) {
    switch (i) {
        case Inject::HardcodedSize: return "hardcoded-size";
        case Inject::BadAllocated: return "bad-allocated";
        case Inject::BadPadding: return "bad-padding";
        case Inject::Overflow: return "overflow";
        default: return "none";
    }
}

struct Counters {
    std::atomic<std::uint64_t> submitted{0};
    std::atomic<std::uint64_t> executed{0};
    std::atomic<std::uint64_t> rejected{0};
};

struct Check {
    std::string what;
    bool ok;
    std::string detail;
};

}  // namespace

int main(int argc, char** argv) {
    std::string contract_path;
    std::string memory_text;
    std::uint64_t frames = 2000;
    int requested = 0;   // 0 = the contract's default
    Inject inject = Inject::None;

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string k = argv[i];
            auto val = [&]() -> std::string {
                if (i + 1 >= argc) throw std::runtime_error("missing value for " + k);
                return argv[++i];
            };
            if (k == "--contract") contract_path = val();
            else if (k == "--npu-memory") memory_text = val();
            else if (k == "--frames") frames = std::stoull(val());
            else if (k == "--buffers") requested = std::stoi(val());
            else if (k == "--inject") {
                const std::string v = val();
                if (v == "none") inject = Inject::None;
                else if (v == "hardcoded-size") inject = Inject::HardcodedSize;
                else if (v == "bad-allocated") inject = Inject::BadAllocated;
                else if (v == "bad-padding") inject = Inject::BadPadding;
                else if (v == "overflow") inject = Inject::Overflow;
                else throw std::runtime_error("unknown --inject " + v);
            } else if (k == "-h" || k == "--help") {
                std::cout <<
                    "w5_contract -- size and check buffers from a compiled model\n\n"
                    "  --contract FILE   JSON from tools/dxnn_inspect --contract\n"
                    "  --npu-memory S    device memory, e.g. 512M (default: unbounded)\n"
                    "  --buffers N       requested buffer count (default: the contract's)\n"
                    "  --frames N        frames to run (default 2000)\n"
                    "  --inject WHAT     none | hardcoded-size | bad-allocated |\n"
                    "                    bad-padding | overflow\n";
                return 0;
            } else throw std::runtime_error("unknown argument " + k);
        }
        if (contract_path.empty()) throw std::runtime_error("--contract is required");
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }

    // A size suffix, binary for K/M/G. Same convention as the inspector.
    auto parse_size = [](const std::string& s) -> std::uint64_t {
        if (s.empty()) return 0;
        const char last = static_cast<char>(std::toupper(s.back()));
        double n = std::stod(s.substr(0, s.size() - (std::isdigit(static_cast<unsigned char>(s.back())) ? 0 : 1)));
        switch (last) {
            case 'K': return static_cast<std::uint64_t>(n * 1024);
            case 'M': return static_cast<std::uint64_t>(n * 1024 * 1024);
            case 'G': return static_cast<std::uint64_t>(n * 1024 * 1024 * 1024);
            default: return static_cast<std::uint64_t>(std::stod(s));
        }
    };

    std::vector<Check> checks;
    std::string load_error, capacity_error;
    ModelContract contract;

    // --- 1. the contract itself -------------------------------------------
    try {
        contract = ModelContract::load(contract_path);
        if (inject == Inject::BadAllocated && !contract.tensors.empty()) {
            // An allocation below the logical size: the device would read past
            // the end of every buffer, and nothing at run time would say so.
            contract.tensors[0].allocated_bytes = contract.tensors[0].logical_bytes - 1;
            contract.validate();
        }
        if (inject == Inject::BadPadding && !contract.tensors.empty()) {
            // Padding computed on bytes rather than on elements -- the error
            // the inspector's self-test caught, arriving from the other side.
            auto& t = contract.tensors[0];
            t.allocated_bytes = (t.logical_bytes + 63) / 64 * 64;
            contract.validate();
        }
    } catch (const ContractError& e) {
        load_error = e.what();
    }

    const bool contract_loaded = load_error.empty();
    checks.push_back({"the contract loads and agrees with itself", contract_loaded,
                      contract_loaded ? contract.model + ", " + std::to_string(contract.tensors.size())
                                        + " tensors"
                                      : load_error});

    if (!contract_loaded) {
        // Nothing downstream can be trusted, so nothing downstream runs. That
        // is the point: a refusal here costs a message, and a refusal after
        // inference costs a wrong answer nobody notices.
        std::cout << "w5_contract    inject=" << inject_name(inject) << "\n";
        std::cout << "contract       " << contract_path << "\n\n";
        for (const auto& c : checks) {
            std::cout << "  " << (c.ok ? "ok  " : "FAIL") << "  " << c.what
                      << "\n            " << c.detail << "\n";
        }
        const bool predicted = (inject == Inject::BadAllocated || inject == Inject::BadPadding);
        std::cout << "\nwork executed  0 stages -- refused before the pipeline started\n";
        std::cout << "verdict        " << (predicted ? "as predicted" : "NOT as predicted") << "\n";
        return predicted ? 0 : 1;
    }

    std::cout << "w5_contract    inject=" << inject_name(inject) << "  frames=" << frames << "\n";
    std::cout << contract.summary() << "\n\n";

    // --- 2. the buffer count, from the model and the device ----------------
    const std::uint32_t want = requested > 0 ? static_cast<std::uint32_t>(requested)
                                             : static_cast<std::uint32_t>(contract.default_buffers);
    std::uint64_t memory = memory_text.empty() ? 0 : parse_size(memory_text);
    if (inject == Inject::Overflow) {
        // One byte short of a single buffer, which is exactly where DX-RT
        // raises rather than quietly running with none.
        memory = contract.fixed_cost + contract.per_buffer_cost - 1;
    }

    std::uint32_t buffers = want;
    if (memory) {
        try {
            buffers = contract.max_buffers_for(memory, want);
        } catch (const ContractError& e) {
            capacity_error = e.what();
        }
    }

    checks.push_back({"the device can fund at least one buffer", capacity_error.empty(),
                      capacity_error.empty()
                          ? (memory ? std::to_string(buffers) + " of " + std::to_string(want)
                                      + " buffers fit in " + std::to_string(memory) + " B"
                                    : "no device memory given; using the contract's default "
                                      + std::to_string(buffers))
                          : capacity_error});

    if (!capacity_error.empty()) {
        for (const auto& c : checks) {
            std::cout << "  " << (c.ok ? "ok  " : "FAIL") << "  " << c.what
                      << "\n            " << c.detail << "\n";
        }
        const bool predicted = (inject == Inject::Overflow);
        std::cout << "\nwork executed  0 stages -- refused before the pipeline started\n";
        std::cout << "verdict        " << (predicted ? "as predicted" : "NOT as predicted") << "\n";
        return predicted ? 0 : 1;
    }

    // --- 3. the pools, sized from the contract -----------------------------
    SchedulerConfig cfg;
    cfg.frames = frames;
    cfg.buffers = buffers;
    cfg.bytes = (inject == Inject::HardcodedSize) ? kHardCodedBytes
                                                  : contract.buffer_bytes(Direction::Input);
    // The output region a real device is charged for is 56 MB for this model.
    // Allocating that per buffer on the host would be honest and useless, so
    // the harness holds the output tensor and states the difference rather than
    // pretending the host pool is the device's.
    cfg.out_bytes = 0;
    cfg.out_buffers = buffers;
    cfg.q_pre = cfg.q_infer = cfg.q_post = 2;
    cfg.pre_workers = 1;
    cfg.infer_workers = 2;
    cfg.post_workers = 1;
    cfg.warmup = frames / 8;

    std::string why;
    const bool admitted = contract.accepts(Direction::Input, cfg.bytes, &why);
    checks.push_back({"the input pool matches the contract", admitted,
                      admitted ? std::to_string(cfg.bytes) + " B per slot, from the model file"
                               : why});

    Counters counters;

    if (!admitted) {
        // Refused at start-up. No stage ever runs, which is the property the
        // injection exists to demonstrate.
        for (const auto& c : checks) {
            std::cout << "  " << (c.ok ? "ok  " : "FAIL") << "  " << c.what
                      << "\n            " << c.detail << "\n";
        }
        const bool predicted = (inject == Inject::HardcodedSize);
        std::cout << "\nwork executed  " << counters.executed.load()
                  << " stages -- refused before the pipeline started\n";
        std::cout << "verdict        " << (predicted ? "as predicted" : "NOT as predicted") << "\n";
        return predicted ? 0 : 1;
    }

    // --- 4. run it, with the contract checked again at submit --------------
    Stages stages;
    stages.read = [&](std::uint64_t seq, std::uint8_t* dst, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) {
            dst[i] = static_cast<std::uint8_t>(seq * 31u + i * 7u);
        }
    };
    stages.preprocess = [](std::uint8_t* buf, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) buf[i] = static_cast<std::uint8_t>(buf[i] ^ 0x33);
    };
    stages.infer = [&](const std::uint8_t* src, std::uint8_t* dst, std::size_t n) {
        counters.submitted.fetch_add(1, std::memory_order_relaxed);
        // Defence in depth: the start-up check already passed, so this should
        // never fire. It is here because the cost of checking is a comparison
        // and the cost of not checking is a buffer the device reads past.
        if (!contract.accepts(Direction::Input, n)) {
            counters.rejected.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        counters.executed.fetch_add(1, std::memory_order_relaxed);
        for (std::size_t i = 0; i < n; ++i) dst[i] = static_cast<std::uint8_t>(src[i] ^ 0x5A);
    };
    stages.postprocess = [](std::uint64_t seq, const std::uint8_t* out, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) {
            const auto want2 = static_cast<std::uint8_t>(
                (static_cast<std::uint8_t>(seq * 31u + i * 7u) ^ 0x33) ^ 0x5A);
            if (out[i] != want2) return false;
        }
        return true;
    };

    SchedulerStats st = run_pipeline(cfg, stages);

    checks.push_back({"every submitted frame passed the contract at submit",
                      counters.rejected.load() == 0,
                      std::to_string(counters.rejected.load()) + " rejected of "
                          + std::to_string(counters.submitted.load())});
    checks.push_back({"accepted == completed == emitted",
                      st.accepted == st.completed && st.completed == st.emitted,
                      std::to_string(st.accepted) + " / " + std::to_string(st.completed)
                          + " / " + std::to_string(st.emitted)});
    checks.push_back({"every frame carries its own values", st.mismatched == 0,
                      std::to_string(st.mismatched) + " of " + std::to_string(st.emitted)
                          + " wrong"});
    checks.push_back({"every lease returned",
                      st.in_outstanding == 0 && st.out_outstanding == 0,
                      std::to_string(st.in_outstanding) + " input, "
                          + std::to_string(st.out_outstanding) + " output outstanding"});
    checks.push_back({"input leases within the model-derived bound",
                      st.peak_in_leases <= buffers,
                      "peak " + std::to_string(st.peak_in_leases) + " of " + std::to_string(buffers)
                          + " buffers, which the model and the device chose"});

    bool all_ok = true;
    for (const auto& c : checks) all_ok = all_ok && c.ok;

    std::cout << "buffers        " << buffers << " of " << want << " requested";
    if (memory) std::cout << ", limited by " << memory << " B of device memory";
    std::cout << "\n";
    std::cout << "host pools     input " << cfg.bytes << " B x " << buffers << "\n";
    std::cout << "device would   " << contract.per_buffer_cost << " B per buffer, of which "
              << contract.output_buffer_bytes << " B is the output region\n";
    std::cout << "work executed  " << counters.executed.load() << " stages\n\n";
    for (const auto& c : checks) {
        std::cout << "  " << (c.ok ? "ok  " : "FAIL") << "  " << c.what
                  << "\n            " << c.detail << "\n";
    }
    const bool predicted = (inject == Inject::None) ? all_ok : false;
    std::cout << "\nverdict        "
              << (inject == Inject::None
                      ? (all_ok ? "all checks pass" : "a check failed")
                      : "NOT as predicted -- the injection did not break anything")
              << "\n";
    return predicted ? 0 : 1;
}
