// W1 in C++ -- the sequential pipeline, for comparison against run_sync.py.
//
//     Where does one frame spend its time, and does C++ agree with Python
//     about the answer?
//
// The stage boundaries, the warm-up rules and the output schema are the Python
// ones. If the two implementations disagree on stage shares, that is a finding
// about the instrumentation and is worth more than either number alone.
//
//   run_sync_cpp --model models/resnet50_224.onnx --device NPU \
//       --source clips/bbb_360_2MB.mp4 --duration-s 10 --warmup-s 2 \
//       --repeat 3 --cooldown 15 --tag cpp

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "harness.hpp"
#include "ov_backend.hpp"
#include "pipeline.hpp"
#include "sources.hpp"

using namespace eirl;
using nlohmann::json;

namespace {

struct Args {
    std::string model, device = "CPU", source, tag = "cpp", out_dir = "results";
    long long frames = 300;
    double duration_s = 0.0, warmup_s = 0.0, cooldown = 0.0;
    int warmup = 20, top_k = 5, repeat = 1;
    bool fused = false;
};

[[noreturn]] void usage(const char* argv0) {
    std::fprintf(stderr,
        "usage: %s --model M.onnx [--device CPU|GPU|NPU] [--source clip.mp4]\n"
        "          [--frames N] [--duration-s S] [--warmup N] [--warmup-s S]\n"
        "          [--top-k K] [--repeat N] [--cooldown S] [--fused]\n"
        "          [--tag T] [--out-dir D]\n", argv0);
    std::exit(2);
}

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) usage(argv[0]);
            return argv[++i];
        };
        if (k == "--model") a.model = val();
        else if (k == "--device") a.device = val();
        else if (k == "--source") a.source = val();
        else if (k == "--frames") a.frames = std::stoll(val());
        else if (k == "--duration-s") a.duration_s = std::stod(val());
        else if (k == "--warmup") a.warmup = std::stoi(val());
        else if (k == "--warmup-s") a.warmup_s = std::stod(val());
        else if (k == "--top-k") a.top_k = std::stoi(val());
        else if (k == "--repeat") a.repeat = std::stoi(val());
        else if (k == "--cooldown") a.cooldown = std::stod(val());
        else if (k == "--fused") a.fused = true;
        else if (k == "--tag") a.tag = val();
        else if (k == "--out-dir") a.out_dir = val();
        else usage(argv[0]);
    }
    if (a.model.empty()) usage(argv[0]);
    return a;
}

std::string stem(const std::string& path) {
    const size_t slash = path.find_last_of("/\\");
    const size_t start = slash == std::string::npos ? 0 : slash + 1;
    const size_t dot = path.find_last_of('.');
    const size_t end = (dot == std::string::npos || dot < start) ? path.size() : dot;
    return path.substr(start, end - start);
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

struct RunOut {
    std::unique_ptr<Recorder> rec;
    json source;
    int emitted = 0;
};

RunOut measure_once(const OvBackend& backend, const Args& a) {
    auto source = make_source(a.source, a.frames);
    auto rec = std::make_unique<Recorder>(
        std::vector<std::string>{"read", "preprocess", "inference", "postprocess"},
        std::vector<std::string>{}, a.warmup, a.warmup_s, /*pipelined=*/false);

    RequestHandle req = backend.make_request();
    const int h = backend.height(), w = backend.width();
    std::vector<float> input(static_cast<size_t>(h) * w * 3);
    std::vector<float> logits(backend.output_elements());
    cv::Mat frame;

    // A fixed frame count measures for a different length of time at every
    // throughput, so a deadline equalises the wall clock across configurations.
    const double deadline = a.duration_s > 0.0 ? now_s() + a.duration_s : -1.0;
    int emitted = 0;

    while (true) {
        if (deadline > 0.0 && now_s() >= deadline) break;

        Timer t_e2e;
        // Time the pull, not the loop body: the source decodes inside `next`,
        // and a timer started once the frame is in hand measures nothing.
        Timer t_read;
        if (!source->next(frame)) break;
        const double read = t_read.elapsed();

        Timer t_pre;
        if (a.fused) preprocess_fused(frame, h, w, input.data());
        else preprocess(frame, h, w, input.data());
        const double pre = t_pre.elapsed();

        Timer t_inf;
        backend.infer(req.get(), input.data(), logits.data());
        const double inf = t_inf.elapsed();

        Timer t_post;
        const auto topk = postprocess(logits.data(), logits.size(), a.top_k);
        const double post = t_post.elapsed();
        (void)topk;

        ++emitted;
        rec->add_frame({{"read", read}, {"preprocess", pre},
                        {"inference", inf}, {"postprocess", post}},
                       t_e2e.elapsed());
    }

    RunOut out;
    out.rec = std::move(rec);
    out.source = source->describe();
    out.emitted = emitted;
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    const Args a = parse(argc, argv);

    try {
        const auto devices = available_devices();
        std::printf("available devices:");
        for (const auto& d : devices) std::printf(" %s", d.c_str());
        std::printf("\n");

        OvBackend backend(a.model, a.device);
        std::printf("model input %dx%d  device=%s  engine=openvino-c-api\n",
                    backend.height(), backend.width(), a.device.c_str());

        json runs = json::array(), sources = json::array();
        const int repeats = std::max(1, a.repeat);

        for (int i = 0; i < repeats; ++i) {
            RunOut r = measure_once(backend, a);

            char title[256];
            std::snprintf(title, sizeof(title),
                          "%s | openvino/%s | source=%s | run %d/%d",
                          stem(a.model).c_str(), a.device.c_str(),
                          r.source["kind"].get<std::string>().c_str(),
                          i + 1, repeats);
            r.rec->print_summary(title);

            // A sequential pipeline cannot emit out of order, so this must
            // stay zero -- which is what makes the async pipeline's zero mean
            // something. Recorded in the same shape bench/reorder.py uses.
            std::printf(" order         emitted %d  violations 0\n", r.emitted);

            json res = r.rec->result();
            res["order"] = {{"emitted", r.emitted}, {"order_violations", 0},
                            {"first_violation_at_frame", nullptr},
                            {"in_submission_order", true}};
            runs.push_back(std::move(res));
            sources.push_back(r.source);

            if (i + 1 < repeats && a.cooldown > 0.0) {
                std::this_thread::sleep_for(std::chrono::duration<double>(a.cooldown));
            }
        }

        json meta = {
            {"tag", a.tag}, {"mode", "sync"}, {"impl", "cpp"},
            {"args", {{"model", a.model}, {"backend", "openvino"},
                      {"ov_device", a.device}, {"source", a.source},
                      {"frames", a.frames}, {"duration_s", a.duration_s},
                      {"warmup", a.warmup}, {"warmup_s", a.warmup_s},
                      {"top_k", a.top_k}, {"repeat", a.repeat},
                      {"cooldown", a.cooldown}, {"fused", a.fused},
                      {"tag", a.tag}, {"out_dir", a.out_dir},
                      {"intra_threads", nullptr}}},
            {"backend", backend.describe()},
            {"sources", sources},
            {"model_input_hw", json::array({backend.height(), backend.width()})},
            {"idle_gate", {{"checked", false}, {"reason", "not implemented in cpp"}}},
            {"env", {{"impl", "cpp"}, {"openvino_devices", devices}}},
        };

        const std::string src_slug = a.source.empty() ? "synthetic" : stem(a.source);
        const std::string path = a.out_dir + "/" + a.tag + "__cppsync__openvino-"
                                 + lower(a.device) + "__" + stem(a.model) + "__"
                                 + src_slug + ".json";
        std::ofstream f(path);
        if (!f) { std::fprintf(stderr, "cannot write %s\n", path.c_str()); return 1; }
        f << json{{"meta", meta}, {"runs", runs}}.dump(2) << "\n";
        std::printf("\nsaved -> %s\n", path.c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
