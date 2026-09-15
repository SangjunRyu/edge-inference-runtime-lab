// Phase 0 -- the whole point of which is to fail fast.
//
// Loads the model on a device, reports what OpenVINO sees, runs one inference
// over a synthetic buffer and prints the top-5. No video, no timing, no
// pipeline: if this does not build and run, the problem is the toolchain or
// the runtime, and no amount of pipeline code would have revealed it.
//
// It needs no OpenCV, so it can be built before the rest of the dependencies
// are in place.
//
//   probe --model models/resnet50_224.onnx --device NPU

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "harness.hpp"
#include "ov_backend.hpp"
#include "postprocess.hpp"

using namespace eirl;

int main(int argc, char** argv) {
    std::string model, device = "CPU";
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        if (k == "--model" && i + 1 < argc) model = argv[++i];
        else if (k == "--device" && i + 1 < argc) device = argv[++i];
        else {
            std::fprintf(stderr, "usage: %s --model M.onnx [--device CPU|GPU|NPU]\n",
                         argv[0]);
            return 2;
        }
    }
    if (model.empty()) {
        std::fprintf(stderr, "usage: %s --model M.onnx [--device CPU|GPU|NPU]\n",
                     argv[0]);
        return 2;
    }

    try {
        ov_version_t v{};
        check(ov_get_openvino_version(&v), "ov_get_openvino_version");
        std::printf("OpenVINO %s\n", v.buildNumber);
        ov_version_free(&v);

        std::printf("devices:");
        for (const auto& d : available_devices()) std::printf(" %s", d.c_str());
        std::printf("\n");

        OvBackend backend(model, device);
        std::printf("model    %s\n", model.c_str());
        std::printf("device   %s\n", device.c_str());
        std::printf("input    [");
        for (size_t i = 0; i < backend.input_shape().size(); ++i) {
            std::printf("%s%lld", i ? ", " : "",
                        static_cast<long long>(backend.input_shape()[i]));
        }
        std::printf("]  ->  %zu logits\n", backend.output_elements());

        // One inference over a fixed pattern. The values are meaningless; that
        // the call completes and the probabilities sum to 1 is the point.
        const size_t n = static_cast<size_t>(backend.height()) * backend.width() * 3;
        std::vector<float> input(n);
        for (size_t i = 0; i < n; ++i) {
            input[i] = static_cast<float>((i % 255)) / 255.0f - 0.5f;
        }
        std::vector<float> logits(backend.output_elements());

        RequestHandle req = backend.make_request();
        Timer t;
        backend.infer(req.get(), input.data(), logits.data());
        const double ms = t.elapsed() * 1000.0;

        const auto topk = postprocess(logits.data(), logits.size(), 5);
        std::printf("inference %.2f ms (cold, single call -- not a measurement)\n", ms);
        std::printf("top-5   ");
        double sum = 0.0;
        for (const auto& [cls, p] : topk) {
            std::printf(" %d:%.4f", cls, static_cast<double>(p));
            sum += p;
        }
        std::printf("\n        top-5 mass %.4f\n", sum);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
