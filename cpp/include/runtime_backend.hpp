// Small compile-time adapter: both engines use the same W4 pipeline sources.
#ifndef EIRL_RUNTIME_BACKEND_HPP
#define EIRL_RUNTIME_BACKEND_HPP
#include <cctype>
#include "postprocess.hpp"
#ifdef EIRL_USE_ORT
#include "ort_backend.hpp"
#define EIRL_CALLBACK
namespace eirl {
using RuntimeBackend = OrtBackend;
struct CompletionCallback { void (*callback_func)(void*) = nullptr; void* args = nullptr; };
inline void bind_input(OrtRequest* req, const Ort::Value* tensor) { req->input = tensor; }
inline void set_callback(OrtRequest* req, const CompletionCallback* cb) { req->callback = cb->callback_func; req->callback_arg = cb->args; }
inline void wait_request(OrtRequest* req) { req->wait(); }
inline auto request_topk(OrtRequest* req, size_t size, int k) { return postprocess(req->output_data.data(), size, k); }
inline const char* runtime_name() { return "onnxruntime"; }
inline std::string runtime_slug(const std::string& device) { return device == "CUDA" ? "cuda" : "cpu"; }
inline std::string runtime_backend_arg(const std::string& device) { return runtime_slug(device); }
inline auto available_runtime_devices() { return Ort::GetAvailableProviders(); }
inline nlohmann::json runtime_intra_threads(int count) { return count; }
}
#else
#include "ov_backend.hpp"
#define EIRL_CALLBACK OPENVINO_C_API_CALLBACK
namespace eirl {
class RuntimeBackend : public OvBackend {
public:
    RuntimeBackend(const std::string& model, const std::string& device, int = 4,
                   const std::string& profile = "") : OvBackend(model, device) {
        if (!profile.empty()) throw std::invalid_argument("--profile is available on ORT targets only");
    }
    void start_async(ov_infer_request_t* req) const { check(ov_infer_request_start_async(req), "ov_infer_request_start_async"); }
};
using CompletionCallback = ov_callback_t;
inline void bind_input(ov_infer_request_t* req, const ov_tensor_t* tensor) { check(ov_infer_request_set_input_tensor(req, tensor), "ov_infer_request_set_input_tensor"); }
inline void set_callback(ov_infer_request_t* req, const CompletionCallback* cb) { check(ov_infer_request_set_callback(req, cb), "ov_infer_request_set_callback"); }
inline void wait_request(ov_infer_request_t* req) { check(ov_infer_request_wait(req), "ov_infer_request_wait"); }
inline auto request_topk(ov_infer_request_t* req, size_t size, int k) {
    ov_tensor_t* raw = nullptr;
    check(ov_infer_request_get_output_tensor(req, &raw), "ov_infer_request_get_output_tensor");
    TensorHandle out(raw); void* data = nullptr;
    check(ov_tensor_data(out.get(), &data), "ov_tensor_data");
    return postprocess(static_cast<const float*>(data), size, k);
}
inline const char* runtime_name() { return "openvino"; }
inline const char* runtime_backend_arg(const std::string&) { return "openvino"; }
inline auto available_runtime_devices() { return available_devices(); }
inline nlohmann::json runtime_intra_threads(int) { return nullptr; }
inline std::string runtime_slug(const std::string& device) {
    std::string d = device;
    for (auto& c : d) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return "openvino-" + d;
}
}
#endif
#endif
