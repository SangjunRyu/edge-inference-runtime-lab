// ONNX Runtime CPU/CUDA backend. Every in-flight request owns its output and
// keeps the caller's input tensor alive until the completion callback returns.
#ifndef EIRL_ORT_BACKEND_HPP
#define EIRL_ORT_BACKEND_HPP
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <nlohmann/json.hpp>

namespace eirl {
struct OrtRequest : std::enable_shared_from_this<OrtRequest> {
    std::unique_ptr<Ort::Value> owned_input;
    const Ort::Value* input = nullptr;
    std::vector<float> output_data;
    Ort::Value output{nullptr};
    std::mutex mutex;
    std::condition_variable completed;
    bool running = false;
    std::shared_ptr<OrtRequest> callback_keepalive;
    std::string error;
    void (*callback)(void*) = nullptr;
    void* callback_arg = nullptr;
    void wait();
    ~OrtRequest() { wait_no_throw(); }
private:
    void wait_no_throw() noexcept;
};
using RequestHandle = std::shared_ptr<OrtRequest>;
using TensorHandle = std::unique_ptr<Ort::Value>;

class OrtBackend {
public:
    OrtBackend(const std::string& model, const std::string& device, int threads = 4,
               const std::string& profile = "");
    int height() const { return static_cast<int>(shape_[2]); }
    int width() const { return static_cast<int>(shape_[3]); }
    size_t output_elements() const { return output_elements_; }
    RequestHandle make_request() const;
    TensorHandle wrap_input(const float* input) const;
    void infer(OrtRequest* req, const float* input, float* output) const;
    void infer_bound(OrtRequest* req, float* output) const;
    void start_async(OrtRequest* req) const;
    nlohmann::json describe() const;
private:
    static void on_complete(void*, OrtValue**, size_t, OrtStatus*) noexcept;
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "eirl-cpp"};
    std::unique_ptr<Ort::Session> session_;
    std::string device_, model_, input_name_, output_name_;
    std::vector<int64_t> shape_, output_shape_;
    size_t input_elements_ = 0, output_elements_ = 0;
    int threads_;
    const char* input_names_[1]{};
    const char* output_names_[1]{};
};
} // namespace eirl
#endif
