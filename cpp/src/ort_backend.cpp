#include "ort_backend.hpp"
#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace eirl {
namespace {
void enable_cuda_dlls() {
#ifdef _WIN32
    // cuDNN loads additional DLLs lazily on the first convolution. Keep all
    // pip nvidia/*/bin directories registered for the process lifetime.
    static const bool enabled = [] {
        if (!SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS |
                                      LOAD_LIBRARY_SEARCH_USER_DIRS))
            throw std::runtime_error("SetDefaultDllDirectories failed");
#ifdef EIRL_NVIDIA_ROOT
        const std::filesystem::path root(EIRL_NVIDIA_ROOT);
        if (std::filesystem::exists(root)) {
            for (const auto& entry : std::filesystem::directory_iterator(root)) {
                const auto bin = entry.path() / "bin";
                if (std::filesystem::is_directory(bin) && !AddDllDirectory(bin.c_str()))
                    throw std::runtime_error("AddDllDirectory failed: " + bin.string());
            }
        }
#endif
        return true;
    }();
    (void)enabled;
#endif
}
}

void OrtRequest::wait() {
    std::unique_lock<std::mutex> lock(mutex);
    completed.wait(lock, [&] { return !running; });
    if (!error.empty()) throw std::runtime_error(error);
}
void OrtRequest::wait_no_throw() noexcept {
    try { std::unique_lock<std::mutex> lock(mutex);
          completed.wait(lock, [&] { return !running; }); }
    catch (...) {}
}

OrtBackend::OrtBackend(const std::string& model, const std::string& device,
                       int threads, const std::string& profile)
    : device_(device), model_(model), threads_(threads) {
    std::transform(device_.begin(), device_.end(), device_.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    if (device_ != "CPU" && device_ != "CUDA")
        throw std::invalid_argument("ORT device must be CPU or CUDA (GPU is ambiguous)");
    if (threads < 1) throw std::invalid_argument("intra-threads must be positive");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(threads);
    options.SetInterOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (!profile.empty()) options.EnableProfiling(std::filesystem::path(profile).c_str());
    if (device_ == "CUDA") {
        enable_cuda_dlls();
        Ort::CUDAProviderOptions cuda;
        cuda.Update({{"device_id", "0"}});
        // Append failure throws; the C++ API does not silently retry on CPU.
        options.AppendExecutionProvider_CUDA_V2(*cuda);
    }
    session_ = std::make_unique<Ort::Session>(env_, std::filesystem::path(model).c_str(), options);
    if (session_->GetInputCount() != 1 || session_->GetOutputCount() != 1)
        throw std::runtime_error("Expected one model input and one output");
    Ort::AllocatorWithDefaultOptions allocator;
    input_name_ = session_->GetInputNameAllocated(0, allocator).get();
    output_name_ = session_->GetOutputNameAllocated(0, allocator).get();
    input_names_[0] = input_name_.c_str(); output_names_[0] = output_name_.c_str();
    auto in_type = session_->GetInputTypeInfo(0);
    auto out_type = session_->GetOutputTypeInfo(0);
    auto in = in_type.GetTensorTypeAndShapeInfo();
    auto out = out_type.GetTensorTypeAndShapeInfo();
    shape_ = in.GetShape(); output_shape_ = out.GetShape();
    if (shape_.size() != 4 || shape_[0] != 1 || shape_[1] != 3 ||
        in.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        out.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        throw std::runtime_error("Expected static FP32 NCHW classifier with batch 1");
    input_elements_ = 1; output_elements_ = 1;
    for (auto n : shape_) { if (n <= 0) throw std::runtime_error("Dynamic input is unsupported"); input_elements_ *= static_cast<size_t>(n); }
    for (auto n : output_shape_) { if (n <= 0) throw std::runtime_error("Dynamic output is unsupported"); output_elements_ *= static_cast<size_t>(n); }
}

RequestHandle OrtBackend::make_request() const {
    auto req = std::make_shared<OrtRequest>();
    req->output_data.resize(output_elements_);
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    req->output = Ort::Value::CreateTensor<float>(memory, req->output_data.data(),
        output_elements_, output_shape_.data(), output_shape_.size());
    return req;
}
TensorHandle OrtBackend::wrap_input(const float* input) const {
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    return std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(memory,
        const_cast<float*>(input), input_elements_, shape_.data(), shape_.size()));
}
void OrtBackend::infer(OrtRequest* req, const float* input, float* output) const {
    req->owned_input = wrap_input(input); req->input = req->owned_input.get();
    infer_bound(req, output);
}
void OrtBackend::infer_bound(OrtRequest* req, float* output) const {
    if (!req->input) throw std::runtime_error("Request input is not bound");
    req->wait();
    session_->Run(Ort::RunOptions{nullptr}, input_names_, req->input, 1,
                  output_names_, &req->output, 1);
    std::memcpy(output, req->output_data.data(), output_elements_ * sizeof(float));
}
void OrtBackend::start_async(OrtRequest* req) const {
    if (threads_ < 2) throw std::invalid_argument("ORT async needs intra-threads >= 2");
    if (!req->input) throw std::runtime_error("Request input is not bound");
    { std::lock_guard<std::mutex> lock(req->mutex);
      if (req->running) throw std::runtime_error("Request already running");
      req->error.clear(); req->running = true;
      req->callback_keepalive = req->shared_from_this(); }
    try {
        session_->RunAsync(Ort::RunOptions{nullptr}, input_names_, req->input, 1,
                           output_names_, &req->output, 1, &OrtBackend::on_complete, req);
    } catch (...) {
        { std::lock_guard<std::mutex> lock(req->mutex); req->running = false; req->callback_keepalive.reset(); }
        req->completed.notify_all(); throw;
    }
}
void OrtBackend::on_complete(void* data, OrtValue**, size_t, OrtStatus* status) noexcept {
    auto* req = static_cast<OrtRequest*>(data);
    auto keepalive = req->callback_keepalive;
    // The preallocated output OrtValue is owned by the request, not callback.
    if (status) {
        req->error = Ort::GetApi().GetErrorMessage(status);
        Ort::GetApi().ReleaseStatus(status);
    }
    try { if (req->callback) req->callback(req->callback_arg); }
    catch (...) { req->error = "Completion callback failed"; }
    // Publish completion only after handing the slot to the output queue.
    // Otherwise the submitter could close that queue before this callback.
    { std::lock_guard<std::mutex> lock(req->mutex);
      req->callback_keepalive.reset(); req->running = false; }
    req->completed.notify_all();
}
nlohmann::json OrtBackend::describe() const {
    const bool cuda = device_ == "CUDA";
    nlohmann::json shape = nlohmann::json::array();
    for (auto n : shape_) shape.push_back(std::to_string(n));
    return {{"engine", "onnxruntime-cpp"}, {"onnxruntime_version", Ort::GetVersionString()},
        {"requested_backend", cuda ? "cuda" : "cpu"}, {"openvino_device", nullptr},
        {"actual_providers", cuda ? nlohmann::json::array({"CUDAExecutionProvider", "CPUExecutionProvider"})
                                 : nlohmann::json::array({"CPUExecutionProvider"})},
        {"provider_note", "Configured providers; inspect optional ORT profile for per-node assignment"},
        {"fell_back_to_cpu", false}, {"input_shape", shape}, {"model_path", model_},
        {"intra_threads", threads_}, {"inter_threads", 1}};
}
} // namespace eirl
