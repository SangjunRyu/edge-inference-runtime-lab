// OpenVINO backend -- the C++ side of bench/backends.py.
//
// Built on OpenVINO's **C** API rather than its C++ one, for two reasons that
// are not compromises:
//
//   * MinGW and MSVC share no C++ ABI, so the MSVC-built `openvino.lib` cannot
//     be linked from GCC. C has a stable ABI, so `openvino_c.lib` can -- which
//     is what lets this build with no Microsoft toolchain present, and what
//     lets the same source build under MSVC later.
//   * Wrapping a C API in RAII is the thing W4 set out to demonstrate.
//     Ownership of every handle is explicit and every release path is the
//     destructor, so there is no leak to find on an error return.
//
// Every entry point returns `ov_status_e`; ignoring one silently produces a
// null handle that fails much later somewhere else, so `check()` converts a
// bad status into an exception carrying the call that produced it.

#ifndef EIRL_OV_BACKEND_HPP
#define EIRL_OV_BACKEND_HPP

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <openvino/c/openvino.h>

#include <nlohmann/json.hpp>

namespace eirl {

class OvError : public std::runtime_error {
public:
    OvError(const std::string& call, ov_status_e status);
    ov_status_e status() const { return status_; }

private:
    ov_status_e status_;
};

void check(ov_status_e status, const char* call);

/// unique_ptr with the matching `ov_*_free` as its deleter. The C API's
/// ownership rules are the type system's job here, not the caller's memory.
template <typename T, void (*Free)(T*)>
struct OvDeleter {
    void operator()(T* p) const noexcept { if (p) Free(p); }
};
template <typename T, void (*Free)(T*)>
using OvHandle = std::unique_ptr<T, OvDeleter<T, Free>>;

using CoreHandle = OvHandle<ov_core_t, ov_core_free>;
using ModelHandle = OvHandle<ov_model_t, ov_model_free>;
using CompiledHandle = OvHandle<ov_compiled_model_t, ov_compiled_model_free>;
using RequestHandle = OvHandle<ov_infer_request_t, ov_infer_request_free>;
using TensorHandle = OvHandle<ov_tensor_t, ov_tensor_free>;
using OutputHandle = OvHandle<ov_output_const_port_t, ov_output_const_port_free>;
using ShapeOwner = std::unique_ptr<ov_shape_t, void (*)(ov_shape_t*)>;

/// Devices OpenVINO can see on this machine, e.g. {"CPU", "GPU", "NPU"}.
std::vector<std::string> available_devices();

/// One compiled model on one device. Infer requests are created from it; the
/// compiled model outlives them all, which is why it owns nothing they need.
class OvBackend {
public:
    /// `device` is "CPU", "GPU", "NPU", ... exactly as OpenVINO names them.
    OvBackend(const std::string& model_path, const std::string& device);

    int height() const { return height_; }
    int width() const { return width_; }
    const std::string& device() const { return device_; }
    const std::vector<int64_t>& input_shape() const { return input_shape_; }
    size_t output_elements() const { return output_elements_; }

    /// A request bound to this model. Cheap to hold, not cheap to create, so
    /// the sequential runner makes one and the async runner makes `inflight`.
    RequestHandle make_request() const;

    /// Synchronous inference. `input` is `height*width*3` floats in NCHW
    /// order; `output` is filled with `output_elements()` logits.
    void infer(ov_infer_request_t* req, const float* input, float* output) const;

    /// Inference on a request whose input tensor is already bound. Skips the
    /// per-frame wrap and set, which is the difference phase 3 measures.
    void infer_bound(ov_infer_request_t* req, float* output) const;

    /// The input tensor handle over caller-owned memory, so the buffer can be
    /// reused across frames instead of reallocated -- kept separate from
    /// `infer` because W4 phase 3 measures exactly that difference.
    TensorHandle wrap_input(const float* input) const;

    nlohmann::json describe() const;

private:
    CoreHandle core_;
    CompiledHandle compiled_;
    std::string device_, model_path_;
    std::vector<int64_t> input_shape_;
    int height_ = 0, width_ = 0;
    size_t input_elements_ = 0, output_elements_ = 0;
};

}  // namespace eirl

#endif  // EIRL_OV_BACKEND_HPP
