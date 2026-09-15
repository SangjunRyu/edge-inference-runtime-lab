#include "ov_backend.hpp"

#include <cstring>

namespace eirl {

OvError::OvError(const std::string& call, ov_status_e status)
    : std::runtime_error(call + " failed, ov_status_e=" + std::to_string(static_cast<int>(status))),
      status_(status) {}

void check(ov_status_e status, const char* call) {
    if (status != OK) throw OvError(call, status);
}

std::vector<std::string> available_devices() {
    CoreHandle core;
    {
        ov_core_t* raw = nullptr;
        check(ov_core_create(&raw), "ov_core_create");
        core.reset(raw);
    }
    ov_available_devices_t devs{};
    check(ov_core_get_available_devices(core.get(), &devs),
          "ov_core_get_available_devices");
    std::vector<std::string> out;
    out.reserve(devs.size);
    for (size_t i = 0; i < devs.size; ++i) out.emplace_back(devs.devices[i]);
    ov_available_devices_free(&devs);
    return out;
}

OvBackend::OvBackend(const std::string& model_path, const std::string& device)
    : device_(device), model_path_(model_path) {
    {
        ov_core_t* raw = nullptr;
        check(ov_core_create(&raw), "ov_core_create");
        core_.reset(raw);
    }

    ModelHandle model;
    {
        ov_model_t* raw = nullptr;
        // Weights path empty: an .onnx carries its own.
        check(ov_core_read_model(core_.get(), model_path.c_str(), nullptr, &raw),
              "ov_core_read_model");
        model.reset(raw);
    }

    {
        ov_compiled_model_t* raw = nullptr;
        check(ov_core_compile_model(core_.get(), model.get(), device.c_str(), 0, &raw),
              "ov_core_compile_model");
        compiled_.reset(raw);
    }

    // Input shape, so the caller never has to agree with the model by hand.
    OutputHandle in_port;
    {
        ov_output_const_port_t* raw = nullptr;
        check(ov_compiled_model_input(compiled_.get(), &raw), "ov_compiled_model_input");
        in_port.reset(raw);
    }
    ov_shape_t shape{};
    check(ov_const_port_get_shape(in_port.get(), &shape), "ov_const_port_get_shape");
    input_shape_.assign(shape.dims, shape.dims + shape.rank);
    input_elements_ = 1;
    for (int64_t d : input_shape_) input_elements_ *= static_cast<size_t>(d);
    if (input_shape_.size() == 4) {
        height_ = static_cast<int>(input_shape_[2]);
        width_ = static_cast<int>(input_shape_[3]);
    }
    ov_shape_free(&shape);

    OutputHandle out_port;
    {
        ov_output_const_port_t* raw = nullptr;
        check(ov_compiled_model_output(compiled_.get(), &raw), "ov_compiled_model_output");
        out_port.reset(raw);
    }
    ov_shape_t oshape{};
    check(ov_const_port_get_shape(out_port.get(), &oshape), "ov_const_port_get_shape(output)");
    output_elements_ = 1;
    for (size_t i = 0; i < static_cast<size_t>(oshape.rank); ++i) {
        output_elements_ *= static_cast<size_t>(oshape.dims[i]);
    }
    ov_shape_free(&oshape);
}

RequestHandle OvBackend::make_request() const {
    ov_infer_request_t* raw = nullptr;
    check(ov_compiled_model_create_infer_request(compiled_.get(), &raw),
          "ov_compiled_model_create_infer_request");
    return RequestHandle(raw);
}

TensorHandle OvBackend::wrap_input(const float* input) const {
    ov_shape_t shape{};
    check(ov_shape_create(static_cast<int64_t>(input_shape_.size()),
                          const_cast<int64_t*>(input_shape_.data()), &shape),
          "ov_shape_create");
    ov_tensor_t* raw = nullptr;
    // from_host_ptr: OpenVINO reads the caller's buffer and does not take
    // ownership of it, which is what makes reusing one buffer across frames
    // possible at all. The buffer must outlive the request that reads it --
    // the constraint phase 3's optimisation has to respect, and the one the
    // equivalence check exists to police.
    const ov_status_e st = ov_tensor_create_from_host_ptr(
        ov_element_type_e::F32, shape, const_cast<float*>(input), &raw);
    ov_shape_free(&shape);
    check(st, "ov_tensor_create_from_host_ptr");
    return TensorHandle(raw);
}

void OvBackend::infer(ov_infer_request_t* req, const float* input, float* output) const {
    TensorHandle in = wrap_input(input);
    check(ov_infer_request_set_input_tensor(req, in.get()),
          "ov_infer_request_set_input_tensor");
    check(ov_infer_request_infer(req), "ov_infer_request_infer");

    ov_tensor_t* out_raw = nullptr;
    check(ov_infer_request_get_output_tensor(req, &out_raw),
          "ov_infer_request_get_output_tensor");
    TensorHandle out(out_raw);

    void* data = nullptr;
    check(ov_tensor_data(out.get(), &data), "ov_tensor_data");
    std::memcpy(output, data, output_elements_ * sizeof(float));
}

void OvBackend::infer_bound(ov_infer_request_t* req, float* output) const {
    check(ov_infer_request_infer(req), "ov_infer_request_infer");
    ov_tensor_t* out_raw = nullptr;
    check(ov_infer_request_get_output_tensor(req, &out_raw),
          "ov_infer_request_get_output_tensor");
    TensorHandle out(out_raw);
    void* data = nullptr;
    check(ov_tensor_data(out.get(), &data), "ov_tensor_data");
    std::memcpy(output, data, output_elements_ * sizeof(float));
}

nlohmann::json OvBackend::describe() const {
    // Same field names as bench/backends.py so summarize.py and the rest read
    // C++ results unchanged. `engine` is the one addition: these runs go
    // through OpenVINO directly, while the Python accelerator rows went
    // through it as an ONNX Runtime execution provider, and that is a
    // different code path rather than the same one renamed.
    return {
        {"engine", "openvino-c-api"},
        {"requested_backend", "openvino"},
        {"openvino_device", device_},
        {"actual_providers", nlohmann::json::array({"OpenVINOExecutionProvider"})},
        {"fell_back_to_cpu", false},
        {"input_shape", [&] {
            nlohmann::json a = nlohmann::json::array();
            for (int64_t d : input_shape_) a.push_back(std::to_string(d));
            return a;
        }()},
        {"model_path", model_path_},
    };
}

}  // namespace eirl
