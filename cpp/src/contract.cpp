#include "contract.hpp"

#include <fstream>
#include <map>
#include <sstream>

#include <nlohmann/json.hpp>

namespace eirl {
namespace {

using nlohmann::json;

const std::map<std::string, std::size_t> kDtypeBytes = {
    {"INT8", 1},  {"UINT8", 1},
    {"INT16", 2}, {"UINT16", 2}, {"FLOAT16", 2}, {"BFLOAT16", 2},
    {"INT32", 4}, {"UINT32", 4}, {"FLOAT32", 4},
    {"INT64", 8}, {"UINT64", 8}, {"FLOAT64", 8},
};

/// How many trailing axes each layout folds into the padded row. The same table
/// the inspector uses, restated here on purpose: two independent derivations
/// that agree are evidence, and one shared implementation would only ever agree
/// with itself.
const std::map<std::string, std::size_t> kLayoutRowAxes = {
    {"ALIGNED", 1}, {"PRE_IM2COL", 2},
};

std::size_t align_up(std::size_t v, std::size_t unit) {
    return unit ? (v + unit - 1) / unit * unit : v;
}

std::size_t predict(const TensorSpec& t) {
    auto w = kDtypeBytes.find(t.dtype);
    auto a = kLayoutRowAxes.find(t.layout);
    if (w == kDtypeBytes.end() || a == kLayoutRowAxes.end()) return 0;
    if (t.align_unit == 0 || t.shape.size() < a->second) return 0;
    std::size_t row = 1;
    for (std::size_t i = t.shape.size() - a->second; i < t.shape.size(); ++i) {
        row *= static_cast<std::size_t>(t.shape[i]);
    }
    std::size_t outer = 1;
    for (std::size_t i = 0; i + a->second < t.shape.size(); ++i) {
        outer *= static_cast<std::size_t>(t.shape[i]);
    }
    return outer * align_up(row, t.align_unit) * w->second;
}

TensorSpec read_spec(const json& j, Direction d) {
    TensorSpec t;
    t.direction = d;
    t.name = j.value("name", std::string("?"));
    t.dtype = j.value("dtype", std::string(""));
    t.layout = j.value("layout", std::string(""));
    t.align_unit = j.value("align_unit", std::size_t{0});
    t.logical_bytes = j.value("logical_bytes", std::size_t{0});
    t.allocated_bytes = j.value("allocated_bytes", std::size_t{0});
    t.predicted_bytes = j.value("predicted_bytes", std::size_t{0});
    if (j.contains("shape") && j["shape"].is_array()) {
        for (const auto& d2 : j["shape"]) t.shape.push_back(d2.get<std::int64_t>());
    }
    return t;
}

}  // namespace

std::string TensorSpec::describe() const {
    std::ostringstream o;
    o << name << " " << dtype << " [";
    for (std::size_t i = 0; i < shape.size(); ++i) o << (i ? "," : "") << shape[i];
    o << "] " << layout << " align " << align_unit
      << "  logical " << logical_bytes << " B, allocated " << allocated_bytes << " B";
    return o.str();
}

ModelContract ModelContract::load(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw ContractError("cannot open contract: " + path);

    json j;
    try {
        f >> j;
    } catch (const std::exception& e) {
        throw ContractError(std::string("contract is not valid JSON: ") + e.what());
    }

    const std::string schema = j.value("schema", std::string(""));
    if (schema != "eirl/dxnn-contract/1") {
        throw ContractError("unknown contract schema: '" + schema + "'");
    }

    ModelContract c;
    c.model = j.value("model", std::string("?"));
    c.target = j.value("target", std::string("?"));
    c.container_version = j.value("container_version", 0);
    c.input_buffer_bytes = j["buffer_bytes"].value("input", std::size_t{0});
    c.output_buffer_bytes = j["buffer_bytes"].value("output", std::size_t{0});
    c.fixed_cost = j.value("fixed_cost", std::uint64_t{0});
    c.per_buffer_cost = j.value("per_buffer_cost", std::uint64_t{0});
    c.default_buffers = j.value("default_buffers", 6);
    c.max_buffers = j.value("max_buffers", 100);

    for (const auto& task : j.value("tasks", json::array())) {
        for (const auto& t : task.value("inputs", json::array())) {
            c.tensors.push_back(read_spec(t, Direction::Input));
        }
        for (const auto& t : task.value("outputs", json::array())) {
            c.tensors.push_back(read_spec(t, Direction::Output));
        }
    }
    if (c.tensors.empty()) throw ContractError("contract describes no tensors");

    c.validate();
    return c;
}

void ModelContract::validate() const {
    std::ostringstream bad;
    int faults = 0;
    auto fail = [&](const std::string& why) {
        if (faults++) bad << "; ";
        bad << why;
    };

    if (input_buffer_bytes == 0) fail("input buffer size is zero");
    if (output_buffer_bytes == 0) fail("output buffer size is zero");
    if (per_buffer_cost == 0) fail("per-buffer cost is zero");

    std::size_t widest_input = 0;
    for (const auto& t : tensors) {
        if (t.align_unit == 0) fail(t.name + ": alignment unit is zero");
        if (t.allocated_bytes < t.logical_bytes) {
            fail(t.name + ": allocation " + std::to_string(t.allocated_bytes)
                 + " is smaller than the logical size " + std::to_string(t.logical_bytes));
        }
        // The loader predicts the padded size from the layout rule and compares
        // it with what the file declares. Two derivations that agree are
        // evidence; one that disagrees means the rule or the file is wrong, and
        // either way the size is not safe to allocate from.
        const std::size_t mine = predict(t);
        if (mine != 0 && mine != t.allocated_bytes) {
            fail(t.name + ": layout " + t.layout + " implies "
                 + std::to_string(mine) + " B but the contract says "
                 + std::to_string(t.allocated_bytes));
        }
        if (t.predicted_bytes != 0 && mine != 0 && t.predicted_bytes != mine) {
            fail(t.name + ": the producer predicted " + std::to_string(t.predicted_bytes)
                 + " B and this loader predicts " + std::to_string(mine));
        }
        if (t.direction == Direction::Input && t.allocated_bytes > widest_input) {
            widest_input = t.allocated_bytes;
        }
    }

    if (widest_input > input_buffer_bytes) {
        fail("an input tensor needs " + std::to_string(widest_input)
             + " B but the input buffer is " + std::to_string(input_buffer_bytes) + " B");
    }

    if (faults) throw ContractError("contract failed validation: " + bad.str());
}

bool ModelContract::accepts(Direction d, std::size_t bytes, std::string* why) const {
    const std::size_t want = buffer_bytes(d);
    if (bytes == want) return true;
    if (why) {
        std::ostringstream o;
        o << name(d) << " buffer is " << bytes << " B; the model requires " << want
          << " B (" << (bytes < want ? "too small, the device would read past it"
                                     : "too large, the extra is never written")
          << ")";
        *why = o.str();
    }
    return false;
}

std::uint32_t ModelContract::max_buffers_for(std::uint64_t npu_memory,
                                             std::uint32_t requested) const {
    std::uint32_t adjusted = requested;
    while (adjusted > 1) {
        if (fixed_cost + per_buffer_cost * adjusted <= npu_memory) break;
        --adjusted;
    }
    if (fixed_cost + per_buffer_cost > npu_memory) {
        throw ContractError(
            "NPU memory insufficient for model: required "
            + std::to_string(fixed_cost + per_buffer_cost) + " bytes (fixed "
            + std::to_string(fixed_cost) + " + per-buffer "
            + std::to_string(per_buffer_cost) + "), available "
            + std::to_string(npu_memory) + " bytes");
    }
    return adjusted;
}

std::string ModelContract::summary() const {
    std::ostringstream o;
    o << model << "  container v" << container_version << "  target " << target << "\n";
    for (const auto& t : tensors) {
        o << "  " << name(t.direction) << "   " << t.describe() << "\n";
    }
    o << "  buffers   input " << input_buffer_bytes << " B, output "
      << output_buffer_bytes << " B\n";
    o << "  costs     fixed " << fixed_cost << " B, per buffer " << per_buffer_cost << " B";
    return o.str();
}

}  // namespace eirl
