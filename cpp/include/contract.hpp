// W5-3 -- the compiled model as the single source of truth for buffers.
//
// W5-1 gave buffers an owner and W5-2 gave the pipeline a bound. Both took the
// buffer *size* as a constant: 150528, written in a default argument, correct
// for exactly one model at exactly one resolution. That is the shape of the
// mistake this file exists to remove.
//
// A compiled `.dxnn` already states what a buffer has to be: the dtype, the
// shape, the layout, the alignment unit, the padded size the device will read,
// and the region the runtime is charged for per buffer. A runtime that carries
// its own copy of those numbers has two sources of truth and no way to notice
// when they disagree -- and they disagree silently, because a buffer of the
// wrong size is still a valid buffer.
//
// So `tools/dxnn_inspect.py --contract` writes them out once, and everything
// here is derived:
//
//   * pools are sized from the contract, never from a literal;
//   * a buffer that does not match the contract is refused **before** any work
//     is submitted, because after inference the only evidence left is a wrong
//     answer that looks like a right one;
//   * the buffer count comes from the contract's own cost model and the
//     device's memory, which is W5-2's resource bound arriving from the model
//     file instead of from a command line.
//
// The contract is also checked against itself on load. A file that says a
// tensor's allocation is smaller than its logical size, or that disagrees with
// the padding rule its own layout implies, is rejected rather than trusted --
// the inspector and this loader derive the same quantities independently, so
// agreement between them is evidence and disagreement is a defect in one of
// the two.
//
// Nothing here is copied from DEEPX. The contract is produced from a file the
// compiler wrote, by a parser in this repository.

#ifndef EIRL_CONTRACT_HPP
#define EIRL_CONTRACT_HPP

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "ownership.hpp"

namespace eirl {

/// The contract was unusable. Thrown on load, never during a run.
class ContractError : public std::runtime_error {
public:
    explicit ContractError(const std::string& what) : std::runtime_error(what) {}
};

/// One tensor, exactly as the compiled model describes it.
struct TensorSpec {
    std::string name;
    std::string dtype;
    std::string layout;
    Direction direction = Direction::Input;
    std::vector<std::int64_t> shape;
    std::size_t align_unit = 0;
    std::size_t logical_bytes = 0;
    std::size_t allocated_bytes = 0;
    std::size_t predicted_bytes = 0;   // 0 when the layout has no stated rule

    std::string describe() const;
};

/// What a runtime must honour to run this model.
struct ModelContract {
    std::string model;
    std::string target;
    int container_version = 0;
    std::vector<TensorSpec> tensors;

    std::size_t input_buffer_bytes = 0;
    std::size_t output_buffer_bytes = 0;
    std::uint64_t fixed_cost = 0;
    std::uint64_t per_buffer_cost = 0;
    int default_buffers = 6;
    int max_buffers = 100;

    /// Read and validate. Throws ContractError rather than returning something
    /// half-trusted; a contract that cannot be believed is not a smaller
    /// contract.
    static ModelContract load(const std::string& path);

    /// Bytes a pool slot must have for this direction.
    std::size_t buffer_bytes(Direction d) const {
        return d == Direction::Input ? input_buffer_bytes : output_buffer_bytes;
    }

    /// Would this buffer be accepted? Called before submitting, never after.
    bool accepts(Direction d, std::size_t bytes, std::string* why = nullptr) const;

    /// W5-2's resource bound, taken from the model rather than a flag.
    /// Reproduces DX-RT's walk-down; throws ContractError when not even one
    /// buffer fits, which is what `InsufficientMemoryException` reports.
    std::uint32_t max_buffers_for(std::uint64_t npu_memory, std::uint32_t requested) const;

    /// The checks `load` runs. Exposed so a harness can corrupt a loaded
    /// contract and show that they bite.
    void validate() const;

    std::string summary() const;
};

}  // namespace eirl

#endif  // EIRL_CONTRACT_HPP
