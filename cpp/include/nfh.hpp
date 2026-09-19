// W5-3 -- host-side NPU format conversion, re-implemented from the rules.
//
// A compiled DEEPX model stores its tensors padded: the alignment unit counts
// *elements*, and the layout says which trailing axes form the row that gets
// padded. `tools/dxnn_inspect.py` reads those numbers out of a real `.dxnn`
// and can print them as arguments for this code, so the geometry here is the
// geometry of a model that was actually compiled rather than a plausible one.
//
// Nothing is copied from DEEPX. Every `dx_rt` file states that the software is
// supplied to customers of a DEEPX NPU and may not be redistributed, so what is
// reproduced is the behaviour, cited by file and line, and checked against the
// container's own declared sizes.
//
// Two facts about `dx_rt/lib/npu_format_handler.cpp` motivate the pair of
// transposes below, and both were read rather than assumed:
//
//   * the fallback at :433 copies one element at a time with `memcpy` into a
//     strided destination -- no blocking, no vectorisation;
//   * the accelerated paths at :456-460 exist only under `USE_IPP` (Intel IPP)
//     or `USE_NEON` (ARM), and `USE_NPU_FORMAT_CONVERSION_ACCELERATION` is OFF
//     by default (`cmake/dxrt.cfg.cmake:19`), gated again at run time.
//
// So on a plain x86 build with default options, the scalar path is the path.
// `transpose_naive` reproduces its shape; `transpose_blocked` is the ordinary
// cache-blocked alternative. Neither uses intrinsics: W5's plan rules out SIMD
// as decoration, and the point here is that the *access order* is what costs,
// which is a claim a portable implementation can make and defend.
//
// This entire header is a function from bytes to bytes. It needs no NPU, and
// its correctness oracle is its own inverse.

#ifndef EIRL_NFH_HPP
#define EIRL_NFH_HPP

#include <cstddef>
#include <cstdint>

namespace eirl::nfh {

/// Round a row up to a whole number of `unit` ELEMENTS. Not bytes -- reading
/// the unit as bytes gives 4032 where a real container says 4096.
std::size_t aligned_row_elems(std::size_t row_elems, std::size_t unit);

/// Bytes a padded tensor occupies.
std::size_t encoded_bytes(std::size_t rows, std::size_t row_elems,
                          std::size_t unit, std::size_t elem_bytes);

/// Dense rows -> padded rows. Padding is zeroed so the result is a function of
/// the input alone, which is what lets two runs be compared byte for byte.
void encode(const std::uint8_t* src, std::uint8_t* dst, std::size_t rows,
            std::size_t row_elems, std::size_t unit, std::size_t elem_bytes);

/// Padded rows -> dense rows.
void decode(const std::uint8_t* src, std::uint8_t* dst, std::size_t rows,
            std::size_t row_elems, std::size_t unit, std::size_t elem_bytes);

/// Transpose and pad in a single pass: dense `rows x cols` in, padded
/// `cols x aligned(rows)` out. Exists so the cost of doing the two separately
/// can be stated as a difference rather than asserted.
void encode_transposed(const std::uint8_t* src, std::uint8_t* dst, std::size_t rows,
                       std::size_t cols, std::size_t unit, std::size_t elem_bytes);

/// One element at a time into a strided destination: the shape of the fallback
/// in dx_rt's format handler when no IPP or NEON path is compiled in.
void transpose_naive(const std::uint8_t* src, std::uint8_t* dst, std::size_t rows,
                     std::size_t cols, std::size_t elem_bytes);

/// The same result, walked in tiles so both sides stay in cache.
void transpose_blocked(const std::uint8_t* src, std::uint8_t* dst, std::size_t rows,
                       std::size_t cols, std::size_t elem_bytes, std::size_t block = 32);

}  // namespace eirl::nfh

#endif  // EIRL_NFH_HPP
