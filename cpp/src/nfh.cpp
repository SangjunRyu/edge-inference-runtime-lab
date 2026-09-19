#include "nfh.hpp"

#include <cstring>

namespace eirl::nfh {

std::size_t aligned_row_elems(std::size_t row_elems, std::size_t unit) {
    if (unit == 0) return row_elems;
    return (row_elems + unit - 1) / unit * unit;
}

std::size_t encoded_bytes(std::size_t rows, std::size_t row_elems,
                          std::size_t unit, std::size_t elem_bytes) {
    return rows * aligned_row_elems(row_elems, unit) * elem_bytes;
}

void encode(const std::uint8_t* src, std::uint8_t* dst, std::size_t rows,
            std::size_t row_elems, std::size_t unit, std::size_t elem_bytes) {
    const std::size_t padded = aligned_row_elems(row_elems, unit);
    const std::size_t dense_row = row_elems * elem_bytes;
    const std::size_t padded_row = padded * elem_bytes;
    for (std::size_t r = 0; r < rows; ++r) {
        std::uint8_t* out = dst + r * padded_row;
        std::memcpy(out, src + r * dense_row, dense_row);
        if (padded_row > dense_row) std::memset(out + dense_row, 0, padded_row - dense_row);
    }
}

void decode(const std::uint8_t* src, std::uint8_t* dst, std::size_t rows,
            std::size_t row_elems, std::size_t unit, std::size_t elem_bytes) {
    const std::size_t padded = aligned_row_elems(row_elems, unit);
    const std::size_t dense_row = row_elems * elem_bytes;
    const std::size_t padded_row = padded * elem_bytes;
    for (std::size_t r = 0; r < rows; ++r) {
        std::memcpy(dst + r * dense_row, src + r * padded_row, dense_row);
    }
}

void encode_transposed(const std::uint8_t* src, std::uint8_t* dst, std::size_t rows,
                       std::size_t cols, std::size_t unit, std::size_t elem_bytes) {
    const std::size_t padded = aligned_row_elems(rows, unit);
    const std::size_t padded_row = padded * elem_bytes;
    for (std::size_t j = 0; j < cols; ++j) {
        std::uint8_t* out = dst + j * padded_row;
        for (std::size_t i = 0; i < rows; ++i) {
            std::memcpy(out + i * elem_bytes, src + (i * cols + j) * elem_bytes, elem_bytes);
        }
        if (padded > rows) std::memset(out + rows * elem_bytes, 0, (padded - rows) * elem_bytes);
    }
}

void transpose_naive(const std::uint8_t* src, std::uint8_t* dst, std::size_t rows,
                     std::size_t cols, std::size_t elem_bytes) {
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            std::memcpy(dst + (j * rows + i) * elem_bytes,
                        src + (i * cols + j) * elem_bytes,
                        elem_bytes);
        }
    }
}

void transpose_blocked(const std::uint8_t* src, std::uint8_t* dst, std::size_t rows,
                       std::size_t cols, std::size_t elem_bytes, std::size_t block) {
    if (block == 0) block = 32;
    for (std::size_t ii = 0; ii < rows; ii += block) {
        const std::size_t i_end = (ii + block < rows) ? ii + block : rows;
        for (std::size_t jj = 0; jj < cols; jj += block) {
            const std::size_t j_end = (jj + block < cols) ? jj + block : cols;
            for (std::size_t i = ii; i < i_end; ++i) {
                for (std::size_t j = jj; j < j_end; ++j) {
                    std::memcpy(dst + (j * rows + i) * elem_bytes,
                                src + (i * cols + j) * elem_bytes,
                                elem_bytes);
                }
            }
        }
    }
}

}  // namespace eirl::nfh
