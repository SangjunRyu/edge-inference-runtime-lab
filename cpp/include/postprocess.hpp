// Post-processing -- kept apart from preprocess because it needs no OpenCV.
//
// That split is not cosmetic: it lets the phase 0 probe exercise the backend,
// the harness and the top-k path before OpenCV is installed at all, which is
// what makes "fail fast on the toolchain" possible rather than aspirational.

#ifndef EIRL_POSTPROCESS_HPP
#define EIRL_POSTPROCESS_HPP

#include <cstddef>
#include <utility>
#include <vector>

namespace eirl {

/// Logits -> top-k (class index, probability), highest first.
/// top_k defaults to 5 to match DX-APP's resnet50 config.json, so the
/// postprocess stage does the same amount of work as the reference.
std::vector<std::pair<int, float>> postprocess(const float* logits, size_t n,
                                               int top_k = 5);

}  // namespace eirl

#endif  // EIRL_POSTPROCESS_HPP
