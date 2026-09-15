#include "postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace eirl {

std::vector<std::pair<int, float>> postprocess(const float* logits, size_t n,
                                               int top_k) {
    if (n == 0) return {};
    const int k = std::min<int>(top_k, static_cast<int>(n));

    // Softmax with the max subtracted, matching the Python exactly: the
    // subtraction is not only for stability, it changes the rounding, and the
    // two implementations are compared to 1e-4.
    const float mx = *std::max_element(logits, logits + n);
    std::vector<float> probs(n);
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        probs[i] = std::exp(logits[i] - mx);
        sum += probs[i];
    }
    const float inv = static_cast<float>(1.0 / sum);
    for (size_t i = 0; i < n; ++i) probs[i] *= inv;

    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return probs[a] > probs[b]; });

    std::vector<std::pair<int, float>> out;
    out.reserve(k);
    for (int i = 0; i < k; ++i) out.emplace_back(idx[i], probs[idx[i]]);
    return out;
}

}  // namespace eirl
