// Pre/post-processing -- the C++ side of bench/pipeline.py.
//
// Deliberately the *same* five passes the Python does: resize, BGR->RGB,
// /255, normalise, transpose to NCHW. Writing the fused single-pass version
// first would confound the port with the optimisation, and W4 phase 3 exists
// to measure that difference against a baseline that matches Python.
//
// `preprocess_fused` is that optimisation, kept beside the plain one so the
// two can be compared in a paired run rather than across a rewrite.

#ifndef EIRL_PIPELINE_HPP
#define EIRL_PIPELINE_HPP

#include <opencv2/core.hpp>

#include "postprocess.hpp"

namespace eirl {

/// ImageNet normalisation, matching bench/pipeline.py.
constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};

/// BGR uint8 HWC -> normalised float32 NCHW, written into `out`
/// (3*height*width floats, caller-owned).
void preprocess(const cv::Mat& frame_bgr, int height, int width, float* out);

/// One pass over the resized image instead of five. Same arithmetic, same
/// result; the difference is how many times the data is walked.
void preprocess_fused(const cv::Mat& frame_bgr, int height, int width, float* out);

}  // namespace eirl

#endif  // EIRL_PIPELINE_HPP
