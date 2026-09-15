#include "pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#include <opencv2/imgproc.hpp>

namespace eirl {

void preprocess(const cv::Mat& frame_bgr, int height, int width, float* out) {
    // The five passes the Python makes, in the same order, so the baseline is
    // the baseline and not already half-optimised.
    cv::Mat resized, rgb, f32, normalised;
    cv::resize(frame_bgr, resized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(f32, CV_32FC3, 1.0 / 255.0);
    cv::subtract(f32, cv::Scalar(kMean[0], kMean[1], kMean[2]), normalised);
    cv::divide(normalised, cv::Scalar(kStd[0], kStd[1], kStd[2]), normalised);

    // HWC -> CHW.
    const size_t plane = static_cast<size_t>(height) * width;
    std::vector<cv::Mat> chans{
        cv::Mat(height, width, CV_32F, out),
        cv::Mat(height, width, CV_32F, out + plane),
        cv::Mat(height, width, CV_32F, out + 2 * plane)};
    cv::split(normalised, chans);
}

void preprocess_fused(const cv::Mat& frame_bgr, int height, int width, float* out) {
    cv::Mat resized;
    cv::resize(frame_bgr, resized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);

    // Colour swap, scale, normalise and transpose in one walk over the pixels.
    // The resize stays OpenCV's: reimplementing the interpolation would change
    // the numbers, and this has to stay bit-comparable with the plain path for
    // verify_equivalence to mean anything.
    const size_t plane = static_cast<size_t>(height) * width;
    const float inv255 = 1.0f / 255.0f;
    const float r_scale = inv255 / kStd[0], g_scale = inv255 / kStd[1],
                b_scale = inv255 / kStd[2];
    const float r_off = kMean[0] / kStd[0], g_off = kMean[1] / kStd[1],
                b_off = kMean[2] / kStd[2];

    for (int y = 0; y < height; ++y) {
        const unsigned char* row = resized.ptr<unsigned char>(y);
        float* r = out + static_cast<size_t>(y) * width;
        float* g = r + plane;
        float* b = g + plane;
        for (int x = 0; x < width; ++x) {
            // OpenCV rows are BGR; the model wants RGB planes.
            const unsigned char bb = row[3 * x + 0];
            const unsigned char gg = row[3 * x + 1];
            const unsigned char rr = row[3 * x + 2];
            r[x] = static_cast<float>(rr) * r_scale - r_off;
            g[x] = static_cast<float>(gg) * g_scale - g_off;
            b[x] = static_cast<float>(bb) * b_scale - b_off;
        }
    }
}

}  // namespace eirl
