#include "sources.hpp"

#include <random>
#include <stdexcept>

namespace eirl {

VideoSource::VideoSource(std::string path, long long n)
    : path_(std::move(path)), n_(n), cap_(path_) {
    if (!cap_.isOpened()) throw std::runtime_error("cannot open video: " + path_);
    frame_count_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_COUNT));
    fps_ = cap_.get(cv::CAP_PROP_FPS);
    width_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    height_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
}

bool VideoSource::next(cv::Mat& out) {
    if (produced_ >= n_) return false;
    if (!cap_.read(out) || out.empty()) {
        cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
        ++restarts_;
        if (!cap_.read(out) || out.empty()) {
            throw std::runtime_error("video produced no frames: " + path_);
        }
    }
    ++produced_;
    return true;
}

nlohmann::json VideoSource::describe() const {
    return {{"kind", "video"}, {"path", path_}, {"frame_count", frame_count_},
            {"fps", fps_}, {"width", width_}, {"height", height_},
            {"restarts", restarts_}};
}

SyntheticSource::SyntheticSource(long long n, int height, int width, int seed)
    : n_(n), height_(height), width_(width) {
    base_.create(height, width, CV_8UC3);
    std::mt19937 rng(static_cast<unsigned>(seed));
    std::uniform_int_distribution<int> dist(0, 255);
    for (int y = 0; y < height; ++y) {
        unsigned char* row = base_.ptr<unsigned char>(y);
        for (int x = 0; x < width * 3; ++x) row[x] = static_cast<unsigned char>(dist(rng));
    }
}

bool SyntheticSource::next(cv::Mat& out) {
    if (produced_ >= n_) return false;
    // Copy rather than hand back the same buffer: the copy is the only work
    // this source does, and sharing it would let a later stage see a frame
    // mutate underneath it.
    base_.copyTo(out);
    ++produced_;
    return true;
}

nlohmann::json SyntheticSource::describe() const {
    // No decode happens here: `read` measures the copy of an h x w x 3 uint8
    // array and nothing else. Never use this as the baseline for a claim about
    // decode cost.
    return {{"kind", "synthetic"}, {"width", width_}, {"height", height_},
            {"restarts", 0}};
}

std::unique_ptr<Source> make_source(const std::string& path, long long n) {
    if (path.empty() || path == "synthetic") {
        return std::make_unique<SyntheticSource>(n);
    }
    return std::make_unique<VideoSource>(path, n);
}

}  // namespace eirl
