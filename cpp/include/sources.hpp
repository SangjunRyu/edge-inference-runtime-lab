// Frame sources -- the C++ side of bench/sources.py.
//
// `read` is a measured stage, so decoding must be real work on every frame.
// Decoding once and replaying the decoded array would delete the stage.

#ifndef EIRL_SOURCES_HPP
#define EIRL_SOURCES_HPP

#include <memory>
#include <string>

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

namespace eirl {

class Source {
public:
    virtual ~Source() = default;
    /// Next frame into `out`. False when the source is finished.
    virtual bool next(cv::Mat& out) = 0;
    virtual nlohmann::json describe() const = 0;
};

/// Decodes from a video file, looping until `n` frames have been produced.
///
/// Looping costs a seek to frame 0 and a keyframe re-decode, which lands in
/// `read` as a single large outlier. That is real work, so it is not hidden --
/// but it is an artefact of the clip being shorter than the run, so `restarts`
/// is reported and the outlier in `read.max` is attributable rather than
/// mysterious.
class VideoSource : public Source {
public:
    VideoSource(std::string path, long long n);
    bool next(cv::Mat& out) override;
    nlohmann::json describe() const override;

private:
    std::string path_;
    long long n_, produced_ = 0;
    int restarts_ = 0, frame_count_ = 0, width_ = 0, height_ = 0;
    double fps_ = 0.0;
    cv::VideoCapture cap_;
};

/// Random frames, no decode. Use to isolate what `read` costs; never as the
/// baseline for a claim about decode.
class SyntheticSource : public Source {
public:
    SyntheticSource(long long n, int height = 1080, int width = 1920, int seed = 0);
    bool next(cv::Mat& out) override;
    nlohmann::json describe() const override;

private:
    long long n_, produced_ = 0;
    int height_, width_;
    cv::Mat base_;
};

/// Empty or "synthetic" gives random frames; anything else is a video path.
std::unique_ptr<Source> make_source(const std::string& path, long long n);

}  // namespace eirl

#endif  // EIRL_SOURCES_HPP
