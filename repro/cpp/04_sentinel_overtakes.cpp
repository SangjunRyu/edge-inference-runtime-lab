// W2, and again in the C++ port: the end-of-stream marker overtook work that
// was still in flight.
//
// SYMPTOM   A 200-frame run reported 196 frames. No error, no warning. The
//           shortfall was small, it moved between runs, and it looked exactly
//           like something being dropped for a good reason.
//
// CAUSE     When the reader reached the end of the source it closed the
//           pipeline immediately. Requests already submitted had not completed,
//           so their results arrived after the collector had already seen
//           end-of-stream and stopped. The marker travelled a path with no
//           queueing on it and beat the work down the pipe.
//
// FIX       Before announcing the end, take back every in-flight permit. The
//           slot pool is a counting semaphore of depth `inflight`; acquiring
//           all of them means every submitted request has completed, because a
//           permit is only released on completion. Then, and only then, close.
//
// WHY IT MATTERS BEYOND ITSELF
//           Shutdown is where pipelines lose data, and a frame count is a poor
//           alarm: 196 of 200 reads as "roughly right". The assertion worth
//           having is `emitted == submitted`, checked in the run itself rather
//           than by a person reading a number.
//
// NEEDS NO ACCELERATOR. Pure threading, so it reproduces anywhere -- which is
// the point, because this is the defect most likely to come back.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include "safe_queue.hpp"
#include "repro.hpp"

namespace {

/// The slot pool: `inflight` permits, one taken per submission and released on
/// completion. This is the structure the fix leans on.
class Permits {
public:
    explicit Permits(int n) : available_(n), total_(n) {}

    void acquire() {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&] { return available_ > 0; });
        --available_;
    }
    void release() {
        {
            std::lock_guard<std::mutex> lk(m_);
            ++available_;
        }
        cv_.notify_one();
    }
    /// Blocks until nothing is in flight. Every permit back means every
    /// submitted request completed.
    void drain() {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&] { return available_ == total_; });
    }

private:
    std::mutex m_;
    std::condition_variable cv_;
    int available_, total_;
};

}  // namespace

int main(int argc, char** argv) {
    repro::Opts o = repro::parse(
        argc, argv, "04_sentinel_overtakes -- end-of-stream announced while work was in flight");
    const int frames = o.n > 4 ? o.n : 200;
    const int inflight = 4;

    Permits permits(inflight);
    eirl::SafeQueue<uint64_t> output(1024);
    std::atomic<int> submitted{0};
    std::atomic<int> completed{0};

    // The collector: drains until the queue closes.
    std::atomic<int> emitted{0};
    std::thread collector([&] {
        while (true) {
            auto item = output.pop();
            if (!item) break;          // closed and drained: end of stream
            emitted.fetch_add(1);
        }
    });

    // The submitter. Each "inference" is a detached worker that takes a while
    // and then reports, which is what the runtime's completion thread does.
    std::vector<std::thread> workers;
    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> jitter(1, 6);

    for (int i = 0; i < frames; ++i) {
        permits.acquire();
        submitted.fetch_add(1);
        const int delay = jitter(rng);
        workers.emplace_back([&, i, delay] {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay));
            output.push(static_cast<uint64_t>(i));
            completed.fetch_add(1);
            permits.release();
        });
    }

    // End of source. The whole defect is in these three lines.
    if (o.mode != "broken") {
        permits.drain();     // <-- the fix: nothing is still in flight
    }
    output.close();

    collector.join();
    for (auto& w : workers) w.join();

    const int sub = submitted.load(), emi = emitted.load(), comp = completed.load();
    std::printf("submitted %d   completed %d   emitted %d\n", sub, comp, emi);
    if (emi < sub) {
        std::printf("%d frame(s) were produced and thrown away: the marker\n"
                    "reached the collector before the work did.\n", sub - emi);
    }

    char detail[256];
    std::snprintf(detail, sizeof(detail),
                  "emitted %d of %d submitted%s", emi, sub,
                  emi == sub ? " -- the assertion worth having in the runner"
                             : " -- a count that looks roughly right and is not");
    return repro::verdict(o, emi != sub, detail);
}
