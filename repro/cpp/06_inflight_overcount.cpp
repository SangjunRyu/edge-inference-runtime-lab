// W5-2: a concurrency counter reports more frames in flight than there are
// buffers to hold them.
//
// SYMPTOM   The first run of `w5_scheduler` printed `peak in-flight 9` against
//           a pool of 8 buffers, and the check that the ceiling could not be
//           exceeded failed. Every other check passed: no leak, no duplicate,
//           no wrong value, every lease returned. Only the instrument was
//           wrong, and only by one.
//
// CAUSE     The counter was bracketed around the *frame* and not around the
//           *lease*. It went up when the reader took a buffer and came down
//           after postprocessing had already given the buffer back:
//
//               ctx.out.release();
//               ctx.in.release();          <-- the buffer is free from here
//               inflight.fetch_sub(1);     <-- but still counted until here
//
//           Between those two lines another thread can take the buffer that
//           was just returned and increment the counter for it. Both frames
//           are counted; only one buffer is held. With eight buffers and four
//           threads the counter can therefore read nine, and the number it
//           reports is not a number of buffers at all.
//
// FIX       Record the high-water mark inside the pool, under the pool's own
//           lock, at the instant the buffer leaves the free stack. Then the
//           count and the thing it counts change together and the reading
//           cannot exceed the capacity, because it is computed from the free
//           stack rather than tracked alongside it.
//
// WHY IT MATTERS BEYOND ITSELF
//           The bound was real; the measurement of it was not, and the failure
//           pointed at the design instead of at the instrument. That is the
//           expensive direction to be wrong in. W4's closing session found
//           three faults of the same kind -- ratios formed outside their
//           pairs, one variant always measured first, a control that was not
//           controlling anything -- and each one had been reporting confident
//           numbers for days.
//
//           The general rule this repository keeps arriving at: an observation
//           taken next to the state it describes, rather than at the same
//           instant and under the same lock, is a different observation.
//
// NEEDS NO ACCELERATOR.

#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

#include "ownership.hpp"
#include "repro.hpp"

int main(int argc, char** argv) {
    repro::Opts o = repro::parse(
        argc, argv, "06_inflight_overcount -- a peak above the pool's capacity");

    // Threads must outnumber buffers, or the defect cannot appear: the
    // over-count needs someone already waiting to take a buffer the instant it
    // is returned, in the window before the previous holder stops being
    // counted. With four threads and eight buffers nobody ever waits, the
    // counter never exceeds four, and the run reports NOT REPRODUCED -- which
    // is what the first version of this program did.
    const std::uint32_t buffers = 4;
    const int threads_n = o.n > 3 * static_cast<int>(buffers)
                              ? o.n : 3 * static_cast<int>(buffers);
    const int iterations = 200000;

    eirl::InputPool pool(buffers, 64);

    // The broken instrument: kept beside the pool rather than inside it.
    std::atomic<std::uint32_t> counter{0};
    std::atomic<std::uint32_t> peak_outside{0};

    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(threads_n));
    for (int t = 0; t < threads_n; ++t) {
        workers.emplace_back([&] {
            for (int i = 0; i < iterations; ++i) {
                eirl::InputLease lease = pool.acquire();
                if (!lease) break;

                const std::uint32_t now = counter.fetch_add(1) + 1;
                std::uint32_t seen = peak_outside.load();
                while (now > seen && !peak_outside.compare_exchange_weak(seen, now)) {
                }

                // Touch the buffer so the loop is not optimised into nothing.
                lease.data()[0] = static_cast<std::uint8_t>(i);

                lease.release();      // the buffer is available again here
                counter.fetch_sub(1); // and only stops being counted here
            }
        });
    }
    for (auto& w : workers) w.join();

    const std::uint32_t outside = peak_outside.load();
    const std::uint32_t inside = pool.peak_outstanding();

    std::printf("pool capacity          %u buffers, %d threads, %d iterations each\n",
                buffers, threads_n, iterations);
    std::printf("peak, counted outside  %u%s\n", outside,
                outside > buffers ? "   <-- more frames than buffers" : "");
    std::printf("peak, counted inside   %u\n", inside);
    std::printf("leases outstanding     %u   (acquires %llu, releases %llu)\n",
                pool.outstanding(),
                static_cast<unsigned long long>(pool.acquires()),
                static_cast<unsigned long long>(pool.releases()));

    const bool reproduced = (o.mode == "broken") ? (outside > buffers) : (inside > buffers);

    char detail[256];
    if (o.mode == "broken") {
        std::snprintf(detail, sizeof detail,
                      "the outside counter peaked at %u against %u buffers", outside, buffers);
    } else {
        std::snprintf(detail, sizeof detail,
                      "the pool's own high-water mark peaked at %u against %u buffers",
                      inside, buffers);
    }
    return repro::verdict(o, reproduced, detail);
}
