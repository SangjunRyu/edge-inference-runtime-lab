// W2/W4: concurrent postprocessing emits frames out of order, and the frame
// count stays exactly right.
//
// SYMPTOM   None. That is the finding. The pipeline produced every frame, the
//           totals matched, the throughput was better, and the output was
//           wrong.
//
// CAUSE     Postprocessing runs off the completion path. With more than one
//           collector, a lighter frame finishes behind a heavier predecessor
//           and is emitted first. Nothing in a frame count can see this.
//
// FIX       A reorder buffer keyed on submission index -- DEEPX's
//           `FrameReorderBuffer`, reimplemented here as `eirl::ReorderBuffer`.
//           It holds completions until their predecessors arrive, and past a
//           cap it releases the lowest index anyway, trading strict order for
//           guaranteed progress and counting every time it does.
//
// WHY IT MATTERS BEYOND ITSELF
//           The buffer is the PRICE of postprocessing concurrently, not an
//           accident of DX-APP's design. Phase 2 of the port could not produce
//           a single out-of-order completion, and the reason was structural:
//           one collector draining a FIFO cannot overtake anything, so
//           delaying frame 20 delays frame 21 with it. The defect needs
//           concurrency to exist: this program runs four collectors, and with
//           one the violation count is structurally zero.
//
// NEEDS NO ACCELERATOR.

#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include "reorder.hpp"
#include "safe_queue.hpp"
#include "repro.hpp"

int main(int argc, char** argv) {
    repro::Opts o = repro::parse(
        argc, argv, "05_reorder_off -- right count, wrong order");
    const int frames = o.n > 8 ? o.n : 400;
    const int workers_n = 4;          // one collector cannot reorder at all

    eirl::SafeQueue<uint64_t> to_post(4);   // DX-APP uses 4
    eirl::OrderChecker checker;
    eirl::ReorderBuffer<uint64_t> reorder(4 * (workers_n + 4));
    std::mutex emit_mu;

    const bool use_reorder = (o.mode != "broken");

    // Postprocess workers. Each takes a variable time, which is what lets a
    // successor overtake its predecessor.
    std::vector<std::thread> workers;
    std::atomic<int> done{0};
    for (int w = 0; w < workers_n; ++w) {
        workers.emplace_back([&, w] {
            std::mt19937 rng(9000 + w);
            std::uniform_int_distribution<int> cost(0, 400);
            while (true) {
                auto item = to_post.pop();
                if (!item) break;
                std::this_thread::sleep_for(std::chrono::microseconds(cost(rng)));
                done.fetch_add(1);

                std::lock_guard<std::mutex> lk(emit_mu);
                if (use_reorder) {
                    for (auto& kv : reorder.push(*item, *item)) {
                        checker.observe(kv.first);
                    }
                } else {
                    checker.observe(*item);   // straight out, as it finished
                }
            }
        });
    }

    for (int i = 0; i < frames; ++i) to_post.push(static_cast<uint64_t>(i));
    to_post.close();
    for (auto& w : workers) w.join();
    if (use_reorder) {
        for (auto& kv : reorder.drain()) checker.observe(kv.first);
    }

    const auto cs = checker.stats();
    const auto rs = reorder.stats();
    std::printf("%d frames, %d postprocess workers, reorder %s\n",
                frames, workers_n, use_reorder ? "on" : "off");
    std::printf("  emitted                %llu\n",
                (unsigned long long)checker.emitted());
    std::printf("  order violations       %llu\n",
                (unsigned long long)checker.violations());
    std::printf("  out-of-order arrivals  %s\n",
                rs["out_of_order_arrivals"].dump().c_str());
    std::printf("  max displacement       %s\n",
                rs["max_displacement"].dump().c_str());
    std::printf("  forced flushes         %s\n",
                rs["forced_flushes"].dump().c_str());

    const bool count_right = (checker.emitted() == (uint64_t)frames);
    const bool order_wrong = checker.violations() > 0;
    char detail[256];
    std::snprintf(detail, sizeof(detail),
                  "count %s (%llu of %d), order %s (%llu violations)",
                  count_right ? "correct" : "WRONG",
                  (unsigned long long)checker.emitted(), frames,
                  order_wrong ? "WRONG" : "correct",
                  (unsigned long long)checker.violations());
    if (order_wrong && count_right) {
        std::printf("\nEvery frame is present and the total is exact. Only the\n"
                    "sequence is wrong, and no count can see that.\n");
    }
    (void)cs;
    return repro::verdict(o, order_wrong || !count_right, detail);
}
