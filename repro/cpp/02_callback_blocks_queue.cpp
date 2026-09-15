// W4 phase 2, defect 2: the completion callback blocked on a bounded queue.
//
// SYMPTOM   The overlapped runner stopped dead once more frames were in flight
//           than the output queue could hold. No error, no CPU load, no
//           progress.
//
// CAUSE     The completion callback ran on an OpenVINO worker thread and
//           pushed its result into a bounded SafeQueue. When that queue filled,
//           the callback blocked -- inside the runtime. The collector that
//           would have drained the queue was itself parked in
//           `ov_infer_request_wait`, waiting for a request whose completion
//           path was the blocked callback. Each side held what the other
//           needed.
//
//           The bound was not an accident either: bounded queues are the
//           backpressure mechanism the whole pipeline is built on. The mistake
//           was applying backpressure at a point where the thread being pushed
//           back on belongs to the runtime rather than to the pipeline.
//
// FIX       The output queue is unbounded and the callback does the least
//           possible: stamp the time, hand over the slot, return. Backpressure
//           is not lost -- it lives in the slot pool, where blocking is the
//           submitting thread's own problem and the runtime's threads are
//           never held.
//
// WHY IT MATTERS BEYOND ITSELF
//           "Never block in a callback you do not own the thread of" is the
//           rule, and DX-APP's design obeys it: it postprocesses off the
//           completion path precisely so the completion path stays short. The
//           cost of that choice is the reorder buffer (see 05).

#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include <openvino/c/openvino.h>

#include "ov_backend.hpp"
#include "safe_queue.hpp"
#include "repro.hpp"

using namespace eirl;

namespace {

struct Ctx {
    SafeQueue<int>* bounded = nullptr;   // the defect: capacity 1
    std::vector<int>* unbounded = nullptr;
    std::mutex* mu = nullptr;
    std::atomic<int> entered{0};
    std::atomic<int> returned{0};
};

struct Slot {
    RequestHandle req;
    ov_callback_t cb{};
    Ctx* ctx = nullptr;
    int index = 0;
};

// One callback for both modes, so the only difference measured is where the
// result goes -- into a queue that can block the caller, or into one that
// cannot.
void on_done(void* args) {
    auto* s = static_cast<Slot*>(args);
    s->ctx->entered.fetch_add(1);
    if (s->ctx->bounded) {
        s->ctx->bounded->push(s->index);      // blocks once full: on OV's thread
    } else {
        std::lock_guard<std::mutex> lk(*s->ctx->mu);
        s->ctx->unbounded->push_back(s->index);
    }
    s->ctx->returned.fetch_add(1);
}

}  // namespace

int main(int argc, char** argv) {
    repro::Opts o = repro::parse(
        argc, argv, "02_callback_blocks_queue -- backpressure applied to the runtime's thread");
    if (o.model.empty()) {
        std::fprintf(stderr, "--model is required (models/resnet50_224.onnx)\n");
        return 2;
    }
    if (o.n < 2) o.n = 4;

    try {
        OvBackend backend(o.model, o.device);
        const size_t n_in = static_cast<size_t>(backend.height()) * backend.width() * 3;

        // Capacity one, and no consumer drains it until every request has been
        // waited on -- which is the shape the runner had, with the collector
        // parked in wait.
        SafeQueue<int> bounded(1);
        std::vector<int> unbounded;
        std::mutex mu;

        Ctx ctx;
        ctx.mu = &mu;
        if (o.mode == "broken") ctx.bounded = &bounded;
        else ctx.unbounded = &unbounded;

        std::vector<Slot> slots(o.n);
        std::vector<std::vector<float>> inputs;
        inputs.reserve(o.n);

        for (int i = 0; i < o.n; ++i) {
            slots[i].req = backend.make_request();
            slots[i].ctx = &ctx;
            slots[i].index = i;
            slots[i].cb.callback_func = on_done;
            slots[i].cb.args = &slots[i];
            check(ov_infer_request_set_callback(slots[i].req.get(), &slots[i].cb),
                  "set_callback");
            inputs.emplace_back(n_in, 0.5f);
        }

        std::printf("%d requests, output queue %s\n", o.n,
                    o.mode == "broken" ? "bounded at 1 (the defect)"
                                       : "unbounded (the fix)");

        for (int i = 0; i < o.n; ++i) {
            TensorHandle t = backend.wrap_input(inputs[i].data());
            check(ov_infer_request_set_input_tensor(slots[i].req.get(), t.get()),
                  "set_input_tensor");
            check(ov_infer_request_start_async(slots[i].req.get()), "start_async");
        }

        // The collector, parked in wait exactly as the runner's was. It cannot
        // drain the queue while it is here, and in broken mode the callback it
        // is waiting for cannot finish until the queue drains.
        repro::Deadline dl(o.timeout_ms);
        int completed = 0;
        for (int i = 0; i < o.n; ++i) {
            const long left = dl.left_ms();
            if (left <= 0) break;
            if (ov_infer_request_wait_for(slots[i].req.get(), left) == OK) ++completed;
        }

        const int entered = ctx.entered.load(), returned = ctx.returned.load();
        char detail[256];
        std::snprintf(detail, sizeof(detail),
                      "%d of %d waits returned; %d callbacks entered, %d returned"
                      " -- %d parked inside the runtime",
                      completed, o.n, entered, returned, entered - returned);

        // Release anything still blocked so the process can exit cleanly; the
        // measurement above has already been taken.
        bounded.close();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        return repro::verdict(o, completed != o.n || entered != returned, detail);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "failed: %s\n", e.what());
        return 3;
    }
}
