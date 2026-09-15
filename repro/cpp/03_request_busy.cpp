// W4 phase 2, defect 3: reading the output of a completed request returns
// REQUEST_BUSY.
//
// SYMPTOM   `ov_infer_request_get_output_tensor` returned -8 from the
//           collector. The same call in the sequential runner had worked since
//           phase 1, and the request in question had definitely finished --
//           its completion callback is what handed the slot to the collector.
//
// CAUSE     "The callback fired" and "the request has left its running state"
//           are different facts. The callback fires as the work completes; the
//           request is still formally running, and the C API refuses accessors
//           on a running request.
//
// FIX       Call `ov_infer_request_wait` in the COLLECTOR before reading. It
//           returns immediately -- the work is done -- and what it changes is
//           the request's state.
//
//           Not in the callback. Calling `wait` from inside the completion
//           callback re-enters the runtime and deadlocks; that was tried while
//           building this reproduction and it hangs with no diagnostic at all.
//           The wait belongs on the collector's thread, which is another way
//           of saying the submit/collect split is not optional.
//
// WHY IT MATTERS BEYOND ITSELF
//           The API is right to separate the two facts, and the separation is
//           the same one the whole pipeline rests on. It is also why DX-APP
//           has a collect step rather than doing the work in the completion
//           handler -- and why 05's reorder buffer has to exist.
//
//           This defect fails loudly with a status code instead of hanging,
//           which makes it the cheapest of the three to keep running.
//
// STRUCTURE The shape matters, so it is the runner's: the callback hands the
//           slot over and a separate collector thread reads the output.
//           `--collect callback` moves the read into the completion callback
//           instead, because a reproduction that changes the structure is
//           testing a different program and the difference should be a flag
//           rather than an edit.
//
// IT DOES NOT CURRENTLY REPRODUCE, and that is reported rather than hidden.
//           Six configurations were tried on this build -- {CPU, NPU} x
//           {collector thread, inside the callback} x {4, 16 requests}, on
//           OpenVINO 2025.4.1 -- and every one of them read the output cleanly
//           with no `wait`. The -8 was definitely observed during phase 2, so
//           either the condition is narrower than any of those six, or it
//           depended on the state the dangling-descriptor bug (see 01) left
//           the request in, which was still present at the time. That second
//           possibility is untested and is the honest first guess.
//
//           The `wait` stays in the runner either way. The API says accessors
//           on a running request are invalid; code that relies on an
//           undocumented grace period is not code worth keeping, and `wait`
//           costs nothing here because the work is already done. What this
//           program demonstrates today is the shape of the collect half, and
//           what it records is that the failure mode it was written for does
//           not currently fire.

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <openvino/c/openvino.h>

#include "ov_backend.hpp"
#include "repro.hpp"

using namespace eirl;

namespace {

struct Shared {
    std::mutex m;
    std::condition_variable cv;
    std::deque<int> ready;        // slots handed over by the callback
    bool closed = false;
    std::atomic<int> handled{0};
    std::atomic<int> busy{0};
    std::atomic<int> ok{0};
    std::atomic<int> last_status{0};
};

struct Slot {
    RequestHandle req;
    ov_callback_t cb{};
    Shared* sh = nullptr;
    int index = 0;
    bool read_here = false;   // read the output inside the callback itself
};

/// The read, wherever it happens. `wait_first` is the fix.
void collect(Slot* s, bool wait_first);

/// Exactly what the runner's callback does: stamp, hand over, return. No
/// blocking, no accessors, nothing that can re-enter the runtime.
void collect(Slot* s, bool wait_first) {
    if (wait_first) {
        // Returns immediately -- the work is done -- and moves the request out
        // of its running state so the accessor below is legal.
        ov_infer_request_wait(s->req.get());
    }
    ov_tensor_t* raw = nullptr;
    const ov_status_e st = ov_infer_request_get_output_tensor(s->req.get(), &raw);
    s->sh->last_status.store(static_cast<int>(st));
    if (st == OK) {
        s->sh->ok.fetch_add(1);
        ov_tensor_free(raw);
    } else {
        s->sh->busy.fetch_add(1);
    }
    s->sh->handled.fetch_add(1);
}

void on_done(void* args) {
    auto* s = static_cast<Slot*>(args);
    if (s->read_here) {
        // NEVER call `wait` from here: it re-enters the runtime and hangs with
        // no diagnostic. That is why the fix lives on the collector's thread.
        collect(s, false);
    }
    {
        std::lock_guard<std::mutex> lk(s->sh->m);
        s->sh->ready.push_back(s->index);
    }
    s->sh->cv.notify_one();
}

}  // namespace

int main(int argc, char** argv) {
    repro::Opts o = repro::parse(
        argc, argv,
        "03_request_busy -- the callback fires before the request leaves running state");
    if (o.model.empty()) {
        std::fprintf(stderr, "--model is required (models/resnet50_224.onnx)\n");
        return 2;
    }

    try {
        OvBackend backend(o.model, o.device);
        const size_t n_in = static_cast<size_t>(backend.height()) * backend.width() * 3;
        const bool wait_first = (o.mode != "broken");

        Shared sh;
        std::vector<Slot> slots(o.n);
        std::vector<std::vector<float>> inputs;
        inputs.reserve(o.n);

        for (int i = 0; i < o.n; ++i) {
            slots[i].req = backend.make_request();
            slots[i].sh = &sh;
            slots[i].index = i;
            slots[i].read_here = (o.collect == "callback");
            slots[i].cb.callback_func = on_done;
            slots[i].cb.args = &slots[i];
            check(ov_infer_request_set_callback(slots[i].req.get(), &slots[i].cb),
                  "set_callback");
            inputs.emplace_back(n_in, 0.5f);
        }

        // The collector, on its own thread, as in the runner.
        std::thread collector([&] {
            while (true) {
                int idx;
                {
                    std::unique_lock<std::mutex> lk(sh.m);
                    sh.cv.wait(lk, [&] { return !sh.ready.empty() || sh.closed; });
                    if (sh.ready.empty()) return;
                    idx = sh.ready.front();
                    sh.ready.pop_front();
                }
                if (!slots[idx].read_here) collect(&slots[idx], wait_first);
                sh.cv.notify_all();
            }
        });

        std::printf("%d requests on %s; the collector %s before reading the output\n",
                    o.n, o.device.c_str(),
                    wait_first ? "calls ov_infer_request_wait"
                               : "reads straight away (the defect)");

        for (int i = 0; i < o.n; ++i) {
            TensorHandle t = backend.wrap_input(inputs[i].data());
            check(ov_infer_request_set_input_tensor(slots[i].req.get(), t.get()),
                  "set_input_tensor");
            check(ov_infer_request_start_async(slots[i].req.get()), "start_async");
        }

        {
            std::unique_lock<std::mutex> lk(sh.m);
            sh.cv.wait_for(lk, std::chrono::milliseconds(o.timeout_ms),
                           [&] { return sh.handled.load() >= o.n; });
            sh.closed = true;
        }
        sh.cv.notify_all();
        collector.join();

        // Everything must be settled before the handles are destroyed.
        for (int i = 0; i < o.n; ++i) ov_infer_request_wait(slots[i].req.get());

        char detail[256];
        std::snprintf(detail, sizeof(detail),
                      "%d outputs read, %d refused with REQUEST_BUSY;"
                      " last status %d",
                      sh.ok.load(), sh.busy.load(), sh.last_status.load());
        return repro::verdict(o, sh.busy.load() > 0, detail);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "failed: %s\n", e.what());
        return 3;
    }
}
