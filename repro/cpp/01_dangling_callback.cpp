// W4 phase 2, defect 1: the completion callback descriptor was a stack local.
//
// SYMPTOM   The overlapped runner worked with one request in flight and hung
//           with two or more. Nothing in the logs distinguished the two cases;
//           the second request simply never completed.
//
// CAUSE     `ov_infer_request_set_callback` takes a `const ov_callback_t*` and
//           KEEPS THE POINTER. It does not copy the struct. The descriptor was
//           a local in the loop that constructed the slots, so its address
//           dangled the moment the loop advanced. With one slot the loop body
//           never ran again and the memory happened to survive, so the bug was
//           invisible. With two, the second iteration reused that stack slot
//           and the first request's retained pointer aimed at whatever landed
//           there.
//
// FIX       The descriptor lives in the Slot, beside the request it belongs
//           to, so it outlives every inference that can reference it.
//
// WHY IT MATTERS BEYOND ITSELF
//           This is the ownership hazard the RAII wrapper exists to surface,
//           and the wrapper did not surface it: `unique_ptr` guards the
//           handles the C API hands back and says nothing about a struct the
//           C API borrows. A borrow with no owner is invisible to a type
//           system that only tracks transfers.
//
// ON REPRODUCING THIS
//           `--mode broken` is undefined behaviour, so whether it hangs,
//           crashes or silently works depends on the compiler, the optimiser
//           and the stack layout on the day. It does NOT currently reproduce
//           on this toolchain -- GCC 16.2 / OpenVINO 2025.4.1, CPU and NPU,
//           four requests -- and that is worth knowing rather than hiding: a
//           defect that appears only sometimes is exactly what made this one
//           expensive, and a broken mode that passes says the reproduction has
//           stopped exercising the condition, not that the code is safe.
//
//           `--mode probe` is the part that does not depend on luck, and it is
//           the diagnostic that actually found the bug. It proves the
//           retention contract with no undefined behaviour at all, so it
//           answers the same way every time. When `--mode broken` goes quiet,
//           the probe is what still tells you the fix is required.

#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <openvino/c/openvino.h>

#include "ov_backend.hpp"
#include "repro.hpp"

using namespace eirl;

namespace {

struct Completion {
    std::atomic<int> count{0};
};

void on_done(void* args) {
    static_cast<Completion*>(args)->count.fetch_add(1);
}

/// The original shape: the descriptor is a local of the function that sets it.
/// It is dead the moment this returns, and the runtime still holds its address.
void set_callback_from_a_local(ov_infer_request_t* req, Completion* c) {
    ov_callback_t cb{};            // <-- lives until this function returns
    cb.callback_func = on_done;
    cb.args = c;
    check(ov_infer_request_set_callback(req, &cb), "set_callback");
}                                   // <-- and the runtime still points here

/// Make the dangling read see something other than what it was given, the way
/// a second loop iteration would.
void scribble_over_that_stack_frame() {
    volatile unsigned char pad[sizeof(ov_callback_t) * 8];
    std::memset(const_cast<unsigned char*>(pad), 0xA5, sizeof(pad));
    (void)pad;
}

int run_inference_mode(const repro::Opts& o) {
    OvBackend backend(o.model, o.device);
    const size_t n_in = static_cast<size_t>(backend.height()) * backend.width() * 3;

    Completion done;
    std::vector<RequestHandle> reqs;
    std::vector<std::vector<float>> inputs;
    std::vector<ov_callback_t> owned;   // the fix: one descriptor per request

    reqs.reserve(o.n);
    inputs.reserve(o.n);
    owned.resize(o.n);

    for (int i = 0; i < o.n; ++i) {
        reqs.push_back(backend.make_request());
        inputs.emplace_back(n_in, 0.5f);

        if (o.mode == "broken") {
            set_callback_from_a_local(reqs.back().get(), &done);
            scribble_over_that_stack_frame();
        } else {
            owned[i].callback_func = on_done;
            owned[i].args = &done;
            check(ov_infer_request_set_callback(reqs.back().get(), &owned[i]),
                  "set_callback");
        }
    }

    for (int i = 0; i < o.n; ++i) {
        TensorHandle t = backend.wrap_input(inputs[i].data());
        check(ov_infer_request_set_input_tensor(reqs[i].get(), t.get()),
              "set_input_tensor");
        check(ov_infer_request_start_async(reqs[i].get()), "start_async");
    }

    repro::Deadline dl(o.timeout_ms);
    int waited_ok = 0;
    for (int i = 0; i < o.n; ++i) {
        const long left = dl.left_ms();
        if (left <= 0) break;
        if (ov_infer_request_wait_for(reqs[i].get(), left) == OK) ++waited_ok;
    }

    const int got = done.count.load();
    char detail[256];
    std::snprintf(detail, sizeof(detail),
                  "%d of %d requests completed, %d callbacks fired"
                  " (deadline %ld ms)", waited_ok, o.n, got, o.timeout_ms);
    // The defect is "the callbacks the runtime was told to make did not all
    // arrive": a hang, a crash, or a count that is quietly short.
    return repro::verdict(o, got != o.n || waited_ok != o.n, detail);
}

/// Does `ov_infer_request_set_callback` copy the struct, or keep the pointer?
///
/// Answered with no undefined behaviour, and therefore without luck: hand the
/// runtime a descriptor this program owns, then MUTATE it in place before
/// starting the inference. If the runtime copied the struct at set time, the
/// callback arrives with the first `args`. If it kept the pointer and reads it
/// at completion time, the callback arrives with the second.
int run_probe(const repro::Opts& o) {
    OvBackend backend(o.model, o.device);
    const size_t n_in = static_cast<size_t>(backend.height()) * backend.width() * 3;
    RequestHandle req = backend.make_request();
    std::vector<float> input(n_in, 0.5f);

    Completion before_mutation, after_mutation;
    ov_callback_t cb{};
    cb.callback_func = on_done;
    cb.args = &before_mutation;
    check(ov_infer_request_set_callback(req.get(), &cb), "set_callback");

    // Still ours, still alive, entirely defined behaviour: point the descriptor
    // at a different object after having told the runtime about it.
    cb.args = &after_mutation;

    TensorHandle t = backend.wrap_input(input.data());
    check(ov_infer_request_set_input_tensor(req.get(), t.get()), "set_input_tensor");
    check(ov_infer_request_start_async(req.get()), "start_async");
    ov_infer_request_wait(req.get());

    const int a = before_mutation.count.load();
    const int b = after_mutation.count.load();
    std::printf("args set BEFORE the mutation, callbacks received: %d\n", a);
    std::printf("args set AFTER  the mutation, callbacks received: %d\n", b);
    std::printf("\n%s\n", b > 0
        ? "The runtime read the descriptor at COMPLETION time, so it kept the\n"
          "pointer. Its lifetime is the caller's problem for as long as the\n"
          "request can complete -- which is the whole defect."
        : "The runtime used the values from set time, so it copied the struct.");
    std::printf("\nEither way the signature is the thing to read:\n"
                "  ov_infer_request_set_callback(ov_infer_request_t*,"
                " const ov_callback_t*)\n"
                "`const T*` is a borrow. Nothing in it promises a copy, so code\n"
                "that keeps the descriptor alive is correct under both answers\n"
                "and code that does not is correct under only one.\n");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    repro::Opts o = repro::parse(
        argc, argv, "01_dangling_callback -- a retained pointer to a dead local");
    if (o.model.empty()) {
        std::fprintf(stderr, "--model is required (models/resnet50_224.onnx)\n");
        return 2;
    }
    try {
        return o.mode == "probe" ? run_probe(o) : run_inference_mode(o);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "failed: %s\n", e.what());
        return 3;
    }
}
