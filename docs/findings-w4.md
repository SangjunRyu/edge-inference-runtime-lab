# W4 findings -- the C++ port

Phases and dependencies: [`w4-plan.md`](w4-plan.md), [`w4-setup.md`](w4-setup.md).

## Phase 0 -- the toolchain, cross-checked

Built with MSYS2 UCRT64 GCC 16.2, no Microsoft software involved, against
OpenVINO's **C** API because MinGW and MSVC share no C++ ABI. The probe loads
the model, enumerates devices and runs one inference:

```
OpenVINO 2025.4.1-20426-82bbf0292c5
devices: CPU GPU NPU
input    [1, 3, 224, 224]  ->  1000 logits
top-5    754:0.1191 905:0.0660 753:0.0574 904:0.0365 921:0.0264
```

The same synthetic input through the Python OpenVINO API and
`bench.pipeline.postprocess` gives those five values exactly. **The C API path
and the C++ softmax agreed with Python before a single stage was timed.** The
GPU returns 754:0.1202 against the NPU's 0.1191, the same 4-6e-3 spread
between accelerators that W3 measured.

## Phase 1 -- the sequential pipeline

NPU, 360p, 10 s per run, paired and alternating in one session
(`bench/compare_impl.py`):

| stage | cpp ms | python ms | cpp/python |
|---|---|---|---|
| read | 0.511 | 0.676 | 0.76 |
| preprocess | 1.095 | 1.093 | **1.00** |
| inference | 6.980 | 7.088 | **0.98** |
| postprocess | 0.016 | 0.148 | **0.11** |

**The cross-validation is the `inference` row.** It is the same model on the
same silicon and neither implementation performs it, so the two must agree
there or one of them is mistiming and nothing else means anything. They agree
to 2%.

The rest is what the port was expected to show and roughly the size expected:

* **`postprocess` 0.11x** -- a softmax and a top-5 over 1000 floats. NumPy
  allocates several temporaries to do it; the C++ does one pass and a partial
  sort. Nine times faster, and 0.13 ms saved, which is the honest way to state
  it: large as a ratio, small as a number.
* **`read` 0.76x** -- the same OpenCV `VideoCapture` either way. The difference
  is the Python call overhead around it.
* **`preprocess` 1.00x** -- identical, because both call the same OpenCV
  primitives in the same order. That is the baseline phase 3's fused path has
  to beat, and it is a fair one.

Throughput is **not** reported. Three of four pairs drifted outside 0.95-1.05x
and were excluded, leaving one usable pair, and one pair has no spread: its CV
is 0.0% and every steadiness test passes trivially. The tool says "an
observation, not a measurement" rather than printing 1.05x as though it were
supported.

### A check that was wrong, and what replaced it

The first version cross-validated on stage **shares** and flagged a 6.8
percentage-point difference as an instrumentation fault. That was backwards.
Shares are normalised by E2E, so an implementation that genuinely speeds up the
host stages necessarily shifts every share -- the check would report any real
improvement as a defect.

The criterion is now the absolute time of the stage neither implementation
performs. Shares are still printed, because they are informative, but a
difference in them is not evidence of anything.

## Phase 2 -- the overlapped pipeline

NPU, 360p, 10 s, in-flight 4:

```
 preprocess     1.22 ms        q_read      0.14 ms
 inference     21.28 ms        q_inflight  5.29 ms   (18.5%)
 postprocess    0.04 ms        q_output    0.05 ms
                               q_reorder   0.04 ms
 E2E           28.64 ms        throughput  169.72 FPS
 resident       4.86 frames (Little's Law, in-flight 4 plus queues)
 order         emitted 1789  violations 0  IN SUBMISSION ORDER
```

Against the C++ sequential run on the same device and clip (~142 FPS), overlap
buys throughput and costs latency, the same shape W2 measured in Python.
`inference` rises from ~7 ms to 21 ms because four requests share one NPU --
the per-request cost of concurrency, visible here as it was there.

### Completions arrive in order, so the reorder buffer never fires

`out_of_order_arrivals 0, max_displacement 0`. The Python pipeline saw 152 of
200 arrive out of order on the same clip.

The structures differ. In C++ the completion callback does nothing but stamp
the time and hand the slot to one postprocess worker, which drains its queue in
FIFO order; OpenVINO completes the requests in submission order, so nothing
ever overtakes. In Python the postprocessing ran off ONNX Runtime's completion
path and genuinely raced.

**So the reorder machinery here is correct but untested by this workload.** The
Python side has `--stall-every` to inject an asymmetry and force the case; the
C++ side does not yet, and until it does, "0 violations" says less than the same
words said in W2.

### Three bugs, and the one that matters

- **The callback descriptor was a stack local.**
  `ov_infer_request_set_callback` takes a `const ov_callback_t*` and keeps the
  pointer, so passing the address of a loop-local left it dangling the moment
  the constructor advanced. With one slot it worked by accident; with two or
  more the pipeline hung. Fixed by giving each slot its own descriptor. This is
  exactly the ownership hazard the RAII wrapper exists to surface, and it was
  found by a hang rather than by the wrapper -- worth remembering when phase 3
  starts reusing buffers.
- **The completion callback blocked on a bounded queue.** It runs on an
  OpenVINO thread; blocking there deadlocks the runtime against its own
  `wait`. The output queue is unbounded now and the callback does the least
  possible. Backpressure is not lost -- it lives in the slot pool.
- **`ov_infer_request_get_output_tensor` returns `REQUEST_BUSY` from the
  callback path.** The callback runs before the request leaves its running
  state, so the collector calls `ov_infer_request_wait` first -- which returns
  immediately, the work being done. The API makes the submit/collect split
  explicit whether or not the structure does.

A fourth, of my own diagnosis rather than the code: the first attempts to
narrow the hang reported failures that were actually exit 127, the executable
not finding the OpenCV DLLs. The Windows loader names only the first missing
library, so an OpenCV problem reads as an OpenVINO one. Two rounds of reasoning
went into a hang that had not happened.

## Still to do

- **Throughput on a machine that holds still.** Every pair drifted; the numbers
  above are stage ratios, which survive drift far better than a rate does.
- **Port `--stall-every`** so the C++ reorder path is exercised rather than
  merely present.
- **The paired C++/Python async comparison**, which is the GIL measurement W2
  could not separate from ONNX Runtime's thread-pool subdivision.
- **Phase 3**: the fused preprocess against the 1.00x baseline above, the input
  tensor over a reused buffer (`ov_tensor_create_from_host_ptr` is already in
  the backend for it), and `verify_equivalence` run against the C++ build from
  the first version that produces predictions -- buffer reuse is exactly the
  optimisation that would introduce the bug only that check detects.
