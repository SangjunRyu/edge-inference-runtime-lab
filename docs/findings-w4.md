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

## Phase 3 -- tuning, and what the measurements say

Measured at 1080p, where host work is comparable to inference and a host-side
win therefore raises the ceiling. Paired: the two configurations alternate in
one session and the ratio inside each pair is reported
(`bench/compare_variants.py`).

| candidate | stage it targets | B/A | per-pair CV | verdict |
|---|---|---|---|---|
| fused preprocess | `preprocess` | **0.61** | 45% | real, magnitude imprecise |
| input tensor reuse | `inference` | 0.98 | 25% | **not supported** |

**The fused preprocess is faster.** One pass over the resized image instead of
five -- colour swap, scale, normalise and transpose folded into the loop that
writes the NCHW planes. Two independent measurements put it at 0.61x and 0.40x
of the baseline, so the direction is not in doubt even though the machine was
too unsteady to pin the size.

**Reusing the input tensor is not.** Binding one `ov_tensor` over the caller's
buffer at startup, instead of wrapping the same buffer again every frame, moves
`inference` by 2% against a 25% run-to-run spread. That is not a result, it is
a number smaller than the noise around it. The wrap is a shape allocation and a
pointer -- cheap beside a 7 ms inference -- so this is what the hypothesis
deserved rather than a disappointment. The repository said measuring a
hypothesis dead is a result; here is one.

### The measurement that had to be thrown away first

The four variants were run back to back before any of this. That version said
`--reuse-tensor` made *preprocessing* 2.3x faster, which it cannot -- it does
not touch preprocessing -- and showed `inference` swinging 1.9x between
variants that none of the flags affect. It was drift, read as a result.

`compare_variants` therefore alternates the configurations and adds a
**control**: the stage neither side changes. `--fused` must leave `inference`
alone and `--reuse-tensor` must leave `read` alone; if the control moves more
than 10% the pairing has not cancelled the drift and the other rows are
describing the machine. Both runs passed, but `--fused` passed at 1.09 -- close
enough to the bar to be worth saying out loud rather than reporting as clean.

### Every variant still produces the same answers

`verify_equivalence` now accepts the C++ runners' `--dump-topk` output, so the
C++ build is checked by the same comparison, the same self-test and the same
sensitivity figure as the Python. Against the Python sequential reference on
300 frames:

| build | top-1 | top-k order | ties | max prob delta |
|---|---|---|---|---|
| cpp sync | 0 | 0 | 3 | 5.96e-08 |
| cpp + reuse-tensor | 0 | 0 | 3 | 5.96e-08 |
| cpp + fused | 0 | 0 | 3 | 5.96e-08 |
| cpp async, 8 in flight, 4 collectors | 0 | 0 | 3 | 5.96e-08 |

This is the net that had to exist before phase 3 touched a buffer, and it is
why the fused path can be trusted: it rewrote the arithmetic and the answers
did not move.

**A criterion was corrected on the way.** The first run reported all four as
DIFFERS over three frames' top-5 *ordering*, at a maximum probability
difference of 5.96e-08 -- float32 epsilon. Two classes with equal probabilities
are a tie, and which one `std::partial_sort` puts first need not match NumPy's
argsort; neither has a claim to being right about a 6e-08 gap. Ties are now
counted separately from mismatches, the same principle `compare_backends`
already applies to the accelerators.

## Phase 2 addendum: making the reorder path real

W2 recorded that completions arrived in submission order, so the reorder buffer
was correct but never exercised. Injecting a stall did not help: with one
collector draining its queue FIFO, delaying frame 20 delays frame 21 as well
and nothing can overtake **by construction**.

The number of collectors is now a flag, and it is the whole story:

| post-workers | reorder | order violations | out-of-order arrivals |
|---|---|---|---|
| 1 | on | 0 | **0** |
| 4 | on | **0** | 96 |
| 4 | off | **72 / 479** | -- |

One collector cannot reorder and needs no buffer. Four collectors postprocess
concurrently, 96 completions overtake a predecessor, and without the buffer 72
frames of 479 are emitted out of order with the count perfectly correct.

**That is the trade DX-APP makes.** It postprocesses off the completion path,
which is why `FrameReorderBuffer` exists: the buffer is the price of
postprocessing concurrently, not an accident of their design.

## Still to do

- **Everything above, on a machine that holds still.** Every pair in every
  phase-3 run drifted out of the 0.95-1.05 band after hours of measurement.
  Stage ratios survive that far better than rates do, which is why they are
  what gets quoted, but the fused preprocess deserves a tight number and does
  not have one.
- **The paired C++/Python async comparison**, which is the GIL measurement W2
  could not separate from ONNX Runtime's thread-pool subdivision.
- **The fused path in the async runner.** `--fused` is wired through both, but
  only the sequential one has been measured with it.
