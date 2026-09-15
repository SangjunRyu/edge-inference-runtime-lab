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

## Still to do

- **Throughput on a machine that holds still.** Every pair drifted; the numbers
  above are stage ratios, which survive drift far better than a rate does.
- **Phase 2**: the async pipeline in C++, and with it the GIL measurement W2
  could not separate from ONNX Runtime's thread-pool subdivision.
- **Phase 3**: the fused preprocess against the 1.00x baseline above, the input
  tensor over a reused buffer (`ov_tensor_create_from_host_ptr` is already in
  the backend for it), and `verify_equivalence` run against the C++ build from
  the first version that produces predictions -- buffer reuse is exactly the
  optimisation that would introduce the bug only that check detects.
