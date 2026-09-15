# W4 — the C++ port

The question W4 answers:

> Can the same pipeline be expressed with explicit concurrency and memory
> ownership, and does having that control reveal improvements Python could not
> reach?

The point is not that C++ is faster. Three things make the port worth the time,
in this order:

1. **A second implementation cross-validates the measurements.** If C++ and
   Python agree on stage shares from the same clip on the same device, both are
   validated. If they disagree, that is a finding about the instrumentation and
   is worth more than either number alone.
2. **Tuning becomes possible.** The optimisations the stage table points at —
   fusing the five preprocessing passes into one, constructing the input tensor
   over a reused buffer — cannot be expressed in the Python version at all.
3. **It measures the GIL.** W2 attributed the CPU async loss to ONNX Runtime
   subdividing its thread pool, but Python's orchestration overhead was never
   separated out. The same async pipeline in C++ on the same device separates
   them.

## Phase 0 — toolchain, and nothing else

This is where the time actually goes, so it gets its own phase and its own
success criterion.

**Success criterion:** a program that opens `models/resnet50_224.onnx`, prints
the input and output names and shapes, and exits. No pipeline, no timing.

Take the lightest path. Both dependencies ship prebuilt for Windows, and
building either from source on an 8 GB laptop is a day lost:

| | what to get | why not the alternative |
|---|---|---|
| ONNX Runtime | the official release **zip** (headers + `onnxruntime.lib` + `.dll`) | building from source needs hours and a large toolchain |
| OpenCV | the prebuilt Windows package (`opencv_world4xx.lib`) | same |
| Build | CMake + **MSVC Build Tools** | full Visual Studio is not needed |
| Packages | none | vcpkg compiles from source; slow here, and not needed for two prebuilt libraries |

The ORT zip must match the provider: the plain CPU zip for the CPU rows, the
DirectML zip if the MX150 is to be reached over DML.

## Phase 1 — sequential pipeline, identical output schema

Port in the order the Python grew, which is also dependency order:

| Python | C++ | what it holds |
|---|---|---|
| `harness.py` | `harness.hpp/.cpp` | `Timer`, `Recorder`: stage stats, p50/p95/max, `unaccounted`, overlap ceiling |
| `pipeline.py` | `pipeline.cpp` | preprocess, postprocess |
| `sources.py` | `sources.cpp` | `cv::VideoCapture`, restart counting |
| `backends.py` | `backend.cpp` | one `Ort::Session`, provider selection, fell-back detection |
| `run_sync.py` | `run_sync.cpp` | the loop |

**The single highest-leverage decision: emit the same JSON schema.** Then
`bench/summarize.py` reads C++ results unchanged, the two implementations land
in one table, and the comparison costs nothing. Deviating here means writing a
second analysis path and never quite trusting the comparison.

**Deliverable:** C++ sync against Python sync, same clip, same device, paired.
Stage shares should agree. A gap is a finding, not a defeat.

## Phase 2 — overlapped pipeline

| Python | C++ | note |
|---|---|---|
| `queue.Queue(maxsize=n)` | `SafeQueue<T>` | `std::mutex` + `std::condition_variable`, bounded, sentinel |
| `reorder.py` | `reorder.hpp` | `std::map` keyed on submit index, `cap` force-flush, `OrderChecker` |
| `session.run_async(..., callback)` | `Ort::Session::RunAsync` | same submit/collect split |
| `threading.Semaphore(inflight)` | `std::counting_semaphore` | the backpressure knob |

The four service stages and four wait stages stay exactly as they are. Order
verification stays on; a C++ pipeline that reorders its output is no more
acceptable than a Python one.

**Deliverable:** the A/B and depth-sweep tables regenerated from C++, plus the
Python-vs-C++ async comparison on one device, which is the GIL measurement.

## Phase 3 — tuning, aimed by the measurements already taken

The stage table says where the host time is. On the NPU at 1080p: `read` 4.8 ms,
`preprocess` 0.7 ms, `inference` 5.3 ms, `postprocess` 0.1 ms. Host work is
roughly equal to inference there, so host-side wins raise the ceiling directly.
At 360p host work is ~1 ms against 3 ms of inference and there is much less to
gain — **tune against the 1080p case, and check the 360p case does not regress.**

Candidates, most likely to pay first:

1. **Fuse preprocessing into one pass.** Python does resize → BGR2RGB → `/255`
   → normalise → transpose as five separate passes over the image. One loop
   writing straight into the NCHW tensor buffer does it in one. This is the
   largest structural difference between the two implementations.
2. **Construct the input tensor over a reused buffer.** `Ort::Value::CreateTensor`
   over memory the caller owns, reused every frame, instead of allocating and
   letting the runtime copy.
3. **Pre-allocate the output / use IoBinding**, so the runtime stops allocating
   an output tensor per call.
4. **Buffer reuse across the pipeline.** The hypothesis raised before W1 and
   never tested, because Python could not test it.
5. **Thread affinity.** Worth doing on a hybrid part; the 255H's P/E cores were
   the cause of W1 finding 4. The 550XBE's cores are uniform, so this one
   largely goes away with the loaner machine.

**Expect some of these to be rejected.** If buffer reuse comes out under 1% it
is written down as under 1%. The repository already says that measuring a
hypothesis dead is a result; that has to survive contact with an optimisation
someone wants to work.

## The protocol does not change

Everything the Python side established applies unchanged, and the C++ numbers
are worthless without it:

- warm-up discarded, `--duration-s` rather than a frame count
- p50/p95/max, never a bare mean; `unaccounted` published
- every optimisation is a **paired** A/B against the previous build in one
  session — not two numbers from different days
- **a change smaller than the run-to-run CV is not a change**
- ratios, not absolutes
- order verified on every run

## Risks

- **Toolchain, and it is the big one.** Phase 0 exists to fail fast. If it is
  not done in a day, the answer is prebuilt binaries rather than more effort.
- **The 550XBE is 4C/8T and 8 GB.** Threading experiments have less headroom
  than the 255H had, and depth sweeps will saturate earlier.
- **No NPU.** Phase 2's headline needs an off-host device; on that machine that
  means DirectML on the MX150, which is a different engine as well as a
  different device — a `dml` row and a `cuda` row are not interchangeable
  evidence, exactly as ORT-CPU and OpenVINO-CPU were not.
