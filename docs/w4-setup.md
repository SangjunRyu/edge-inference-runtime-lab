# W4 setup — dependencies and order of work

## Do the port on the gram, not after it

The plan said the C++ work follows the machine change. That is backwards. Code
does not need the target machine to be written, and the gram is the only
machine that will ever have the NPU: porting now means the C++ pipeline can be
**cross-validated against the Python numbers on the device the whole W2 result
rests on**. After handover that check is impossible forever.

The 550XBE is where tuning *continues*, not where the port starts.

## What is already here

The `openvino` pip wheel ships the complete C++ SDK, not just the Python
bindings. Confirmed in this repository's own venv:

```
.venv/Lib/site-packages/openvino/
  include/openvino/openvino.hpp        470 headers
  libs/openvino.lib                     15 import libraries
  cmake/OpenVINOConfig.cmake            find_package(OpenVINO) works
  cmake/TBBConfig.cmake                 TBB comes with it
```

So `find_package(OpenVINO)` against
`.venv/Lib/site-packages/openvino/cmake` links C++ straight to the runtime
already installed and already version-matched to the Python results. Nothing to
download, and no chance of the C++ and Python sides running different OpenVINO
builds.

## What has to be installed

| # | what | size | needed by | note |
|---|---|---|---|---|
| 1 | **MSVC Build Tools**, "Desktop development with C++" | ~3-6 GB | phase 0 | the actual gate; `cl` is not on this machine |
| 2 | **CMake** | ~50 MB | phase 0 | not on this machine either |
| 3 | **OpenCV 5.0.0 for Windows** (`opencv-5.0.0-windows.exe`) | 186 MB | phase 1 | self-extracting, prebuilt `opencv_world500.lib` |
| 4 | `nlohmann/json.hpp` | 900 KB | phase 1 | single header, drop in `cpp/third_party/` |

Full Visual Studio is not needed — Build Tools is the compiler without the IDE.
`opencv-python` does **not** provide C++ headers, so item 3 is a real download
even though Python already has OpenCV.

Nothing here is built from source. That is deliberate: building ONNX Runtime or
OpenCV would cost a day and buy nothing.

### Later, on the 550XBE only

| what | why |
|---|---|
| ONNX Runtime `onnxruntime-win-x64-1.24.4.zip` (71 MB) | C++ headers + libs for the CPU provider |
| `Microsoft.ML.OnnxRuntime.DirectML` NuGet 1.24.4 | the only prebuilt C++ path to the MX150 |

## Which engine, and the honesty cost

OpenVINO reaches Intel CPU, the Arc iGPU and the NPU — the gram's three
backends — and it is already installed. ONNX Runtime reaches the CPU from its
release zip and the MX150 through DirectML. Reaching the NPU *through ONNX
Runtime* in C++ would need ORT built from source with the OpenVINO EP, which is
exactly the day this setup is arranged to avoid.

So: **OpenVINO C++ now, ONNX Runtime C++ added later**, behind one backend
interface — the same shape `bench/backends.py` already has.

The cost has to be stated. The Python GPU and NPU rows ran OpenVINO *as an ONNX
Runtime execution provider*; native OpenVINO is a different code path, so C++
against Python on those devices is not perfectly like-for-like. That is a
measurable difference rather than a fatal one: `compare_backends` already shows
OpenVINO-CPU and ORT-CPU agreeing to 2.6e-06, and the same tool can be pointed
at the C++ build. Measure it; do not assume it away.

## Order of work

Phases 0-3 are described in [`w4-plan.md`](w4-plan.md). What matters here is
what blocks on what.

**Writing the C++ does not block on the toolchain.** The source can exist before
a compiler does, so the install and the code can proceed in parallel.

| step | needs | blocks on install? |
|---|---|---|
| Write `backend`, `pipeline`, `sources`, `harness`, `run_sync` | nothing | **no** |
| Phase 0: compile a 20-line session probe | 1, 2 | yes |
| Phase 1: build and run the sequential pipeline | 1, 2, 3, 4 | yes |
| **Cross-validate C++ against Python on CPU / GPU / NPU** | all | yes — **and on the gram** |
| Phase 2: async, reorder, order checking | all | yes |
| Phase 3: tuning | all | can continue on the 550XBE |

### Suggested sequence

1. **Start the MSVC Build Tools download now.** It is the long pole and it is
   pure waiting.
2. While it downloads, write the C++ sources. They are a port of code that
   already exists and whose behaviour is already pinned by the Python results.
3. Phase 0 the moment `cl` works: open the model, print input and output names
   and shapes, exit. Nothing else. If that does not build, the problem is the
   toolchain and no amount of pipeline code will reveal it.
4. Phase 1, then **immediately** cross-validate on the NPU. That is the
   deadline-bound step; everything after it survives the handover.
5. Phase 2 and 3 wherever the machine is.

## Layout

```
cpp/
  CMakeLists.txt
  third_party/json.hpp
  include/           harness.hpp  backend.hpp  pipeline.hpp  sources.hpp
                     safe_queue.hpp  reorder.hpp
  src/               harness.cpp  backend.cpp  pipeline.cpp  sources.cpp
                     run_sync.cpp  run_async.cpp
```

`bench/` keeps the Python. The C++ writes the **same JSON schema**, so
`bench/summarize.py`, `bench/compare_backends.py` and the existing analysis read
both without modification — the single decision that makes the comparison cost
nothing.

## The protocol does not change

Warm-up discarded, `--duration-s` rather than a frame count, p50/p95/max,
`unaccounted` published, paired A/B in one session, ratios not absolutes, order
verified on every run, and a change smaller than the run-to-run CV is not a
change. `verify_equivalence` runs against the C++ build from the first version
that produces predictions — buffer reuse and a fused preprocessing pass are
precisely the optimisations that would introduce the bug it is the only thing
here that detects.
