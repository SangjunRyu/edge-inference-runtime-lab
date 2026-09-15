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

## No Microsoft toolchain is needed, and phase 0 already passed

Installing MSVC on a company laptop is a policy question as much as a licensing
one, and it turns out to be unnecessary. MSYS2 is already on this machine from
earlier work, and its **UCRT64** environment already carries the toolchain:

```
gcc    16.1.0   (MinGW-w64 UCRT64)   installed
cmake   4.4.0                        installed
ninja   1.13.2                       installed
```

GCC under MinGW is GPL-with-runtime-exception; nothing here is Microsoft's.

### The risk that had to be tested first

MinGW and MSVC do not share a C++ ABI, so **MSVC-built C++ libraries cannot be
linked from GCC.** Every prebuilt library here -- OpenVINO, OpenCV's official
Windows package, the ONNX Runtime zip -- is built with MSVC. That would normally
end the idea.

It does not, because OpenVINO ships a **C** API alongside the C++ one, and C has
a stable ABI. `openvino_c.dll` and `openvino_c.lib` are both in the pip wheel,
with headers under `include/openvino/c/`.

Tested rather than assumed:

```
gcc -O2 -I<openvino>/include probe.c -o probe.exe     -L<openvino>/libs -l:openvino_c.dll

OpenVINO 2025.4.1-20426-82bbf0292c5-releases/2025/4
  OpenVINO Runtime
devices: CPU GPU NPU
```

GCC links the MSVC-built import library, and all three devices enumerate.
**Phase 0's success criterion is met on a toolchain with no Microsoft software
in it.**

Every call the port needs exists in the C API, including the two that matter
most:

| call | needed for |
|---|---|
| `ov_infer_request_start_async` + `ov_infer_request_set_callback` | the submit/collect split that makes W2 asynchronous rather than threaded |
| `ov_tensor_create_from_host_ptr` | phase 3's input tensor over a reused buffer |
| `ov_infer_request_wait`, `ov_infer_request_set_input_tensor`, `ov_core_compile_model` | the rest of the pipeline |

### C API, and why that is not a consolation prize

The C++ API would be less verbose, but wrapping a C API in RAII is itself the
thing W4 set out to demonstrate: `std::unique_ptr` with custom deleters over
`ov_core_t*`, `ov_compiled_model_t*`, `ov_infer_request_t*`, with ownership
explicit at every boundary. It is also the portable choice -- the same source
builds under MSVC on the 550XBE later, where the C++ API would not have built
here at all.

## What is still to install

| what | download | installed | needed by |
|---|---|---|---|
| `mingw-w64-ucrt-x86_64-opencv` 4.13.0 | 20 MB | 166 MB | phase 1 (decode, resize) |
| `mingw-w64-ucrt-x86_64-nlohmann-json` | 118 KB | 938 KB | phase 1 (the shared JSON schema) |

```bash
pacman -S mingw-w64-ucrt-x86_64-opencv mingw-w64-ucrt-x86_64-nlohmann-json
```

Both are permissive-licensed (Apache 2.0 and MIT) and built for the MinGW ABI,
so the C++ linking problem above does not arise for them.

**One caution, because this MSYS2 is used for other work.** MSYS2 does not
support partial upgrades: if the package database is stale, `pacman -S` can pull
libraries newer than the installed base and break the existing toolchain. Run
`pacman -Syu` first, close the shell when it asks, and reopen before installing.
That is the standard MSYS2 procedure and it is worth following on a machine
whose GCC is already doing another job.

### Later, on the 550XBE only

| what | why |
|---|---|
| MSVC Build Tools + Visual Studio | the personal machine, where this concern does not apply |
| ONNX Runtime `onnxruntime-win-x64-1.24.4.zip` (71 MB) | C++ headers and libs for the CPU provider |
| `Microsoft.ML.OnnxRuntime.DirectML` NuGet 1.24.4 | the only prebuilt path to the MX150 |

## Building

```bash
# MSYS2 UCRT64 shell
cd cpp
cmake -S . -B build -G Ninja
cmake --build build

./build/probe.exe --model ../models/resnet50_224.onnx --device NPU
```

`OV_ROOT` defaults to the project's own venv. Point it elsewhere with
`-DOV_ROOT=...` if the OpenVINO runtime lives somewhere else.

CMake degrades on purpose: without OpenCV it builds `probe` alone and says so,
so phase 0 can be finished while the rest of the dependencies are still
installing. The OpenVINO DLLs are copied next to each executable after linking
-- `openvino_c.dll` pulls in `openvino.dll` and the device plugins beside it, so
the whole `libs` directory goes rather than the one import target.

### Phase 0 result

```
OpenVINO 2025.4.1-20426-82bbf0292c5-releases/2025/4
devices: CPU GPU NPU
model    models/resnet50_224.onnx
device   NPU
input    [1, 3, 224, 224]  ->  1000 logits
top-5    754:0.1191 905:0.0660 753:0.0574 904:0.0365 921:0.0264
```

Cross-checked rather than merely run: the same synthetic input through the
Python OpenVINO API and `bench.pipeline.postprocess` gives
`754:0.1191 905:0.0660 753:0.0574 904:0.0365 921:0.0264`, identical to four
decimals. The C API path and the C++ softmax agree with the Python ones before
a single stage has been timed.

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
