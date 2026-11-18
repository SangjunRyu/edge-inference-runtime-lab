# W4 on the MX150: native ONNX Runtime CPU and CUDA

The existing C++ W4 runners now also build against ONNX Runtime. OpenVINO is
optional for this build. The two engines compile the same runner, preprocessing,
queues, ownership and measurement code through a small backend adapter.

## Build on Windows x64

Requirements: Visual Studio 2022 C++ tools, Windows SDK, CMake, and the working
`.venv-cuda` from the Python CUDA tests. From the repository root in PowerShell:

```powershell
& .\tools\build_cpp_ort.ps1
```

The helper downloads the official OpenCV 4.12.0 Windows C++ SDK and ONNX Runtime
1.26.0 C/C++ SDK into ignored `cpp/build/deps/`. It links the SDK import library
but deploys the **existing .venv-cuda ORT runtime DLLs**, so the C++ and Python
inference engine versions match. CUDA/cuDNN directories from that environment
remain on the executable's DLL search path, including lazily loaded cuDNN DLLs.
No packages are added to or replaced in a Python environment.

Do not mix an arbitrary SDK with a different runtime: these instructions pin
1.26.0 and CUDA 12 for the current MX150 setup. The original OpenVINO build is
still selected with `EIRL_BUILD_OPENVINO=ON` and its existing `OV_ROOT`.

## Run CPU or CUDA

```powershell
$bin = '.\cpp\build\ort-msvc\Release'

& "$bin\run_sync_ort.exe" --model models/resnet50_224.onnx `
    --backend cpu --source clips/bbb_360_2MB.mp4 `
    --frames 300 --warmup 100 --intra-threads 4 --out-dir results/cpp-ort

& "$bin\run_sync_ort.exe" --model models/resnet50_224.onnx `
    --backend cuda --source clips/bbb_1080_10MB.mp4 `
    --frames 300 --warmup 100 --intra-threads 4 --out-dir results/cpp-ort

& "$bin\run_async_ort.exe" --model models/resnet50_224.onnx `
    --backend cuda --source clips/bbb_1080_10MB.mp4 `
    --duration-s 15 --warmup-s 5 --inflight 4 --queue-size 4 `
    --intra-threads 4 --repeat 3 --cooldown 5 --out-dir results/cpp-ort
```

`--backend cpu|cuda` and `--device CPU|CUDA` select the same provider. `GPU` is
rejected on the ORT targets to prevent confusion with OpenVINO's Intel GPU.
CPU-only installations work with `--backend cpu`; CUDA requests fail when their
provider dependencies cannot load. Ordinary CPU graph partitions remain allowed
behind CUDA, matching the Python runner. `actual_providers` lists the configured
providers; use `--profile PREFIX` in a separate diagnostic run to inspect actual
per-node assignment. Profiling should be off for performance measurement.

Async uses ORT `RunAsync` and requires at least two intra-op threads. Each slot
owns an output tensor and retains its input until the collector finishes.
Callbacks enqueue completion without blocking; all callbacks finish before the
output queue is closed. Worker errors wake blocked workers and return nonzero.
Callbacks retain request ownership until their final notification completes.

The existing `--fused`, `--reuse-tensor` (sync), `--post-workers`, reordering and
top-5 dump options remain available. Use `--repeat 1` when collecting a dump,
since a dump path is reused across repeats.

## Verify predictions and failure handling

```powershell
& .\.venv-cuda\Scripts\python.exe tools/validate_cpp_ort.py `
    --out-dir results/cpp-ort-validation
```

The check compares all 300 frames at both source resolutions and both EPs with
Python, including a multi-collector async path, then checks tensor reuse, fused
preprocessing and depth 1. It verifies frame counts/order, error handling and
CUDA node assignment in separate diagnostic profiles. Top-5 probabilities use
the repository's existing tolerance and comparator self-tests.

The native SDK is OpenCV 4.12.0 while the current Python wheel is OpenCV 5.0.0;
their decoder/preprocessing builds are therefore an additional variable in a
Python/C++ performance comparison. Prediction checks cover the current clips,
not every codec or image. The C++ pipeline also acquires a total slot before
reading and returns it after collection, whereas Python caps outstanding
inference separately from stage queues. Equal `inflight` numbers do not mean
equal total buffering or equivalent latency. Keep these differences explicit
when interpreting results. These executables are Windows x64 deployments;
Linux runtime SDK discovery/deployment is not implemented by this CMake path.

For a short repeated timing matrix, with profiling disabled:

```powershell
& .\.venv-cuda\Scripts\python.exe tools/benchmark_cpp_ort.py `
    --out-dir results/cpp-ort-benchmark
```

It alternates sync/async ordering across three pairs at 360p and 1080p for both
EPs, with 10-second runs and 3-second warm-up. Each executable has its own
session; this is not the single-session Python W2 instrument. Start-of-run CPU
load, drift, FPS ranges and raw stage summaries are retained, so unstable runs
remain visible rather than silently becoming language-speedup claims.
