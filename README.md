# edge-inference-runtime-lab

A study of inference pipeline execution models, inspired by the architecture of
the open-source [DEEPX DX-APP](https://github.com/DEEPX-AI/dx_app) runtime and
reimplemented with ONNX Runtime on commodity hardware.

DX-APP ships a synchronous and an asynchronous runner for every model, reports
Read / Preprocess / Inference / Postprocess as separate rows, and keeps two
different metric types for the two runners — `ProfilingMetrics.get_fps()` for
sync, `AsyncProfilingMetrics.get_throughput()` plus queue depth for async. The
naming is not incidental. This repository works out why.

## Questions, in order

| Phase | Question |
|---|---|
| W1 | Where does one frame spend its time? |
| W2 | What changes when stages overlap? |
| W3 | Does higher throughput still produce correct output? |
| W4 | Can the same pipeline be expressed with explicit C++ concurrency and memory ownership, and does that control reveal improvements Python cannot reach? |

W3 did not stay a separate phase — ordering had to be checked in the same run
that reported the speedup, so the checks were built into W2.
[`docs/findings-w3.md`](docs/findings-w3.md) records what it set out to ask,
what was answered, and the part that was not: **nothing yet verifies that the
async pipeline produces the same predictions, only that it emits them in the
right order.** W4 is planned in [`docs/w4-plan.md`](docs/w4-plan.md).

## The hypothesis under test

Pipeline overlap does not make inference faster. It removes idle time between
stages. So the achievable gain is set by **stage balance**, not by hardware
speed:

```
gain  ~  sum(all stages) / max(single stage)
```

The faster the accelerator, the smaller inference's share, and the *larger* the
overlap gain. DEEPX's own figures for YOLOv8n at 640x640 on a DX-M1 are 25-35
fps sync against 60-90 fps async — the same NPU, so in sync mode the
accelerator was idle for most of the wall clock and the host pipeline was the
ceiling.

On slow inference hardware the prediction inverts: inference dominates, and
overlap should buy close to nothing. Measuring both regimes on one machine is
the point of this repository, not a workaround for the machine.

## Measurement protocol

See [`docs/protocol.md`](docs/protocol.md). Fixed before the first run: Read
timed as its own stage; service time never mixed with wait time; warm-up frames
discarded; p50/p95/max always reported alongside the mean; and
`E2E - sum(stages)` published as `unaccounted` rather than silently absorbed.

## Quick start

Call the venv interpreter directly instead of activating it. On Windows
`activate` requires a relaxed PowerShell execution policy, and on any machine an
environment activated earlier in `PATH` can silently win.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt

.venv\Scripts\python.exe -m bench.env_fingerprint           # what am I running on?

.venv\Scripts\python.exe -m pip install torch torchvision   # export only
.venv\Scripts\python.exe -m bench.prepare_models --models resnet50 --sizes 224

.venv\Scripts\python.exe -m bench.run_sync --model models/resnet50_224.onnx `
    --backend cpu --frames 300 --tag w1-baseline
```

On Linux/macOS substitute `.venv/bin/python`.

## Results

W1 is measured: a 3 backend x 3 resolution grid, 5 runs per cell.
Full write-up in [`docs/findings-w1.md`](docs/findings-w1.md).

```
inference p50 (ms)      360p    720p   1080p      overlap ceiling (x)
cpu (ORT CPU EP)       14.39   11.87   11.91      1.10   1.29   1.41
openvino/CPU           17.39   17.61   17.36      1.06   1.17   1.29
openvino/GPU (Arc)      2.39    2.53    3.43      1.83   2.19   1.64
openvino/NPU            3.05    3.42    5.32      1.39   1.98   2.14
```

1. **Making inference faster moves the bottleneck to the host.** Inference
   falls 4-6x on the accelerators and the overlap ceiling roughly doubles. On
   CPU an async pipeline is not worth building: the ceiling at 360p is 1.10x
   against 30% run-to-run noise.
2. **Measured "inference" is not device time.** It rises with *source*
   resolution even though the model always sees 224x224 — +74% on the NPU from
   360p to 1080p, against 2% within-cell noise. `session.run()` on an offloaded
   device includes host copy, submit and **wait**. The `openvino/CPU` row is the
   control that confirms it: a CPU backend has no submit-and-wait path, and it
   is the flattest row in the grid at **CV 1%**. DEEPX's SDK already splits this
   into `get_latency()` and `get_npu_inference_time()`; this grid reached the
   same distinction from the other side.
   *Engine and device also separate now:* OpenVINO's CPU kernels are 1.46x
   slower than ORT's, and the NPU is 3.3x faster than OpenVINO on CPU — the
   "4-6x" quoted earlier was one stacked on the other.
3. **Inference on the iGPU contends with video decode.** `read` agrees across
   backends to within 3% at 720p, but at 1080p the GPU row is double the CPU
   row — same silicon doing decode and inference. It is the only cell in the
   grid where `read` is the bottleneck, and the only backend whose ceiling
   *falls* at 1080p.
4. **The CPU cells are bimodal, not noisy** — per-run inference switches
   between two levels rather than drifting, consistent with a hybrid P/E-core
   part and an unpinned thread pool. W2 confirmed the fix: pinning
   `--intra-threads 4` removed it and cut the paired-ratio CV from 23.8% to
   6.2%, and changed the CPU async answer from 1.37x to 0.85x.

### W2 — overlapped pipeline

Paired measurement: sync and async alternate in one process sharing one
session, and the ratio within each pair is reported. Full write-up in
[`docs/findings-w2.md`](docs/findings-w2.md).

| device | clip | throughput | latency (E2E p50) | ratio CV |
|---|---|---|---|---|
| CPU (ORT), 4 threads pinned | 360p | **0.85x** | 12.90x worse | 6.2% |
| CPU (ORT), threads unpinned | 360p | 1.37x | 7.9x worse | 23.8% |
| Arc iGPU (OpenVINO) | 360p | 1.06x | 9.89x worse | 1.6% |
| Arc iGPU (OpenVINO) | 1080p | 1.57x | 1.98x worse | 10.2% |
| NPU (OpenVINO) | 360p | 1.44x | 7.25x worse | 12.7% |
| **NPU (OpenVINO)** | **1080p** | **3.09x** | **0.83x better** | 6.5% |

Every run in every configuration emitted all frames in submission order;
`run_ab` refuses to report a speedup for any configuration that did not.

The CPU rows use ONNX Runtime's kernels and the accelerator rows use OpenVINO,
so the same engine was run on the CPU to close that confound. It came out too
noisy to quote (5 pairs, mean 1.01x at 29.1% CV), but the qualitative form
holds: both CPU engines land at or below parity while both accelerators are
above it, so **the CPU loss is not an artefact of the engine.**

Two variables move the answer and they separate cleanly: **where inference
runs** (on CPU with a pinned pool async *loses*) and **how much host work there
is to hide** (360p leaves almost nothing to overlap, so even the NPU manages
only 1.44x; 1080p decode costs 5-8 ms and the same device reaches 3.09x).
W1's overlap ceiling predicts it from sequential timings alone — NPU at 360p,
ceiling 1.39x against 1.44x measured.

**Overlap pays only when inference runs somewhere other than the host.**
`session.run_async` executes on ONNX Runtime's intra-op pool — the pool that
also does the computation — so on CPU in-flight requests do not add
parallelism, they subdivide it: four requests on four threads make each
inference ~4.5x slower and the pipeline loses to the sequential one. On the
NPU inference is off the host, queue waits fall to ~0.05 ms, and throughput
moves from one frame per sum of stages (1/19.4 ms = 52 FPS) to one frame per
slowest stage (1/6.16 ms = 162 FPS, 176 measured). Little's Law confirms it
independently: 1.00 frames resident sequentially, 3.13 overlapped.

The latency improvement is **not** an overlap effect — overlap cannot shorten a
frame's own service time. The stage times themselves fell (`read` 9.75 → 5.05
ms for identical work), which is recorded as an open question rather than
claimed as a pipeline result.

### In-flight depth (NPU, 360p, 10 s per point)

| depth | FPS | E2E p50 | E2E p95 | inference p50 | resident |
|---|---|---|---|---|---|
| 1 | 226.7 | 28.1 | 45.9 | 3.52 | 6.96 |
| 2 | 363.3 | 21.1 | 25.8 | 5.07 | 7.95 |
| **4** | **383.8** | **25.8** | 28.7 | 10.14 | 10.08 |
| 8 | 383.6 | 38.5 | 47.9 | 20.39 | 15.12 |
| 16 | 386.8 | 74.1 | 104.1 | 36.27 | 29.65 |

Throughput saturates at depth 4; going on to 16 buys **+0.8% throughput for
2.87x the latency**. `inference` grows linearly with depth because the stage
timer spans submit to completion and queueing inside the runtime lands in it,
and `resident` tracks depth exactly as Little's Law requires.

The same sweep at 1080p finds no knee: `q_inflight` is 0.00 ms from depth 2
upward and `resident` stays near 2.8 whatever the limit says, because 1080p
decode cannot feed more. A limit that never binds cannot shape anything —
whether depth is worth tuning is a property of the workload, not the pipeline.

With reordering disabled, 138 of 200 frames are emitted out of submission order
while the frame count is exactly right — the defect DEEPX's `FrameReorderBuffer`
prevents. Restoring order costs ~7 ms of mean latency and no throughput.

## Findings from reading DX-APP

See [`docs/dxapp-callgraph.md`](docs/dxapp-callgraph.md) and
[`docs/dxapp-sync-async.md`](docs/dxapp-sync-async.md).

## Licence note

No DEEPX source is vendored or redistributed here. `dx_rt` is licensed to
holders of DEEPX hardware; this repository contains an independent
implementation and reading notes only.
