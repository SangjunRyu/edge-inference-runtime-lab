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
| W4+ | Can the same pipeline be expressed with explicit C++ concurrency and memory ownership? |

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

_Pending first run._

## Findings from reading DX-APP

See [`docs/dxapp-callgraph.md`](docs/dxapp-callgraph.md) and
[`docs/dxapp-sync-async.md`](docs/dxapp-sync-async.md).

## Licence note

No DEEPX source is vendored or redistributed here. `dx_rt` is licensed to
holders of DEEPX hardware; this repository contains an independent
implementation and reading notes only.
