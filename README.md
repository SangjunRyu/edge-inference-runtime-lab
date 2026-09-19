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
| W5 | What must a pipeline own to be able to refuse work, and what does refusing cost? ([plan](docs/w5-plan.md), [findings](docs/findings-w5.md)) |

W3 did not stay a separate phase — ordering had to be checked in the same run
that reported the speedup, so the checks were built into W2.
[`docs/findings-w3.md`](docs/findings-w3.md) records what it set out to ask,
what was answered. The gap it identified -- nothing verified that the async
pipeline produced the same *predictions*, only that it emitted them in the
right order -- is now closed by `bench/verify_equivalence.py`: bit-identical
top-5 across 300 frames at every depth tested, on CPU, Arc iGPU and NPU alike,
with a self-test proving the check can detect a fault and a sensitivity figure
stating how much of the pairing it actually covers.

`bench/compare_backends.py` asks the neighbouring question — whether the
backends agree with *each other*. OpenVINO's CPU kernels match ONNX Runtime's
exactly; the two accelerators each flip the same two frames of 300, both at a
margin of 0.0003 against a numerical spread of 0.004-0.006. They broke a tie
differently rather than disagreeing about the image, and these FP32 numbers are
the floor any later quantisation work would have to beat. W4 is planned in [`docs/w4-plan.md`](docs/w4-plan.md).

## What this found

Six results, in the order they were reached. Each links to the measurement that
supports it; each was measured on one thermally unsteady laptop, which is why
the discipline below matters as much as the numbers.

1. **Overlap pays only when inference runs somewhere other than the host.**
   On the NPU at 1080p a pipelined runner reaches **3.09x** the sequential
   throughput. On the CPU with a pinned thread pool it reaches **0.85x** — it
   loses. `run_async` executes on the same intra-op pool that does the
   computation, so concurrent requests do not add parallelism, they subdivide
   it. [W2](#w2--overlapped-pipeline)

2. **The size of the gain is set by stage balance, and sequential timings
   predict it.** `sum(stages) / max(stage)` gave 1.39x for the NPU at 360p
   before any pipeline existed; the measurement came back 1.44x. The same
   ceiling says CPU at 360p is not worth pipelining, and it was not.
   [W1](#w1--where-one-frame-spends-its-time)

3. **What the stage table calls "inference" is not device time.** It rises 74%
   with *source* resolution though the model always sees 224x224, because
   `session.run()` on an offloaded device includes copy, submit and **wait**.
   The `openvino/CPU` row is the control that proves it: no submit-and-wait
   path, and the flattest row in the grid at CV 1%. [W1](#w1--where-one-frame-spends-its-time)

4. **The reorder buffer is the price of postprocessing concurrently, not an
   accident of DX-APP's design.** One collector cannot reorder and needs no
   buffer; four collectors produce 96 out-of-order completions, and with the
   buffer off, **72 of 479 frames are emitted in the wrong order while the
   frame count stays exactly right.** A silently wrong pipeline, right down to
   the total. [W4](#w4--the-c-port)

5. **Python's orchestration costs ~25% sequentially and 4% overlapped.** The
   host stages are 2-10x faster in C++ either way; overlap moves them off the
   critical path, and a cost that is not on the critical path is not a cost.
   This is result 1 arriving from the other direction — *the reason to overlap
   is not that the host is slow, it is that the host stops mattering.*
   [W4](#w4--the-c-port)

6. **One tuning candidate lived and one died.** Fusing five preprocessing
   passes into one is real (B/A 0.61 at 1080p, direction confirmed twice);
   binding the input tensor once instead of wrapping per frame moves inference
   2% against a 25% spread, which is **smaller than the noise around it** and
   is recorded as unsupported. [W4](#w4--the-c-port)

## How the measurements are defended

The machine moves 10-30% by itself, so the instrument is most of the work.
Fixed before the runs, in [`docs/protocol.md`](docs/protocol.md): Read timed as
its own stage, service time never mixed with wait time, warm-up discarded,
p50/p95/max beside every mean, and `E2E - sum(stages)` published as
`unaccounted` rather than absorbed.

What the comparison tools add:

- **Everything is paired.** Configurations alternate inside one session and the
  ratio is formed **within a pair**, never as a median of one side over a
  median of the other — the two halves of such a ratio come from different
  moments. **Order inside a pair alternates too**, because a paired design
  cannot cancel a bias it applies uniformly.
- **A control stage neither side changes**, judged by its **spread** rather
  than its level: a control detects the machine *changing* between the halves
  of a pair, so a constant offset in it is a finding and a moving one is a
  defect. Where no honest control exists — comparing two overlapped pipelines,
  where the interpreter lock touches every host stage — **the tool says so**
  instead of choosing one it would have to explain away.
- **A pair counts only if both halves reached steady state**, and the figures
  over every pair are printed underneath, so the cost of that rule is on the
  page rather than in a decision nobody can see.
- **One pair claims nothing.** A single pair has no spread, so every steadiness
  test passes trivially; the tools report an observation and refuse the
  measurement.
- **Correctness gates speed.** No speedup is reported for a run that emitted
  frames out of order, and `verify_equivalence` checks that the answers match a
  sequential reference — with a self-test proving it can detect a fault, and
  tie handling so two classes 6e-08 apart are not called a mismatch.
- **Saved runs can be re-read.** `--reanalyse` applies today's criteria to
  yesterday's pairs at no cost in machine time, which matters because the
  criteria below were all corrected *after* the runs they apply to.

Every defect found while building this is a **runnable program** under
[`repro/`](repro/), not only a paragraph: each takes `--mode broken` and
`--mode fixed` and exits 0 when it behaved as that mode predicts, so a rewrite
that reintroduces one is caught rather than described. `python repro/run_all.py`
runs them all. Two of the eight currently report `NOT REPRO` — the condition
stopped firing on this build — and the table says so instead of rounding it to
a pass.

Things this repository retracted rather than kept: a cross-validation on stage
*shares* (which would flag any real improvement as a fault), a knee test that
compared gain to raw CV, a summary that judged cross-axis effects against the
worst cell in the grid, "the timers agree to 2%" (one pair, coincidence), and
"preprocess is identical in both" (also one pair). The Arc iGPU produced
398-second driver stalls under concurrent load and is reported as unmeasurable
rather than plotted.

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

### W1 — where one frame spends its time

A 3 backend x 3 resolution grid, 5 runs per cell.
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

### W4 — the C++ port

Why port at all: a second implementation cross-validates the measurements,
tuning that Python cannot express becomes possible, and the same overlapped
pipeline in C++ separates Python's own overhead from ONNX Runtime's thread-pool
behaviour. Full write-up in [`docs/findings-w4.md`](docs/findings-w4.md).

Built with MSYS2 UCRT64 GCC against OpenVINO's **C** API — no Microsoft
software involved, and the C ABI is the bridge because MinGW and MSVC share no
C++ ABI.

**Sequential, C++ against Python** (NPU, 360p, 6 counterbalanced pairs):

| stage | cpp/python | per-pair CV |
|---|---|---|
| read | 0.70 | 12% |
| preprocess | 0.66 | 2% |
| inference *(control)* | 0.86 | 3% |
| postprocess | 0.09 | 5% |
| **throughput** | **1.28** (1.21 - 1.36) | 4.6% |

The control's level is not 1.00 and need not be: **Python reaches the NPU
through ONNX Runtime's OpenVINO provider and C++ through OpenVINO directly**,
and that costs 10-16% per inference: **0.86** here at CV 3%, and **0.92** at
CV 6% in the overlapped run below. What a control has to hold is its spread,
and 3% is what it held.

**Overlapped, C++ against Python** — the GIL measurement (NPU, 360p, in-flight
4, one collector each so the structures match):

```
throughput  cpp/python  x1.04   (1.00 - 1.09, CV 5.8%)
```

Host stages 2-10x faster, four percent of throughput, because inference is 77%
of the frame and the device does not care which language asked. **W2's
attribution survives**: the CPU async loss was ONNX Runtime splitting its
thread pool, not the interpreter lock. Worth checking — "Python is slow" is the
explanation everyone reaches for, and here it would have been wrong.

**C++ sequential against C++ overlapped**, paired: `x1.32` (1.14 - 1.65),
consistent with Python's 1.44x in the same cell and W1's 1.39x ceiling.

**Tuning.** Measured at 1080p, where host work is comparable to inference:

| candidate | stage | B/A | per-pair CV | verdict |
|---|---|---|---|---|
| fused preprocess | preprocess | **0.61** | 45% | real, size not pinned |
| input tensor reuse | inference | 0.98 | 25% | **unsupported** |

**Ordering, made real.** Phase 2 saw zero out-of-order completions and could
not force one, because a single collector draining FIFO makes overtaking
impossible by construction. With the collector count as a flag:

| post-workers | reorder | order violations | out-of-order arrivals |
|---|---|---|---|
| 1 | on | 0 | 0 |
| 4 | on | 0 | 96 |
| 4 | off | **72 / 479** | — |

**Ownership bugs the port surfaced**, all three from the same submit/collect
split DX-APP has: `ov_callback_t` allocated as a stack local while the API
keeps the pointer (worked with one slot, hung with two); the completion
callback blocking on a bounded queue and deadlocking the runtime against its
own `wait`; and `get_output_tensor` returning `REQUEST_BUSY` from the callback
path until `wait` is called first.

All four C++ builds — sync, reuse-tensor, fused, and async at 8 in flight with
4 collectors — match the Python sequential reference across 300 frames.

## Findings from reading DX-APP

See [`docs/dxapp-callgraph.md`](docs/dxapp-callgraph.md) and
[`docs/dxapp-sync-async.md`](docs/dxapp-sync-async.md).

## Licence note

No DEEPX source is vendored or redistributed here. `dx_rt` is licensed to
holders of DEEPX hardware; this repository contains an independent
implementation and reading notes only.
