# W1 findings

Backend x resolution grid, ResNet50 @224, Core Ultra 7 255H, Big Buck Bunny
H.264 clips, 5 runs per cell x 200 counted frames after 100 warm-up, 60 s
cooldown between runs and between cells. Raw data: `results/gram__*.json`.
Regenerate with `python -m bench.summarize --tag gram`.

Rows are the median across the five runs of each run's p50, so one bad run
cannot define a cell.

## 1. Making inference faster moves the bottleneck to the host

```
inference p50 (ms)      360p    720p   1080p
cpu (ORT CPU EP)       14.39   11.87   11.91
openvino/GPU (Arc)      2.39    2.53    3.43
openvino/NPU            3.05    3.42    5.32

overlap ceiling (x)     360p    720p   1080p
cpu                     1.10    1.29    1.41
openvino/GPU            1.83    2.19    1.64
openvino/NPU            1.39    1.98    2.14
```

Inference drops 4-6x on the accelerators and the overlap ceiling roughly
doubles, from 1.10-1.41x on CPU to as much as 2.19x. That is the whole
hypothesis: overlap does not make inference faster, it removes idle time, so
the gain available is set by stage balance. The faster the accelerator, the
smaller inference's share, and the more there is to overlap.

This is the regime DEEPX's own published figures sit in — 25-35 fps sync
against 60-90 fps async for YOLOv8n on a DX-M1, the same NPU in both rows,
which puts the accelerator idle for most of the wall clock in sync mode.

On CPU it is not worth building an async pipeline at all: at 360p the ceiling
is 1.10x, and the machine's own run-to-run noise in that cell is 30%.

## 2. Measured "inference" is not device time

`inference` should not depend on the source clip — the model always receives a
224x224 tensor. It does:

| | cross-clip CV | within-cell CV | verdict |
|---|---|---|---|
| cpu | 11% | 30% | can't tell, cell too noisy |
| openvino/GPU | 20% | 10% | **real** |
| openvino/NPU | 31% | 6% | **real** |

The NPU 360p cell is the tightest in the grid — five runs at 2.97 / 3.14 / 3.02
/ 3.08 / 3.05 ms, CV 2%. Against that, 1080p at 5.32 ms is +74%, far outside
the noise.

The stage timer wraps `session.run()`, which is input copy, submit and wait —
host work — not just device compute. Decoding 1080p keeps the CPU busy and the
host side of the call slows down with it.

**DEEPX's SDK already separates these.** `dx_rt` exposes `get_latency()` and
`get_npu_inference_time()` as different calls, and their performance notes use
the gap between them to tell a CPU-side bottleneck from an NPU-side one. This
grid arrived at the same distinction from the other direction: measure at the
application boundary and the two are silently mixed.

## 3. Inference on the iGPU contends with video decode

`read` should depend only on the clip. At 720p it does, almost perfectly:

```
read p50 (ms)           360p    720p   1080p
cpu                     0.37    2.38    3.86
openvino/NPU            0.47    2.35    4.78
openvino/GPU            0.71    2.24    7.70
                               CV 3%
```

At 720p all three backends agree within 3%. At 1080p the GPU row is double the
CPU row. The Arc iGPU is running inference while OpenCV decodes, and on an
integrated part the media engine, the execution units and the memory bandwidth
are the same silicon. 720p decode is light enough not to collide; 1080p is not.
The NPU sits in between, as a separate block that still shares memory.

The consequence shows up in the ceiling table: GPU is the only backend whose
ceiling *falls* at 1080p (2.19x at 720p, 1.64x at 1080p), and the only cell in
the grid where `read` rather than `inference` is the bottleneck.

For an edge SoC this is the interesting case, not a nuisance: the accelerator
and the video path are not independent, so a pipeline tuned on one resolution
does not transfer to another.

## 4. The CPU cells are bimodal, not noisy

| cell | fps CV | inference CV | read CV |
|---|---|---|---|
| cpu x 1080p | 30% | 54% | 32% |
| cpu x 360p | 30% | 27% | 6% |
| openvino/GPU x 360p | 25% | 9% | 35% |
| openvino/NPU x 360p | 9% | 2% | 10% |
| cpu x 720p | 8% | 8% | 9% |
| openvino/GPU x 720p | 6% | 8% | 5% |
| openvino/NPU x 720p | 3% | 1% | 5% |

Per-run `inference` p50 in the worst cells:

```
cpu x 1080p:  12.28  30.11  11.31  11.91  11.19
cpu x 360p:   18.08  10.31  10.34  18.01  14.39
```

That is not thermal drift, which moves gradually. It switches between two
levels. The 255H is a hybrid part — P-cores, E-cores and low-power E-cores —
and these runs had `intra_threads` unset, so ONNX Runtime sizes its pool from
the machine and the OS is free to land it on different core types from run to
run. The OpenVINO cells do not show it; that execution provider manages its own
threading and affinity.

**Untested fix:** pin `--intra-threads` and re-measure the CPU row. Until that
is done the correct statement is "the CPU cells were unstable in this
configuration", not "the CPU cannot be measured".

The idle gate reported all nine cells below threshold (6-22% background), so
this is not contention from other processes.

## 5. Report ratios, not absolutes

From the five-run NPU set (`results/gram-npu-1080_*.json`): throughput CV 9.7%,
E2E p50 CV 10.1%, overlap ceiling CV 2.2%. The ratio is roughly eight times
more stable, because the clock falls during a run (2000 -> 1500 MHz in four
runs out of five) and every stage slows together.

Every headline here is therefore a ratio or a share. No bare FPS number in this
document should be quoted on its own.

## Not established

- **Engine vs device is confounded in the CPU row.** `cpu` is ONNX Runtime's
  own CPU kernels; `openvino/GPU` and `openvino/NPU` are the OpenVINO provider.
  Comparing them mixes a change of engine with a change of device. An
  `openvino/CPU` row would separate the two and has not been run.
- **CPU row with pinned threads** (finding 4).
- **`read` across backends at 1080p** came out at 37% cross-axis against 24%
  within-cell — under the 2x bar this report uses, so finding 3 rests on the
  360p and 720p columns plus the mechanism, not on the 1080p column alone.
- **Async has not been built.** Every number here is the sequential pipeline.
  The ceiling is what overlap could reach, not what it does reach.
