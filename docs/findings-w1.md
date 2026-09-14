# W1 findings

Backend x resolution grid, ResNet50 @224, Core Ultra 7 255H, Big Buck Bunny
H.264 clips, 5 runs per cell x 200 counted frames after 100 warm-up, 60 s
cooldown between runs and between cells. Raw data: `results/gram__*.json`.
Regenerate with `python -m bench.summarize --tag gram`.

Rows are the median across the five runs of each run's p50, so one bad run
cannot define a cell.

## 1. Making inference faster moves the bottleneck to the host

```
inference p50 (ms)      360p    720p   1080p      overlap ceiling (x)
cpu (ORT CPU EP)       14.39   11.87   11.91      1.10   1.29   1.41
openvino/CPU           17.39   17.61   17.36      1.06   1.17   1.29
openvino/GPU (Arc)      2.39    2.53    3.43      1.83   2.19   1.64
openvino/NPU            3.05    3.42    5.32      1.39   1.98   2.14
```

Inference drops 3-7x once it leaves the host and the overlap ceiling roughly
doubles, from 1.06-1.41x on the two CPU rows to as much as 2.19x. That is the whole
hypothesis: overlap does not make inference faster, it removes idle time, so
the gain available is set by stage balance. The faster the accelerator, the
smaller inference's share, and the more there is to overlap.

This is the regime DEEPX's own published figures sit in — 25-35 fps sync
against 60-90 fps async for YOLOv8n on a DX-M1, the same NPU in both rows,
which puts the accelerator idle for most of the wall clock in sync mode.

On CPU it is not worth building an async pipeline at all: at 360p the ceiling
is 1.10x, and the machine's own run-to-run noise in that cell is 30%.

## 2. Measured "inference" is not device time

`inference` should not depend on the source clip -- the model always receives a
224x224 tensor. It does, on the accelerators only:

| | cross-clip CV | within-cell CV | verdict |
|---|---|---|---|
| cpu (ORT CPU EP) | 11% | 30% | can't tell, cell too noisy |
| **openvino/CPU** | **1%** | 1% | **flat -- the control** |
| openvino/GPU | 20% | 10% | **real** |
| openvino/NPU | 31% | 6% | **real** |

```
inference p50 (ms)      360p    720p   1080p
cpu (ORT CPU EP)       14.39   11.87   11.91
openvino/CPU           17.39   17.61   17.36     <- CV 1%
openvino/GPU            2.39    2.53    3.43
openvino/NPU            3.05    3.42    5.32
```

The NPU 360p cell is the tightest in the grid -- five runs at 2.97 / 3.14 / 3.02
/ 3.08 / 3.05 ms, CV 2%. Against that, 1080p at 5.32 ms is +74%, far outside the
noise.

The stage timer wraps `session.run()`, which for an offloaded device is input
copy, submit, **wait**, output copy -- host work -- not just device compute.
Decoding 1080p keeps the CPU busy and the host side of the call slows with it.

**`openvino/CPU` is the control that confirms the mechanism.** On a CPU backend
there is no submit-and-wait path: `session.run()` is computation on the calling
thread. If the resolution effect came from anything else -- cache pressure,
memory bandwidth, thermal -- it should appear here too, and more strongly, since
decode and inference are competing for the same cores. It does not appear at
all: 17.39 / 17.61 / 17.36 ms, **CV 1%**, the flattest row in the grid.

**DEEPX's SDK already separates these.** `dx_rt` exposes `get_latency()` and
`get_npu_inference_time()` as different calls, and their performance notes use
the gap between them to tell a CPU-side bottleneck from an NPU-side one. This
grid arrived at the same distinction from the other direction: measure at the
application boundary and the two are silently mixed.

## 2b. Engine and device, finally separated

The `cpu` row uses ONNX Runtime's own kernels while the GPU and NPU rows use
OpenVINO, so every earlier CPU-versus-NPU statement carried an uncontrolled
second variable. The `openvino/CPU` row removes it:

| comparison | holds constant | effect |
|---|---|---|
| ORT CPU -> OpenVINO CPU | the device | OpenVINO is **1.46x slower** (11.9 -> 17.4 ms) |
| OpenVINO CPU -> OpenVINO NPU | the engine | the NPU is **3.3x faster** (17.4 -> 5.32 ms) |

The "4-6x" quoted before this row existed was a 1.46x engine penalty stacked on
a 3.3x device gain. The device advantage is real but smaller than it looked.

## 3. Inference on the iGPU contends with video decode

`read` should depend only on the clip. At 720p it does, almost perfectly:

```
read p50 (ms)           360p    720p   1080p
cpu (ORT CPU EP)        0.37    2.38    3.86
openvino/CPU            0.37    2.39    4.75
openvino/NPU            0.47    2.35    4.78
openvino/GPU            0.71    2.24    7.70
                               CV 3%
```

At 720p all four backends agree within 3%. At 1080p three of the four land
between 3.86 and 4.78 ms and **only the GPU is double** at 7.70.

The Arc iGPU is running inference while OpenCV decodes, and on an
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

**Confirmed in W2:** pinning `--intra-threads 4` removed the two-level
behaviour and cut the paired-ratio CV from 23.8% to 6.2%. It also changed the
answer -- the CPU async result went from 1.37x to 0.85x -- which is why the
confound mattered. The correct statement was "the CPU cells were unstable in
this configuration", not "the CPU cannot be measured".

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

- **The W1 grid's CPU row still has unpinned threads.** W2 showed pinning fixes
  the bimodality, but the grid above was measured before that and has not been
  re-run, so its `cpu` row carries 27-54% within-cell noise.
- **`read` across backends at 1080p** came out at 37% cross-axis against 24%
  within-cell -- under the 2x bar this report uses. With the `openvino/CPU` row
  added, three of four backends now agree at 1080p and only the GPU stands
  apart, which is stronger than the statistic alone suggests, but the formal
  test is still not met.

## What W2 did with this

Every number above is the sequential pipeline, and the overlap ceiling is what
overlap *could* reach. `docs/findings-w2.md` measures what it does reach, and
the ceiling turns out to predict it: NPU at 360p, ceiling 1.39x against 1.44x
measured.
