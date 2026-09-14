# Results

Every file is the raw JSON a run wrote, including the environment fingerprint
and the arguments that produced it. Nothing here is edited by hand.

Read `../docs/protocol.md` first — on this instrument a single run is not
evidence, and absolute throughput carries roughly 10% run-to-run noise.

## Validated

| file | what it is |
|---|---|
| `gram-npu-1080_openvino-npu_resnet50_224.json` | ResNet50 @224 on the Intel NPU, 1080p source, **5 runs × 200 frames**, 60 s cooldown. The noise baseline the protocol quotes: throughput CV 9.7%, overlap ceiling 2.09x at CV 2.2%. |

## Exploratory

Single runs taken while the harness was being built. They are kept because the
protocol's conclusions were derived from them, not because their absolute
numbers should be quoted. All were taken with the machine at ~19% background
load and without repeats, so by this repository's own rules they are
contaminated.

| file | why it exists |
|---|---|
| `smoke_cpu_resnet50_224.json` | first end-to-end run; confirmed `unaccounted` ≈ 0.01 ms, i.e. the stage timers account for all of E2E |
| `w1_cpu_resnet50_224.json` | synthetic source, 1080p frames — showed `read` = 2.5 ms is an array copy, not a decode |
| `videocheck_cpu_resnet50_224.json` | first real video; `read` fell to 0.45 ms, proving the synthetic `read` was never decode cost |
| `w1-360_cpu_resnet50_224.json` | 360p, warm-up 150; overlap ceiling 1.12x with inference at 89.5% |
| `ceilcheck_cpu_resnet50_224.json` | same, after the harness began reporting the ceiling and the noise-vs-gain check |
| `npucheck_openvino-npu_resnet50_224.json` | first NPU run; bottleneck moved from inference to `read` |
| `noisetest_openvino-npu_resnet50_224.json` | 3 runs, 10 s cooldown; produced the first between-run CV and showed 10 s is not enough |

## The W1 grid

`gram__*.json` -- 9 cells, 3 backends x 3 resolutions, 5 runs each, 200 counted
frames after 100 warm-up, 60 s cooldown between runs and between cells. All on
ONNX Runtime 1.24.1. Analysed in `../docs/findings-w1.md`.

| | 360p | 720p | 1080p |
|---|---|---|---|
| cpu (ORT CPU EP) | done | done | done |
| openvino/GPU (Arc iGPU) | done | done | done |
| openvino/NPU | done | done | done |

Per-cell noise, worst first -- the CPU cells are bimodal rather than noisy and
should be re-run with `--intra-threads` pinned:

| cell | fps CV | inference CV | read CV |
|---|---|---|---|
| cpu x 1080p | 30% | 54% | 32% |
| cpu x 360p | 30% | 27% | 6% |
| openvino/GPU x 360p | 25% | 9% | 35% |
| openvino/GPU x 1080p | 17% | 13% | 23% |
| openvino/NPU x 1080p | 14% | 15% | 18% |
| openvino/NPU x 360p | 9% | 2% | 10% |
| cpu x 720p | 8% | 8% | 9% |
| openvino/GPU x 720p | 6% | 8% | 5% |
| openvino/NPU x 720p | 3% | 1% | 5% |

The `openvino/CPU` row was added later with the same settings and drops
straight into the grid (`gram__openvino-cpu__*.json`), so it is 4 backends x 3
resolutions now.

## W2 -- overlapped pipeline

| file | what it is |
|---|---|
| `ab-*.json` | paired sync-vs-async, alternating in one process. cpu/gpu/npu x 360p/1080p |
| `sweep-npu360-10s__*.json` | in-flight depth sweep, NPU 360p, 10 s per point. The clean curve: knee at 4 |
| `sweep-npu__*.json` | same at 1080p. No knee -- the limit never binds |
| `sweep-gpu360__*.json` | Arc iGPU, re-run with a stall watchdog and one process per depth |
| `sweep-npu360__*.json` | superseded by the `-10s` file: fixed frame count, 0.55-0.98 s per run |

## Still to run

- CPU row of the W1 grid with `--intra-threads` pinned. W2 showed pinning
  removes the bimodality, but the grid was measured before that.
