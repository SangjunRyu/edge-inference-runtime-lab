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

## Not yet measured

The backend × resolution matrix. So far CPU has only been measured at 360p and
the NPU only at 1080p, so backend and resolution are confounded — `inference`
should respond only to the backend and `read` only to the resolution, and that
has not been shown.

| | 360p | 720p | 1080p |
|---|---|---|---|
| CPU | partial | — | — |
| GPU (Arc iGPU) | — | — | — |
| NPU | — | — | ✅ |
