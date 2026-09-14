# Measurement protocol

Fixed before the first measurement, because a protocol chosen after seeing the
numbers is not a protocol.

## Stage boundaries

| Stage | Starts | Ends |
|---|---|---|
| `read` | pull from the frame source | decoded BGR frame in hand |
| `preprocess` | resize | normalised NCHW float32 tensor |
| `inference` | `session.run` entry | outputs returned |
| `postprocess` | logits in hand | top-k list |
| `e2e` | before `read` | after `postprocess` |

`unaccounted = e2e - sum(stages)`. Reported on every run. It is where copies,
allocation and runtime overhead live, and it is the first thing to explain if
it is large.

## Rules

1. **Discard warm-up.** Default 20 frames. First-call allocation, lazy kernel
   selection and page faults are not steady-state behaviour.
2. **Never report a bare mean.** p50/p95/max on every stage. Overlap and
   queueing move the tail long before they move the mean.
3. **Service time is not wait time.** W1 has no queues so only service time
   exists; the field names are already separated so W2 adds rows rather than
   redefining columns.
4. **Pin thread counts** (`--intra-threads`) whenever two runs are compared.
   Otherwise ONNX Runtime sizes its pool from the machine and the CPU stages
   become incomparable.
5. **Check for drift.** The harness compares the first and last decile of E2E.
   Above 1.05x the run was not in steady state — a thermally throttling laptop
   is two different machines and its average describes neither.
6. **Record what actually ran.** Requesting CUDA and silently getting CPU is
   the standard way to publish a wrong number, so `fell_back_to_cpu` is stored
   with every result.
7. **One variable per run.** Model, resolution, backend, thread count, source.
   Change one.

## The instrument is a laptop

Measured on this machine (Core Ultra 7 255H, ResNet50 @224 on the Intel NPU via
OpenVINO, 1080p H.264 source, 200 counted frames after 100 warm-up, five
identical runs, 60 s cooldown between them —
`results/gram-npu-1080_openvino-npu_resnet50_224.json`):

| metric | mean | min | max | CV |
|---|---|---|---|---|
| throughput FPS | 80.19 | 74.34 | 92.62 | **9.7%** |
| E2E p50 ms | 12.36 | 10.49 | 13.41 | **10.1%** |
| overlap ceiling | 2.09 | 2.01 | 2.13 | **2.2%** |

CPU clock across a single run: **2000 → 1500 MHz** in four runs out of five.

An earlier three-run set with only 10 s of cooldown
(`results/noisetest_openvino-npu_resnet50_224.json`) gave 14.9% / 16.7% / 1.8%
and its slowest run was the last — heat accumulates *between* repeats, not only
within one. Raising the cooldown from 10 s to 60 s cut throughput CV from 14.9%
to 9.7%. The ratio's CV barely moved either way.

Two conclusions, and they govern everything measured here:

1. **Absolute numbers on this machine carry ~15% run-to-run noise.** Any claim
   of an improvement smaller than that is unsupported. `--repeat 3` is the
   minimum; the runner prints the spread and states the detection floor.
2. **Ratios are ~8x more stable than absolutes.** The clock drops during a run
   and every stage slows together, so the ratio survives what the absolute does
   not. Headline claims are therefore stated as ratios — overlap ceiling, stage
   share, sync-vs-async speedup measured back to back — never as a bare FPS.

Supporting mechanisms, all reported with the result:

- **Idle gate.** Refuses to start until system CPU is below `--max-busy-pct`.
  This machine idles near 19%, so the 15% default cannot be met without closing
  background applications — a real finding about the instrument, not a bug.
  When the gate gives up it records `settled: false` so the run is marked
  contaminated rather than silently accepted.
- **Clock sampling.** CPU MHz before and after each run, so a slowdown is
  attributed to the clock rather than guessed at.
- **Trimmed mean.** `mean_trim10` drops the outer 5% each side. The gap between
  it and the plain mean measures how much a few outliers are distorting the
  headline.
- **Cooldown.** `--cooldown` between repeats. 10 s was not enough here: run 3
  was the slowest of three. Heat accumulates across repeats, not just within
  one.

## Known confounds

- **GIL.** The W2 Python pipeline runs its stages as threads under one
  interpreter lock. NumPy and OpenCV release it for their heavy calls, the
  orchestration around them does not. Expected to cap scaling well before the
  C++ port does; must be measured, not assumed.
- **Decode cost is source-dependent.** A 4K video and a 640x480 JPEG put very
  different loads on `read`. Fix the source when comparing anything else.
- **`SyntheticSource` has no decode.** Use it to isolate what `read` costs,
  never as the baseline for a pipeline claim.
