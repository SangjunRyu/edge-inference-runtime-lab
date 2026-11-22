# Actual-image pipeline comparison

- Input: Imagenette-320 validation subset, 500 unique images/run.
- Warmup: 50; measured: 450; repeats: 3.
- Batch 1, identical manifest order, same preprocessing, Top-5 collection.
- File mode includes read+decode with OS-managed warm cache; no cold-disk claim.
- Predecoded mode yields read-only references; preload excluded from timing.
- Async inference time is submit-to-callback latency, including runtime waiting.

| Input | Runner | Images/s mean [min, max] | E2E p95 ms mean [min, max] | Value parity |
|---|---|---:|---:|---|
| files | sync | 29.77 [27.62, 32.98] | 37.20 [33.36, 41.84] | all pass |
| files | async1 | 28.65 [25.23, 30.42] | 297.72 [251.94, 377.96] | all pass |
| files | async2 | 31.68 [29.58, 33.02] | 319.98 [272.51, 393.61] | all pass |
| predecoded | sync | 28.59 [27.81, 29.44] | 39.27 [37.29, 42.35] | all pass |
| predecoded | async1 | 28.70 [26.81, 29.64] | 328.78 [284.68, 410.73] | all pass |
| predecoded | async2 | 30.66 [26.17, 33.38] | 357.83 [268.11, 504.51] | all pass |
