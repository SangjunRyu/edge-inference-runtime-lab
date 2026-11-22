# Actual-image pipeline comparison

- Input: Imagenette-320 validation subset, 500 unique images/run.
- Warmup: 50; measured: 450; repeats: 3.
- Batch 1, identical manifest order, same preprocessing, Top-5 collection.
- File mode includes read+decode with OS-managed warm cache; no cold-disk claim.
- Predecoded mode yields read-only references; preload excluded from timing.
- Async inference time is submit-to-callback latency, including runtime waiting.

| Input | Runner | Images/s mean [min, max] | E2E p95 ms mean [min, max] | Value parity |
|---|---|---:|---:|---|
| files | sync | 232.40 [219.38, 240.19] | 5.24 [4.91, 5.68] | all pass |
| files | async1 | 353.54 [337.19, 365.12] | 22.76 [21.89, 24.05] | all pass |
| files | async2 | 372.65 [322.06, 399.46] | 25.07 [23.35, 28.38] | all pass |
| files | async4 | 426.59 [384.05, 473.13] | 30.01 [25.47, 32.45] | all pass |
| files | async8 | 454.02 [410.11, 482.90] | 49.50 [44.04, 55.49] | all pass |
| files | async16 | 453.26 [434.44, 475.28] | 96.35 [87.84, 102.70] | all pass |
| files | async32 | 477.61 [461.30, 492.30] | 148.20 [118.10, 184.62] | all pass |
| predecoded | sync | 269.11 [238.42, 300.07] | 4.50 [4.09, 4.85] | all pass |
| predecoded | async1 | 364.80 [355.61, 370.82] | 21.42 [20.31, 22.07] | all pass |
| predecoded | async2 | 421.86 [413.11, 426.39] | 21.15 [20.87, 21.52] | all pass |
| predecoded | async4 | 420.20 [364.41, 466.13] | 30.40 [27.30, 35.23] | all pass |
| predecoded | async8 | 459.33 [406.25, 488.85] | 46.34 [42.79, 51.99] | all pass |
| predecoded | async16 | 451.57 [398.63, 485.06] | 99.45 [91.23, 113.09] | all pass |
| predecoded | async32 | 465.97 [411.94, 493.63] | 150.33 [136.45, 175.41] | all pass |
