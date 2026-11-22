# Imagenette runtime policy validation

Age starts at the scheduled arrival, including upstream waiting in Block.
Ingress holds only image IDs. All emitted images were decoded and inferred with ResNet50.
Order violations count departures from the next expected dense submission ID.
Input pool holds the reusable CPU FP32 tensor through the ORT callback.
The lifetime test hashes staged tensors before submission and at callback.
The paced policy runs retain the same pool without per-image hash overhead.

| Reorder | Emitted | Order violations | Pool peak / capacity | Acquire / release | Outstanding | Top-5 parity |
|---|---:|---:|---:|---:|---:|---|
| off #1 | 500 | 477 | 16 / 16 | 500 / 500 | 0 | True |
| on #1 | 500 | 0 | 16 / 16 | 500 / 500 | 0 | True |
| on #2 | 500 | 0 | 16 / 16 | 500 / 500 | 0 | True |
| off #2 | 500 | 476 | 16 / 16 | 500 / 500 | 0 | True |
| on #3 | 500 | 0 | 16 / 16 | 500 / 500 | 0 | True |
| off #3 | 500 | 480 | 16 / 16 | 500 / 500 | 0 | True |

| Target images/s | Policy | Emitted mean | Dropped mean | Age p95 mean (ms) | Producer block mean (ms) |
|---:|---|---:|---:|---:|---:|
| 100 | block | 500.0 | 0.0 | 6.19 | 0.00 |
| 100 | drop-oldest | 500.0 | 0.0 | 6.40 | 0.00 |
| 100 | drop-newest | 500.0 | 0.0 | 6.24 | 0.00 |
| 800 | block | 500.0 | 0.0 | 466.96 | 965.20 |
| 800 | drop-oldest | 318.7 | 181.3 | 88.11 | 0.00 |
| 800 | drop-newest | 315.3 | 184.7 | 93.57 | 0.00 |
