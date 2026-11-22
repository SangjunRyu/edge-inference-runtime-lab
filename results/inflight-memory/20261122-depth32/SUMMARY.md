# Fresh-process inflight sweep

File+decode mode, same image manifest/reference; batch 1; fresh session/process per depth.
VRAM is NVML device-wide reserved/allocated usage sampled at 100ms, not live activations.
inflight is outstanding requests, not proven concurrent GPU executions.

| Depth | Images/s mean [min, max] | E2E p95 mean ms | Device VRAM peak MiB | Minimum free MiB | RSS peak MiB |
|---|---:|---:|---:|---:|---:|
| 0 | 29.71 [24.49, 34.09] | 41.01 | 389.7 | 1658.3 | 653.5 |
| 1 | 28.66 [27.75, 29.79] | 309.64 | 427.7 | 1620.3 | 563.1 |
| 2 | 31.39 [30.78, 32.61] | 366.17 | 427.7 | 1620.3 | 563.1 |
| 4 | 22.54 [14.49, 31.76] | 800.26 | 693.7 | 1354.3 | 574.2 |
| 8 | 19.78 [16.24, 24.22] | 1462.98 | 435.7 | 1612.3 | 567.8 |
| 16 | 33.52 [32.09, 34.77] | 1179.89 | 691.7 | 1356.3 | 581.7 |
| 32 | 35.07 [33.17, 38.06] | 1885.47 | 695.7 | 1352.3 | 591.9 |
