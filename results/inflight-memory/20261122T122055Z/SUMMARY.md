# Fresh-process inflight sweep

File+decode mode, same image manifest/reference; batch 1; fresh session/process per depth.
VRAM is NVML device-wide reserved/allocated usage sampled at 100ms, not live activations.
inflight is outstanding requests, not proven concurrent GPU executions.

| Depth | Images/s mean [min, max] | E2E p95 mean ms | Device VRAM peak MiB | Minimum free MiB | RSS peak MiB |
|---|---:|---:|---:|---:|---:|
| 0 | 255.00 [252.81, 257.08] | 4.52 | 1028.6 | 7159.4 | 875.3 |
| 1 | 371.90 [364.56, 380.60] | 20.29 | 1036.6 | 7151.4 | 881.1 |
| 2 | 423.15 [412.56, 428.52] | 21.38 | 1058.5 | 7129.5 | 884.9 |
| 4 | 446.45 [428.92, 456.54] | 27.79 | 1073.0 | 7115.0 | 886.8 |
| 8 | 475.01 [440.27, 492.92] | 45.79 | 1077.8 | 7110.2 | 892.5 |
| 16 | 493.83 [487.16, 497.58] | 80.70 | 1062.4 | 7125.6 | 901.8 |
| 32 | 495.33 [487.19, 503.59] | 142.14 | 1062.4 | 7125.6 | 923.2 |
