# Benchmark modules

Run commands from the repository root with `python -m bench.<group>.<module>`.
The former flat `bench.run_sync`-style paths have moved; examples throughout
the repository use the new paths.

| Directory | Responsibility | Entry points / modules |
|---|---|---|
| `runtime/` | Shared inference and frame processing | `backends`, `pipeline`, `sources`, `reorder` |
| `measurement/` | Environment, noise control and metrics | `env_fingerprint`, `noise`, `harness`, `statistics` |
| `classification/` | ResNet sync/async baselines and value checks | `run_sync`, `run_async`, `verify_equivalence` |
| `detection/` | Actual-video YOLO execution and verification | `run_video`, `verify`, `yolo` |
| `experiments/` | Paired comparisons, matrix and depth sweeps | `run_ab`, `run_matrix`, `run_sweep`, `compare_*` |
| `analysis/` | Saved-result summaries and decode diagnostics | `summarize`, `read_rate` |
| `data/` | Download inputs and export classification models | `get_clips`, `get_detection_video`, `prepare_models` |
| `workloads/` | Reusable source-clock/admission protocol | `clocked` (library; no synthetic image generator) |

Start with [the actual-video guide](../docs-ko/11-real-video.md).
The executable synthetic-bus scenario and its asset preparation/rendering tools
have been removed. Its published measurements remain historical evidence;
they are not real-footage validation results.
