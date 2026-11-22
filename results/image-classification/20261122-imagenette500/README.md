# Imagenette ResNet50 evidence, 2026-11-22

`results.json` contains the original 18 runs: files/predecoded x sync/async1/async2 x 3 repeats. Each run processes 500 unique images, with 50 warmup images excluded from timing. All 500 outputs are compared against the sync reference.

`input-manifest.json` freezes selection, order and image hashes. It is an archived manifest, not the execution source path: run the downloader to recreate `data/imagenette-320-subset/manifest.json` and its adjacent images. JPEGs, the 326MiB archive, ONNX weights and venvs are not committed.

The raw evidence was collected before the upstream module reorganization was integrated. Its original paths, environment, timestamps and source hashes are preserved. Current commands live under `bench.classification`, with shared input/backend code under `bench.runtime`. After integration, nine unit checks and short real CUDA sync/async, local-reference, subprocess-memory and value-equivalence checks passed. Short checks are installation/integration checks, not new performance measurements.

The later request-depth sweep is in `results/inflight-memory/20261122-depth32/`; the interleaved GPU-clock/thermal investigation is in `results/inflight-investigation/20261122-interleaved/`. Thermal slowdown and clock changes were observed in the investigation. Do not treat the earlier sequential sweep as a controlled performance ranking.

Start on another machine with [the reproduction guide](../../../docs-ko/cross-machine-images.md).
