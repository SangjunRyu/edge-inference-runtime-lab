# Clock-driven inspection and counting — recorded evidence

Measured on **2026-11-18**: i7-8565U, GeForce MX150 2GB, Windows, ORT 1.26.0.
These are virtual-camera workloads with real CPU/CUDA inference. They are not
factory accuracy tests, DEEPX hardware results or a Python/C++ speed comparison.

Read the [Korean findings](../../docs-ko/10-industrial-scenarios.md) for the
analysis and the [scenario guide](../../docs/industrial-scenarios.md) to rerun.

## What is included

| File | Contents |
|---|---|
| [summary.csv](summary.csv), [summary.json](summary.json) | 21 conditions, two runs each; counts, drop rates, deadline coverage, age p95 |
| [protocol.json](protocol.json) | Environment, source/model hashes, plan and correction history |
| [request-traces.zip](request-traces.zip) | All 42 validated per-request traces, value checks and failure diagnostics |
| [model-provenance.json](model-provenance.json) | Model file sizes and SHA-256 hashes |
| [model-verification.json](model-verification.json) | Three 64-frame CUDA sync/async checks with the shared-stream setting |
| [count-event-validation.json](count-event-validation.json) | 78 matched crossing events across 26 counting runs; FP 0 / FN 0 in this surrogate scene |
| [concurrency-incident.json](concurrency-incident.json) | Controls that narrowed the CUDA output-coordinate mismatch |
| [initial-bad-box-diagnostic.json](initial-bad-box-diagnostic.json) | Example wrong boxes from the rejected configuration |
| [manifest.json](manifest.json) | Checksums for the published data files |

The final set combines 26 unchanged CPU/trigger runs from the initial phase
with **16 remeasured CUDA-stream runs**. Initial CUDA-stream timings were
invalidated after output-value checks failed; they do not contribute to the
summary. Their diagnostics remain available so the rejection can be inspected.

Only local absolute filesystem paths were made relative in the public copies.
Measured numbers, IDs, timestamps, hashes and validation outcomes were retained.
This directory disables Git line-ending normalization so the checksums in
`manifest.json` remain valid after publishing and checkout.
Raw model weights, third-party photographs, rendered videos, SDKs and virtual
environments are excluded. Generate scene assets and videos with the linked
tools instead.

## How to interpret the figures

* Output age starts at the **scheduled acquisition time**, including source
  lateness and all queueing. It is not just the inference-call duration.
* Deadline coverage uses **all offered inputs** as its denominator, including
  dropped work. A fresh surviving result does not mean every product was seen.
* The 200ms trigger and 100ms stream deadlines are illustrative assumptions.
  The 120 FPS stream is an intentional virtual-camera overload condition.
* Two repetitions are exploratory evidence, not a confidence interval or a
  production SLA. Idle-gate outcomes and source lateness remain in the traces.
* Matching three surrogate crossing events does not establish industrial
  product-detection accuracy or actuator timing. The event matcher uses a
  0.5-second tolerance and 1/30-second ground-truth time resolution.

## Correctness before speed

The default concurrent YOLO CUDA configuration intermittently changed box
coordinates even when frame order and total crossing count looked correct.
The scenario runner explicitly selects `use_ep_level_unified_stream=1` after
the documented controls and value checks. This is a measured containment for
this model/runtime/driver/GPU combination, not proof of a universal library
root cause. See [the runnable reproduction](../../repro/yolo_cuda_concurrency.py).
