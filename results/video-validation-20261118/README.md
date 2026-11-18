# Actual-video validation after module reorganization

[Summary](summary.json) records CPU/CUDA x sync/async runs on the first 300
frames of OpenCV vtest.avi, plus a separate 64-frame CUDA value/profile check
sampled across the entire source. The source and model hashes identify inputs.
All four output videos were decoded back and had 300 frames in source order.
Within each backend, sync/async boxes, scores and centroid IDs matched within
the stated tolerances. Profiling recorded CUDA node events and no CPU node
events for the profiled calls. This describes those calls, not every model.

These are single runs for functional validation, not a controlled performance
comparison. No detection/tracking ground-truth accuracy was measured. Model
weights, input footage, output videos and absolute local paths are not included.
See [the guide](../../docs-ko/11-real-video.md) for execution and limitations.
