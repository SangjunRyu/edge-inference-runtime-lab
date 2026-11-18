# Actual video detection demo

The source is OpenCV's real pedestrian footage, not the translated still image
used by the retired controlled conveyor surrogate. Its executable code has been removed:
https://github.com/opencv/opencv/blob/4.x/samples/data/vtest.avi

Download (PowerShell, project root):

```powershell
python -m bench.data.get_detection_video
```

Downloaded SHA256: `45cddc9490be69345cbdab64ca583be65987e864ca408038e648db99e10516cf`.
The upstream sample is kept locally; this change does not redistribute it.
This is a pedestrian scene, not evidence of product detection accuracy.

```powershell
cd C:\Users\ysjun5656\deepX\DEEPX-poc
.\.venv-cuda\Scripts\Activate.ps1
python -m bench.detection.run_video --backend cuda --mode async --inflight 2 --out-dir results/real-video-cuda
Invoke-Item .\results\real-video-cuda\detection.mp4
```

Left: original frame. Right: YOLO11n person detections and simple centroid IDs.
The input is 768x576 at 10 FPS; the detector uses letterboxed 320x320 input.
Use `--display` for an OpenCV preview window (q or ESC closes it), or replace
`--video` with another local video. Only COCO person class 0 is selected.
CPU: replace `--backend cuda` with `--backend cpu` and use a separate output
directory. Synchronous control: `--mode sync`. Limit duration: `--max-frames 300`.
Without activation, replace `python` with `.\.venv-cuda\Scripts\python.exe`.

The async path uses ORT `run_async`, bounds outstanding frames, and restores
frame order before tracking/encoding. The CUDA session uses the previously
validated `use_ep_level_unified_stream=1` workaround. It does not imply
simultaneous GPU kernel execution. `--verify-every 30` compares sampled raw
outputs with synchronous inference and fails on a mismatch.

This runner processes a file as fast as possible, preserves every frame, and
writes the result at source FPS. It is not a clock-driven live-camera overload
test. Reported processing FPS includes decoding, preprocessing, postprocessing
and MP4 encoding, but excludes session startup/warmup. Preview adds playback
waits and verification adds extra inference. Callback latency includes runtime
queueing, transfers and execution. IDs are a visualization aid: occlusion and
nearby pedestrians can cause ID switches. No labeled detection/tracking
accuracy or product counts are claimed.

For request limits versus batching and video/camera input semantics, see the
[Korean explanation](../docs-ko/11-real-video.md) and [module map](../bench/README.md).
