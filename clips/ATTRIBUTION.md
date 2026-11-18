# Test clips

Big Buck Bunny (c) copyright 2008, Blender Foundation — https://peach.blender.org/
Licensed CC-BY 3.0. Re-encoded variants fetched from https://test-videos.co.uk/.

Not committed to this repository (see `.gitignore`). Reproduce with:

    python -m bench.data.get_clips          # 360p / 720p / 1080p
    python -m bench.data.get_clips --all    # full 3 x 6 resolution/bitrate grid

Every clip is 10 s, 300 frames, 30 fps, H.264.

## Actual pedestrian footage

`real-video/vtest.avi` is downloaded locally from OpenCV's public sample:
https://github.com/opencv/opencv/blob/4.x/samples/data/vtest.avi

Fetch and verify: `python -m bench.data.get_detection_video`.
SHA256: `45cddc9490be69345cbdab64ca583be65987e864ca408038e648db99e10516cf`.
768x576, 10 FPS, 795 frames. No footage is committed or redistributed here.
The source is a pedestrian scene, not a factory/conveyor dataset.
