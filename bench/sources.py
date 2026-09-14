"""Frame sources.

``Read`` is a measured stage, so decoding must be real work on every frame.
Decoding once and replaying the decoded array would delete the stage.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class VideoSource:
    """Decodes frames from a video file, looping until ``n`` are produced.

    Looping costs a seek back to frame 0 and a keyframe re-decode, which lands
    in the ``read`` stage as a single large outlier. That is real work, so it is
    not hidden -- but it is an artefact of the clip being shorter than the run,
    not of the pipeline. ``restarts`` is reported with the result so the outlier
    in ``read.max`` is attributable instead of mysterious. Prefer a clip with at
    least ``--frames`` frames and keep restarts at 0.
    """

    kind = "video"

    def __init__(self, path: str, n: int):
        self.path, self.n = str(path), n
        self.restarts = 0
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def frames(self):
        produced = 0
        while produced < self.n:
            ok, frame = self.cap.read()
            if not ok:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.restarts += 1
                ok, frame = self.cap.read()
                if not ok:
                    raise RuntimeError("video produced no frames")
            produced += 1
            yield frame
        self.cap.release()

    def describe(self) -> dict:
        return {"kind": self.kind, "path": self.path,
                "frame_count": self.frame_count, "fps": self.fps,
                "width": self.width, "height": self.height,
                "restarts": self.restarts}


class ImageSource:
    """Re-decodes one image file per frame from its in-memory bytes.

    Re-encoding is avoided but the JPEG/PNG decode happens every iteration, so
    the Read stage measures decode cost with no disk noise.
    """

    kind = "image"

    def __init__(self, path: str, n: int):
        self.path, self.n = str(path), n
        self._buf = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
        if cv2.imdecode(self._buf, cv2.IMREAD_COLOR) is None:
            raise RuntimeError(f"cannot decode image: {path}")

    def frames(self):
        for _ in range(self.n):
            yield cv2.imdecode(self._buf, cv2.IMREAD_COLOR)

    def describe(self) -> dict:
        return {"kind": self.kind, "path": self.path, "restarts": 0}


class SyntheticSource:
    """Random frames, no decode. Use to isolate the cost of Read itself."""

    kind = "synthetic"

    def __init__(self, n: int, height: int = 1080, width: int = 1920, seed: int = 0):
        self.n, self.h, self.w = n, height, width
        self._rng = np.random.default_rng(seed)
        self._base = self._rng.integers(0, 255, (height, width, 3), dtype=np.uint8)

    def frames(self):
        for _ in range(self.n):
            yield self._base.copy()

    def describe(self) -> dict:
        # No decode happens here: `read` measures the copy of an h x w x 3
        # uint8 array and nothing else. Never use this as the baseline for a
        # claim about decode cost.
        return {"kind": self.kind, "width": self.w, "height": self.h,
                "restarts": 0}


def make_source(source: str | None, n: int):
    if source in (None, "", "synthetic"):
        return SyntheticSource(n)
    ext = Path(source).suffix.lower()
    if ext in {".mp4", ".avi", ".mkv", ".mov", ".webm"}:
        return VideoSource(source, n)
    return ImageSource(source, n)
