"""Frame sources.

``Read`` is a measured stage, so decoding must be real work on every frame.
Decoding once and replaying the decoded array would delete the stage.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import time

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


class ImageListSource:
    """A finite, immutable manifest order, with file or predecoded input.

    Predecoded frames are read-only: Read yields references, not synthetic
    full-frame copies. All existing preprocessing starts by allocating a resize.
    Loading/hash verification/predecode finish before any runner starts timing.
    """

    kind = 'image_list'

    def __init__(self, manifest: str, n: int, mode: str = 'files'):
        if mode not in {'files', 'predecoded'}:
            raise ValueError(f'Unknown image read mode: {mode}')
        self.path = Path(manifest).resolve()
        raw = self.path.read_bytes()
        obj = json.loads(raw)
        if obj.get('schema') != 'image-classification-list/v1':
            raise ValueError('Not an image-classification-list/v1 manifest')
        self.entries = obj['images']
        if not 0 < n <= len(self.entries):
            raise ValueError(f'--frames must be 1..{len(self.entries)}; images do not loop')
        self.entries = self.entries[:n]
        self.paths = [(self.path.parent / e['path']).resolve() for e in self.entries]
        for entry, path in zip(self.entries, self.paths):
            if not path.is_relative_to(self.path.parent):
                raise ValueError(f'Image path outside manifest directory: {path}')
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
                raise ValueError(f'Image checksum mismatch: {path}')
        self.mode, self.n = mode, n
        self.manifest_sha = hashlib.sha256(raw).hexdigest()
        self.order_sha = hashlib.sha256(json.dumps(
            [(e['path'], e['sha256']) for e in self.entries],
            separators=(',', ':')).encode()).hexdigest()
        started = time.perf_counter()
        self.decoded = []
        if mode == 'predecoded':
            for path in self.paths:
                frame = self._decode(path)
                frame.flags.writeable = False
                self.decoded.append(frame)
        self.preload_s = time.perf_counter() - started

    @staticmethod
    def _decode(path):
        # imdecode handles non-ASCII Windows paths and makes file I/O explicit.
        frame = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f'Cannot decode image: {path}')
        return frame

    def frames(self):
        if self.mode == 'predecoded':
            yield from self.decoded
        else:
            for path in self.paths:
                yield self._decode(path)

    def describe(self):
        return {'kind': self.kind, 'path': str(self.path), 'mode': self.mode,
                'count': self.n, 'manifest_sha256': self.manifest_sha,
                'input_order_sha256': self.order_sha, 'restarts': 0,
                'read_boundary': ('file_read_and_decode' if self.mode == 'files'
                                  else 'predecoded_reference'),
                'preload_s_excluded': self.preload_s,
                'preloaded_bytes': sum(f.nbytes for f in self.decoded),
                'file_cache': 'OS-managed; checksum validation reads files before timing'}


def make_source(source: str | None, n: int, image_read_mode: str = 'files'):
    if source in (None, "", "synthetic"):
        return SyntheticSource(n)
    ext = Path(source).suffix.lower()
    if ext == '.json':
        return ImageListSource(source, n, image_read_mode)
    if ext in {".mp4", ".avi", ".mkv", ".mov", ".webm"}:
        return VideoSource(source, n)
    return ImageSource(source, n)
