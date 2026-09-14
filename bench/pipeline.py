"""Pre/post-processing for ImageNet classifiers.

Kept deliberately plain. W1 measures where time goes in the obvious
implementation; optimising these is W4's job, and an optimisation with no
baseline is just an assertion.
"""
from __future__ import annotations

import cv2
import numpy as np

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(frame_bgr: np.ndarray, height: int, width: int) -> np.ndarray:
    """BGR uint8 HWC -> normalised float32 NCHW."""
    img = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, 0)


def postprocess(logits: np.ndarray, top_k: int = 5) -> list[tuple[int, float]]:
    """Logits -> top-k (index, probability).

    top_k defaults to 5 to match DX-APP's resnet50 config.json, so the
    postprocess stage is doing the same amount of work as the reference.
    """
    x = logits.reshape(-1).astype(np.float32)
    x = x - x.max()
    e = np.exp(x)
    probs = e / e.sum()
    idx = np.argpartition(-probs, top_k)[:top_k]
    idx = idx[np.argsort(-probs[idx])]
    return [(int(i), float(probs[i])) for i in idx]
