"""Inference backends.

One ONNX model, several execution providers. ONNX Runtime is the single API so
that swapping CPU <-> CUDA is a flag and not a rewrite -- the point of the
comparison is the *pipeline*, and the pipeline must stay identical.

CUDAExecutionProvider on a Pascal card (MX150, sm_61) may simply not be
available depending on the onnxruntime-gpu / CUDA runtime combination. That is
a normal outcome, not a failure: ``available()`` reports it and the run
continues on CPU. Record which provider actually served the session -- asking
for CUDA and silently getting CPU is the classic way to publish a wrong number.
"""
from __future__ import annotations

import numpy as np

BACKENDS = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    # OpenVINO is optional and only reaches Intel CPU/iGPU, never the MX150.
    "openvino": ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
}


def _enable_openvino_libs() -> None:
    """Put the OpenVINO DLLs on the search path.

    `onnxruntime-openvino` does not bundle the OpenVINO runtime; it loads
    `openvino.dll` from the separately installed `openvino` wheel, whose
    binaries live in `site-packages/openvino/libs` and are not on PATH. Without
    this the provider fails to load and ONNX Runtime silently falls back to CPU
    -- which would publish a CPU number under a GPU or NPU label.
    """
    try:
        from openvino import utils
        utils.add_openvino_libs_to_path()
    except Exception:
        pass


def available() -> list[str]:
    import onnxruntime as ort
    _enable_openvino_libs()
    have = set(ort.get_available_providers())
    return [name for name, prov in BACKENDS.items() if prov[0] in have]


class OnnxBackend:
    """Thin wrapper: one session, one input, one output."""

    def __init__(self, model_path: str, backend: str = "cpu",
                 intra_threads: int | None = None, ov_device: str | None = None):
        import onnxruntime as ort

        if backend not in BACKENDS:
            raise ValueError(f"unknown backend {backend!r}; pick from {list(BACKENDS)}")
        if backend == "openvino":
            _enable_openvino_libs()

        opts = ort.SessionOptions()
        # Pin thread counts. Left at the default, ORT sizes its pool from the
        # machine and the CPU-stage timings stop being comparable between runs.
        if intra_threads is not None:
            opts.intra_op_num_threads = intra_threads
            opts.inter_op_num_threads = 1

        providers: list = list(BACKENDS[backend])
        if backend == "openvino" and ov_device:
            # OpenVINO reaches Intel CPU / iGPU / NPU. device_type picks which;
            # without it the plugin chooses and the result says nothing about
            # which unit actually ran the model.
            providers[0] = (providers[0], {"device_type": ov_device.upper()})

        self.requested = backend
        self.ov_device = ov_device.upper() if (backend == "openvino" and ov_device) else None
        self.session = ort.InferenceSession(
            model_path, sess_options=opts, providers=providers
        )
        self.actual_providers = self.session.get_providers()
        self.fell_back = BACKENDS[backend][0] not in self.actual_providers

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_shape = inp.shape          # e.g. [1, 3, 224, 224]
        self.output_name = self.session.get_outputs()[0].name

    @property
    def hw(self) -> tuple[int, int]:
        """(height, width) the model expects, from the ONNX input shape."""
        h, w = self.input_shape[2], self.input_shape[3]
        return int(h), int(w)

    def run(self, tensor: np.ndarray) -> np.ndarray:
        return self.session.run([self.output_name], {self.input_name: tensor})[0]

    def describe(self) -> dict:
        return {
            "requested_backend": self.requested,
            "openvino_device": self.ov_device,
            "actual_providers": self.actual_providers,
            "fell_back_to_cpu": self.fell_back,
            "input_shape": [str(d) for d in self.input_shape],
        }
