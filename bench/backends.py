"""Inference backends.

One ONNX model, several execution providers. ONNX Runtime is the single API so
that swapping CPU <-> CUDA is a flag and not a rewrite -- the point of the
comparison is the *pipeline*, and the pipeline must stay identical.

CUDAExecutionProvider on a Pascal card (MX150, sm_61) may simply not be
available depending on the onnxruntime-gpu / CUDA runtime combination. That is
a normal outcome, not a failure: ``available()`` reports it and the run
continues on CPU. Record which provider actually served the session -- asking
for CUDA and silently getting CPU is the classic way to publish a wrong number.

``dml`` is the same device reached over DirectML instead of CUDA. It needs no
CUDA runtime and no cuDNN, so on a machine where the CUDA EP will not load it is
the remaining way to put inference on a discrete GPU -- which is the property
the whole comparison rests on. It is a different engine as well as a different
path to the same silicon, so a dml row and a cuda row are not interchangeable
evidence about the device.

Two separate things are called "fallback" here and only one of them is wanted.
Listing ``CPUExecutionProvider`` behind the accelerator is graph partitioning:
an op the accelerator has no kernel for runs on the host, which is normal and
is what the provider list is for. The other is ONNX Runtime's Python wrapper
catching an execution-time EP failure inside ``run()``, swapping the whole
session to CPU and retrying -- once, with two lines on stdout. That one turns a
dead accelerator into a complete, plausible, entirely CPU benchmark filed under
the accelerator's name, so it is disabled for every non-CPU backend.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# os.add_dll_directory returns a handle that removes the directory again when
# it is closed or collected, so the handles have to outlive the call.
_DLL_DIR_HANDLES: list = []

BACKENDS = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    # Any DX12 GPU, from the onnxruntime-directml wheel. Windows only.
    "dml": ["DmlExecutionProvider", "CPUExecutionProvider"],
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


def _enable_cuda_libs(ort) -> None:
    """Make pip-installed CUDA and cuDNN DLLs findable, and keep them findable.

    ``preload_dlls`` LoadLibrary's a fixed list, which covers what ONNX Runtime
    links directly. cuDNN 9 is not fully covered by it: further sublibraries are
    loaded lazily at the first convolution rather than at session creation, and
    ``cudnn_engines_tensor_ir64_9.dll`` is not on the list ORT preloads. By then
    only the Windows DLL search path is consulted, and
    ``site-packages/nvidia/*/bin`` is not on it.

    Measured on the 550XBE with onnxruntime-gpu 1.26.0 and the pip cuDNN 9.26:
    the session created on CUDA, then the first Conv failed with
    ``CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`` after
    ``Could not locate cudnn_engines_tensor_ir64_9.dll``.

    ``directory=""`` rather than the default: the default prefers a PyTorch
    install's own CUDA libraries, and this project pins its CUDA deliberately.
    """
    try:
        ort.preload_dlls(directory="")
    except Exception:
        pass

    if os.name != "nt":
        return
    nvidia_root = Path(ort.__file__).resolve().parent.parent / "nvidia"
    for bin_dir in sorted(nvidia_root.glob("*/bin")):
        try:
            _DLL_DIR_HANDLES.append(os.add_dll_directory(str(bin_dir)))
        except OSError:
            pass


def available() -> list[str]:
    import onnxruntime as ort
    _enable_openvino_libs()
    have = set(ort.get_available_providers())
    return [name for name, prov in BACKENDS.items() if prov[0] in have]


class OnnxBackend:
    """Thin wrapper: one session, one input, one output."""

    def __init__(self, model_path: str, backend: str = "cpu",
                 intra_threads: int | None = None, ov_device: str | None = None,
                 allow_fallback: bool = False):
        import onnxruntime as ort

        if backend not in BACKENDS:
            raise ValueError(f"unknown backend {backend!r}; pick from {list(BACKENDS)}")
        if backend == "openvino":
            _enable_openvino_libs()
        if backend == "cuda":
            _enable_cuda_libs(ort)

        opts = ort.SessionOptions()
        # Pin thread counts. Left at the default, ORT sizes its pool from the
        # machine and the CPU-stage timings stop being comparable between runs.
        if intra_threads is not None:
            opts.intra_op_num_threads = intra_threads
            opts.inter_op_num_threads = 1

        if backend == "dml":
            # DirectML supports neither the memory-pattern planner nor the
            # parallel execution mode. Left on, they are a documented way to get
            # a crash or a wrong answer rather than a slow one.
            opts.enable_mem_pattern = False
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        providers: list = list(BACKENDS[backend])
        if backend == "openvino" and ov_device:
            # OpenVINO reaches Intel CPU / iGPU / NPU. device_type picks which;
            # without it the plugin chooses and the result says nothing about
            # which unit actually ran the model.
            providers[0] = (providers[0], {"device_type": ov_device.upper()})

        self.requested = backend
        self.allow_fallback = allow_fallback
        self.ov_device = ov_device.upper() if (backend == "openvino" and ov_device) else None
        self.session = ort.InferenceSession(
            model_path, sess_options=opts, providers=providers
        )

        # ONNX Runtime's Python InferenceSession.run() and run_async() catch an
        # EP failure, call set_providers(["CPUExecutionProvider"]) and retry the
        # call. The measurement then continues to completion, on the CPU, under
        # this backend's name -- the failure class this project keeps meeting,
        # where the count is right and nothing looks wrong. An accelerator that
        # dies mid-run is a failed run, not a slow one.
        if backend != "cpu":
            self.session.disable_fallback()

        self.actual_providers = self.session.get_providers()
        self.fell_back = BACKENDS[backend][0] not in self.actual_providers
        # Checked once, after the first inference: a swap can only show up then.
        self._check_pending = backend != "cpu"

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
        out = self.session.run([self.output_name], {self.input_name: tensor})[0]
        if self._check_pending:
            self._check_pending = False
            self._assert_providers_intact()
        return out

    def _assert_providers_intact(self) -> None:
        """Belt to disable_fallback's braces.

        The providers recorded at construction are what the session *started*
        with. Anything that swaps them later -- a future ONNX Runtime, a path
        this version does not take -- would otherwise leave the swap invisible
        in the result file, because nothing reads them again.
        """
        now = self.session.get_providers()
        if now == self.actual_providers:
            return
        self.actual_providers = now
        self.fell_back = BACKENDS[self.requested][0] not in now
        if self.fell_back and not self.allow_fallback:
            raise RuntimeError(
                f"{self.requested!r} collapsed to {now} during execution. "
                "Every frame after that point is a CPU frame under an "
                "accelerator's label; refusing to keep measuring."
            )

    def describe(self) -> dict:
        return {
            "requested_backend": self.requested,
            "openvino_device": self.ov_device,
            "actual_providers": self.actual_providers,
            "fell_back_to_cpu": self.fell_back,
            "input_shape": [str(d) for d in self.input_shape],
        }
