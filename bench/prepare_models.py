"""Export torchvision classifiers to ONNX at several input resolutions.

Model choice and input resolution are the *balance axis* of this project: they
move where time is spent between the host stages and inference. Exporting a
grid up front means W2 can sweep that axis without touching code.

    python -m bench.prepare_models --models resnet50 resnet18 mobilenet_v3_small \
        --sizes 224 160 128
"""
from __future__ import annotations

import argparse
from pathlib import Path

BUILDERS = {
    "resnet50": ("ResNet50_Weights", "resnet50"),
    "resnet18": ("ResNet18_Weights", "resnet18"),
    "mobilenet_v3_small": ("MobileNet_V3_Small_Weights", "mobilenet_v3_small"),
}


def export(name: str, size: int, out_dir: Path, opset: int = 13) -> Path:
    import torch
    import torchvision.models as tvm

    weights_attr, fn_name = BUILDERS[name]
    weights = getattr(tvm, weights_attr).DEFAULT
    model = getattr(tvm, fn_name)(weights=weights).eval()

    out = out_dir / f"{name}_{size}.onnx"
    dummy = torch.randn(1, 3, size, size)
    # Static shapes on purpose: dynamic axes let the runtime re-plan per call,
    # which would show up as inference-time variance that has nothing to do
    # with the pipeline.
    #
    # dynamo=False pins the TorchScript exporter. Since torch 2.6 the default
    # is the dynamo exporter, which decomposes operators differently and
    # changes between torch releases -- so the same source model would produce
    # a different graph, and "inference time" would stop meaning one thing
    # across runs. Pin the path rather than inherit a moving default.
    kwargs = dict(
        input_names=["input"], output_names=["logits"],
        opset_version=opset, do_constant_folding=True,
    )
    try:
        torch.onnx.export(model, dummy, str(out), dynamo=False, **kwargs)
    except TypeError:
        # torch < 2.5 has no `dynamo` kwarg; it is the TorchScript path anyway.
        torch.onnx.export(model, dummy, str(out), **kwargs)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["resnet50"], choices=list(BUILDERS))
    ap.add_argument("--sizes", nargs="+", type=int, default=[224])
    ap.add_argument("--out-dir", default="models")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        for size in args.sizes:
            path = export(name, size, out_dir)
            print(f"{path}  ({path.stat().st_size / 1024**2:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
