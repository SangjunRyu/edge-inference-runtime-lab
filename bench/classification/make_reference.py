"""Generate sync Top-5 on this machine/model before a cross-machine sweep."""
import argparse
import hashlib
import json
from pathlib import Path

from bench.classification.run_sync import measure_once
from bench.measurement import env_fingerprint
from bench.runtime.backends import OnnxBackend


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='models/resnet50_224.onnx')
    p.add_argument('--source', default='data/imagenette-320-subset/manifest.json')
    p.add_argument('--frames', type=int, default=500)
    p.add_argument('--backend', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--out', default='results/local-image-reference/top5.json')
    args = p.parse_args()
    out = Path(args.out)
    if out.exists():
        p.error(f'Reference already exists: {out}; choose a new --out')
    backend = OnnxBackend(args.model, args.backend, 4)
    if backend.fell_back:
        raise RuntimeError(f'Provider unavailable: {backend.actual_providers}')
    config = argparse.Namespace(source=args.source, frames=args.frames, image_read_mode='files',
                warmup=0, warmup_s=0., duration_s=0., top_k=5)
    collected = []
    _,source,order = measure_once(backend, config, *backend.hw, collected)
    if len(collected) != args.frames or len(dict(collected)) != args.frames:
        raise RuntimeError('Missing/duplicate reference outputs')
    backend._assert_providers_intact()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(collected), encoding='utf-8')
    metadata = {'model_sha256': hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
                'source': source, 'order': order, 'backend': backend.describe(),
                'env': env_fingerprint.collect()}
    out.with_suffix('.meta.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Saved {len(collected)} sync predictions: {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
