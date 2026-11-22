"""Download official Imagenette-320; freeze a balanced validation subset."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import tarfile
import urllib.request
from pathlib import Path

URL = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz'
CLASSES = {
    'n01440764': ('tench', 0), 'n02102040': ('English springer', 217),
    'n02979186': ('cassette player', 482), 'n03000684': ('chain saw', 491),
    'n03028079': ('church', 497), 'n03394916': ('French horn', 566),
    'n03417042': ('garbage truck', 569), 'n03425413': ('gas pump', 571),
    'n03445777': ('golf ball', 574), 'n03888257': ('parachute', 701),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='data/imagenette-320-subset')
    parser.add_argument('--per-class', type=int, default=50)
    parser.add_argument('--seed', type=int, default=20261122)
    args = parser.parse_args()
    if args.per_class < 1:
        parser.error('--per-class must be positive')
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    archive = out.parent / 'cache' / 'imagenette2-320.tgz'
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        partial = archive.with_suffix('.tgz.part')
        print(f'Downloading official archive: {URL}', flush=True)
        with urllib.request.urlopen(URL, timeout=60) as src, partial.open('wb') as dst:
            expected = int(src.headers.get('Content-Length', 0))
            count = 0
            progress = 0
            while block := src.read(1024 * 1024):
                dst.write(block)
                count += len(block)
                if count // (32 * 1024 * 1024) > progress:
                    progress = count // (32 * 1024 * 1024)
                    print(f'  {count / 1024**2:.0f} MiB', flush=True)
        if expected and count != expected:
            raise RuntimeError(f'Incomplete archive: {count} != {expected}')
        # Reading the members below also verifies that this is a valid tar stream.
        partial.replace(archive)
    archive_sha = digest(archive)
    rng = random.Random(args.seed)
    selected = []
    with tarfile.open(archive, 'r:gz') as tar:
        groups = {key: [] for key in CLASSES}
        for member in tar.getmembers():
            parts = member.name.split('/')
            if (member.isfile() and len(parts) == 4 and parts[1] == 'val'
                    and parts[2] in groups and Path(parts[3]).suffix.lower() in {'.jpg', '.jpeg', '.png'}):
                groups[parts[2]].append(member)
        for synset, members in sorted(groups.items()):
            if len(members) < args.per_class:
                raise RuntimeError(f'{synset}: only {len(members)} validation images')
            chosen = rng.sample(sorted(members, key=lambda m: m.name), args.per_class)
            for member in chosen:
                target = out / 'images' / synset / Path(member.name).name
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as f:
                    content = f.read()
                if not target.exists() or target.read_bytes() != content:
                    target.write_bytes(content)
                selected.append({'path': target.relative_to(out).as_posix(),
                                 'synset': synset, 'class_name': CLASSES[synset][0],
                                 'imagenet_index': CLASSES[synset][1],
                                 'sha256': hashlib.sha256(content).hexdigest(),
                                 'archive_member': member.name})
    rng.shuffle(selected)
    manifest = {'schema': 'image-classification-list/v1', 'dataset': 'Imagenette',
                'variant': '320px', 'split': 'val', 'seed': args.seed,
                'images_per_class': args.per_class, 'count': len(selected),
                'source_url': URL, 'source_archive_sha256': archive_sha,
                'homepage': 'https://github.com/fastai/imagenette',
                'images': selected}
    path = out / 'manifest.json'
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Prepared {len(selected)} images: {path}', flush=True)
    print(f'Manifest SHA-256: {digest(path)}', flush=True)


if __name__ == '__main__':
    main()
