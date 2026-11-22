"""Paired real-image runs: sync/async x file/predecoded input."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

import numpy as np

from bench.measurement import env_fingerprint, noise
from bench.classification import run_async, run_sync
from bench.runtime.backends import OnnxBackend
from bench.runtime.sources import make_source
from bench.classification.verify_equivalence import compare


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='models/resnet50_224.onnx')
    p.add_argument('--source', default='data/imagenette-320-subset/manifest.json')
    p.add_argument('--backend', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--frames', type=int, default=500)
    p.add_argument('--warmup', type=int, default=50)
    p.add_argument('--repeat', type=int, default=3)
    p.add_argument('--depths', nargs='+', type=int, default=[1, 2])
    p.add_argument('--modes', nargs='+', choices=('files', 'predecoded'),
                   default=['files', 'predecoded'])
    p.add_argument('--intra-threads', type=int, default=4)
    p.add_argument('--queue-size', type=int, default=4)
    p.add_argument('--cooldown', type=float, default=2.0)
    p.add_argument('--out-dir', default=None)
    args = p.parse_args()
    if not 0 <= args.warmup < args.frames or args.repeat < 1:
        p.error('require 0 <= warmup < frames and repeat >= 1')
    if args.intra_threads < 2 or any(d < 1 for d in args.depths) or args.queue_size < 1:
        p.error('async requires intra-threads >= 2, positive depth and queue-size')
    out = Path(args.out_dir or ('results/image-classification/' +
               datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
    out.mkdir(parents=True, exist_ok=False)
    # This checks every input, not only the first image, before timing inference.
    files = make_source(args.source, args.frames, 'files')
    memory = make_source(args.source, args.frames, 'predecoded')
    for disk, ram in zip(files.frames(), memory.frames()):
        if not np.array_equal(disk, ram):
            raise RuntimeError('file/predecoded pixel mismatch')
    source = files.describe()
    resident_bytes = memory.describe()['preloaded_bytes']
    del files, memory
    backend = OnnxBackend(args.model, args.backend, args.intra_threads)
    if backend.fell_back:
        raise RuntimeError(f'Requested {args.backend}, got {backend.actual_providers}')
    height, width = backend.hw
    jobs = []
    for rep in range(args.repeat):
        block = [(rep, mode, depth) for mode in args.modes for depth in [0] + args.depths]
        random.Random(20261122 + rep).shuffle(block)
        jobs.extend(block)
    # Establish output reference first; later sync repetitions are self-controls.
    first = (0, args.modes[0], 0)
    jobs.remove(first)
    jobs.insert(0, first)
    report = {'args': vars(args), 'created_utc': datetime.now(timezone.utc).isoformat(),
              'model_sha256': file_hash(args.model), 'source': source,
              'pixel_parity_all_images': True, 'predecoded_bytes': resident_bytes,
              'backend': backend.describe(), 'env': env_fingerprint.collect(),
              'source_sha256': {name: file_hash(Path(__file__).parents[1] / name)
                  for name in ['classification/compare_images.py', 'runtime/sources.py',
                               'runtime/pipeline.py', 'classification/run_sync.py',
                               'classification/run_async.py', 'runtime/backends.py']},
              'schedule': jobs, 'runs': []}
    reference = None
    all_ok = True
    for number, (rep, mode, depth) in enumerate(jobs, 1):
        label = 'sync' if depth == 0 else f'async{depth}'
        print(f'[{number}/{len(jobs)}] {mode} {label} repeat={rep+1}', flush=True)
        ns = argparse.Namespace(source=args.source, frames=args.frames,
             image_read_mode=mode, warmup=args.warmup, warmup_s=0.0,
             duration_s=0.0, top_k=5, inflight=depth or 1,
             queue_size=args.queue_size, reorder='on', reorder_cap=None,
             stall_every=0, stall_ms=0.0, stall_timeout_s=30.0)
        clock_before = noise.clock()
        collected = []
        recorder, src, info = (run_sync if depth == 0 else run_async).measure_once(
            backend, ns, height, width, collected)
        predictions = dict(collected)
        if reference is None:
            reference = predictions
            (out / 'reference-top5.json').write_text(json.dumps(collected), encoding='utf-8')
        comparison = compare(reference, predictions, 1e-4)
        healthy = (len(collected) == len(predictions) == args.frames and
                   comparison['identical'] and recorder.counted == args.frames-args.warmup)
        if depth:
            healthy = healthy and not info['errors'] and not info['stalls']
            healthy = healthy and info['order']['order_violations'] == 0
            healthy = healthy and info['reorder']['forced_flushes'] == 0
        backend._assert_providers_intact()
        healthy = healthy and src['input_order_sha256'] == source['input_order_sha256']
        row = {'repeat': rep+1, 'mode': mode, 'runner': label,
               'source': src, 'metrics': recorder.result(), 'info': info,
               'correctness': comparison, 'collected': len(collected),
               'unique_ids': len(predictions), 'healthy': bool(healthy),
               'cpu_clock_mhz': [clock_before, noise.clock()]}
        report['runs'].append(row)
        all_ok = all_ok and healthy
        report['passed'] = bool(all_ok)
        (out / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        m = row['metrics']
        print(f"  {m['throughput_fps']:.2f} images/s; p95 {m['e2e_ms']['p95']:.2f} ms; "
              f"{len(collected)} results; parity={comparison['identical']}; healthy={healthy}", flush=True)
        if not healthy:
            raise RuntimeError(f'Invalid run; saved evidence in {out}')
        if number < len(jobs):
            time.sleep(args.cooldown)
    lines = ['# Actual-image pipeline comparison', '',
             f'- Input: Imagenette-320 validation subset, {args.frames} unique images/run.',
             f'- Warmup: {args.warmup}; measured: {args.frames-args.warmup}; repeats: {args.repeat}.',
             '- Batch 1, identical manifest order, same preprocessing, Top-5 collection.',
             '- File mode includes read+decode with OS-managed warm cache; no cold-disk claim.',
             '- Predecoded mode yields read-only references; preload excluded from timing.',
             '- Async inference time is submit-to-callback latency, including runtime waiting.',
             '', '| Input | Runner | Images/s mean [min, max] | E2E p95 ms mean [min, max] | Value parity |',
             '|---|---|---:|---:|---|']
    for mode in args.modes:
        for runner in ['sync'] + [f'async{d}' for d in args.depths]:
            rows = [r for r in report['runs'] if r['mode'] == mode and r['runner'] == runner]
            rate = [r['metrics']['throughput_fps'] for r in rows]
            tail = [r['metrics']['e2e_ms']['p95'] for r in rows]
            fmt = lambda v: f'{statistics.mean(v):.2f} [{min(v):.2f}, {max(v):.2f}]'
            lines.append(f'| {mode} | {runner} | {fmt(rate)} | {fmt(tail)} | all pass |')
    (out / 'SUMMARY.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'Completed: {out.resolve()}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
