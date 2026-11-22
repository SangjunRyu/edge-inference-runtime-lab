"""Interleaved fresh-process runs to check the anomalous depth-4/8 slowdown."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames', type=int, default=200)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--cooldown', type=float, default=5)
    p.add_argument('--model', default='models/resnet50_224.onnx')
    p.add_argument('--source', default='data/imagenette-320-subset/manifest.json')
    p.add_argument('--reference', default='results/image-classification/20261122-imagenette500/reference-top5.json')
    p.add_argument('--intra-threads', type=int, default=4)
    p.add_argument('--queue-size', type=int, default=4)
    p.add_argument('--out-dir', default=None)
    args = p.parse_args()
    out = Path(args.out_dir or ('results/inflight-investigation/' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
    out.mkdir(parents=True, exist_ok=False)
    schedule = [[4,2,16,8], [8,16,2,4], [16,4,8,2]]
    (out/'schedule.json').write_text(json.dumps({'args': vars(args), 'schedule': schedule}, indent=2))
    done = []
    for block, depths in enumerate(schedule, 1):
        for depth in depths:
            job = out/f'round-{block}-depth-{depth}'
            job.mkdir()
            print(f'round={block} depth={depth}', flush=True)
            cmd = [sys.executable, '-m', 'bench.classification.sweep_inflight', '--worker', str(depth),
                   '--out-dir', str(job), '--frames', str(args.frames),
                   '--warmup', str(args.warmup), '--repeat', '1',
                   '--intra-threads', str(args.intra_threads), '--queue-size', str(args.queue_size),
                   '--model', args.model, '--source', args.source, '--reference', args.reference]
            try:
                code = subprocess.run(cmd, timeout=180).returncode
            except subprocess.TimeoutExpired:
                code = 'timeout'
            done.append({'round': block, 'depth': depth, 'exit_code': code,
                         'result': str(job/f'depth-{depth}.json')})
            (out/'status.json').write_text(json.dumps(done, indent=2))
            if code != 0:
                return 1
            time.sleep(args.cooldown)
    print(f'Completed: {out.resolve()}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
