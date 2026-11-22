"""Fresh-process inflight sweep with NVML device memory and process RSS sampling."""
from __future__ import annotations

import argparse
import ctypes as c
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

import psutil

from bench.measurement import env_fingerprint
from bench.classification import run_async, run_sync
from bench.runtime.backends import OnnxBackend
from bench.classification.verify_equivalence import compare


class MemorySampler:
    """100ms device-wide VRAM samples; not live activation bytes or per-process VRAM."""
    class Info(c.Structure):
        _fields_ = [('total', c.c_ulonglong), ('free', c.c_ulonglong), ('used', c.c_ulonglong)]

    class Util(c.Structure):
        _fields_ = [('gpu', c.c_uint), ('memory', c.c_uint)]

    def __init__(self):
        self.dll = c.WinDLL('nvml.dll') if os.name == 'nt' else c.CDLL('libnvidia-ml.so.1')
        self.check(self.dll.nvmlInit_v2())
        self.handle = c.c_void_p()
        self.dll.nvmlDeviceGetHandleByIndex_v2.argtypes = [c.c_uint, c.POINTER(c.c_void_p)]
        self.check(self.dll.nvmlDeviceGetHandleByIndex_v2(0, c.byref(self.handle)))
        self.dll.nvmlDeviceGetMemoryInfo.argtypes = [c.c_void_p, c.POINTER(self.Info)]
        self.process = psutil.Process()
        self.process.cpu_percent(None)
        self.rows = []
        self.errors = []
        self.stop = threading.Event()
        self.thread = None

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f'NVML failed: {code}')

    def sample(self):
        info = self.Info()
        self.check(self.dll.nvmlDeviceGetMemoryInfo(self.handle, c.byref(info)))
        row = {'t': time.perf_counter(), 'device_used': info.used,
                'device_free': info.free, 'device_total': info.total,
                'rss': self.process.memory_info().rss}
        def scalar(name, extra=(), wide=False):
            fn = getattr(self.dll, name, None)
            if fn is None:
                return None
            kind = c.c_ulonglong if wide else c.c_uint
            value = kind()
            fn.argtypes = [c.c_void_p] + [c.c_uint]*len(extra) + [c.POINTER(kind)]
            rc = fn(self.handle, *extra, c.byref(value))
            return value.value if rc == 0 else None
        row['gpu_sm_clock_mhz'] = scalar('nvmlDeviceGetClockInfo', (1,))
        row['gpu_memory_clock_mhz'] = scalar('nvmlDeviceGetClockInfo', (2,))
        row['gpu_temperature_c'] = scalar('nvmlDeviceGetTemperature', (0,))
        row['gpu_power_mw'] = scalar('nvmlDeviceGetPowerUsage')
        row['gpu_pstate'] = scalar('nvmlDeviceGetPerformanceState')
        row['gpu_clock_event_bits'] = scalar('nvmlDeviceGetCurrentClocksThrottleReasons', wide=True)
        util = self.Util()
        fn = self.dll.nvmlDeviceGetUtilizationRates
        fn.argtypes = [c.c_void_p, c.POINTER(self.Util)]
        rc = fn(self.handle, c.byref(util))
        row['gpu_util_percent'] = util.gpu if rc == 0 else None
        row['system_cpu_percent'] = psutil.cpu_percent(None)
        row['process_cpu_percent'] = self.process.cpu_percent(None)
        return row

    def _loop(self):
        while not self.stop.wait(.1):
            try:
                self.rows.append(self.sample())
            except Exception as exc:
                self.errors.append(repr(exc))
                return

    def start(self):
        self.rows.append(self.sample())
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def finish(self):
        self.stop.set()
        self.thread.join()
        self.rows.append(self.sample())
        self.check(self.dll.nvmlShutdown())
        if self.errors:
            raise RuntimeError(self.errors)
        return {'method': 'NVML device-wide memory + psutil process RSS; sampled every 100ms',
                'samples': len(self.rows), 'before': self.rows[0], 'after': self.rows[-1],
                'peak_device_used': max(r['device_used'] for r in self.rows),
                'minimum_device_free': min(r['device_free'] for r in self.rows),
                'peak_rss': max(r['rss'] for r in self.rows), 'raw_samples': self.rows}


def worker(args):
    sampler = MemorySampler()
    baseline = sampler.sample()
    backend = OnnxBackend(args.model, 'cuda', args.intra_threads)
    if backend.fell_back:
        raise RuntimeError(f'CUDA unavailable: {backend.actual_providers}')
    after_session = sampler.sample()
    previous = json.loads(Path(args.reference).read_text(encoding='utf-8'))
    meta_path = Path(args.reference).with_suffix('.meta.json')
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        model_hash = hashlib.sha256(Path(args.model).read_bytes()).hexdigest()
        manifest_hash = hashlib.sha256(Path(args.source).read_bytes()).hexdigest()
        if meta['model_sha256'] != model_hash or meta['source']['manifest_sha256'] != manifest_hash:
            raise ValueError('Reference model/manifest differs; generate a local make_reference baseline')
    reference = {int(i): top for i, top in previous if int(i) < args.frames}
    if len(reference) != args.frames:
        raise ValueError('Reference must cover every requested frame')
    config = argparse.Namespace(source=args.source, image_read_mode='files',
             frames=args.frames, warmup=args.warmup, warmup_s=0., duration_s=0.,
             top_k=5, inflight=max(args.worker, 1), queue_size=args.queue_size,
             reorder='on', reorder_cap=None, stall_every=0, stall_ms=0.,
             stall_timeout_s=30.)
    rows = []
    sampler.start()
    for rep in range(args.repeat):
        run_start = time.perf_counter()
        collected = []
        runner = run_sync if args.worker == 0 else run_async
        rec, src, info = runner.measure_once(backend, config, *backend.hw, collected)
        comparison = compare(reference, dict(collected), 1e-4)
        healthy = (len(collected) == len(dict(collected)) == args.frames
                   and comparison['identical'] and rec.counted == args.frames-args.warmup)
        if args.worker:
            healthy = (healthy and not info['errors'] and not info['stalls'] and
                       info['order']['order_violations'] == 0 and
                       info['reorder']['forced_flushes'] == 0)
        backend._assert_providers_intact()
        if not healthy:
            raise RuntimeError(f'Invalid run: {comparison}, {info}')
        rows.append({'repeat': rep+1, 'metrics': rec.result(), 'source': src,
                     'info': info, 'correctness': comparison, 'healthy': bool(healthy),
                     'time_window_perf': [run_start, time.perf_counter()]})
        print(f"depth={args.worker} repeat={rep+1} rate={rows[-1]['metrics']['throughput_fps']:.2f} "
              f"p95={rows[-1]['metrics']['e2e_ms']['p95']:.1f} ms", flush=True)
    memory = sampler.finish()
    memory['baseline_before_session'] = baseline
    memory['after_session_creation'] = after_session
    value = {'depth': args.worker, 'args': vars(args), 'pid': os.getpid(),
             'model_sha256': hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
             'backend': backend.describe(), 'provider_options': backend.session.get_provider_options(),
             'env': env_fingerprint.collect(), 'runs': rows, 'memory': memory,
             'passed': all(r['healthy'] for r in rows)}
    Path(args.out_dir, f'depth-{args.worker}.json').write_text(json.dumps(value, indent=2), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='models/resnet50_224.onnx')
    p.add_argument('--source', default='data/imagenette-320-subset/manifest.json')
    p.add_argument('--reference', default='results/image-classification/20261122-imagenette500/reference-top5.json')
    p.add_argument('--depths', nargs='+', type=int, default=[1, 2, 4, 8, 16, 32])
    p.add_argument('--frames', type=int, default=300)
    p.add_argument('--warmup', type=int, default=30)
    p.add_argument('--repeat', type=int, default=3)
    p.add_argument('--intra-threads', type=int, default=4)
    p.add_argument('--queue-size', type=int, default=4)
    p.add_argument('--out-dir', default=None)
    p.add_argument('--worker', type=int, default=None, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.intra_threads < 2 or args.repeat < 1 or not 0 <= args.warmup < args.frames:
        p.error('intra-threads >= 2, repeat >= 1 and 0 <= warmup < frames required')
    if any(n < 1 for n in args.depths):
        p.error('depths must be positive')
    if args.worker is not None:
        worker(args)
        return
    out = Path(args.out_dir or ('results/inflight-memory/' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
    out.mkdir(parents=True, exist_ok=False)
    done = []
    for depth in [0] + args.depths:
        cmd = [sys.executable, '-m', 'bench.classification.sweep_inflight', '--worker', str(depth),
               '--out-dir', str(out), '--model', args.model, '--source', args.source,
               '--reference', args.reference, '--frames', str(args.frames),
               '--warmup', str(args.warmup), '--repeat', str(args.repeat),
               '--intra-threads', str(args.intra_threads), '--queue-size', str(args.queue_size)]
        started = time.perf_counter()
        try:
            proc = subprocess.run(cmd, timeout=240, check=False)
            exitcode = proc.returncode
        except subprocess.TimeoutExpired:
            exitcode = 'timeout'
        item = {'depth': depth, 'exit_code': exitcode, 'wall_s': time.perf_counter()-started}
        done.append(item)
        (out/'status.json').write_text(json.dumps(done, indent=2), encoding='utf-8')
        if exitcode != 0:
            print(f'Stopping at depth {depth}; inspect logs. No OOM classification inferred from exit code alone.', flush=True)
            return 1
        saved = json.loads((out/f'depth-{depth}.json').read_text())
        # Stop escalation with <256MiB device headroom; do not deliberately exhaust VRAM.
        if saved['memory']['minimum_device_free'] < 256*2**20:
            print('Stopping escalation: sampled device headroom <256 MiB.', flush=True)
            break
        time.sleep(2)
    lines = ['# Fresh-process inflight sweep', '',
        'File+decode mode, same image manifest/reference; batch 1; fresh session/process per depth.',
        'VRAM is NVML device-wide reserved/allocated usage sampled at 100ms, not live activations.',
        'inflight is outstanding requests, not proven concurrent GPU executions.', '',
        '| Depth | Images/s mean [min, max] | E2E p95 mean ms | Device VRAM peak MiB | Minimum free MiB | RSS peak MiB |',
        '|---|---:|---:|---:|---:|---:|']
    for job in done:
        obj = json.loads((out/f"depth-{job['depth']}.json").read_text())
        rate = [r['metrics']['throughput_fps'] for r in obj['runs']]
        tail = [r['metrics']['e2e_ms']['p95'] for r in obj['runs']]
        m = obj['memory']
        lines.append(f"| {job['depth']} | {statistics.mean(rate):.2f} [{min(rate):.2f}, {max(rate):.2f}] | "
              f"{statistics.mean(tail):.2f} | {m['peak_device_used']/2**20:.1f} | "
              f"{m['minimum_device_free']/2**20:.1f} | {m['peak_rss']/2**20:.1f} |")
    (out/'SUMMARY.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'Completed: {out.resolve()}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
