"""Integration check against the real Python runners; no synthetic inference mocks."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.runtime.backends import OnnxBackend
from bench.measurement.env_fingerprint import collect
from bench.classification.run_sync import measure_once
from bench.classification.verify_equivalence import compare, load_dump, self_test, sensitivity


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bin-dir', type=Path, default=ROOT/'cpp/build/ort-msvc/Release')
    p.add_argument('--out-dir', type=Path, default=ROOT/'results/cpp-ort-validation')
    p.add_argument('--frames', type=int, default=300)
    args = p.parse_args()
    out = args.out_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    previous_profiles = set(out.glob('profile-*.json'))
    checks = []

    def execute(mode, backend, res, label, frames, extra=(), expected=0):
        exe = args.bin_dir.resolve()/f'run_{mode}_ort.exe'
        dump = out/f'{label}.jsonl'
        cmd = [str(exe), '--model', str(ROOT/'models/resnet50_224.onnx'),
               '--backend', backend, '--source', str(ROOT/f'clips/bbb_{res}_{"2MB" if res=="360" else "10MB"}.mp4'),
               '--frames', str(frames), '--duration-s', '0', '--warmup', '0', '--warmup-s', '0',
               '--intra-threads', '4', '--dump-topk', str(dump), '--tag', label, '--out-dir', str(out)]
        if mode == 'async': cmd += ['--inflight', '4', '--post-workers', '4']
        cmd += list(extra)
        print('RUN', label, flush=True)
        r = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
        (out/f'{label}.log').write_bytes(r.stdout)
        if (expected == 0 and r.returncode != 0) or (expected != 0 and r.returncode == 0):
            raise RuntimeError(f'{label}: exit={r.returncode}; see log')
        if expected: return None
        result_path = next(out.glob(f'{label}__*.json'))
        result = json.loads(result_path.read_text(encoding='utf-8'))
        run = result['runs'][0]
        if run['order']['emitted'] != frames or run['order']['order_violations']:
            raise RuntimeError(f'{label}: count/order mismatch')
        if mode == 'async' and (run['errors'] or run['submitted'] != frames):
            raise RuntimeError(f'{label}: async failed')
        return load_dump(str(dump))

    for backend in ('cpu', 'cuda'):
        b = OnnxBackend(str(ROOT/'models/resnet50_224.onnx'), backend, 4)
        if b.fell_back: raise RuntimeError(f'{backend} unavailable')
        for res, size in [('360','2MB'),('1080','10MB')]:
            print('PYTHON_REFERENCE', backend, res, flush=True)
            a = SimpleNamespace(source=str(ROOT/f'clips/bbb_{res}_{size}.mp4'), frames=args.frames,
                                warmup=0, warmup_s=0.0, duration_s=0.0, top_k=5)
            collected=[]; measure_once(b,a,224,224,collected)
            ref=dict(collected)
            if len(ref)!=args.frames or not self_test(ref,1e-4): raise RuntimeError('Reference/self-test failed')
            (out/f'python-{backend}-{res}.json').write_text(json.dumps(collected),encoding='utf-8')
            for mode in ('sync','async'):
                label=f'{backend}-{res}-{mode}'
                extra=['--profile', str(out/f'profile-{label}')] if backend=='cuda' else []
                got=execute(mode,backend,res,label,args.frames,extra)
                result=compare(ref,got,1e-4)
                result.update(backend=backend,resolution=res,mode=mode,label=label,sensitivity=sensitivity(ref))
                if not result['identical']: raise RuntimeError(f'Prediction mismatch: {result}')
                checks.append(result); print('PASS',label,'frames',result['frames_compared'],flush=True)
            if res=='360':
                n=min(30,args.frames); prefix={i:ref[i] for i in range(n)}
                for label,mode,extra in [('reuse','sync',['--reuse-tensor']),('fused','sync',['--fused']),
                                         ('depth1','async',['--inflight','1','--post-workers','1'])]:
                    name=f'{backend}-{label}'
                    got=execute(mode,backend,res,name,n,extra)
                    result=compare(prefix,got,1e-4)
                    result.update(label=name)
                    if not result['identical']: raise RuntimeError(f'Variant mismatch: {result}')
                    checks.append(result)
    # Failure must wake all workers and return nonzero, rather than hang or
    # save a plausible success under the requested accelerator's label.
    execute('async','cpu','360','invalid-depth',8,['--inflight','0'],expected=1)
    execute('async','cpu','360','invalid-async-threads',8,['--intra-threads','1'],expected=1)
    execute('sync','GPU','360','invalid-device',8,expected=1)
    profiles={}
    for path in sorted(set(out.glob('profile-*.json')) - previous_profiles):
        counts={}
        for e in json.loads(path.read_text(encoding='utf-8')):
            provider=e.get('args',{}).get('provider')
            if provider: counts[provider]=counts.get(provider,0)+1
        if not counts.get('CUDAExecutionProvider'): raise RuntimeError(f'No CUDA nodes in {path}')
        profiles[path.name]=counts
    if len(profiles)!=4: raise RuntimeError(f'Expected four CUDA profiles, got {len(profiles)}')
    result={'passed':True,'frames_per_main_check':args.frames,'checks':checks,'cuda_profiles':profiles,
            'negative_checks':['invalid-depth','invalid-async-threads','invalid-device'],'environment':collect()}
    (out/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('VALIDATION COMPLETE',len(checks),'prediction checks; CUDA profiles:',profiles,flush=True)

if __name__=='__main__': main()
