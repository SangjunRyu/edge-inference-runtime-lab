"""Short repeated native runs. Same engine, separate process/session per run."""
import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from bench.measurement.env_fingerprint import collect
from bench.measurement.noise import idle_gate

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out-dir',type=Path,default=ROOT/'results/cpp-ort-benchmark')
    p.add_argument('--bin-dir',type=Path,default=ROOT/'cpp/build/ort-msvc/Release')
    p.add_argument('--pairs',type=int,default=3)
    p.add_argument('--duration-s',type=float,default=10)
    p.add_argument('--warmup-s',type=float,default=3)
    a=p.parse_args(); out=a.out_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    meta={'environment':collect(),'duration_s':a.duration_s,'warmup_s':a.warmup_s,
          'pairs':a.pairs,'inflight':4,'queue_size':4,'post_workers':1,'intra_threads':4,'cooldown_s':3,
          'note':'Same ORT 1.26/OpenCV 4.12 C++ engine, separate process/session per run. Not a controlled Python/C++ language comparison. No profiling during timing.'}
    runs=[]
    def save():
        (out/'measurements.json').write_text(json.dumps({'meta':meta,'runs':runs},indent=2),encoding='utf-8')
    cells=[(b,res,size) for b in ('cpu','cuda') for res,size in [('360','2MB'),('1080','10MB')]]
    for rep in range(1,a.pairs+1):
        for b,res,size in (cells if rep%2 else list(reversed(cells))):
            for mode in (('sync','async') if rep%2 else ('async','sync')):
                if runs: time.sleep(3)
                gate=idle_gate(max_busy_pct=15,attempts=3)
                tag=f'p{rep}-{b}-{res}-{mode}'
                cmd=[str(a.bin_dir.resolve()/f'run_{mode}_ort.exe'),'--model',str(ROOT/'models/resnet50_224.onnx'),
                     '--backend',b,'--source',str(ROOT/f'clips/bbb_{res}_{size}.mp4'),'--frames','1000000',
                     '--duration-s',str(a.duration_s),'--warmup-s',str(a.warmup_s),'--intra-threads','4',
                     '--tag',tag,'--out-dir',str(out)]
                if mode=='async': cmd+=['--inflight','4','--queue-size','4','--post-workers','1']
                print('START',tag,flush=True)
                r=subprocess.run(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
                (out/f'{tag}.log').write_bytes(r.stdout)
                if r.returncode: raise RuntimeError(f'{tag}: exit {r.returncode}')
                result=json.loads(next(out.glob(f'{tag}__*.json')).read_text(encoding='utf-8'))
                run=result['runs'][0]
                if run['frames_counted']<2 or run['order']['order_violations']: raise RuntimeError('Invalid count/order')
                if mode=='async' and (run['errors'] or run['submitted']!=run['order']['emitted']): raise RuntimeError('Invalid async run')
                runs.append({'pair':rep,'backend':b,'resolution':res,'mode':mode,'idle_gate':gate,'result':run,'metadata':result['meta']})
                save()
                print(f"DONE {tag}: FPS={run['throughput_fps']:.2f} E2E={run['e2e_ms']['p50']:.2f}ms drift={run['drift']['ratio']:.3f}",flush=True)
    summary=[]
    for b,res,_ in cells:
        pairs=[]
        for rep in range(1,a.pairs+1):
            rs={r['mode']:r for r in runs if r['backend']==b and r['resolution']==res and r['pair']==rep}
            pairs.append(rs['async']['result']['throughput_fps']/rs['sync']['result']['throughput_fps'])
        for mode in ('sync','async'):
            rs=[r for r in runs if r['backend']==b and r['resolution']==res and r['mode']==mode]
            fps=[r['result']['throughput_fps'] for r in rs]
            summary.append({'backend':b,'resolution':res,'mode':mode,'runs':len(rs),
                'fps_median':statistics.median(fps),'fps_min':min(fps),'fps_max':max(fps),
                'fps_cv_pct':100*statistics.stdev(fps)/statistics.mean(fps) if len(fps)>1 else None,
                'e2e_p50_median_ms':statistics.median(r['result']['e2e_ms']['p50'] for r in rs),
                'e2e_p95_median_ms':statistics.median(r['result']['e2e_ms']['p95'] for r in rs),
                'read_p50_median_ms':statistics.median(r['result']['stages_ms']['read']['p50'] for r in rs),
                'preprocess_p50_median_ms':statistics.median(r['result']['stages_ms']['preprocess']['p50'] for r in rs),
                'inference_p50_median_ms':statistics.median(r['result']['stages_ms']['inference']['p50'] for r in rs),
                'idle_gate_passed':sum(bool(r['idle_gate'].get('settled',False)) for r in rs),
                'paired_async_sync_ratio_median':statistics.median(pairs),
                'paired_ratio_min':min(pairs),'paired_ratio_max':max(pairs)})
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with (out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    print('BENCHMARK COMPLETE',json.dumps(summary),flush=True)
if __name__=='__main__':main()
