"""Two industrial-style input contracts, measured with real CPU/CUDA models."""
import argparse,csv,gc,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from bench.backends import OnnxBackend
from bench.env_fingerprint import collect
from bench.noise import idle_gate
from bench.pipeline import preprocess,postprocess
from bench.industrial import Config,run_workload,letterbox,decode_yolo
from bench.surrogate_camera import SurrogateCamera

def main():
    p=argparse.ArgumentParser();p.add_argument('--out-dir',type=Path,default=ROOT/'results/industrial-rerun')
    p.add_argument('--repeat',type=int,default=2);p.add_argument('--smoke',action='store_true')
    p.add_argument('--only',default='',help='Optional scenario-name prefix');a=p.parse_args()
    out=a.out_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    assets=ROOT/'clips/scenario-assets'
    images=[(assets/f'camera-{i}.jpg').read_bytes() for i in range(4)]
    camera=SurrogateCamera(assets/'object-crop.jpg')
    manifest=json.loads((assets/'manifest.json').read_text())
    plan=[]
    for backend in ('cpu','cuda'):
        for rate in (2,12):
            for mode in ('sync','async'):
                plan.append((f'trigger-{backend}-{rate}-{mode}',backend,Config(workload='trigger',mode=mode,
                    duration_s=6,cameras=4,fps_per_camera=rate,inflight=4,capacity=4,deadline_ms=200)))
        stream=[('normal-sync',5,'sync',1,8,'reject-newest'),('normal-async',5,'async',4,8,'reject-newest'),
                ('over-reject-d1',120,'async',1,8,'reject-newest'),('over-fresh-d1',120,'async',1,8,'drop-oldest'),
                ('over-fresh-d4',120,'async',4,8,'drop-oldest')]
        if backend=='cuda':stream += [('over-sync',120,'sync',1,8,'reject-newest'),
            ('over-reject-d4',120,'async',4,8,'reject-newest'),('over-fresh-q2',120,'async',1,2,'drop-oldest')]
        for name,rate,mode,depth,capacity,policy in stream:
            plan.append((f'stream-{backend}-{name}',backend,Config(workload='stream',mode=mode,duration_s=10,
                cameras=1,fps_per_camera=rate,inflight=depth,capacity=capacity,policy=policy,deadline_ms=100,source_fps=120)))
    if a.smoke:plan=[plan[0],plan[1],next(r for r in plan if r[0]=='stream-cuda-normal-async')]
    plan=[r for r in plan if r[0].startswith(a.only)]
    if not plan:raise ValueError('No scenario matches --only')
    meta={'environment':collect(),'assets':manifest,'source':'virtual JPEG trigger channels / procedural raw-BGR camera',
          'implementation':'Python orchestration, real ORT native inference and callbacks',
          'deadline_note':'Trigger 200ms and stream 100ms are illustrative requirements, not measured factory budgets.',
          'plan':[{'name':n,'backend':b,**vars(c)} for n,b,c in plan],
          'model_sha256':{n:hashlib.sha256((ROOT/'models'/n).read_bytes()).hexdigest() for n in ('resnet50_224.onnx','yolo11n_320.onnx')}}
    (out/'protocol.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    results=[];last_key=None;backend=None;references={}
    for repeat in range(1,a.repeat+1):
        for name,ep,c in (plan if repeat%2 else list(reversed(plan))):
            key=(ep,c.workload)
            if key!=last_key:
                backend=None;gc.collect()
                model='resnet50_224.onnx' if c.workload=='trigger' else 'yolo11n_320.onnx'
                options={'use_ep_level_unified_stream':'1'} if ep=='cuda' and c.workload=='stream' else None
                backend=OnnxBackend(str(ROOT/'models'/model),ep,4,cuda_options=options)
                if backend.fell_back:raise RuntimeError('Requested backend fell back')
                references={}
                for i in range(8):
                    if c.workload=='trigger':
                        image=cv2.imdecode(np.frombuffer(images[i%4],np.uint8),cv2.IMREAD_COLOR)
                        tensor=preprocess(image,*backend.hw)
                        references[i%4]=postprocess(backend.run(tensor),5)
                    else:
                        tensor,tr=letterbox(camera[145*4],320);decode_yolo(backend.run(tensor),tr)
                last_key=key
            if results:time.sleep(1)
            gate=idle_gate(max_busy_pct=20,attempts=3)
            print('START',repeat,name,flush=True)
            if a.smoke:c.duration_s=2
            result=run_workload(backend,c,images=images,frames=camera)
            if c.workload=='trigger':
                for r in result['records']:
                    if r['status']=='completed':
                        ref=references[r['camera']]
                        if [x[0] for x in r['top5']]!=[x[0] for x in ref] or max(abs(x[1]-y[1]) for x,y in zip(r['top5'],ref))>1e-4:
                            raise RuntimeError('Camera/result value pairing failed')
                result['prediction_pairing_checked']=True
            result.update(name=name,repeat=repeat,idle_gate=gate)
            filename=f'{repeat:02d}-{name}.json';(out/filename).write_text(json.dumps(result,indent=2),encoding='utf-8')
            results.append(result)
            s=result['summary'];print(f"DONE {name}: offered={s['offered']} completed={s['completed']} drops={s['dropped']} timely={s['timely']} age_p95={s['age_ms']['p95']:.1f}ms count={s['crossing_count']}",flush=True)
            (out/'index.json').write_text(json.dumps([{'file':f"{r['repeat']:02d}-{r['name']}.json",'name':r['name'],'repeat':r['repeat'],'summary':r['summary']} for r in results],indent=2),encoding='utf-8')
    rows=[]
    for name,ep,c in plan:
        rs=[r for r in results if r['name']==name];median=lambda f:float(np.median([f(r) for r in rs]))
        rows.append({'name':name,'backend':ep,'workload':c.workload,'mode':c.mode,'source_rate_total':c.cameras*c.fps_per_camera,
            'inflight':c.inflight if c.mode=='async' else 1,'queue':c.capacity,'policy':c.policy,'repeat':len(rs),
            'offered':median(lambda r:r['summary']['offered']),'completed_median':median(lambda r:r['summary']['completed']),
            'drop_pct':median(lambda r:100*r['summary']['dropped']/r['summary']['offered']),
            'timely_offered_pct':median(lambda r:100*r['summary']['timely_per_offered']),
            'age_p50_ms':median(lambda r:r['summary']['age_ms']['p50']),'age_p95_ms':median(lambda r:r['summary']['age_ms']['p95']),
            'source_late_p95_ms':median(lambda r:r['summary']['source_late_ms']['p95']),
            'count_each_run':','.join(str(r['summary']['crossing_count']) for r in rs) if c.workload=='stream' else '',
            'all_counts_correct':all(r['summary']['crossing_count']==c.expected_crossings for r in rs) if c.workload=='stream' else None,
            'all_idle_gates_passed':all(r['idle_gate'].get('settled',False) for r in rs)})
    (out/'summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    with (out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('COMPLETE',len(results),'runs',flush=True)

if __name__=='__main__':main()
