"""Real YOLO sync/async value check and separate CUDA node-assignment profile."""
import argparse,json,sys,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,onnxruntime as ort
from bench.backends import OnnxBackend
from bench.industrial import letterbox,decode_yolo
from bench.surrogate_camera import SurrogateCamera

def main():
    p=argparse.ArgumentParser();p.add_argument('out_dir',type=Path);p.add_argument('--samples',type=int,default=64)
    p.add_argument('--unsafe-default-stream',action='store_true',help='Negative control; may fail on this MX150 stack')
    a=p.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=True)
    cuda_options=None if a.unsafe_default_stream else {'use_ep_level_unified_stream':'1'}
    b=OnnxBackend(str(ROOT/'models/yolo11n_320.onnx'),'cuda',4,cuda_options=cuda_options)
    if b.fell_back:raise RuntimeError('CUDA unavailable')
    source=SurrogateCamera(ROOT/'clips/scenario-assets/object-crop.jpg')
    indices=[(i*37)%len(source) for i in range(a.samples)]
    inputs=[letterbox(source[i],320) for i in indices]
    refs=[b.run(x) for x,_ in inputs]
    results={};errors=[];done=threading.Event();lock=threading.Lock();slots=threading.Semaphore(4)
    def complete(outputs,index,error):
        with lock:
            if error:errors.append(str(error))
            else:results[index]=outputs[0]
            if len(results)+len(errors)==len(inputs):done.set()
        slots.release()
    for index,(tensor,_) in enumerate(inputs):
        slots.acquire();b.session.run_async([b.output_name],{b.input_name:tensor},complete,index)
    if not done.wait(30) or errors:raise RuntimeError(f'Async stalled or failed: {errors}')
    checks=[]
    for index,(ref,(_,transform)) in enumerate(zip(refs,inputs)):
        got=results[index]
        boxes_ref=decode_yolo(ref,transform);boxes_got=decode_yolo(got,transform)
        diagnostic={'frame':indices[index],'max_raw_abs_difference':float(np.max(np.abs(ref-got))),
                    'max_class_score_abs_difference':float(np.max(np.abs(ref[:,4:]-got[:,4:]))),
                    'raw_allclose_rtol_atol_1e3':bool(np.allclose(ref,got,rtol=1e-3,atol=1e-3)),
                    'reference_detections':boxes_ref,'async_detections':boxes_got}
        print(json.dumps(diagnostic),flush=True)
        checks.append(diagnostic)
        (a.out_dir/'model-verification-diagnostic.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
        if not diagnostic['raw_allclose_rtol_atol_1e3']:raise RuntimeError('Raw output mismatch; refusing this configuration')
        if len(boxes_ref)!=len(boxes_got):raise RuntimeError('NMS detection count mismatch')
        for x,y in zip(boxes_ref,boxes_got):
            if not np.allclose(x['box'],y['box'],atol=.1) or abs(x['score']-y['score'])>1e-4:
                raise RuntimeError('Detection coordinate or score mismatch')
    options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
    options.enable_profiling=True;options.profile_file_prefix=str(a.out_dir.resolve()/'yolo-cuda-profile')
    session=ort.InferenceSession(str(ROOT/'models/yolo11n_320.onnx'),sess_options=options,providers=[('CUDAExecutionProvider',cuda_options or {}),'CPUExecutionProvider'])
    session.disable_fallback()
    if 'CUDAExecutionProvider' not in session.get_providers():raise RuntimeError('Profiling session fell back')
    for tensor,_ in inputs[:2]:session.run(None,{session.get_inputs()[0].name:tensor})
    profile=Path(session.end_profiling());counts={}
    for event in json.loads(profile.read_text()):
        provider=event.get('args',{}).get('provider')
        if provider:counts[provider]=counts.get(provider,0)+1
    if not counts.get('CUDAExecutionProvider'):raise RuntimeError('No CUDA node events')
    result={'passed':True,'application_tolerance':{'box_pixels':.1,'score_absolute':1e-4},
            'all_raw_allclose':all(c['raw_allclose_rtol_atol_1e3'] for c in checks),
            'cuda_options':cuda_options,
            'sync_async_checks':checks,'profile_node_events':counts,'profile':profile.name,'profile_during_benchmarks':False}
    (a.out_dir/'model-verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))
if __name__=='__main__':main()
