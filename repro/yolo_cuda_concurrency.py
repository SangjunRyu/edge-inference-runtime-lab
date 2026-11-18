"""Current MX150/ORT/YOLO combination: correct count/order can hide bad boxes.

Broken is a nondeterministic negative control. NOT REPRO is not a clean bill
of health. Fixed selects one CUDA EP stream; it does not prove a library root
cause or universal correctness on different GPUs/models/drivers.
"""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['broken','fixed'],required=True)
p.add_argument('--out-dir',type=Path,default=ROOT/'results/yolo-concurrency-repro');a=p.parse_args()
a.out_dir.mkdir(parents=True,exist_ok=True)
attempts=3 if a.mode=='broken' else 1
for attempt in range(attempts):
    folder=a.out_dir/f'{a.mode}-{attempt+1}'
    cmd=[sys.executable,str(ROOT/'tools/verify_industrial_model.py'),str(folder),'--samples','64']
    if a.mode=='broken':cmd+=['--unsafe-default-stream']
    r=subprocess.run(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
    (a.out_dir/f'{a.mode}-{attempt+1}.log').write_bytes(r.stdout)
    if a.mode=='fixed':
        print('FIXED CHECK', 'PASS' if r.returncode==0 else 'FAILED');sys.exit(r.returncode)
    diag=folder/'model-verification-diagnostic.json'
    if r.returncode and diag.exists() and any(not x['raw_allclose_rtol_atol_1e3'] for x in json.loads(diag.read_text())):
        print('BROKEN: output-value mismatch reproduced; inspect saved diagnostics');sys.exit(0)
print('NOT REPRO in three attempts; failure is intermittent');sys.exit(77)
