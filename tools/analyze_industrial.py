"""Event-level check against scene geometry, independent of detector totals."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def match_events(expected,observed,tolerance=.5):
    remaining=set(range(len(expected)));matches=[];false=[]
    for timestamp in observed:
        candidates=[(abs(timestamp-expected[i]),i) for i in remaining]
        error,index=min(candidates,default=(float('inf'),None))
        if error<=tolerance:
            remaining.remove(index);matches.append({'expected_s':expected[index],'observed_s':timestamp,'error_ms':error*1000})
        else:false.append(timestamp)
    return {'true_positives':len(matches),'false_positives':len(false),'false_negatives':len(remaining),
            'matches':matches,'false_positive_times_s':false,'tolerance_s':tolerance}

def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
    truth=json.loads((ROOT/'clips/scenario-assets/ground-truth.json').read_text())
    previous={};expected=[]
    for frame in truth:
        for obj in frame['objects']:
            center=(obj['box'][0]+obj['box'][2])/2
            if obj['id'] in previous and previous[obj['id']]<320<=center:expected.append(frame['time_s'])
            previous[obj['id']]=center
    if len(expected)!=3:raise RuntimeError('Scene ground truth invalid')
    results=[]
    for path in sorted(a.directory.glob('[0-9][0-9]-stream-*.json')):
        d=json.loads(path.read_text());observed=[];prev=0;reported={}
        for r in d['records']:
            if r['status']=='completed':
                increment=r['crossing_count']-prev
                observed.extend([r['due_s']]*increment);prev=r['crossing_count']
                if increment:reported[r['due_s']]=r['finished']-(r['due']-r['due_s'])
        check=match_events(expected,observed);check.update(name=d['name'],repeat=d['repeat'],observed_times_s=observed)
        for event in check['matches']:
            event['reported_s']=reported[event['observed_s']]
            event['signed_output_lag_ms']=(event['reported_s']-event['expected_s'])*1000
        results.append(check)
    out={'ground_truth_crossings_s':expected,'ground_truth_time_resolution_s':1/30,
         'note':'Match capture-time crossing events, not output time; deadline misses are measured separately. Controlled surrogate only.',
         'runs':results}
    (a.directory/'count-event-validation.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print('EVENT CHECK',len(results),'runs',sum(r['true_positives'] for r in results),'TP',sum(r['false_positives'] for r in results),'FP',sum(r['false_negatives'] for r in results),'FN')
if __name__=='__main__':main()
