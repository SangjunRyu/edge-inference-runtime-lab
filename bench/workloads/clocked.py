"""Clock-driven protocol engine; sources are supplied by the caller.

The retired synthetic-bus scenario is not an input generator for this module.
"""
from collections import deque
from dataclasses import dataclass
import math
import queue
import threading
import time
import cv2
import numpy as np
from bench.runtime.pipeline import preprocess, postprocess
from bench.measurement.statistics import stats
from bench.detection.yolo import letterbox, decode_yolo, LineCounter


class AdmissionQueue:
    def __init__(self,capacity,policy):
        if capacity<1 or policy not in ('reject-newest','drop-oldest'): raise ValueError('Invalid admission config')
        self.capacity=capacity; self.policy=policy; self.items=deque(); self.closed=False
        self.cv=threading.Condition(); self.peak=0
    def offer(self,item):
        with self.cv:
            if self.closed: raise RuntimeError('Admission after close')
            evicted=None
            if len(self.items)==self.capacity:
                if self.policy=='reject-newest': return item
                evicted=self.items.popleft()
            self.items.append(item); self.peak=max(self.peak,len(self.items)); self.cv.notify()
            return evicted
    def take(self):
        with self.cv:
            self.cv.wait_for(lambda:self.items or self.closed)
            return self.items.popleft() if self.items else None
    def close(self):
        with self.cv: self.closed=True; self.cv.notify_all()


@dataclass
class Config:
    workload:str='trigger'
    mode:str='async'
    duration_s:float=10
    cameras:int=4
    fps_per_camera:float=5
    inflight:int=2
    capacity:int=4
    policy:str='reject-newest'
    deadline_ms:float=200
    source_fps:float=30
    expected_crossings:int|None=None
    detection_class:int=0


def run_workload(backend,config,images=None,frames=None,observer=None):
    """Images are encoded bytes; frames are decoded controlled replay frames.

    Each scheduled item is either completed or explicitly dropped. Permits are
    held through ordered postprocessing, bounding all submitted-frame storage.
    Source emissions may be late under OS load; lateness remains in output age.
    """
    c=config
    if c.mode not in ('sync','async') or c.fps_per_camera<=0 or c.duration_s<=0 or c.cameras<1 or c.inflight<1:
        raise ValueError('Invalid workload configuration')
    depth=c.inflight if c.mode=='async' else 1
    admission=AdmissionQueue(c.capacity,c.policy)
    completions=queue.Queue(); slots=threading.Semaphore(depth)
    records=[]; errors=[]; reorder_pending={}; next_seq=0
    counter=LineCounter() if c.workload=='stream' else None
    scheduled=[]
    for camera in range(c.cameras):
        for local_seq in range(int(math.ceil(c.duration_s*c.fps_per_camera))):
            due=local_seq/c.fps_per_camera
            if due<c.duration_s: scheduled.append((due,camera,local_seq))
    # Synchronous camera bursts, with deterministic rotating tie order rather
    # than always giving camera 0 first access to the last queue slot.
    scheduled.sort(key=lambda x:(x[0],(x[1]-round(x[0]*c.fps_per_camera))%c.cameras))
    start=time.perf_counter()+0.05
    max_active=0; active=0; active_lock=threading.Lock()

    def callback(outputs,item,err):
        item['infer_done']=time.perf_counter()
        if err: item['error']=str(err)
        else: item['output']=outputs[0]
        # Unbounded completion handoff; storage is bounded by held slot permits.
        completions.put(item)

    def produce():
        try:
            for seq,(due,camera,local_seq) in enumerate(scheduled):
                remaining=start+due-time.perf_counter()
                if remaining>0: time.sleep(remaining)
                captured=time.perf_counter()
                item={'seq':seq,'camera':camera,'camera_seq':local_seq,'due_s':due,
                      'capture':captured,'due':start+due,'source_late_ms':max(0,(captured-start-due)*1000)}
                if c.workload=='stream': item['frame_index']=min(len(frames)-1,int(due*c.source_fps))
                item['enqueued']=time.perf_counter()
                dropped=admission.offer(item)
                if dropped is not None:
                    dropped.update(status='rejected' if c.policy=='reject-newest' else 'evicted',
                                   drop_time=time.perf_counter())
                    completions.put(dropped)
        except BaseException as e: errors.append('source: '+repr(e))
        finally: admission.close()

    def process():
        nonlocal active,max_active
        try:
            while True:
                slots.acquire()
                item=admission.take()
                if item is None: slots.release(); break
                item['dequeued']=time.perf_counter()
                with active_lock: active+=1; max_active=max(max_active,active)
                try:
                    read_start=time.perf_counter()
                    if c.workload=='trigger': image=cv2.imdecode(np.frombuffer(images[item['camera']%len(images)],np.uint8),cv2.IMREAD_COLOR)
                    else: image=frames[item['frame_index']]
                    item['read_ms']=(time.perf_counter()-read_start)*1000
                    if image is None: raise RuntimeError('Image decode failed')
                    pre_start=time.perf_counter()
                    if c.workload=='trigger': tensor=preprocess(image,*backend.hw)
                    else: tensor,item['transform']=letterbox(image,backend.hw[0])
                    item['pre_ms']=(time.perf_counter()-pre_start)*1000
                    item['tensor']=tensor # keep input memory alive through completion
                    item['submitted']=time.perf_counter()
                    if c.mode=='async':
                        backend.session.run_async([backend.output_name],{backend.input_name:tensor},callback,item)
                    else: callback([backend.run(tensor)],item,None)
                except BaseException as e:
                    item['error']=repr(e); item['infer_done']=time.perf_counter(); completions.put(item)
        except BaseException as e: errors.append('processor: '+repr(e))

    producer=threading.Thread(target=produce,name='clock-source',daemon=True)
    processor=threading.Thread(target=process,name='preprocess-submit',daemon=True)
    producer.start(); processor.start()
    try:
        while len(records)<len(scheduled):
            try: item=completions.get(timeout=30)
            except queue.Empty: raise RuntimeError('Pipeline stalled for 30s; refusing partial success')
            if item['seq'] in reorder_pending or item['seq']<next_seq: raise RuntimeError('Duplicate completion')
            reorder_pending[item['seq']]=item
            while next_seq in reorder_pending:
                current=reorder_pending.pop(next_seq); next_seq+=1
                if 'drop_time' in current:
                    records.append({k:v for k,v in current.items() if k not in ('tensor','output')})
                    continue
                try:
                    if current.get('error'): raise RuntimeError(current['error'])
                    post_start=time.perf_counter()
                    if c.workload=='trigger': current['top5']=postprocess(current['output'],5)
                    else:
                        detections=decode_yolo(current['output'],current['transform'],class_id=c.detection_class)
                        current['detections']=counter.update(current['seq'],current['due_s'],detections)
                        current['crossing_count']=counter.count
                    now=time.perf_counter()
                    current.update(status='completed',finished=now,post_ms=(now-post_start)*1000,
                        age_ms=(now-current['due'])*1000,
                        capture_e2e_ms=(now-current['capture'])*1000,
                        queue_ms=(current['dequeued']-current['enqueued'])*1000,
                        infer_ms=(current['infer_done']-current['submitted'])*1000,
                        reorder_ms=(post_start-current['infer_done'])*1000)
                    current['deadline_met']=current['age_ms']<=c.deadline_ms
                    if observer is not None: observer(current)
                    records.append({k:v for k,v in current.items() if k not in ('tensor','output')})
                finally:
                    with active_lock: active-=1
                    slots.release()
    finally:
        producer.join(timeout=3); processor.join(timeout=3)
    backend._assert_providers_intact()
    if errors: raise RuntimeError('; '.join(errors))
    if producer.is_alive() or processor.is_alive(): raise RuntimeError('Worker failed to drain')
    completed=[r for r in records if r['status']=='completed']; dropped=len(records)-len(completed)
    assert len(records)==len(scheduled) and len({r['seq'] for r in records})==len(scheduled)
    assert max_active<=depth and admission.peak<=c.capacity
    elapsed=max(c.duration_s,time.perf_counter()-start)
    timely=sum(r['deadline_met'] for r in completed)
    per_camera=[]
    for camera in range(c.cameras):
        group=[r for r in records if r['camera']==camera]; done=[r for r in group if r['status']=='completed']
        per_camera.append({'camera':camera,'offered':len(group),'completed':len(done),'dropped':len(group)-len(done),
                           'timely':sum(r['deadline_met'] for r in done),'age_ms':stats([r['age_ms'] for r in done])})
    summary={'offered':len(records),'completed':len(completed),'dropped':dropped,
        'source_seconds':c.duration_s,'drain_seconds':max(0,elapsed-c.duration_s),
        'elapsed_seconds':elapsed,'completed_per_wall_second':len(completed)/elapsed,
        'timely':timely,'timely_per_offered':timely/len(records),
        'timely_per_completed':timely/len(completed) if completed else 0,
        'peak_waiting':admission.peak,'max_active':max_active,'order_violations':0,
        'age_ms':stats([r['age_ms'] for r in completed]),
        'capture_e2e_ms':stats([r['capture_e2e_ms'] for r in completed]),
        'queue_ms':stats([r['queue_ms'] for r in completed]),
        'inference_ms':stats([r['infer_ms'] for r in completed]),
        'source_late_ms':stats([r['source_late_ms'] for r in records]),'per_camera':per_camera,
        'crossing_count':counter.count if counter else None,
        'expected_crossings':c.expected_crossings if counter else None}
    return {'config':vars(c),'summary':summary,'records':records,'backend':backend.describe()}
