"""Offline visualization of measured results; never included in timing."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from bench.surrogate_camera import SurrogateCamera

def render(result_path,out_path):
    data=json.loads(Path(result_path).read_text());c=data['config'];s=data['summary']
    camera=SurrogateCamera(ROOT/'clips/scenario-assets/object-crop.jpg')
    complete=[r for r in data['records'] if r['status']=='completed']
    start=complete[0]['due']-complete[0]['due_s']
    writer=cv2.VideoWriter(str(out_path),cv2.VideoWriter_fourcc(*'mp4v'),30,(1280,440))
    if not writer.isOpened():raise RuntimeError('Video encoder unavailable')
    pointer=0;latest=None;poster=None
    for index in range(300):
        t=index/30
        while pointer<len(complete) and complete[pointer]['finished']-start<=t:
            latest=complete[pointer];pointer+=1
        live=camera[min(len(camera)-1,round(t*camera.fps))]
        result=np.full_like(live,60)
        age=None;count=0
        if latest:
            result=camera[latest['frame_index']]
            for d in latest['detections']:
                x1,y1,x2,y2=map(int,d['box']);cv2.rectangle(result,(x1,y1),(x2,y2),(0,200,0),2)
                cv2.putText(result,f"bus ID {d['track_id']}",(x1,max(15,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,80,0),1)
            age=(t-latest['due_s'])*1000;count=latest['crossing_count']
        for image in (live,result):cv2.line(image,(320,0),(320,359),(0,0,255),2)
        canvas=np.full((440,1280,3),245,np.uint8);canvas[80:]=np.hstack([live,result])
        text=f"YOLO11n / {data['name']} / virtual camera (surrogate, not factory footage)"
        cv2.putText(canvas,text,(12,22),cv2.FONT_HERSHEY_SIMPLEX,.48,(30,30,30),1)
        cv2.putText(canvas,f"LIVE SOURCE  t={t:.2f}s",(12,58),cv2.FONT_HERSHEY_SIMPLEX,.6,(30,30,30),1)
        age_text='waiting' if age is None else f'{age:.0f}ms old'
        cv2.putText(canvas,f"LATEST RESULT: {age_text} / count {count} (truth total 3)",(652,58),cv2.FONT_HERSHEY_SIMPLEX,.55,
                    (0,120,0) if age is not None and age<=100 else (0,0,180),1)
        writer.write(canvas)
        if index==150:poster=canvas.copy()
    writer.release()
    if poster is not None:cv2.imwrite(str(out_path.with_suffix('.jpg')),poster)
    cap=cv2.VideoCapture(str(out_path));ok,frame=cap.read()
    assert ok and frame.shape[:2]==(440,1280) and int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==300
    cap.release()
    return {'video':str(out_path),'frames':300,'observed_count':s['crossing_count']}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('result',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True);print(render(a.result,a.output))
