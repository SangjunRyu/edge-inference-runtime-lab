"""Public-image surrogate with known trajectories, not an industrial dataset."""
import argparse, hashlib, json, sys, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from bench.backends import OnnxBackend
from bench.industrial import letterbox, decode_yolo, LineCounter

URL='https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg'

def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',default='cuda');a=p.parse_args()
    folder=ROOT/'clips/scenario-assets';folder.mkdir(parents=True,exist_ok=True)
    image_path=folder/'bus.jpg'
    if not image_path.exists(): urllib.request.urlretrieve(URL,image_path)
    image=cv2.imread(str(image_path));h,w=image.shape[:2]
    for i in range(4):
        # Four virtual camera snapshots, all from the same disclosed public image.
        view=image if i==0 else image[round(h*0.05*i):round(h*(1-0.03*i)),round(w*0.02*i):round(w*(1-0.02*i))]
        cv2.imwrite(str(folder/f'camera-{i}.jpg'),view,[cv2.IMWRITE_JPEG_QUALITY,90])
    b=OnnxBackend(str(ROOT/'models/yolo11n_320.onnx'),a.backend,4)
    if b.fell_back: raise RuntimeError('Requested backend unavailable')
    x,transform=letterbox(image,320);detections=decode_yolo(b.run(x),transform)
    if not detections: raise RuntimeError('Public source image did not yield a bus detection')
    best=max(detections,key=lambda d:d['score']);x1,y1,x2,y2=map(int,best['box'])
    crop=image[y1:y2,x1:x2];cw=200;ch=round(crop.shape[0]/crop.shape[1]*cw)
    crop=cv2.resize(crop,(cw,ch))
    cv2.imwrite(str(folder/'object-crop.jpg'),crop)
    fps=30;duration=10;frames=[];truth=[]
    writer=cv2.VideoWriter(str(folder/'conveyor-surrogate.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(640,360))
    if not writer.isOpened(): raise RuntimeError('Video encoder unavailable')
    for frame_id in range(duration*fps):
        t=frame_id/fps;frame=np.full((360,640,3),180,dtype=np.uint8);objects=[]
        cv2.rectangle(frame,(0,260),(639,359),(90,90,90),-1)
        for ident in range(3):
            local=t-ident*duration/3
            if 0<=local<duration/3:
                left=round(-cw+(640+2*cw)*local/(duration/3));top=100
                lo=max(0,left);hi=min(640,left+cw)
                if hi>lo:
                    frame[top:top+ch,lo:hi]=crop[:,lo-left:hi-left]
                    objects.append({'id':ident,'box':[left,top,left+cw,top+ch]})
        truth.append({'frame':frame_id,'time_s':t,'objects':objects})
        writer.write(frame);frames.append(frame)
    writer.release()
    # Real inference on every fifth frame gives an asset sanity check. The
    # ground truth count below comes from trajectories, never from this check.
    tracker=LineCounter();preview=frames[145].copy();evidence=[]
    for index in range(0,len(frames),5):
        tensor,transform=letterbox(frames[index],320)
        boxes=decode_yolo(b.run(tensor),transform)
        tracked=tracker.update(index,index/fps,boxes)
        evidence.append({'frame':index,'detections':len(boxes),'count':tracker.count})
        if index==145:
            for d in tracked:
                x1,y1,x2,y2=map(int,d['box']);cv2.rectangle(preview,(x1,y1),(x2,y2),(0,180,0),2)
                cv2.putText(preview,f"bus track={d['track_id']}",(x1,max(15,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,80,0),1)
    cv2.line(preview,(320,0),(320,359),(0,0,255),2)
    cv2.imwrite(str(folder/'preview.jpg'),preview)
    manifest={'scenario':'controlled surrogate object line crossing','source_image':URL,'source_image_sha256':hashlib.sha256(image_path.read_bytes()).hexdigest(),
        'model':'YOLO11n COCO FP32 320x320 opset17 raw output [1,84,2100]','duration_s':duration,'fps':fps,
        'ground_truth_crossings':3,'line_x':320,'objects':3,'sanity_detected_crossings':tracker.count,
        'sanity_samples':evidence,'limitation':'Same public bus crop repeated on a synthetic background; no industrial product or defect accuracy claim.'}
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (folder/'ground-truth.json').write_text(json.dumps(truth),encoding='utf-8')
    print(json.dumps(manifest),flush=True)

if __name__=='__main__':main()
