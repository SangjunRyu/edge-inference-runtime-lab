"""YOLO preprocessing, NMS and simple centroid tracking."""
import math
import cv2
import numpy as np


def letterbox(frame,size):
    h,w=frame.shape[:2]; scale=min(size/w,size/h)
    nw,nh=round(w*scale),round(h*scale)
    left=(size-nw)//2; top=(size-nh)//2
    image=np.full((size,size,3),114,np.uint8)
    image[top:top+nh,left:left+nw]=cv2.resize(frame,(nw,nh),interpolation=cv2.INTER_LINEAR)
    tensor=np.ascontiguousarray(image[:,:,::-1].transpose(2,0,1),dtype=np.float32)[None]/255.0
    return tensor,(scale,left,top,w,h)


def decode_yolo(output,transform,confidence=0.25,iou=0.45,class_id=0):
    """YOLO11 raw COCO output [1,84,N], class-aware selection before NMS."""
    pred=np.asarray(output)[0]
    if pred.ndim!=2 or pred.shape[0]!=84: raise ValueError(f'Expected [1,84,N], got {output.shape}')
    scores=pred[4+class_id]; selected=np.flatnonzero(scores>=confidence)
    if not len(selected): return []
    xywh=pred[:4,selected].T.copy(); xywh[:,:2]-=xywh[:,2:]/2
    keep=cv2.dnn.NMSBoxes(xywh.tolist(),scores[selected].tolist(),confidence,iou)
    scale,left,top,w,h=transform; result=[]
    for index in np.asarray(keep).reshape(-1):
        x,y,bw,bh=xywh[index]
        box=[max(0.,min(w,(x-left)/scale)),max(0.,min(h,(y-top)/scale)),
             max(0.,min(w,(x+bw-left)/scale)),max(0.,min(h,(y+bh-top)/scale))]
        if box[2]>box[0] and box[3]>box[1]:
            result.append({'box':[float(v) for v in box],'score':float(scores[selected[index]]),'class':class_id})
    return result


class LineCounter:
    """Small centroid tracker; ordered timestamps required, no product semantics."""
    def __init__(self,line_x=320,max_distance=220,max_age_s=1.0):
        self.line=line_x; self.distance=max_distance; self.max_age=max_age_s
        self.tracks={}; self.next_id=1; self.count=0; self.last_seq=-1
    def update(self,seq,timestamp,detections):
        if seq<=self.last_seq: raise ValueError('Tracking requires strictly increasing frame IDs')
        self.last_seq=seq
        self.tracks={i:t for i,t in self.tracks.items() if timestamp-t['time']<=self.max_age}
        available=set(self.tracks); output=[]
        for detection in sorted(detections,key=lambda d:-d['score']):
            box=detection['box']; center=((box[0]+box[2])/2,(box[1]+box[3])/2)
            candidates=[(math.dist(center,self.tracks[i]['center']),i) for i in available]
            best=min(candidates,default=(math.inf,None))
            if best[0]<=self.distance:
                ident=best[1]; available.remove(ident); track=self.tracks[ident]
                if not track['counted'] and track['center'][0]<self.line<=center[0]:
                    self.count+=1; track['counted']=True
                track.update(center=center,time=timestamp)
            else:
                ident=self.next_id; self.next_id+=1
                self.tracks[ident]={'center':center,'time':timestamp,'counted':False}
            output.append(dict(detection,track_id=ident))
        return output
