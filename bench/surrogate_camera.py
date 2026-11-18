"""Deterministic virtual raw-BGR camera; arbitrary rates, no repeated video IDs."""
import cv2
import numpy as np

class SurrogateCamera:
    def __init__(self,crop_path,duration_s=10,fps=120):
        self.crop=cv2.imread(str(crop_path))
        if self.crop is None: raise ValueError('Object crop missing; prepare assets first')
        self.duration=duration_s;self.fps=fps
    def __len__(self):return round(self.duration*self.fps)
    def __getitem__(self,index):
        if not 0<=index<len(self):raise IndexError(index)
        t=index/self.fps;frame=np.full((360,640,3),180,np.uint8)
        cv2.rectangle(frame,(0,260),(639,359),(90,90,90),-1)
        ch,cw=self.crop.shape[:2]
        for ident in range(3):
            local=t-ident*self.duration/3
            if 0<=local<self.duration/3:
                left=round(-cw+(640+2*cw)*local/(self.duration/3));lo=max(0,left);hi=min(640,left+cw)
                if hi>lo:frame[100:100+ch,lo:hi]=self.crop[:,lo-left:hi-left]
        return frame
