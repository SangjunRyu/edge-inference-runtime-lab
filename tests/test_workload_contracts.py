import time
import threading
import unittest
import numpy as np
import cv2
from bench.workloads.clocked import AdmissionQueue, Config, run_workload
from bench.detection.yolo import LineCounter, decode_yolo, letterbox


class FakeBackend:
    """Only protocol tests use this; measured scenarios use actual ORT."""
    hw=(4,4)
    def __init__(self):self.session=self
    def run(self,tensor):time.sleep(.03);return np.arange(10,dtype=np.float32)[None]
    def run_async(self,names,feed,callback,data):raise AssertionError('Not used by protocol test')
    def _assert_providers_intact(self):pass
    def describe(self):return {'test_double':True}


class WorkloadContracts(unittest.TestCase):
    def test_reverse_callbacks_are_consumed_in_source_order(self):
        class ReverseBackend(FakeBackend):
            output_name='output';input_name='input'
            def run_async(self,names,feed,callback,data):
                def finish():
                    output=np.zeros((1,10),np.float32);output[0,data['camera']]=5
                    callback([output],data,None)
                threading.Timer(.04*(4-data['camera']),finish).start()
        _,jpeg=cv2.imencode('.jpg',np.zeros((4,4,3),np.uint8))
        c=Config(mode='async',duration_s=.2,cameras=4,fps_per_camera=1,inflight=4)
        result=run_workload(ReverseBackend(),c,images=[jpeg.tobytes()]*4)
        self.assertEqual([r['seq'] for r in result['records']],[0,1,2,3])
        self.assertEqual([r['top5'][0][0] for r in result['records']],[0,1,2,3])
        self.assertEqual(result['summary']['dropped'],0)

    def test_rejection_preserves_pending_work(self):
        q=AdmissionQueue(2,'reject-newest');q.offer(10);q.offer(20)
        self.assertEqual(q.offer(30),30);q.close()
        self.assertEqual([q.take(),q.take(),q.take()],[10,20,None])

    def test_freshness_replaces_old_pending_work(self):
        q=AdmissionQueue(2,'drop-oldest');q.offer(10);q.offer(20)
        self.assertEqual(q.offer(30),10);q.close()
        self.assertEqual([q.take(),q.take(),q.take()],[20,30,None])

    def test_coordinate_inverse_and_nms(self):
        _,transform=letterbox(np.zeros((360,640,3),np.uint8),320)
        raw=np.zeros((1,84,2),np.float32)
        raw[0,:4,0]=[160,160,100,50];raw[0,:4,1]=[160,160,100,50]
        raw[0,4]=[.9,.8] # COCO person class 0
        boxes=decode_yolo(raw,transform,class_id=0)
        self.assertEqual(len(boxes),1)
        np.testing.assert_allclose(boxes[0]['box'],[220,130,420,230],atol=.01)

    def test_count_once_and_reject_reversed_frames(self):
        tracker=LineCounter(320)
        for index,x in enumerate([250,300,330,325,340]):
            tracker.update(index,index*.1,[{'box':[x-20,100,x+20,140],'score':.9}])
        self.assertEqual(tracker.count,1)
        with self.assertRaises(ValueError):tracker.update(2,.2,[])

    def test_source_clock_and_accounting_under_overload(self):
        ok,jpeg=cv2.imencode('.jpg',np.zeros((4,4,3),np.uint8));self.assertTrue(ok)
        config=Config(mode='sync',duration_s=.5,cameras=1,fps_per_camera=100,capacity=2,deadline_ms=100)
        result=run_workload(FakeBackend(),config,images=[jpeg.tobytes()])
        s=result['summary'];self.assertEqual(s['offered'],50)
        self.assertEqual(s['completed']+s['dropped'],50)
        self.assertGreater(s['dropped'],20)
        self.assertLessEqual(s['peak_waiting'],2)
        self.assertLess(s['elapsed_seconds'],1.5)
        self.assertEqual([r['seq'] for r in result['records']],list(range(50)))

if __name__=='__main__':unittest.main()
