"""Detect people in actual video frames with bounded, ordered ORT requests.

Offline file processing: no source-clock simulation or frame drops. Video
encoding/display are included in elapsed time, so FPS is not infer-only FPS.
"""
import argparse
from collections import deque
from concurrent.futures import Future
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
import cv2
import numpy as np
from bench.runtime.backends import OnnxBackend
from bench.detection.yolo import letterbox, decode_yolo, LineCounter
from bench.measurement.statistics import stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', type=Path, default=ROOT/'clips/real-video/vtest.avi')
    p.add_argument('--model', type=Path, default=ROOT/'models/yolo11n_320.onnx')
    p.add_argument('--backend', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--mode', choices=['sync', 'async'], default='async')
    p.add_argument('--inflight', type=int, default=2)
    p.add_argument('--max-frames', type=int, default=0, help='0: whole video')
    p.add_argument('--confidence', type=float, default=.25)
    p.add_argument('--out-dir', type=Path, default=ROOT/'results/real-video')
    p.add_argument('--display', action='store_true', help='Show preview; q or ESC stops')
    p.add_argument('--verify-every', type=int, default=0,
                   help='Compare every Nth async raw output against sync; affects timing')
    a = p.parse_args()
    if a.inflight < 1 or a.max_frames < 0 or a.verify_every < 0:
        p.error('Invalid frame/request limit')
    a.out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(a.video))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open {a.video}')
    source_backend = cap.getBackendName()
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError('Video has no valid FPS')
    w, h = int(cap.get(3)), int(cap.get(4))
    backend = OnnxBackend(str(a.model), a.backend, 4,
                          cuda_options={'use_ep_level_unified_stream': '1'})
    backend.run(np.zeros((1, 3, 320, 320), np.float32))
    writer = cv2.VideoWriter(str(a.out_dir/'detection.mp4'),
                             cv2.VideoWriter_fourcc(*'mp4v'), fps, (w*2, h))
    if not writer.isOpened():
        raise RuntimeError('Cannot create output video')
    tracker = LineCounter(line_x=w//2, max_distance=60, max_age_s=.5)
    pending = deque()
    records = []
    stop = False
    depth = a.inflight if a.mode == 'async' else 1
    peak = 0
    verified = 0

    def callback(outputs, item, error):
        item['finished'] = time.perf_counter()
        if error:
            item['future'].set_exception(RuntimeError(str(error)))
        else:
            item['future'].set_result(outputs[0])

    def consume(item):
        nonlocal stop, verified
        output = item['future'].result(timeout=60)
        if a.verify_every and item['seq'] % a.verify_every == 0:
            # Drain before synchronous checks to isolate the reference call.
            for other in pending:
                other['future'].result(timeout=60)
            reference = backend.run(item['tensor'])
            if not np.allclose(reference, output, rtol=1e-3, atol=1e-3):
                raise RuntimeError(f'Sync/async mismatch at frame {item["seq"]}')
            verified += 1
        detections = decode_yolo(output, item['transform'], a.confidence, class_id=0)
        tracked = tracker.update(item['seq'], item['seq']/fps, detections)
        annotated = item['frame'].copy()
        for d in tracked:
            x1, y1, x2, y2 = map(round, d['box'])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"person #{d['track_id']} {d['score']:.2f}",
                        (x1, max(40, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, .45, (0,255,0), 1)
        view = np.hstack([item['frame'], annotated])
        cv2.rectangle(view, (0, 0), (w*2, 32), (0, 0, 0), -1)
        cv2.putText(view, f'Original | frame {item["seq"]} | {item["seq"]/fps:.1f}s',
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 1)
        cv2.putText(view, f'{a.backend.upper()} {a.mode} | people: {len(tracked)}',
                    (w+10, 22), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 1)
        writer.write(view)
        if item['seq'] in (0, 100, 300):
            cv2.imwrite(str(a.out_dir/f'preview-{item["seq"]:04d}.jpg'), view)
        records.append({'frame': item['seq'], 'video_time_s': item['seq']/fps,
                        'submit_to_callback_ms': (item['finished']-item['submitted'])*1000,
                        'detections': tracked})
        if a.display:
            cv2.imshow('Actual video: original | detection (q to stop)', view)
            stop = cv2.waitKey(max(1, round(1000/fps))) & 255 in (27, ord('q'))

    start = time.perf_counter()
    seq = 0
    try:
        while not stop and (not a.max_frames or seq < a.max_frames):
            ok, frame = cap.read()
            if not ok:
                break
            tensor, transform = letterbox(frame, 320)
            item = dict(seq=seq, frame=frame, tensor=tensor, transform=transform,
                        future=Future(), submitted=time.perf_counter())
            if a.mode == 'async':
                backend.session.run_async([backend.output_name], {backend.input_name:tensor}, callback, item)
            else:
                callback([backend.run(tensor)], item, None)
            pending.append(item)
            peak = max(peak, len(pending))
            seq += 1
            if len(pending) >= depth:
                consume(pending.popleft())
        while pending:
            consume(pending.popleft())
        backend._assert_providers_intact()
    finally:
        cap.release()
        writer.release()
        if a.display:
            cv2.destroyAllWindows()
    elapsed = time.perf_counter()-start
    summary = dict(source=str(a.video.resolve()), source_sha256=hashlib.sha256(a.video.read_bytes()).hexdigest(),
                   source_fps=fps, source_size=[w,h], decode_backend=source_backend,
                   backend=backend.describe(), mode=a.mode, frames=len(records),
                   video_duration_s=len(records)/fps, processing_seconds=elapsed,
                   processing_fps=len(records)/elapsed, max_outstanding=peak,
                   frames_with_people=sum(bool(r['detections']) for r in records),
                   detection_instances=sum(len(r['detections']) for r in records),
                   submit_to_callback_ms=stats([r['submit_to_callback_ms'] for r in records]),
                   sync_reference_checks=verified,
                   notes='Offline, no frame dropping. FPS includes decode, preprocess, ordered postprocess and video encoding; verification/display add overhead. Person-only detection; simple centroid IDs are not accuracy labels.')
    (a.out_dir/'result.json').write_text(json.dumps(dict(summary=summary, records=records), indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
