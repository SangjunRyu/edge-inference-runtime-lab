# Clock-driven inspection and counting scenarios

[Korean findings](../docs-ko/10-industrial-scenarios.md) |
[Recorded evidence](../results/industrial-20261118-public/)

These experiments turn the existing inference pipeline into two application
contracts. They use Python orchestration and real ONNX Runtime CPU/CUDA
inference. They are independent implementations inspired by DX-APP's public
preprocess / async inference / postprocess structure, not measurements of a
DEEPX device or DX-RT. Existing native C++ runners are unchanged by this suite.

## 1. Triggered multi-camera image classification

Four virtual cameras submit a JPEG when a product event is due. Each request
has a global ID, camera ID, per-camera sequence and scheduled timestamp. Camera
bursts are simultaneous, with rotating tie order to avoid always admitting
camera 0 first. A shared ResNet50 session classifies the images.

* Normal load: 4 cameras × 2 triggers/s = 8 requests/s.
* Overload: 4 cameras × 12 triggers/s = 48 requests/s.
* Compare sync and async (depth 4), queue capacity 4, CPU and CUDA.
* Verdict deadline: 200 ms from the scheduled product event, an experimental
  assumption rather than a requirement from a real production line.
* Full queue: reject the new request and record its product ID. The producer
  never waits, and rejected requests are not presented as inspected products.

Each camera repeats a different crop of the public test image. Top-5 results
are checked against that camera's synchronous reference for every completed
request. This checks value pairing, not normal/defective classification
accuracy. Per-camera completion, rejection and deadline outcomes are retained.

The recorded 8 requests/s case completed every request, yet CPU outputs met
the illustrative 200ms deadline only about 64% (sync) / 46% (async) of the time:
four simultaneous camera arrivals made average capacity an insufficient test
of per-product response time. Consult the published per-camera traces before
generalizing this observation to a different trigger pattern.

## 2. Streaming detection, tracking and line crossing

YOLO11n COCO, static FP32 ONNX 320×320, opset 17, raw output `[1,84,2100]`.
Processing is letterbox → RGB float NCHW → ORT → confidence filter → NMS →
coordinate restoration → ordered tracking → one-way crossing count. Only COCO
class 5 (bus) is used in this controlled example.

On this MX150/ORT 1.26/CUDA-cuDNN combination, concurrent YOLO requests with
the default per-thread CUDA stream configuration produced intermittent box
coordinate mismatches despite matching class scores and correct frame order.
Copying callback outputs did not eliminate it; CPU concurrency and CUDA depth
1 controls passed. `use_ep_level_unified_stream=1` contained the problem in
repeated checks, so all final CUDA streaming scenarios select that option.
This is a measured containment, not a confirmed library/driver root cause.
The initial CUDA streaming timings are invalidated and retained separately.
See `repro/yolo_cuda_concurrency.py --mode broken|fixed` and the saved model
diagnostics. The verifier refuses a mismatching configuration before profiling.

The public bus image supplies an object crop. A deterministic virtual raw-BGR
camera moves three copies one at a time across a 640×360 scene over 10 seconds.
The geometric ground truth is three crossings. Its frame function supports
120 distinct timestamps/s; the high-rate test is not a 30 FPS video duplicated
four times. A 30 FPS MP4 preview is also generated for inspection.

* Normal input: 5 frames/s, sync and async.
* Stress input: 120 frames/s, deliberately above the measured ~60 FPS pilot
  capacity. This is a virtual-camera stress assumption, not observed hardware.
* Compare reject-newest and drop-oldest, depth 1/4, queue capacity 8, plus CUDA
  capacity 2. Deadline for a fresh application output: 100 ms.
* Rejection preserves older queued work; drop-oldest keeps newer pending work.
  Neither policy cancels inference already in progress.
* The collector restores frame order across both inference completions and
  explicit drop markers before passing detections to the tracker.

The centroid tracker is intentionally small. It is not production multi-object
tracking: the scene has no clutter, occlusion, camera motion or domain shift.
Preserving three crossings here does not establish safe frame-dropping rates
for a real conveyor. A COCO bus detector is not an industrial product detector.

## Measurements and invariants

The producer uses its own clock and cannot slow down to hide overload. Every
scheduled input is completed, rejected or evicted; their counts must balance.
Outstanding work is bounded by queue capacity plus active permits (including
postprocessing), and callbacks hand off results without waiting on a full queue.

* **Output age:** application-output time minus scheduled acquisition time.
  Includes source lateness, queueing, preprocessing, runtime completion,
  reordering and postprocessing.
* **Capture E2E:** application-output time minus the observed arrival of the
  virtual camera descriptor. This is not physical sensor exposure latency.
* **Source lateness:** observed descriptor arrival minus scheduled arrival.
* **Timely / offered:** timely outputs divided by all scheduled inputs; drops
  and late outputs both reduce this coverage metric.
* **Timely / completed:** freshness of retained outputs. A high value can
  coexist with many dropped inputs, so report it beside drop rate/coverage.
* Queue wait, inference-call duration, bounded occupancy, explicit drop IDs,
  per-camera outcomes and crossing count are also saved.

No FPS claim is made by measuring only surviving requests. Idle-gate outcomes
and source lateness remain in the data. Two repetitions support an exploratory
scenario demonstration, not a confidence interval or a production SLA.

In the corrected CUDA 120 FPS experiment, depth 1 / queue 8 with new-request
rejection had output-age p95 about 222ms. Replacing the oldest queued frame
reduced it to about 88ms; a queue of 2 reduced it to about 37ms. Those settings
dropped about 53/55/56% of offered frames respectively. This trades coverage
for freshness; it is not a lossless inference speedup. All figures and the
16-run CUDA correction history are in the published evidence linked above.

## Reproduce

Use the already working `.venv-cuda`; neither PyTorch nor Ultralytics is needed
to run ONNX experiments on Windows. The initial ONNX export was done in a
separate model-export step with Ultralytics 8.4.183 / torch 2.3.1 / ONNX 1.23.2,
`imgsz=320`, `batch=1`, `opset=17`, `dynamic=False`, `simplify=False`, `nms=False`.
Export example in an environment containing those tools:

```python
from ultralytics import YOLO
YOLO('yolo11n.pt').export(format='onnx', imgsz=320, batch=1,
    opset=17, dynamic=False, simplify=False, nms=False, device='cpu')
```

Place the resulting model at `models/yolo11n_320.onnx` alongside ResNet50, then:

```powershell
.\.venv-cuda\Scripts\python.exe tools/prepare_industrial_assets.py
.\.venv-cuda\Scripts\python.exe -m unittest discover -s tests -p test_industrial.py -v
.\.venv-cuda\Scripts\python.exe tools/run_industrial_scenarios.py --repeat 2 `
    --out-dir results/industrial-rerun
.\.venv-cuda\Scripts\python.exe tools/verify_industrial_model.py results/yolo-verification
.\.venv-cuda\Scripts\python.exe tools/analyze_industrial.py results/industrial-rerun
```

The source image is downloaded from the upstream Ultralytics repository:
https://github.com/ultralytics/ultralytics/blob/main/ultralytics/assets/bus.jpg.
Record its hash and the model hash with the experiment; weights and generated
assets are local experiment inputs. Consult upstream terms before distributing
the model/image as part of another product. Source scene assets/metadata are
under ignored `clips/scenario-assets/`; results include per-request traces.

Public architectural references:

* https://github.com/DEEPX-AI/dx_app — task-specific sync/async runners,
  input sources, model processors and tracker separation.
* https://developer.deepx.ai/modelzoo/ — classification and detection models.
* https://docs.ultralytics.com/modes/export/ — ONNX export and input-size options.

## What a subsequent real deployment must supply

Actual product images/video and labels; a suitable trained detector or defect
classifier; physical trigger/exposure timestamps; product spacing/belt speed;
actuator deadline; permissible reject/drop rules; occlusion and missed-count
tests. A DEEPX backend would then need actual SDK/hardware integration and new
measurements rather than relabelling these CPU/CUDA numbers.
