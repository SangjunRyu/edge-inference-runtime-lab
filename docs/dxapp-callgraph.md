# DX-APP: where one frame goes (sync path)

Reading notes. Paths relative to `DEEPX-AI/dx_app` @ main.

## Entry is thin by design

```
src/python_example/classification/resnet50/resnet50_sync.py      31 lines
    factory = Resnet50Factory()
    SyncRunner(factory).run(args)

src/cpp_example/classification/resnet50/resnet50_sync.cpp        13 lines
    auto factory = std::make_unique<dxapp::ResNet50Factory>();
    dxapp::SyncClassificationRunner<dxapp::ResNet50Factory> runner(std::move(factory));
    return runner.run(argc, argv);
```

The entire model-specific surface is the factory (29 lines) and `config.json`,
whose whole content is `{"top_k": 5}`.

`factory/resnet50_factory.py` only assembles — `SimpleResizePreprocessor`,
`ClassificationPostprocessor`, `ClassificationVisualizer` — and answers
`get_model_name()` / `get_task_type()`.

## Call graph — fill in while reading

```
resnet50_sync.py
  └─ Resnet50Factory
  └─ SyncRunner.run(args)
       ├─ input source            <- common/inputs/image_source.py
       ├─ [timer] Read
       ├─ [timer] Preprocess      <- SimpleResizePreprocessor
       ├─ [timer] Inference       <- ie.run()
       ├─ [timer] Postprocess     <- ClassificationPostprocessor (top_k=5)
       └─ ProfilingMetrics -> print_sync_performance_summary
```

## TODO

- [ ] Which call exactly does each of the 7 timing sums in `sync_runner.py`
      wrap? (`sum_read`, `sum_preprocess`, ...)
- [ ] For each arrow: dtype / shape / layout before and after.
- [ ] Does the reported `Inference` row include the output copy, or stop at the
      runtime boundary?
- [ ] Is visualisation inside or outside the reported FPS?

## Structural asymmetry between the two languages

| | Python | C++ |
|---|---|---|
| Runner | 2 generic: `common/runner/sync_runner.py`, `async_runner.py` | 16 task-specific pairs: `sync_classification_runner.hpp`, `sync_detection_runner.hpp`, `sync_pose_runner.hpp`, ... |
| Dispatch | runtime polymorphism, factory passed as an object | template parameter, `SyncClassificationRunner<ResNet50Factory>` |
| Ownership | GC | `std::unique_ptr` + `std::move` into the runner |

Not a 1:1 port. The same five layers (entry → runner → factory → component →
interface) expressed with each language's own dispatch and ownership model.

## Where DX-RT sits

`dx_app` is the application layer. `run()` / `run_async()` / `wait()` /
`get_latency()` / `get_npu_inference_time()` come from
`dx_rt/python_package/src/dx_engine/inference_engine.py`, a separate
repository. `InferenceOption` exposes `use_ort`, `devices`, `bound_option`,
`buffer_count` — `buffer_count` being the in-flight depth knob.

W1 deliberately stops at this boundary. The question is not how the runtime
drives the NPU; it is how an application must call a runtime to keep an
accelerator busy.
