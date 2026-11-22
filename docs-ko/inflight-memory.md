# 동시 요청 수와 메모리 사용 비교

> 후속 재측정에서 GPU의 SW Thermal Slowdown과 큰 SM 클럭 변화, 일부 실행의 높은 CPU 부하를 확인했습니다. 아래 처리율 표는 요청 수만의 효과로 해석하지 않습니다. 메모리·정합성 확인과 성능 순위를 분리합니다. 조사 결과는 `results/inflight-investigation/20261122-interleaved/`에 보존했습니다.

## 실행 조건

MX150 2GiB, ONNX Runtime CUDA EP, ResNet50 batch 1, Imagenette 실제 이미지 300장, warmup 30장 제외, 요청 수별 3회 반복입니다. 이미지 파일 읽기·JPEG 디코딩을 포함합니다. 요청 수마다 새 프로세스·새 세션을 사용해 앞선 요청 수에서 확장된 GPU arena가 다음 조건에 남지 않도록 했습니다. 각 프로세스 안에서는 하나의 모델 세션을 공유합니다. ORT intra-op threads=4, pipeline queue size=4를 유지했습니다.

## 결과

| 미완료 요청 상한 | 처리율 평균 [최소,최대] (장/s) | E2E p95 평균 (ms) | GPU 메모리 관측 최고 (MiB) | 관측 최소 여유 (MiB) | 프로세스 RSS 최고 (MiB) |
|---|---:|---:|---:|---:|---:|
| 동기 | 29.71 [24.49,34.09] | 41.0 | 389.7 | 1658.3 | 653.5 |
| 1 | 28.66 [27.75,29.79] | 309.6 | 427.7 | 1620.3 | 563.1 |
| 2 | 31.39 [30.78,32.61] | 366.2 | 427.7 | 1620.3 | 563.1 |
| 4 | 22.54 [14.49,31.76] | 800.3 | 693.7 | 1354.3 | 574.2 |
| 8 | 19.78 [16.24,24.22] | 1463.0 | 435.7 | 1612.3 | 567.8 |
| 16 | 33.52 [32.09,34.77] | 1179.9 | 691.7 | 1356.3 | 581.7 |
| 32 | 35.07 [33.17,38.06] | 1885.5 | 695.7 | 1352.3 | 591.9 |

현재 설정에서 미완료 요청 상한 32까지 OOM 없이 완료했습니다. 모든 조건에서 300개 frame ID와 Top-5를 기존 동기 기준과 대조했고, 누락·중복·순서 위반·오류·stall 없이 통과했습니다. 32는 이번에 확인한 값이며 메모리가 허용하는 최대치로 측정한 값은 아닙니다.

GPU 메모리는 NVML로 100ms마다 관측한 장치 전체 사용량입니다. 다른 프로세스·그래픽 사용도 포함되고, 짧은 할당 peak를 놓칠 수 있습니다. 활성 tensor 크기나 이 프로세스만의 VRAM 사용량이 아닙니다. RAM은 해당 프로세스의 RSS입니다. 원본 JSON에 세션 생성 전·후 값과 샘플 전체를 보존했습니다.

## 요청 수와 실제 실행 수

`inflight=8`은 run_async에 제출한 뒤 아직 callback으로 회수되지 않은 요청을 최대 8개 허용한다는 뜻입니다. GPU에서 8개 모델 실행이 동시에 진행된다는 뜻은 아닙니다. ORT run_async는 intra-op C++ threadpool에서 실행되며, 현재 threads=4 설정에서는 더 큰 요청 상한의 일부가 runtime 내부에서 기다릴 수 있습니다. GPU stream·커널의 실제 중첩은 별도의 타임라인 계측 대상입니다.

따라서 GPU 메모리가 요청 상한과 비례하지 않아도 이상하지 않습니다. 스레드 수·세션 수·batch·모델·입력 크기를 바꾸면 이 메모리 결과를 그대로 적용할 수 없습니다.

## 메모리 구성

같은 세션을 공유하는 추론의 대략적인 구성은 다음과 같습니다.

```text
GPU = 세션 공유 가중치 + CUDA/런타임 상태
    + 실제 겹쳐 실행되는 요청의 activation·입출력·workspace
    + allocator가 확보해 둔 여유·캐시

RAM = 런타임 상태 + 이미지 큐 + 제출 대기 입력 tensor + 결과 큐
```

요청마다 모델 가중치 전체를 다시 로딩하지 않습니다. forward 추론은 중간 tensor의 수명이 끝나면 메모리를 재사용하므로 모든 layer의 activation을 단순 합산해 곱하는 계산도 맞지 않습니다. cuDNN workspace와 allocator의 예약 메모리도 따로 고려해야 합니다.

CPU에 생성하는 FP32 NCHW 입력은 이미지당 `1×3×224×224×4 = 602,112 bytes`, 약 0.574MiB입니다. 제출 대기 입력 32개만 계산하면 약 18.4MiB입니다. 이것은 GPU activation 메모리 계산과 별개입니다.

## 재실행

프로젝트 루트에서 실행합니다. 기본 출력은 새 timestamp 디렉터리입니다.

```powershell
cd C:\Users\ysjun5656\deepX\edge-inference-runtime-lab
& .\.venv-cuda\Scripts\python.exe -m bench.classification.sweep_inflight --depths 1 2 4 8 16 32 --frames 300 --warmup 30 --repeat 3
```

이 도구는 프로세스 분리, 입력·출력 정합성 검사, NVML/RSS 샘플링을 수행합니다. 실패·timeout 또는 관측 GPU 여유 256MiB 미만이면 요청 수 증가를 중단합니다. 이는 OOM을 사전에 완전히 보장해 막는 VRAM 예약 기능은 아닙니다.

[ORT run_async와 threadpool](https://onnxruntime.ai/docs/api/python/api_summary.html#onnxruntime.InferenceSession.run_async)
[CUDA EP arena·stream 설정](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
