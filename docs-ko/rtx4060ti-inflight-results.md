# ResNet50 CUDA 실측 · RTX 4060 Ti 비동기 깊이 스윕

2026-11-22 · Intel Core i5-14400F / GeForce RTX 4060 Ti 8 GiB / Windows / ONNX Runtime 1.32.0 CUDA EP / NVIDIA 드라이버 610.88

동시 요청 상한 8에서 상한 32 처리량의 **96%**를 얻었다. 상한 16에서 32로 늘리면 처리량은 **493.83 → 495.33 images/s (+0.3%)**, E2E p95는 **80.70 → 142.14 ms (+76%)**였다. 단기 NVML 샘플에서 SW Thermal Slowdown은 관측되지 않았다.

## 실험 구성과 원시 결과

| 실험 | 조건 | 실행 수와 검증 |
|---|---|---|
| [입력 방식 비교](../results/image-classification/20261122T115718Z/SUMMARY.md) | 동일 Imagenette-320 입력 500장, 워밍업 50장. 파일 읽기+디코딩 / 사전 디코딩 × 동기 및 비동기 깊이 1·2·4·8·16·32 × 3회 | [42회](../results/image-classification/20261122T115718Z/results.json) 모두 Top-5 정합성과 실행 검증 통과 |
| [깊이별 새 프로세스 스윕](../results/inflight-memory/20261122T122055Z/SUMMARY.md) | 파일 읽기+디코딩 300장, 워밍업 30장. 깊이마다 새 프로세스·세션을 열고 3회 실행 | [21회](../results/inflight-memory/20261122T122055Z/status.json) 모두 [별도로 생성한 CUDA 동기 기준](../results/pcbang-reference/top5.json)의 해당 300장과 일치. 비동기 18회 모두 출력 순서 위반·오류·정체·강제 플러시 0건 |

두 실험 모두 배치 크기 1이었다. 파일 입력은 OS 캐시가 데워진 상태에서 읽기·디코딩을 포함했고, 사전 디코딩 준비는 측정 구간 밖에서 수행했다. 표의 수치는 회차별 통계의 산술평균이다. 동시 요청 상한은 진행 중인 추론 요청의 한도이며, GPU 커널 동시 실행 수는 계측하지 않았다.

입력 방식 비교에서 파일 입력의 동기 / 깊이 8 / 깊이 32는 각각 **232.40 / 454.02 / 477.61 images/s**, E2E p95는 **5.24 / 49.50 / 148.20 ms**였다. 사전 디코딩에서도 깊이 8 → 32의 처리량은 **459.33 → 465.97 images/s (+1.4%)**였고 p95는 **46.34 → 150.33 ms**로 늘었다. 3회차는 14개 중 13개 조건에서 1회차보다 처리량이 낮았다. 이 실행은 GPU 열 이벤트를 수집하지 않아 회차별 하락의 원인은 미확인이다.

## 깊이별 새 프로세스 스윕

| 깊이 | 처리량 평균 [최소, 최대] (images/s) | E2E p95 평균 (ms) | 장치 VRAM 피크 (MiB) |
|---:|---:|---:|---:|
| 0 (동기) | 255.00 [252.81, 257.08] | 4.52 | 1028.6 |
| 1 | 371.90 [364.56, 380.60] | 20.29 | 1036.6 |
| 2 | 423.15 [412.56, 428.52] | 21.38 | 1058.5 |
| 4 | 446.45 [428.92, 456.54] | 27.79 | 1073.0 |
| 8 | 475.01 [440.27, 492.92] | 45.79 | 1077.8 |
| 16 | 493.83 [487.16, 497.58] | 80.70 | 1062.4 |
| 32 | 495.33 [487.19, 503.59] | 142.14 | 1062.4 |

깊이 16과 32의 처리량 범위는 겹친다. 같은 구간에서 추론 제출 → 콜백 평균은 **32.64 → 65.81 ms**, 재정렬 대기 평균은 **20.68 → 37.15 ms**로 늘었다. 읽기·전처리·후처리 평균은 각각 0.02 ms 이내에서 유지됐다. 처리 용량에 가까워진 상태에서 추가 요청이 런타임에 머무는 시간과 순서 복원 대기를 늘렸다는 설명이 계측값과 일치한다. 제출 → 콜백에는 런타임 대기가 포함되므로 CUDA 커널, 전송, 런타임 내부 큐 중 병목 위치를 확인하려면 타임라인 계측이 필요하다. VRAM은 장치 전체의 100 ms 샘플 피크로, 활성 텐서 메모리의 구성은 확인할 수 없다.

[깊이 32 원시 기록](../results/inflight-memory/20261122T122055Z/depth-32.json)에서 3회 모두 300개 결과를 출력했다. 재정렬 버퍼는 회차당 **284~291건**을 순서 밖으로 받았으며 최종 출력 순서 위반은 0건이었다. 깊이 8의 첫 실행과 두 번째 실행 사이에는 약 44초의 간격이 있어, 프로세스 전체 경과 시간과 각 회차의 처리량을 구분했다. 간격의 원인은 기록에서 확인되지 않는다.

## MX150 기록과 열 상태 비교

| 환경 | 파일 입력 스윕 깊이 0 / 16 / 32 처리량 (images/s) | 깊이 0 / 16 / 32 E2E p95 (ms) | 열 이벤트 확인 범위 |
|---|---:|---:|---|
| [MX150 / ORT 1.26.0](../results/inflight-memory/20261122-depth32/SUMMARY.md) | 29.71 / 33.52 / 35.07 | 41.01 / 1179.89 / 1885.47 | 별도 [교차 실행 조사](../results/inflight-investigation/20261122-interleaved)에서 SW Thermal Slowdown 463/792 샘플, 최고 82°C |
| [RTX 4060 Ti / ORT 1.32.0](../results/inflight-memory/20261122T122055Z/SUMMARY.md) | 255.00 / 493.83 / 495.33 | 4.52 / 80.70 / 142.14 | 이번 스윕에서 SW Thermal Slowdown 0/589 샘플, 최고 56°C |

두 스윕은 동일한 300장 입력 순서, 요청 상한, 워밍업 수, 반복 수를 사용했다. CPU·GPU·ORT·드라이버와 ONNX 파일 SHA-256이 달라 GPU 단독 성능 배수는 산출하지 않았다. 모델 아티팩트는 각 실행의 JSON에 기록되어 있다. RTX 샘플에서 확인된 클럭 이벤트는 0과 1뿐이었다. [NVML 정의](https://docs.nvidia.com/deploy/archive/R550/nvml-api/group__nvmlClocksEventReasons.html)에서 1은 GPU idle이고 SW Thermal Slowdown 비트는 0x20이다. 깊이 16·32의 실행 창은 회당 약 0.66~0.98초였다. 열 제한 미관측은 이 샘플링 범위에 해당한다.

## 재현

[환경·모델·데이터 준비 가이드](cross-machine-images.md)에 따라 CUDA 가상 환경과 고정 Imagenette 매니페스트를 준비한다. 같은 환경에서 로컬 기준 예측을 만든 뒤 스윕을 실행한다.

```powershell
$benchPython = '.\.venv-cuda\Scripts\python.exe'
& $benchPython -m bench.classification.compare_images --backend cuda --frames 500 --warmup 50 --repeat 3 --depths 1 2 4 8 16 32
& $benchPython -m bench.classification.make_reference --backend cuda --frames 500 --out results/local-image-reference/top5.json
& $benchPython -m bench.classification.sweep_inflight --reference results/local-image-reference/top5.json --depths 1 2 4 8 16 32 --frames 300 --warmup 30 --repeat 3
```
