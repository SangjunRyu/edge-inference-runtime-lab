# 다수 이미지 분류의 동기·비동기 파이프라인

## 실험 목적

실제 이미지 여러 장을 ResNet50으로 분류하면서, 추론 제출과 완료 수집을 분리했을 때 처리율·프레임 지연·큐 대기가 어떻게 달라지는지 확인합니다. 합성 1080p 프레임 복사를 대표 입력으로 사용하지 않습니다.

## 데이터 준비

[fastai Imagenette](https://github.com/fastai/imagenette)는 ImageNet의 10개 클래스로 구성된 데이터셋입니다. 공식 320px 버전의 validation split에서 클래스당 50장, 총 500장을 선택했습니다. 320px는 짧은 변의 크기이며, 원본 종횡비와 이미지별 크기는 유지됩니다. 실제 입력을 임의의 1080p 프레임으로 늘리지 않습니다.

PowerShell, 프로젝트 루트 기준:

```powershell
& .\.venv-cuda\Scripts\python.exe tools/prepare_imagenette.py --per-class 50 --seed 20261122
```

다운로더는 공식 `imagenette2-320.tgz`를 내려받고 validation 이미지 일부만 추출합니다. 최초 다운로드는 약 326 MiB이며 캐시를 재사용합니다. train 이미지는 추출하지 않습니다. 선택은 파일명 정렬·고정 seed로 재현할 수 있습니다.

- `data/imagenette-320-subset/images/`: 선택된 JPEG 500장
- `data/imagenette-320-subset/manifest.json`: 순서·원본 이름·클래스·ImageNet index·SHA-256
- `data/cache/imagenette2-320.tgz`: 공식 아카이브 캐시

아카이브 SHA-256은 내려받은 파일의 식별용 기록입니다. 배포자의 별도 서명 검증을 했다는 의미는 아닙니다. 데이터와 가상환경은 Git에서 제외합니다.

## 두 입력 모드

| 모드 | Read 계측 경계 | 전처리 |
|---|---|---|
| `files` | 이미지 파일 읽기 + JPEG 디코딩 | 매 이미지 resize·정규화·NCHW 변환 |
| `predecoded` | 사전 디코딩한 읽기 전용 이미지 참조 전달 | 위와 동일 |

`predecoded`의 디코딩과 메모리 준비는 계측 전에 완료됩니다. 준비 시간과 상주 이미지 bytes도 기록합니다. Read에서 이미지를 다시 복사하지 않습니다. 기존 preprocess는 resize로 새 배열을 생성하므로 입력 이미지를 수정하지 않습니다.

`files`도 hash 검증과 반복 실행으로 OS 파일 캐시가 따뜻해진 조건입니다. 저장장치의 cold read 성능 실험이 아닙니다. 두 모드 모두 동일한 manifest를 사용하고, 마지막 이미지에 도달하면 종료합니다. 부족한 입력을 조용히 반복하지 않습니다.

## 실제 처리 흐름

1. manifest 순서에 따른 이미지 입력
2. OpenCV 224×224 resize, BGR→RGB, float32/255, ImageNet mean/std 정규화, NCHW 변환
3. ResNet50 ONNX Runtime 실행: batch 1, 입력 `[1,3,224,224]`, 출력 `[1,1000]`
4. 안정화 softmax, Top-5 index/probability 선택·정렬
5. frame ID로 결과 수집·순서 확인

이 실험은 ImageNet 정답 정확도를 평가하지 않습니다. 기존 모델의 전처리 코드를 유지해 실행 구조를 비교합니다. 선택한 가중치의 정확도 benchmark를 하려면 그 가중치에 맞는 resize/center-crop recipe와 정답 label 대응을 별도로 검증해야 합니다. manifest의 ImageNet index는 출처 추적용이며, 10개 class만으로 output을 축소하지 않습니다.

동기 실행은 한 이미지의 후처리까지 끝낸 뒤 다음 이미지를 처리합니다. 비동기 실행은 read/preprocess/완료 수집을 분리하고, Semaphore로 동시에 제출된 요청 수를 제한합니다. callback이 결과를 output queue로 전달하고 postprocess 이후 frame ID 순서로 회수합니다. Python 코드에 구현한 경로입니다. 기존 C++ OpenVINO adapter를 MX150용 ORT C++로 바꾼 실적은 포함하지 않습니다.

## 반복 비교 실행

```powershell
& .\.venv-cuda\Scripts\python.exe -m bench.classification.compare_images --frames 500 --warmup 50 --repeat 3 --depths 1 2
```

두 입력 모드 각각 sync/async1/async2를 3회 반복합니다. 첫 sync로 reference Top-5를 만든 뒤 반복 블록 안의 실행 순서를 고정 seed로 섞습니다. 같은 모델 세션과 ORT intra-op threads=4, queue size=4를 사용합니다. 각 실행은 같은 500장을 처리하고 첫 50장을 warmup으로 제외해 450장의 timing을 집계합니다. 결과 검증은 warmup을 포함한 500장 모두를 사용합니다.

실행마다 raw JSON을 저장하므로 실패 직전 결과도 남습니다. 기본 출력은 새 UTC timestamp 디렉터리입니다.

- `results.json`: 단계 시간·대기·E2E·처리율·환경·소스 해시·입력 순서 hash·검증 결과
- `reference-top5.json`: frame ID별 기준 예측
- `SUMMARY.md`: 입력 모드/동시 요청 수별 반복 평균과 최소·최대

검증 항목은 모든 이미지의 files/predecoded 픽셀 일치, sync 반복 및 async 결과의 Top-5 일치(확률 절대오차 1e-4), 누락·중복·순서 위반·forced flush·오류·stall 여부입니다. 확률이 같다는 것은 정답 label을 맞혔다는 뜻과 다릅니다.

## 개별 실행

```powershell
& .\.venv-cuda\Scripts\python.exe -m bench.classification.run_sync --model models/resnet50_224.onnx --backend cuda --source data/imagenette-320-subset/manifest.json --image-read-mode files --frames 500 --warmup 50 --repeat 3 --intra-threads 4 --tag images-sync
& .\.venv-cuda\Scripts\python.exe -m bench.classification.run_async --model models/resnet50_224.onnx --backend cuda --source data/imagenette-320-subset/manifest.json --image-read-mode files --frames 500 --warmup 50 --repeat 3 --intra-threads 4 --inflight 2 --tag images-async
```

`--image-read-mode predecoded`로 바꾸면 파일 읽기·디코딩을 제외합니다. 결과 파일명에 모드가 포함돼 덮어쓰지 않습니다. 엄격한 A/B 비교에는 실행 순서도 섞는 `compare_images` 사용을 권합니다.

## 결과 해석

처리율은 images/s, 지연은 이미지 한 장의 E2E p50/p95로 확인합니다. 비동기는 처리율이 비슷해도 큐 대기와 한 이미지의 지연이 늘 수 있습니다. 속도 향상이 작거나 없으면 그 결과를 그대로 기록합니다.

동기 Inference는 `session.run()` 반환까지의 시간입니다. 비동기 Inference는 제출부터 callback까지이며 내부 실행 대기가 포함됩니다. 두 수치를 순수 CUDA 커널 시간의 개선으로 해석하지 않습니다. ORT가 관리하는 H2D/D2H는 이 API 경계에 포함되며 별도 CUDA 이벤트를 사용한 전송 계측은 아닙니다.

## 확인 명령

```powershell
& .\.venv-cuda\Scripts\python.exe -m unittest repro.test_image_list_source -v
& .\.venv-cuda\Scripts\python.exe -m bench.classification.verify_equivalence --model models/resnet50_224.onnx --backend cuda --source data/imagenette-320-subset/manifest.json --image-read-mode predecoded --frames 500 --intra-threads 4 --depths 1 2 --self-test --tag images-verify
```
