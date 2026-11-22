# 다른 환경에서 이미지 분류 실험 재현

이 가이드는 Imagenette 실제 이미지 500장을 ResNet50으로 분류하는 Python 동기·비동기 실험입니다. 영상 검출 demo 및 C++ ORT 경로는 기존 문서를 그대로 유지합니다.

## 1. 저장소와 실행 환경

```powershell
git clone https://github.com/SangjunRyu/edge-inference-runtime-lab.git
cd edge-inference-runtime-lab
python -m venv .venv-cuda
& .\.venv-cuda\Scripts\python.exe -m pip install -r requirements-cuda.txt
```

이미 clone했다면 작업 폴더에서 `git pull --ff-only`로 갱신합니다. 아래 명령은 모두 프로젝트 루트에서 실행합니다.

NVIDIA 드라이버와 ORT에 맞는 CUDA/cuDNN 라이브러리가 필요합니다. `requirements-cuda.txt`만으로 드라이버·CUDA/cuDNN 전체가 설치되는 것은 아닙니다. 새 환경의 GPU/OS에 맞춰 [ORT CUDA EP 설치 안내](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)를 따릅니다. CPU용 `onnxruntime`와 GPU용 `onnxruntime-gpu`는 한 venv에 같이 설치하지 않습니다.

기존 MX150 Windows 환경의 정확한 패키지 목록은 `requirements-cuda-mx150-windows.txt`입니다. Python 3.14.8, ORT 1.26.0, CUDA runtime 12.9, cuDNN 9.10.2.21에서 실행했습니다. 이 파일을 다른 GPU의 보편적인 권장 조합으로 간주하지 않습니다. 특히 최신 cuDNN을 설치했을 때 MX150에서 첫 Conv가 실패했던 이력이 있어 검증된 목록을 따로 남겼습니다.

Linux에서는 venv Python 경로를 `.venv-cuda/bin/python`으로 바꿉니다. CPU만 사용할 때는 별도 venv에 `requirements.txt`를 설치하고 비교 도구에 `--backend cpu`를 지정할 수 있습니다. NVML 메모리 sweep은 NVIDIA CUDA 장치와 NVML을 요구합니다.

## 2. 모델과 이미지 준비

모델 파일은 Git에 넣지 않습니다. 기존 `models/resnet50_224.onnx`가 있다면 그대로 사용할 수 있습니다. 새로 export할 때는 별도 환경에서 PyTorch·Torchvision·ONNX를 설치합니다.

```powershell
python -m venv .venv-export
& .\.venv-export\Scripts\python.exe -m pip install torch torchvision onnx
& .\.venv-export\Scripts\python.exe -m bench.data.prepare_models --models resnet50 --sizes 224
& .\.venv-cuda\Scripts\python.exe tools/prepare_imagenette.py --per-class 50 --seed 20261122
```

export 도구는 Torchvision의 pretrained DEFAULT 가중치와 TorchScript ONNX export 경로를 사용합니다. export 환경·라이브러리 버전에 따라 ONNX 파일 hash가 달라질 수 있어 각 실행 결과에 model SHA-256을 기록합니다. 정확한 기존 모델 파일을 별도로 옮겼다면 저장 결과의 `2ec1c6007b64f29b9e9920f6d104b3d3438fc38586d6f3629d3fc543ac48dcbe`와 비교할 수 있습니다.

이미지는 공식 Imagenette-320 아카이브 약 326MiB를 받아 validation 10개 클래스당 50장만 추출합니다. 선택한 JPEG는 약 13MB입니다. 공식 아카이브가 동일하면 생성 manifest SHA-256은 `f0f8809bcd26f482c64a228ca453c496e214ea3a9469aa809d97fe5c2b53b29a`입니다. 기존 manifest 증거는 `results/image-classification/20261122-imagenette500/input-manifest.json`에 있습니다.

## 3. 짧은 실행 확인 후 본 측정

```powershell
& .\.venv-cuda\Scripts\python.exe -m unittest repro.test_image_list_source -v
& .\.venv-cuda\Scripts\python.exe -m bench.classification.compare_images --frames 32 --warmup 4 --repeat 1 --depths 1 2 --modes files
```

CUDA provider 생성과 첫 Conv 실행, 이미지 읽기, callback 완료, 결과 회수가 되는지 확인합니다. provider 생성만 성공한 것은 전체 추론 성공과 다릅니다. 짧은 실행은 설치 확인용이며 속도 비교 결과로 사용하지 않습니다.

```powershell
& .\.venv-cuda\Scripts\python.exe -m bench.classification.compare_images --frames 500 --warmup 50 --repeat 3 --depths 1 2 4 8
```

파일+디코딩과 사전 디코딩 모드를 모두 비교합니다. 같은 입력·전처리·모델·후처리를 사용하고 결과 수·frame ID·Top-5를 검사합니다. 결과는 새 `results/image-classification/<UTC timestamp>/` 디렉터리의 `SUMMARY.md`, `results.json`, `reference-top5.json`에 저장됩니다.

## 4. 요청 수와 메모리 비교

다른 환경에서는 그 환경의 모델과 실행 결과로 동기 기준값부터 새로 만듭니다. 기존 MX150 예측을 새 모델에 무조건 적용하지 않습니다.

```powershell
& .\.venv-cuda\Scripts\python.exe -m bench.classification.make_reference --frames 500 --out results/local-image-reference/top5.json
& .\.venv-cuda\Scripts\python.exe -m bench.classification.sweep_inflight --reference results/local-image-reference/top5.json --depths 1 2 4 8 16 32 --frames 300 --warmup 30 --repeat 3
```

각 요청 수에서 새 프로세스·새 모델 세션을 사용합니다. CUDA 메모리·프로세스 RSS·GPU SM/메모리 클럭·온도·사용률·clock-event를 100ms 간격으로 기록합니다. 일부 장치에서 지원하지 않는 항목은 null입니다. 미완료 요청 상한은 실제 GPU 동시 실행 개수와 다릅니다. threads=4와 queue size=4가 기본값이며 별도 옵션으로 변경할 수 있습니다.

`--reference`를 생략하면 저장소에 포함된 기존 MX150 기준 결과를 사용합니다. 새 머신·새 export에서는 위와 같이 local reference를 명시하는 편이 맞습니다. 기준 파일은 자동 덮어쓰지 않으므로 재생성할 때는 새로운 `--out`을 사용합니다.

## 5. 순서를 바꾼 온도·클럭 조사

```powershell
& .\.venv-cuda\Scripts\python.exe -m bench.classification.investigate_inflight --reference results/local-image-reference/top5.json
```

요청 2·4·8·16을 다른 순서로 3회 측정하고 실행 사이 5초를 둡니다. 5초 대기가 완전한 냉각을 보장하지는 않습니다. 원본은 `results/inflight-investigation/<UTC timestamp>/`에 저장됩니다.

## 기존 MX150 결과의 결론

- 요청 32까지 OOM 없이 실행하고 출력 일치를 확인했습니다. 최대 지원 요청 수를 찾은 실험은 아닙니다.
- 초기 sweep의 요청 4·8 저하는 순서를 바꾸면 해당 요청 수에 고정해 재현되지 않았습니다.
- 재측정에서 80~82°C, SW Thermal Slowdown, 큰 SM 클럭 변동을 관측했습니다. 일부 실행에는 높은 CPU 부하도 섞였습니다.
- 따라서 초기 표를 “32개가 최적” 또는 확정적인 속도 향상 수치로 제시하지 않습니다.

[입력·파이프라인 설명](imagenette-pipeline.md), [메모리 결과](inflight-memory.md), [성능 저하 조사](inflight-investigation.md)를 함께 확인합니다. 다른 환경에서는 GPU·드라이버·ORT·CUDA/cuDNN·model/manifest hash·threads·queue size·온도·클럭을 함께 남겨 비교합니다.
