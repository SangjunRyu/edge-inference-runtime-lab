# 실제 동영상으로 실행하는 YOLO 비동기 추론

이번 입력은 OpenCV 공개 샘플인 [`vtest.avi`](https://github.com/opencv/opencv/blob/4.x/samples/data/vtest.avi)다.
고정 카메라 앞에서 여러 사람이 걷는 실제 영상이며, 768×576, 10 FPS,
795프레임으로 약 79.5초다. 버스 이미지를 이동시키던 모사 코드는 제거했다.
이전 결과는 실험 이력으로 보존하며, 공장 영상이나 제품 인식 성능으로 해석하지 않는다.

## 실행 방법

PowerShell에서 저장소 루트로 이동한다. 모델은 기존의
`models/yolo11n_320.onnx`를 사용한다. 실행에 PyTorch나 Ultralytics 설치는 필요 없다.

```powershell
cd C:\Users\ysjun5656\deepX\DEEPX-poc
.\.venv-cuda\Scripts\Activate.ps1
python -m bench.data.get_detection_video
python -m bench.detection.run_video --backend cuda --mode async --inflight 2 --out-dir results/real-video-cuda
Invoke-Item .\results\real-video-cuda\detection.mp4
```

활성화가 막히면 `python` 대신 `.\.venv-cuda\Scripts\python.exe`를 쓴다.
`--display`를 추가하면 실행 중 창이 열리고, `q`나 ESC로 종료한다.
CPU는 `--backend cpu`, 동기는 `--mode sync`로 바꾸고 결과 폴더도 다르게 지정한다.
`--max-frames 300`은 앞의 30초만 처리한다. 본인 영상은 `--video "C:\영상\sample.mp4"`로 지정한다.
현재 후처리는 COCO의 사람 클래스만 선택한다.

## 프레임이 처리되는 과정

```text
VideoCapture.read(): 영상 읽기·디코딩
  → 비율 유지 리사이즈·패딩: 320×320
  → BGR→RGB, float32 NCHW, 0~1 정규화
  → ONNX Runtime: YOLO11n 추론
  → 사람 클래스 선택·NMS·원본 좌표 복원
  → 프레임 순서대로 중심점 기반 추적
  → 원본과 박스·ID를 나란히 그려 MP4 저장
```

이 노트북에서 입력 디코딩 백엔드는 FFMPEG다. 추론 백엔드는 CPU 또는 CUDA이며,
디코딩 백엔드와 별개다. 모델은 이미 학습된 가중치를 사용한다. 이번 실행에서 새로 학습하지 않았다.
중심점 추적은 간단한 시각화용 구현이라 가림이나 밀집 상황에서 ID가 바뀔 수 있다.
정답 라벨과 비교한 탐지·추적 정확도는 측정하지 않았다.

## `inflight=2`와 배치 크기의 차이

현재 모델 입력은 `[1, 3, 320, 320]`이다. 요청 하나에 이미지 한 장이 들어간다.
`inflight=2`는 두 장을 묶는 배치가 아니라, 결과를 순서대로 소비하기 전까지
유지하는 프레임 항목의 상한이다. 여기에는 완료됐지만 아직 소비하지 않은 결과도 포함된다.

실행 루프는 `session.run_async()`로 첫 프레임을 제출하고 다음 프레임의 읽기·전처리를 진행한다.
`pending`에 두 항목이 쌓이면 가장 먼저 제출한 프레임의 완료를 기다려 후처리한다.
추론이 다른 순서로 끝나도 추적과 저장은 입력 순서를 따른다.
별도로 계속 입력을 쌓는 캡처 스레드나 입력 대기 큐는 없다.

배치 크기 2라면 `[2, 3, 320, 320]`을 한 요청으로 처리한다.
현재 정적 배치 1 모델과 다른 구성이다. 비동기 요청이 두 개 있다고 해서
GPU 커널 두 개가 동시에 실행된다고 볼 수도 없다.
CUDA에는 이 환경에서 출력 불일치를 완화한 `use_ep_level_unified_stream=1`을 적용했다.
라이브러리나 드라이버의 근본 원인이 확인됐다는 뜻은 아니다.

## 일반적인 YOLO의 영상·카메라 입력

이 프로젝트의 YOLO11n 탐지기는 현재 프레임의 이미지를 처리한다.
동영상 파일, USB 카메라, RTSP에서 프레임을 꺼내는 일은 주변 입력 코드가 담당한다.
이전 프레임의 물체와 같은 ID로 연결하는 일도 별도의 추적기가 담당한다.

Ultralytics는 이 주변 코드를 `model.predict(source="video.mp4", stream=True)`나
`source=0` 같은 인터페이스로 제공한다. 여기서 `stream=True`는 결과를 하나씩
반환하는 제너레이터 방식이며, 비동기 추론이나 GPU 동시 실행 설정은 아니다.
이 프로젝트는 요청 수와 완료 순서 등을 직접 관찰하려고 ONNX Runtime으로 구성했다.

## 현재 측정이 다루는 범위

파일을 가능한 빠르게 읽고 모든 프레임을 보존하는 오프라인 실행이다.
처리가 느려지면 읽기도 기다린다. 원본이 10 FPS라고 매초 10장의 입력을 독립적으로
발생시키지는 않는다. `--display`도 표시 대기를 추가할 뿐, 카메라 입력 시계를 재현하지 않는다.
실시간 입력의 지연 누적·프레임 드롭·대기열 정책은 별도의 실험이 필요하다.

`processing_fps`는 읽기·전처리·추론·후처리·영상 인코딩을 포함하며,
세션 생성과 워밍업은 제외한다. `submit_to_callback_ms`는 런타임 대기·전송 등을 포함하므로
GPU 연산 시간만을 뜻하지 않는다. 화면 표시와 값 검증을 켜면 측정 시간이 추가된다.
결과 MP4는 처리 속도와 관계없이 원본의 10 FPS로 저장된다.

2026-11-18의 최초 CUDA 전체 실행은 795프레임을 약 12.69초에 처리했다.
이 값은 단일 실행 관찰치다. 기존 300프레임 실행에서는 30프레임마다 비교한
동기·비동기 원시 출력 10건이 허용 오차 안에서 일치했다.
각 프레임의 박스·점수·추적 ID와 실행 정보는 결과 폴더의 `result.json`에 저장된다.

실제 영상 전체에 걸쳐 64개 프레임을 골라 CUDA 동기·비동기 출력과 노드 배치를 검증하려면:

```powershell
python -m bench.detection.verify results/yolo-verification --samples 64
```

코드 경로는 [`bench/README.md`](../bench/README.md)에 정리했다.
공식 문서: [Ultralytics 영상 입력](https://docs.ultralytics.com/modes/predict/),
[ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html),
[CUDA EP 옵션](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#use_ep_level_unified_stream).

## 경로 개편 후 확인한 내용

CPU·CUDA 각각 동기와 비동기로 300프레임씩 실행했다. 네 결과 영상 모두
300프레임이 정상적으로 읽혔다. 각 백엔드 안에서 동기·비동기의 박스·점수·추적 ID가
일치했다. 별도로 영상 전체에서 고른 64프레임의 CUDA 원시 출력과 노드 배치를 확인했다.
대기열·순서·좌표 복원 등의 계약 테스트 6개와 주요 CLI 도움말 15개도 통과했다.
[검증 요약](../results/video-validation-20261118/summary.json)에 조건과 허용 오차를 남겼다.
실행별 처리 시간은 반복 측정한 성능 비교로 해석하지 않는다.
