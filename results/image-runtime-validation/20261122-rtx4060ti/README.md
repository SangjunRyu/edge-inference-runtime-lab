# 실제 이미지의 비동기 순서·버퍼 수명·입력 과부하 검증

RTX 4060 Ti, ONNX Runtime CUDAExecutionProvider, ResNet50 FP32 입력 224×224, Imagenette-320 500장. 포트폴리오 9·10쪽의 실측 근거다.

## 9쪽 측정

같은 ONNX 모델과 매니페스트의 CUDA 동기 Top-5 기준을 사용했다. JPEG 파일에서 이미지를 읽고 전처리·추론·후처리를 수행했다. 인플라이트 요청 상한은 16, 각 단계 큐 용량은 4였다. 50장 워밍업 후 재정렬 OFF와 ON을 각각 3회 실행했다.

- OFF와 ON 모두 회차당 500장을 출력했고, 매니페스트 이미지 ID별 Top-5 검사를 통과했다.
- OFF 출력의 다음 기대 순번 불일치는 476~480건이었다. ON은 0건이었다. ON에서도 재정렬 버퍼의 도착 순번 불일치는 467~483건이었다.
- 재사용 CPU 입력 슬롯 16개의 동시 점유 최고치는 16이었다. 각 회차에서 500번 확보하고 500번 반환했다. 최종 잔여와 제출 전·콜백 시 SHA-256 불일치는 0건이었다.
- 버퍼 수명 계측 범위는 애플리케이션이 관리하는 CPU FP32 입력이다. CUDA 런타임 내부의 VRAM 버퍼와 복사 수명은 별도 계측이 필요하다.

## 10쪽 측정

동일한 파일 중 500개의 이미지 ID를 고정 예정 시각에 공급하도록 설정했다. 예정률은 100장/s와 800장/s이며 정책별로 3회씩 반복했다. 생산자가 넘기는 것은 이미지 ID이고, 용량 4인 입구 큐에서 정책을 적용한다. 디코더가 가져간 뒤에는 기존 비동기 파이프라인으로 처리한다. 폐기 후 남은 외부 이미지 ID는 검증에 유지하고, 내부 재정렬에는 제출 순서대로 연속 번호를 할당했다.

| 예정 도착률 | 정책 | 출력 평균 | 폐기 평균 | 회차별 p95 평균 |
|---|---|---:|---:|---:|
| 100장/s | Block | 500.0 | 0.0 | 6.19 ms |
| 100장/s | DropOldest | 500.0 | 0.0 | 6.40 ms |
| 100장/s | DropNewest | 500.0 | 0.0 | 6.24 ms |
| 800장/s | Block | 500.0 | 0.0 | 466.96 ms |
| 800장/s | DropOldest | 318.7 | 181.3 | 88.11 ms |
| 800장/s | DropNewest | 315.3 | 184.7 | 93.57 ms |

p95는 이미지의 예정 도착 시각에서 최종 출력까지의 지연이다. Block으로 생산자가 밀린 시간도 포함된다. 예정률 800장/s에서 Block의 생산자 누적 대기 평균은 965.20 ms다. DropOldest와 DropNewest의 p95는 각각 81.52~98.24 ms와 90.01~95.60 ms로 겹친다. 이번 실행에서 확인된 큰 차이는 Block과 Drop 사이의 이미지 보존율·지연 교환관계다.

18개 정책 실행 모두 출력 ID와 의도적 폐기 ID를 합치면 500장이었으며, 출력 순서 위반은 0건이고 살아남은 이미지의 Top-5는 동기 기준과 일치했다. 정책 측정에는 입력 풀을 계속 쓰고, 버퍼 수명 검증용 해시는 적용하지 않았다. 입력은 실제 JPEG 파일의 예정 시각 재생이다. 장시간 열 상태는 이 결과의 범위에 포함되지 않는다.

## DEEPX 스트리밍과의 연결

9쪽은 제출한 요청의 출력 순서를 다룬다. DEEPX DX-APP의 C++ 예제도 표시·저장 전에 제출 순번을 기준으로 비동기 완료 결과를 재정렬한다. 독립 이미지 분류에서는 이미지 ID를 기준으로 결과를 대조할 수 있고, 영상 표시·저장처럼 순서가 필요한 소비자는 재정렬된 출력을 받을 수 있다. [DX-APP FrameReorderBuffer](https://github.com/DEEPX-AI/dx_app/blob/main/src/cpp_example/common/utility/frame_reorder.hpp)

10쪽은 입구 큐의 수용 정책을 다룬다. 데이터셋 평가에서는 전체 이미지 결과를 보존해야 하므로 Block이 적합하다. 실시간 미리보기에서는 최신 프레임을 우선할 수 있다. GStreamer queue는 블로킹, 새 버퍼 폐기, 오래된 버퍼 폐기를 지원한다. DEEPX DX-STREAM의 DxRate는 RTSP 입력률을 제한하고, DX-APP의 drop-frames 옵션은 C++ 미리보기에서 오래된 프레임을 건너뛴다. [GStreamer queue](https://gstreamer.freedesktop.org/documentation/coreelements/queue.html) · [DEEPX DX-STREAM 가이드](https://github.com/DEEPX-AI/dx_stream/blob/main/docs/source/docs/08_DX-STREAM_Agent_Driven_Development.md) · [DEEPX DX-APP 옵션](https://github.com/DEEPX-AI/dx_app#command-line-arguments)

이 실험은 RTX에서 실제 JPEG에 ResNet50 추론을 수행하면서 큐가 가득 찬 경우를 재현했다. 폐기 조건은 큐 용량이며, p95 지연은 정책이 낸 결과다. 라이브 영상 입력이나 지연 시간 한도를 기준으로 폐기하는 정책에는 별도의 검증이 필요하다.

## 재현

edge-inference-runtime-lab 루트에서 실행한다. CUDA 가상 환경, ONNX 모델, Imagenette 매니페스트가 준비되어 있어야 한다. [검증 러너](../../../bench/classification/validate_image_runtime.py)와 [정책·버퍼 테스트](../../../tests/test_image_runtime_validation.py)를 함께 제공한다.

```powershell
$python = '.\.venv\Scripts\python.exe'
& $python -m bench.classification.make_reference --backend cuda --frames 500 --out results/local-image-reference/top5.json
& $python -m bench.classification.validate_image_runtime --reference results/local-image-reference/top5.json --frames 500 --warmup 50 --inflight 16 --queue-size 4 --repeat 3 --rates 100 800
& $python -m unittest discover -s tests
```

상세 실행은 results.json, 출력 표는 SUMMARY.md에 있다. NVML 환경 조회가 실패했으므로 GPU 모델은 CUDA Driver API로 직접 확인한 cuda-device.json에 보존했다.
