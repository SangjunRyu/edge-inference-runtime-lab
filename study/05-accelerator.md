# 05 — 가속기 동시성: CUDA streams, OpenVINO 요청 풀, DX-RT

**우선순위 5.** 앞의 넷이 호스트 쪽 동시성이라면 여기는 장치 쪽이다. 이미 절반은
W2와 W4에서 했으니 나머지를 채우는 단계다.

## 세 시스템이 같은 말을 한다

| 개념 | CUDA | OpenVINO | DX-RT |
|---|---|---|---|
| 동시 실행 단위 | stream | `InferRequest` | task / request |
| 요청 풀 | 사용자가 만듦 | `AsyncInferQueue`(Python) | `bufferCount` 만큼 |
| 런타임이 알려주는 최적 동시성 | — | `ov::optimal_number_of_infer_requests` | `bufferCount` (메모리에서 유도) |
| 완료 알림 | event, callback | `set_callback` | 완료 콜백 |
| 호스트-장치 전송 | `cudaMemcpyAsync` + pinned | 자동 | NFH + DMA |

**가운데 줄이 핵심이다.** OpenVINO 는 `ov::optimal_number_of_infer_requests` 로
"이 장치에서는 요청 몇 개가 적당하다"를 알려주고, DX-RT 는 `bufferCount` 를 장치
메모리에서 계산한다. **둘 다 런타임이 사용자의 천장을 대신 정해 준다.** 내
W5-2 가 유도한 `min(자원, 구조)` 에서 자원 쪽을 장치가 말해 주는 셈이다.

W2 의 in-flight 깊이 스윕에서 무릎이 4 였는데, OpenVINO 에 물어봤으면 숫자를
말해 줬을 것이다. **물어보지 않고 재서 찾은 것이고, 둘이 일치하는지는 확인하지
않았다.** 확인할 가치가 있다.

## CUDA streams

없는 장비지만 개념은 가져가야 한다. 면접에서 가장 자주 나오는 가속기 동시성 어휘다.

- stream 은 순서가 보장되는 작업 큐다. 서로 다른 stream 끼리는 순서가 없다.
- `cudaMemcpyAsync` 는 **pinned(page-locked) 호스트 메모리**에서만 진짜 비동기다.
  pageable 메모리면 드라이버가 내부 스테이징 버퍼로 복사하면서 동기화된다.
- 전형적 패턴은 copy-compute-copy 를 여러 stream 으로 겹치는 것이다. 내 W2 가
  read-preprocess-infer 를 겹친 것과 같은 구조이고, **여기서는 세 단계가 전부 장치
  안에 있다.**
- event 로 stream 간 의존을 건다. graph capture 로 반복되는 패턴을 하나로 묶는다.

**pinned memory 가 내 프로젝트의 빈칸과 정확히 이어진다.** W5-3 이 포맷 변환은
다뤘는데 전송은 안 다뤘다. [06 지도에 없는 것](06-gaps.md)에 적어뒀다.

## OpenVINO 쪽에서 아직 안 본 것

W2 와 W4 에서 `InferRequest` 를 썼지만 다음은 안 봤다.

- `ov::hint::performance_mode` 의 THROUGHPUT 대 LATENCY. 플러그인이 무엇을 바꾸나.
- `ov::hint::num_requests` 로 힌트를 주면 `optimal_number_of_infer_requests` 가
  바뀌나.
- `AsyncInferQueue` (Python) 의 구현. **유한한 요청 풀에 idle 핸들러**이고, 이건
  내 `BufferPool` + 완료 콜백과 같은 물건이다. C++ 에는 없어서 직접 만들어야 하는데
  내가 이미 만들었다.
- CPU 플러그인의 stream 개념(`ov::num_streams`). TBB 아레나와 어떻게 연결되나.

## DX-RT 쪽에서 아직 안 본 것

- `device_pool.cpp` 와 `device_dispatcher.cpp` 의 장치 선택. 여러 NPU 가 있을 때
  누가 어떻게 고르나. 내 PoC 에 없는 층이다.
- `N_BOUND_INF_ONLY_NPU0` 같은 바인딩 옵션. 사용자가 장치를 고정할 수 있게 하는
  이유가 무엇인가.
- `dxrt_service` 와 `dynamic_ipc`, `device_shm`. **여러 프로세스가 하나의 NPU 를
  공유하는 구조**이고, 이 층은 내 PoC 보다 Triton 에 가깝다.
- `npu_memory_cache.h`, `memory_interface_multiprocess.h`. 이름만 봐도 위와 이어진다.

## 확인할 질문

1. W2 의 in-flight 무릎 4 와 `ov::optimal_number_of_infer_requests` 가 같은 값인가.
   **이건 기계 시간 거의 없이 지금 확인할 수 있다.**
2. pinned memory 없이 `cudaMemcpyAsync` 를 부르면 왜 동기가 되나. 드라이버가
   무엇을 하나. 같은 문제가 NPU 의 DMA 경로에도 있나.
3. OpenVINO 의 THROUGHPUT 모드는 무엇을 바꾸나. 요청 수인가, 스레드 배치인가, 둘 다인가.
4. DX-RT 에서 장치가 둘일 때 요청이 어떻게 배분되나. 라운드로빈인가 부하 기반인가.
5. `dxrt_service` 가 있을 때와 없을 때(`no_service_layer.cpp`) 무엇이 달라지나.
   왜 두 경로를 다 유지하나.

## 산출물

1번 질문이 그대로 산출물이다. **W2 가 측정으로 찾은 무릎과 OpenVINO 가 스스로
말하는 최적 요청 수를 비교한다.** 일치하면 측정이 런타임의 내부 모델을 재발견한
것이고, 다르면 그 차이가 결과다. 둘 다 쓸 만한 문장이 된다.

## 시간

2일. CUDA 는 장비가 없으니 읽기만이고, OpenVINO 확인이 반나절, DX-RT 소스 읽기가
나머지다.
