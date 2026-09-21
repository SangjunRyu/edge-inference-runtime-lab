# study — 유한 동시성이 다섯 군데에서 다른 이름으로 나타나는 방식

이 디렉터리는 프로젝트 문서가 아니라 **학습 계획**이다. `docs/`가 이 저장소가
무엇을 했는지 기록한다면 여기는 다음에 무엇을 볼지를 적는다. 한글로 쓴다.

## 왜 이 지형인가

W5에서 유도한 것이 하나 있다.

```
in-flight 천장 = min(자원 용량, 구조 용량)
             = min(버퍼 수, S + ΣQi + ΣWj)
```

이건 이 프로젝트의 발명이 아니다. **같은 문제를 여러 산업이 오래 다뤄왔고, 각자
다른 이름을 붙였다.** 그 이름들을 아는 것이 이 디렉터리의 목적이다.

| 개념 | 내 PoC | oneTBB | Taskflow | GStreamer | Triton | DX-RT |
|---|---|---|---|---|---|---|
| 동시 실행 상한 | ceiling | live tokens | lines | `max-size-buffers` | `max_queue_size` | `bufferCount` |
| 승인 거부 | 계약 불일치 거부 | `limiter_node` | lines 소진 | `leaky` | `timeout_action: REJECT` | `InsufficientMemory` |
| 오래된 것 버리기 | DropOldest | — | — | `leaky=downstream` | 타임아웃 | — |
| 순서 복원 | reorder ring | `sequencer_node` | — | 큐가 FIFO | sequence batcher | reorder buffer |
| 고정 버퍼 풀 | `BufferPool` | — | — | 버퍼 풀 | pinned/CUDA pool | `FixedSizeBuffer` |
| 대기와 계산 분리 | (미완) | — | — | latency query | `queue_duration` 메트릭 | `get_npu_inference_time` |

**마지막 줄이 특히 중요하다.** W1 발견 2가 "스테이지 표의 inference는 장치 시간이
아니다"라고 한 그 문제를, Triton은 메트릭을 둘로 나눠서, DX-RT는 API를 둘로 나눠서
푼다. 같은 문제를 셋이 같은 방식으로 인정한다.

## 읽는 순서

| 문서 | 내용 | 시간 |
|---|---|---|
| [00 공통 모델](00-common-model.md) | 다섯을 묶는 큐잉 이론. **여기부터** | 반나절 |
| [01 GStreamer](01-gstreamer.md) | W5-4와 가장 가깝다. 엣지 비디오의 표준 | 1~2일 |
| [02 oneTBB](02-onetbb.md) | 토큰과 `limiter_node`. 이미 내 프로세스 안에 있다 | 2~3일 |
| [03 Triton](03-triton.md) | 서빙 계층의 같은 표. 가장 production에 가깝다 | 2일 |
| [04 Taskflow](04-taskflow.md) | 같은 것을 lines 로 부른다 | 1일 |
| [05 가속기 동시성](05-accelerator.md) | CUDA streams, OpenVINO 요청 풀, DX-RT | 2일 |
| [06 지도에 없는 것](06-gaps.md) | 전송·MPMC·표준 비동기·프로파일링 | 상시 |
| [07 실습 계획](07-plan.md) | **읽기가 프로그램으로 끝나게 하는 규칙** | — |

시간이 없으면 00과 01과 07이다. 07이 없으면 나머지가 독후감이 된다.

## 왜 TBB가 자꾸 나왔나

개념적인 이유도 있지만 구체적인 이유가 더 세다. **OpenVINO CPU 플러그인이 TBB 위에
지어져 있어서 이미 내 프로세스 안에 들어와 있었다.** 지금 `cpp/build/`에
`tbb12.dll`, `tbbbind_2_5.dll`, `tbbmalloc.dll`이 OpenVINO libs에서 복사돼 앉아
있다. W4에서 만든 스레드 풀이 OpenVINO의 intra-op 풀과 경쟁한 그 문제도 여기서
나온다.

## 이 디렉터리가 지켜야 하는 규칙

`docs/w5-plan.md`가 W5에 건 규칙과 같다.

> 읽기는 산출물이 아니다. 모든 학습은 내 프로그램이 계산한 수치나 내 프로그램이
> 검사하는 단언으로 끝나야 한다.

프레임워크마다 노트가 아니라 **W5-4의 표를 그 프레임워크 어휘로 재현하는 짧은
프로그램**이 산출물이다. 자세한 것은 [07 실습 계획](07-plan.md)에 있다.
