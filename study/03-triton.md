# 03 — NVIDIA Triton Inference Server

**우선순위 3.** AI 추론에서 production 에 가장 가까운 비교 대상이고, 내 W5-4 표를
서빙 계층의 어휘로 그대로 쓴다.

## 겹치는 층과 갈라지는 층

```
                 Triton          내 PoC          DX-RT
네트워크·모델 저장소   O               -              -
동적 배칭            O               -              -
다중 프로세스 공유     O               -         O (dxrt_service)
─────────────────────────────────────────────────────
아티팩트 = 계약       O               O              O
승인 제어·유한 동시성   O               O              O
고정 풀·수명          O               O              O
요청 수명·완료 콜백    O               O              O
─────────────────────────────────────────────────────
포맷 변환 계층         백엔드 안        O              O
장치 선택·바인딩       O               -              O
ioctl·DMA·펌웨어      -               -              O
```

**가운데 네 줄이 내가 만든 것이고 그게 둘의 교집합이다.**

## 대응표

| 내 개념 | Triton |
|---|---|
| 계약 | `config.pbtxt` 의 input/output dims, data_type, max_batch_size |
| 승인 거부 | `default_queue_policy { timeout_action: REJECT }` |
| 큐 용량 | `default_queue_policy { max_queue_size }` |
| 마감 기반 드롭 | `timeout_action: DELAY` + `default_timeout_microseconds` |
| 동시 실행 수 | `instance_group { count }` |
| 자원 기반 승인 | rate limiter 의 인스턴스별 자원 선언 |
| 고정 풀 | `--pinned-memory-pool-byte-size`, `--cuda-memory-pool-byte-size` |
| 순서 보장 | `sequence_batching` (correlation id 별) |
| 대기와 계산 분리 | `nv_inference_queue_duration_us` 대 `nv_inference_compute_infer_duration_us` |

**마지막 줄이 이 문서에서 가장 값나가는 한 줄이다.** W1 발견 2가 "스테이지 표의
inference 는 장치 시간이 아니다"라고 한 그 문제를 Triton 은 메트릭을 둘로 나눠서
푼다. DX-RT 는 API 를 둘로 나눠서 푼다(`get_latency()` 대
`get_npu_inference_time()`). 내 하네스만 아직 안 나눴고, 그게
`handoff/open-threads.md` 의 미해결 항목이다. **세 시스템이 같은 문제를 같은 방식으로
인정한다는 건 그 문제가 실재한다는 강한 증거다.**

## 내게 없는 다섯 번째 정책

Triton 의 `timeout_action: DELAY` + 타임아웃은 **마감 기반 드롭**이다. 정해진 시간
안에 실행되지 못하면 버린다. 내 W5-4 에는 Block, DropOldest, DropNewest, 무한 큐만
있고 이건 없다. 실제 서빙 시스템이 쓰는 정책이니 추가할 가치가 있다.

신선도 관점에서 보면 DropOldest 의 일반화다. DropOldest 는 "큐가 찼을 때" 버리고,
마감 드롭은 "너무 늙었을 때" 버린다. 후자가 더 직접적으로 나이를 통제한다.

## 동적 배칭은 왜 내게 없나

Triton 의 핵심 차별점이고 내게도 DX-RT 에도 없다. 이유가 있다.

**배칭은 배치 1 에서 장치가 놀 때만 성립하는 지연-처리량 교환이다.** 서버 가정이지
엣지 가정이 아니다. 엣지에서는 프레임이 한 번에 하나씩 오고 신선도가 중요하다.
`max_queue_delay_microseconds` 가 존재한다는 사실 자체가 "모으려면 기다려야 한다"를
인정하는 것이다.

이걸 설명할 수 있으면 "왜 안 했나"가 "왜 안 하는 게 맞나"가 된다.

## 무엇을 읽나

1. Model Configuration 문서. `instance_group`, `dynamic_batching`,
   `sequence_batching`, `ensemble_scheduling` 순으로.
2. `default_queue_policy` 전체. 네 필드가 전부 내 W5-4 어휘다.
3. Architecture 문서의 스케줄러 절.
4. Metrics 문서. 어떤 것을 재고 어떻게 나누는지. **내 하네스에 없는 것을 찾아라.**
5. (여유가 있으면) Backend API. `TRITONBACKEND_ModelInstanceExecute` 시그니처만 봐도
   요청 배열과 응답 팩토리 구조가 보인다.

검색어: `Triton max_queue_size`, `Triton queue policy timeout_action`,
`Triton dynamic batching max_queue_delay_microseconds`,
`Triton instance_group count`, `Triton rate limiter resources`,
`Triton metrics queue duration compute duration`

## 확인할 질문

1. `max_queue_size` 를 넘겨 거부된 요청은 클라이언트에게 무엇으로 보이나. 내 계약
   위반 거부는 프로세스 안에서 끝나는데 Triton 은 네트워크 너머로 말해야 한다.
2. `instance_group { count: 4 }` 는 내 `--infer-workers 4` 와 같은가 다른가.
   인스턴스가 모델 사본을 각각 갖는가.
3. rate limiter 의 자원 선언이 DX-RT 의 `bufferCount` 와 같은 발상인가. 둘 다 자원
   기반 승인인데 하나는 사람이 적고 하나는 모델에서 계산된다.
4. `sequence_batching` 이 순서를 보장하는 방식이 내 `RingReorder` 와 같은가.
   correlation id 라는 개념이 내게 없는데 필요한가.
5. queue duration 과 compute duration 의 경계가 정확히 어디인가. 내 스케줄러에서
   같은 경계를 그으려면 어디에 타이머를 넣어야 하나.

## 산출물

Triton 을 설치해 모델 하나를 띄우고 `max_queue_size` 와 `instance_group` 을 바꿔가며
과부하를 걸어 메트릭에서 queue duration 과 drop 을 읽는다. **W5-4 표와 같은 열을
Triton 어휘로 채운다.** GPU 가 없어도 CPU 백엔드(ONNX Runtime)로 돌아간다.

## 시간

2일. 설치와 모델 리포지터리 구성이 절반이다.
