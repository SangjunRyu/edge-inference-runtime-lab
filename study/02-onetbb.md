# 02 — oneTBB

**우선순위 2.** 내 `BufferLease`와 천장의 가장 직접적인 형식적 대응물이 여기 있다.
그리고 이미 내 프로세스 안에 들어와 있다.

## 왜 이미 들어와 있나

OpenVINO CPU 플러그인이 TBB 위에 지어져 있다. `cpp/build/`에 `tbb12.dll`,
`tbbbind_2_5.dll`, `tbbmalloc.dll`이 OpenVINO libs에서 복사돼 앉아 있다.

W2 발견 B에서 CPU 비동기가 진 이유도 여기서 나온다. `run_async`가 계산을 수행하는
바로 그 intra-op 풀에서 돌아 병렬성을 더하지 않고 나눴다. **내가 만든 스레드 풀과
OpenVINO의 풀이 같은 코어를 두고 경쟁한 것이고, 그 풀이 TBB다.**

## 두 갈래를 봐야 한다

### `parallel_pipeline` — 토큰이 곧 천장

```cpp
tbb::parallel_pipeline(
    max_number_of_live_tokens,          // <- 이게 내 ceiling 이다
    tbb::make_filter<void, Frame>(tbb::filter_mode::serial_in_order, reader)
  & tbb::make_filter<Frame, Tensor>(tbb::filter_mode::parallel,      preprocess)
  & tbb::make_filter<Tensor, Result>(tbb::filter_mode::parallel,     infer)
  & tbb::make_filter<Result, void>(tbb::filter_mode::serial_in_order, writer));
```

대응이 거의 일대일이다.

| 내 개념 | parallel_pipeline |
|---|---|
| 천장 | `max_number_of_live_tokens` |
| 스테이지 | filter |
| 워커 수 | `filter_mode::parallel` 이면 스케줄러가 정함 |
| 순서 복원 | `filter_mode::serial_in_order` |
| 순서 무관 직렬 | `filter_mode::serial_out_of_order` |

**차이도 분명하다.** TBB는 스테이지별 워커 수를 내가 정하지 않는다. `parallel`이라고
선언하면 아레나가 알아서 배분한다. 내 스케줄러는 `--pre-workers`처럼 명시한다.
어느 쪽이 나은지가 아니라 **무엇을 통제권으로 남길지의 선택**이고, 그걸 설명할 수
있어야 한다.

### flow graph — `limiter_node`가 크레딧 기반 백프레셔

```cpp
tbb::flow::graph g;
tbb::flow::limiter_node<Frame> limiter(g, threshold);
tbb::flow::function_node<Frame, Result> work(g, concurrency, body);
tbb::flow::sequencer_node<Result> order(g, [](const Result& r){ return r.seq; });

tbb::flow::make_edge(limiter, work);
tbb::flow::make_edge(work, order);
tbb::flow::make_edge(order, limiter.decrementer());   // <- 크레딧 반환
```

마지막 줄이 핵심이다. **완료가 리미터에게 크레딧을 돌려준다.** 내 `BufferLease`가
소멸하며 풀에 돌아가는 것과 같은 구조이고, 다른 점은 TBB가 자원이 아니라 카운트를
센다는 것이다. 내 쪽이 실제 버퍼를 세니 메모리 예산과 연결되고, TBB 쪽이 더 가볍다.

`sequencer_node`는 순번으로 순서를 복원한다. **내 `RingReorder`와 같은 역할**이다.
TBB가 어떻게 무한히 자라지 않게 막는지 보면 내 링 크기 유도와 비교할 거리가 된다.

## 무엇을 읽나

1. `parallel_pipeline` 레퍼런스와 토큰 설명. 토큰이 왜 필요한지, 없으면 무엇이
   자라는지.
2. `filter_mode` 세 가지의 차이와 각각이 만드는 순서 보장.
3. `limiter_node` 레퍼런스, 특히 `decrementer()` 포트.
4. `sequencer_node`.
5. `task_arena`와 `global_control`. OpenVINO와 코어를 두고 싸우는 문제를 다루는 곳이
   여기다. W2 발견 B의 처방이 결국 이쪽이다.

검색어: `oneTBB parallel_pipeline live tokens`, `oneTBB flow graph limiter_node`,
`oneTBB sequencer_node`, `oneTBB task_arena constraints`,
`OpenVINO TBB thread pool contention`

## 확인할 질문

1. `max_number_of_live_tokens`를 스테이지 수보다 작게 주면 어떻게 되나. 내 천장
   공식의 "구조 용량"에 대응하는 개념이 TBB에 있나, 아니면 토큰이 전부인가.
2. `limiter_node`의 threshold를 넘는 입력이 오면 거부되나 대기하나. 그게 내
   Block인가 DropNewest인가.
3. `sequencer_node`가 빠진 순번을 영원히 기다리면 어떻게 되나. 내 `RingReorder`는
   강제 방출과 카운트로 처리했는데 TBB는 무엇을 하나.
4. `task_arena`로 OpenVINO와 코어를 분리할 수 있나. W2 발견 B의 1.37배가 그냥 남는
   병렬성이었다는 결론이 이걸로 바뀌나.
5. TBB의 토큰은 메모리를 세지 않는다. DX-RT의 `bufferCount`는 센다. 어떤 시스템에서
   어느 쪽이 맞나.

## 산출물

`parallel_pipeline`으로 W5-4의 네 정책 중 최소한 Block과 무한 큐를 재현하고, 토큰
수를 바꿔가며 in-flight 천장이 토큰 수를 따라가는지 센다. 드롭 정책은 TBB에 없으니
**없다는 것 자체가 결과다.** 이유를 쓰면 된다.

## 시간

2~3일. `parallel_pipeline`은 반나절이고 flow graph가 나머지다.
