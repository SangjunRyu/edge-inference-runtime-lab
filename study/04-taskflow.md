# 04 — Taskflow

**우선순위 4.** API 가 깔끔하고 같은 문제에 또 다른 이름을 붙이지만, 생태계가 작아
실무에서 만날 확률은 낮다. **하루짜리다.** 개념만 가져오고 넘어간다.

## 왜 그래도 보나

TBB 가 토큰이라 부르는 것을 Taskflow 는 **line** 이라 부른다. 같은 것을 두 이름으로
보면 그게 라이브러리의 성질이 아니라 문제의 성질이라는 게 확실해진다. 그 확신이
이 디렉터리 전체의 목적이다.

## 대응표

```cpp
tf::Pipeline pl(num_lines,                       // <- 이게 내 ceiling 이다
  tf::Pipe{tf::PipeType::SERIAL,   [](tf::Pipeflow& pf){ /* read */ }},
  tf::Pipe{tf::PipeType::PARALLEL, [](tf::Pipeflow& pf){ /* preprocess */ }},
  tf::Pipe{tf::PipeType::PARALLEL, [](tf::Pipeflow& pf){ /* infer */ }},
  tf::Pipe{tf::PipeType::SERIAL,   [](tf::Pipeflow& pf){ /* write */ }});
```

| 내 개념 | Taskflow |
|---|---|
| 천장 | `num_lines` |
| 스테이지 | `tf::Pipe` |
| 직렬 스테이지 | `tf::PipeType::SERIAL` |
| 병렬 스테이지 | `tf::PipeType::PARALLEL` |
| 프레임 번호 | `pf.token()` |
| 어느 라인에 실렸나 | `pf.line()` |
| 스테이지 종료 | `pf.stop()` |

**line 이 토큰보다 직관적이다.** 파이프라인을 격자로 보고 라인 하나가 한 프레임의
수평 경로다. 라인 수가 동시에 흐를 수 있는 프레임 수이므로 곧 천장이다.

## 세 가지 변형

- `tf::Pipeline` — 파이프 구성이 컴파일 타임에 고정
- `tf::ScalablePipeline` — 파이프를 런타임에 바꿀 수 있음. 내 `--pre-workers` 같은
  구성 가능성에 대응
- `tf::DataPipeline` — 스테이지 사이에 데이터를 실어 나름. 내 `FrameContext` 가
  하는 일을 라이브러리가 해줌

세 번째가 흥미롭다. 내 설계에서는 프레임이 리스를 들고 다니는데, `DataPipeline` 은
값을 들고 다닌다. **소유권을 누가 쥐느냐의 차이**이고, 그게 W5-1 전체의 주제다.

## 무엇을 읽나

1. Pipeline 튜토리얼과 `Pipeflow` API. 한 시간이면 된다.
2. `ScalablePipeline` 과 `DataPipeline` 의 차이.
3. Executor 와 work stealing. W5 계획서가 work stealing 을 명시적으로 배제했으니
   **배제한 것이 무엇인지는 알아야 한다.**

검색어: `Taskflow pipeline lines tokens`, `Taskflow scalable pipeline`,
`Taskflow data pipeline`, `Taskflow executor work stealing`

## 확인할 질문

1. `num_lines` 를 늘리면 메모리가 얼마나 늘어나나. Taskflow 가 라인당 무엇을 잡나.
   내 천장은 버퍼 수와 묶여 있는데 Taskflow 는 묶여 있나.
2. `SERIAL` 파이프가 순서를 보장하나, 아니면 한 번에 하나만 돈다는 뜻인가.
   TBB 의 `serial_in_order` 와 `serial_out_of_order` 구분에 대응하는 게 있나.
3. work stealing 이 있는 스케줄러에서 내 "구조 용량" 개념이 성립하나. 워커가 일을
   훔치면 "스테이지마다 워커 하나"라는 셈이 달라지나.
4. `DataPipeline` 에서 큰 텐서를 스테이지 사이로 옮기면 복사가 생기나. 내
   `BufferLease` 가 move-only 인 이유가 여기서도 유효한가.

## 산출물

`tf::Pipeline` 로 W5-2 의 스윕을 재현한다. `num_lines` 를 바꿔가며 관측 최대
동시 프레임 수를 세고, **천장 공식이 여기서도 성립하는지** 본다. 내 공식은 큐와
워커를 따로 세는데 Taskflow 는 라인 하나로 합쳐 버리니, 차이가 나면 그게 결과다.

## 시간

1일. 이 디렉터리에서 가장 짧다.
