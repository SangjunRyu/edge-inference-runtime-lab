# 01 — GStreamer

**우선순위 1.** W5-4와 가장 가깝고, 엣지 비디오 파이프라인에 어디에나 있고, 실제
업무에서 만날 확률이 가장 높다. DEEPX의 dx_app 생태계도 비디오 입력에 GStreamer를
쓴다.

## 왜 먼저인가

W5-4에서 소스에 시계를 주고 `Block` / `DropOldest` / `DropNewest`를 비교해서 얻은
결론이, GStreamer에서는 **라이브 소스의 표준 레시피**다.

```
queue leaky=downstream max-size-buffers=1
```

오래된 버퍼를 버리고 큐를 거의 비워둔다. 즉 drop-oldest에 얕은 큐다. 합성 실험으로
독립적으로 도달한 결론을 업계가 기본값으로 쓰고 있다는 뜻이고, **이건 면접에서 쓸 수
있는 종류의 수렴이다.**

## 대응표

| 내 개념 | GStreamer |
|---|---|
| 큐 용량 | `max-size-buffers`, `max-size-bytes`, `max-size-time` |
| Block | `leaky=no` (기본값) |
| DropOldest | `leaky=downstream` |
| DropNewest | `leaky=upstream` |
| 소스가 자기 시계를 가짐 | live source, `is-live=true`, 파이프라인 클럭 |
| 출력 시점의 나이 | latency query, 버퍼 PTS 대 실행 시각 |
| 유한 풀 | 버퍼 풀(`GstBufferPool`), `min-buffers` / `max-buffers` |

**`leaky` 의 이름이 직관과 반대다.** `downstream`이 오래된 것을 버리고 `upstream`이
새 것을 버린다. 값으로는 1이 upstream, 2가 downstream이다. 헷갈리기로 유명하니
**쓸 때 enum 을 확인하라.** 이 문서의 서술도 검증 대상이다.

## 무엇을 읽나

1. `queue` 엘리먼트 문서 전체. 프로퍼티가 열 개 남짓이고 그중 절반이 내 W5-4다.
2. `queue2` 와의 차이. `queue2`는 링 버퍼와 다운로드 버퍼링용이고 라이브에는 보통
   `queue`다.
3. `appsrc` / `appsink`. 내 코드와 GStreamer를 잇는 지점이고, `appsink`의
   `max-buffers` + `drop=true`가 또 하나의 drop-oldest다.
4. latency query(`GST_QUERY_LATENCY`)와 라이브 소스의 지연 보상. **파이프라인이
   자기 지연을 스스로 계산해서 소스에 알려준다**는 발상이 내 프로젝트에 없는 것이다.
5. `GST_DEBUG` 와 `gst-launch-1.0` 로 손으로 파이프라인을 조립해 보기.

검색어: `GStreamer queue leaky downstream`, `GStreamer max-size-buffers`,
`GStreamer drop old frames latency`, `GStreamer live source latency`,
`appsink max-buffers drop`

## 확인할 질문

1. `leaky=downstream`과 `leaky=upstream` 중 어느 쪽이 오래된 버퍼를 버리나.
   문서가 아니라 `gst-inspect-1.0 queue` 출력으로 확인하라.
2. `max-size-buffers=1`과 `max-size-time=0`을 같이 쓰면 무슨 일이 생기나.
   세 한계(buffers/bytes/time)는 OR인가 AND인가.
3. 라이브 소스에서 파이프라인이 지연을 어떻게 계산하고 누가 그걸 보상하나.
4. `appsink drop=true`로 버려진 버퍼를 셀 수 있나. 내 W5-4는 드롭 수를 세는데
   GStreamer는 세어 주나.
5. 버퍼 풀은 누가 소유하나. 요소마다인가 파이프라인마다인가. 내 `BufferPool`의
   대응물이 정확히 어디인가.

## 산출물

```
gst-launch-1.0 videotestsrc is-live=true ! \
  queue leaky=downstream max-size-buffers=1 ! \
  identity sleep-time=25000 ! fakesink sync=false
```

`identity sleep-time`으로 싱크를 느리게 만들어 과부하를 유도하고, `leaky`를 바꿔가며
드롭 수와 지연을 비교한다. **W5-4 표와 같은 열을 GStreamer 어휘로 채우는 것이
산출물이다.** 자세한 것은 [07 실습 계획](07-plan.md)에 있다.

## 시간

1~2일. 파이프라인을 손으로 몇 개 조립해 보는 것이 문서를 읽는 것보다 빠르다.
