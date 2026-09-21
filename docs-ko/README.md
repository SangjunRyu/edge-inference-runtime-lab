# docs-ko — 한글 정리본

`docs/`, `README.md`, `RUNBOOK.md`, `repro/README.md`의 내용을 한글로 다시 쓴 것이다.
축약 요약이 아니라 각 문서가 스스로 완결되도록 썼고, 수치와 판정은 원문과 같은 값을
쓴다.

**원문이 권위다.** 코드가 바뀌면 영문 쪽이 먼저 갱신되고 여기는 뒤따른다. 값이
어긋나 보이면 원문을 믿으면 된다. 각 문서 맨 위에 대응하는 원문 경로를 적어 뒀다.

## 읽는 순서

| 문서 | 내용 |
|---|---|
| [00 개요](00-overview.md) | 프로젝트가 무엇이고 무엇을 알아냈는지. **여기부터 읽으면 된다** |
| [01 W1](01-w1.md) | 프레임 하나는 시간을 어디에 쓰는가 |
| [02 W2](02-w2.md) | 스테이지를 겹치면 무엇이 달라지는가 |
| [03 W3](03-w3.md) | 처리량이 올라도 출력이 여전히 맞는가 |
| [04 W4](04-w4.md) | 명시적 C++ 동시성과 소유권이 바꾸는 것 |
| [05 W5](05-w5.md) | 파이프라인이 소유해야 하는 것과 거절해야 하는 것 |
| [06 재현 프로그램](06-repro.md) | 발견한 결함 11개를 실행 가능한 형태로 |
| [07 실행법](07-runbook.md) | W1부터 W5까지 어떤 명령으로 어떻게 돌리는가 |
| [08 측정 규율](08-protocol.md) | 첫 측정 전에 고정한 규칙과 계측기의 잡음 특성 |
| [09 read 호출 빈도](09-read-rate.md) | W2 가 남긴 미귀속 수치의 결말, 그리고 통제가 여섯 번 깨진 원인 |

시간이 없으면 [00 개요](00-overview.md)와 [07 실행법](07-runbook.md) 둘이면 된다.
개요에 결과 열 개가 한 화면에 있고, 실행법에 그걸 다시 만드는 명령이 있다.

## 원문 대응표

| 한글 | 원문 |
|---|---|
| 00 개요 | `README.md` |
| 01 W1 | `docs/findings-w1.md` |
| 02 W2 | `docs/findings-w2.md` |
| 03 W3 | `docs/findings-w3.md` |
| 04 W4 | `docs/findings-w4.md`, `docs/w4-plan.md`, `docs/w4-setup.md` |
| 05 W5 | `docs/findings-w5.md`, `docs/w5-plan.md` |
| 06 재현 프로그램 | `repro/README.md` |
| 07 실행법 | `RUNBOOK.md` |
| 08 측정 규율 | `docs/protocol.md` |
| 09 read 호출 빈도 | `docs/findings-read-rate.md` |

여기에 대응본이 없는 원문도 있다. `docs/dxapp-callgraph.md`와
`docs/dxapp-sync-async.md`는 DX-APP 소스를 읽으며 남긴 분석 노트이고,
`tools/README.md`는 DXNN 인스펙터 사용법이다. 필요한 내용은
[05 W5](05-w5.md)와 [07 실행법](07-runbook.md)에 녹여 뒀다.
