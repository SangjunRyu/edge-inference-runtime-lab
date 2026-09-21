# 실행법 — 단계별로 다시 돌리기

원문: [`RUNBOOK.md`](../RUNBOOK.md)

`docs/`의 모든 결과는 명령에서 나왔다. 여기 그 명령들이 순서대로 있다. 각 명령이
무엇을 묻고 정상 응답이 어떤 모양인지 함께 적었다.

스크립트가 아니라 실습 노트처럼 읽으면 된다. 단계는 서로 쌓이지만 각각 독립적으로
돌릴 수 있고, 여럿은 가속기가 아예 필요 없다.

## 무엇이 무엇을 필요로 하는가

기계가 바뀌어도 대부분이 살아남도록 일부러 층을 나눴다. 하드웨어가 떨어질 때까지
아래로 내려가면 되고, 멈춘 줄 위는 전부 돈다.

| 층 | 필요 | 포함 |
|---|---|---|
| 0 | 파이썬만 | `tools/dxnn_inspect`(그리고 `.dxnn` 파일), `repro/stats` |
| 1 | C++17 컴파일러 | W5 하네스 전부, `repro/cpp/04`~`08` |
| 2 | 파이썬 + OpenVINO 휠 | W1, W2, W3 |
| 3 | + OpenCV 와 OpenVINO C SDK | W4 러너, `repro/cpp/01`~`03` |

1층이 흥미로운 지점이다. W5 하네스는 OpenVINO도 OpenCV도 링크하지 않는다. 그래서
어디서나 돌고, 그 사실이 할당 0 주장에 의미를 준다.

## 준비

### 파이썬

환경을 활성화하지 말고 인터프리터를 직접 부른다. 윈도우에서 `activate`는 실행 정책을
완화해야 하고, 어느 기계에서든 `PATH` 앞쪽에 먼저 활성화된 환경이 조용히 이길 수
있다.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m bench.env_fingerprint
```

리눅스나 macOS에서는 `.venv/bin/python`으로 바꾼다.

`env_fingerprint`는 CPU, GPU, RAM, 패키지 버전을 출력한다. **어떤 숫자를 인용하기
전에 먼저 기록한다.** 이 저장소의 모든 측정 결과는 그것을 만든 기계에 대한 이야기다.

### C++

마이크로소프트 툴체인이 필요 없다. 빌드는 OpenVINO의 **C** API를 쓰고 그 ABI는
안정적이라 GCC가 MSVC로 만든 임포트 라이브러리를 링크한다.

```bash
# MSYS2 UCRT64
pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-cmake \
          mingw-w64-ucrt-x86_64-ninja mingw-w64-ucrt-x86_64-opencv

cmake -S cpp -B cpp/build -G Ninja
cmake --build cpp/build
```

OpenCV는 선택이다. 없어도 `probe`와 W5 타깃 전부가 빌드되고, 무엇을 건너뛰었는지
경고로 알려준다.

**실행 전에 툴체인 런타임을 `PATH`에 넣어야 한다.** 없으면 실행 파일이 메시지 없이
127로 끝난다. 이 프로젝트가 OpenVINO의 DLL 누락에 대해 기록해 둔 것과 같은 무음
실패다.

```bash
export PATH="/c/msys64/ucrt64/bin:$PATH"     # Git Bash
$env:PATH = "C:\msys64\ucrt64\bin;$env:PATH" # PowerShell
```

### 자산

```powershell
.venv\Scripts\python.exe -m pip install torch torchvision   # 내보내기 전용
.venv\Scripts\python.exe -m bench.prepare_models --models resnet50 --sizes 224
.venv\Scripts\python.exe -m bench.get_clips
```

`get_clips`는 Big Buck Bunny를 세 해상도로 받는다. 같은 내용의 클립 셋이라 해상도만
변하고 내용은 변하지 않는다. `read`는 이 프로젝트가 재는 스테이지이고, 다른 영상을
쓰면 파이프라인과 무관한 이유로 값이 달라진다.

## W1 — 프레임 하나는 시간을 어디에 쓰는가

```powershell
.venv\Scripts\python.exe -m bench.run_sync --model models/resnet50_224.onnx `
    --backend cpu --source clips/bbb_360_2MB.mp4 --frames 300 --warmup 150 --tag w1-360

.venv\Scripts\python.exe -m bench.run_matrix --tag gram --repeat 5 --cooldown 60 --max-busy-pct 25
.venv\Scripts\python.exe -m bench.summarize --tag gram
```

**정상 응답의 모양.** `summarize`가 표 둘과 실행 간 변동계수를 출력한다. `inference`는
백엔드에 따라 변하고 클립에 따라 평평해야 한다. `read`는 그 반대다. 본 것 중 최악의
CV도 함께 출력하는데, 그게 크면 격자가 백엔드가 아니라 기계를 설명하고 있는 것이고
정직한 선택은 평균을 인용하지 않고 그렇다고 말하는 것이다.

## W2 — 겹치면 무엇이 달라지는가

```powershell
.venv\Scripts\python.exe -m bench.run_async --model models/resnet50_224.onnx `
    --backend openvino --ov-device NPU --source clips/bbb_360_2MB.mp4 `
    --frames 600 --inflight 4 --tag w2-npu

.venv\Scripts\python.exe -m bench.run_ab --model models/resnet50_224.onnx `
    --backend openvino --ov-device NPU --tag w2-ab

.venv\Scripts\python.exe -m bench.run_sweep --depths 1,2,4,8,16 --tag w2-sweep
```

믿을 것은 `run_ab`다. 순차와 겹침을 **쌍으로**, 순서를 교대하며 돌린다. 세션 중에
드리프트하는 기계를 속도 향상으로 착각할 수 없다.

**처리량만 보지 말고 순서 위반을 본다.** 순서 없이 프레임을 내놓는 겹침은 속도
향상을 만든 게 아니라 다른 프로그램을 만든 것이다. 러너가 둘 다 보고한다.

## W3 — 답이 여전히 맞는가

```powershell
.venv\Scripts\python.exe -m bench.verify_equivalence --model models/resnet50_224.onnx `
    --source clips/bbb_360_2MB.mp4 --frames 200

.venv\Scripts\python.exe -m bench.compare_backends --model models/resnet50_224.onnx
```

`verify_equivalence`는 자체 시험을 들고 있다. 일부러 불일치를 주입하고 잡히는지
확인할 수 있어서, 통과는 검사가 돌았다는 뜻이 아니라 검사가 동작한다는 뜻이 된다.

## W4 — 명시적 C++ 동시성

```bash
cpp/build/probe.exe --model models/resnet50_224.onnx      # 툴체인 확인
cpp/build/run_sync_cpp.exe  --model models/resnet50_224.onnx \
    --source clips/bbb_360_2MB.mp4 --frames 300 --tag w4-sync
cpp/build/run_async_cpp.exe --model models/resnet50_224.onnx \
    --source clips/bbb_360_2MB.mp4 --frames 600 --inflight 4 --tag w4-async
```

핵심인 비교들이다.

```powershell
.venv\Scripts\python.exe -m bench.compare_impl --mode sync   --tag w4-impl
.venv\Scripts\python.exe -m bench.compare_variants --variant fused --tag w4-fused
.venv\Scripts\python.exe -m bench.compare_variants --reanalyse results/w4-fused__*.json
```

`--reanalyse`는 저장된 쌍을 다시 읽어 현재 기준을 **기계 시간 0으로** 적용한다. W4가
사후에 결론 둘이 고친 통제를 통과하지 못한다는 걸 알아낸 방법이다.

**일부는 답하기를 거부하는 게 정상이다.** `compare_variants`는 통제를 수준이 아니라
분산으로 판정해서, 하루 종일 돌린 노트북에서는 그 셀을 측정 불가로 보고한다. 설계된
동작이지 실패가 아니다.

## W5 — 파이프라인은 무엇을 소유하고 무엇을 거절하는가

여기는 가속기가 전혀 필요 없다. GPU 없는 노트북에서 전부 돈다.

### W5-0 — 컴파일된 모델의 메모리 예산

```bash
python -m tools.dxnn_inspect --self-test
python -m tools.dxnn_inspect path/to/model.dxnn --npu-memory 128M
python -m tools.dxnn_inspect path/to/model.dxnn --nfh-args
```

모델 파일에서 DX-RT의 승인 산술을 재현한다. `--npu-memory`를 주면 버퍼 개수가
런타임과 똑같이 깎이는 과정과 런타임이 남길 경고를 보여준다.

매 실행이 통과·실패 개수로 끝나고 하나라도 실패하면 종료 코드가 0이 아니다. 가장
강한 검사는 컨테이너가 스스로 선언한 총량과 재구성한 단일 버퍼 footprint를 비교하는
것이다. 바이트 단위로 맞으면 비용 모델이 옳다는 증거다.

`.dxnn` 파일이 필요하다. DX-COM으로 컴파일하거나 배포된 것을 쓴다.

### W5-1 — 소유권

```bash
cpp/build/w5_ownership.exe --frames 20000 --inflight 8 --workers 4
cpp/build/w5_ownership.exe --frames 1000 --interrupt-at 437
```

정상 상태 할당 0, 모든 리스 반환, `accepted == completed == emitted`가 나와야 한다.

그다음 일부러 깨뜨린다. 검증마다 주입이 있고, **예측한 검증만 정확히** 실패할 때
종료 코드가 0이다.

```bash
cpp/build/w5_ownership.exe --inject leak
cpp/build/w5_ownership.exe --inject alloc
cpp/build/w5_ownership.exe --inject early-release
```

`early-release`를 보면 된다. 프레임의 1~2%만 깨뜨리고 나머지는 멀쩡하다. 개수도
순서도 맞고 누수도 없이 다른 프레임의 데이터를 싣는다.

컴파일러가 검사하는 주장은 빌드에 **실패해야** 한다.

```bash
cmake -S cpp -B cpp/build -DEIRL_NEGATIVE_COMPILE=ON
cmake --build cpp/build --target negative_direction   # 에러가 나야 정상
cmake -S cpp -B cpp/build -DEIRL_NEGATIVE_COMPILE=OFF
```

### W5-2 — 스케줄러와 두 상한

```bash
cpp/build/w5_scheduler.exe --frames 2000 --buffers 8 --queue-capacity 2
cpp/build/w5_scheduler.exe --sweep --frames 4000
```

스윕이 버퍼, 큐 깊이, 워커 수를 바꿔가며 두 상한을 함께 보고한다. 표를 읽을 때
구분을 기억해야 한다. **천장 열은 리스를 묶고**, in-flight 열은 그보다 한둘 위에
정당하게 앉을 수 있다.

### W5-3 — 포맷 변환

```bash
cpp/build/w5_format.exe --verify
cpp/build/w5_format.exe --counts --frames 200
cpp/build/w5_format.exe --bench --pairs 10 --reps 30

# 실제 모델 형상으로 구동
cpp/build/w5_format.exe --bench $(python -m tools.dxnn_inspect model.dxnn --nfh-args)
```

`--verify`는 하드웨어도 참조 출력도 필요 없다. 왕복이 스스로 정답지다. `--bench`는
통제가 쌍 사이에서 15% 넘게 움직이면 UNMEASURABLE을 출력하고 비율 인용을 거부한다.
바쁜 노트북에서는 큰 형상 대부분이 그렇게 된다.

### W5-3 두 번째 — 모델이 버퍼를 정한다

버퍼 크기가 코드의 상수가 아니라 컴파일된 모델에서 한 번 온다.

```bash
python -m tools.dxnn_inspect model.dxnn --contract contracts/model.json
cpp/build/w5_contract.exe --contract contracts/model.json --npu-memory 512M
```

그다음 각 성질을 일부러 깨뜨린다. 넷 모두 **한 스테이지도 실행되기 전에** 거부해야
하고, `work executed  0 stages`를 출력한다.

```bash
cpp/build/w5_contract.exe --contract contracts/model.json --inject hardcoded-size
cpp/build/w5_contract.exe --contract contracts/model.json --inject bad-allocated
cpp/build/w5_contract.exe --contract contracts/model.json --inject bad-padding
cpp/build/w5_contract.exe --contract contracts/model.json --inject overflow
```

`hardcoded-size`를 보면 된다. 풀을 `150528`로 잡는데 이건 예전 코드가 쓰던 값이고
조밀 텐서 크기다. 모델은 `157696`이 필요하고 차이가 패딩이다. 그냥 시작했다면 장치가
모든 버퍼의 끝에서 7,168바이트를 넘겨 읽었을 것이고 런타임에는 아무 표시도 없다.

### W5-4 — 백프레셔

소스에 자기 시계가 없으면 정책들이 구분되지 않는다.

```bash
cpp/build/w5_backpressure.exe --compare --frames 240 --repeat 3
cpp/build/w5_backpressure.exe --policy block --frames 200
cpp/build/w5_backpressure.exe --policy drop-oldest --frames 200
```

처리량 열이 아니라 **출력 시점의 나이** 열을 본다. 나이는 프레임이 나왔어야 할
시각부터 세므로 파이프라인 안에 있던 시간뿐 아니라 캡처를 기다린 시간도 센다. 그게
정책 간의 차이이고, 파이프라인 지연만으로는 감춰진다.

## 재현 프로그램

```bash
python repro/run_all.py                  # 여기서 돌릴 수 있는 전부
python repro/run_all.py --skip-openvino  # 이식 가능한 부분집합

cpp/build/repro/repro_05_reorder_off.exe --mode broken
cpp/build/repro/repro_05_reorder_off.exe --mode fixed
```

자세한 내용은 [재현 프로그램](06-repro.md)에 있다.

## 10분 경로

모델도 영상도 내려받지 않고 프로젝트 모양만 보려면 이렇게 한다.

```bash
export PATH="/c/msys64/ucrt64/bin:$PATH"
cmake -S cpp -B cpp/build -G Ninja && cmake --build cpp/build

cpp/build/w5_ownership.exe --frames 5000 --inflight 4
cpp/build/w5_ownership.exe --inject early-release
cpp/build/w5_scheduler.exe --sweep --frames 1000
cpp/build/w5_format.exe --verify
cpp/build/w5_backpressure.exe --policy drop-oldest --frames 120
python repro/run_all.py --skip-openvino
python -m tools.dxnn_inspect --self-test
```

소유권 모델, 동시성 상한, 포맷 규칙, 재현 프로그램, 컨테이너 산술이다. 가속기도
모델 파일도 영상도 필요 없다.

## 결과 읽기

결과는 `results/`에 하나의 스키마를 가진 JSON으로 떨어진다. 그래서 파이썬과 C++
실행이 같은 표에 앉고 `--reanalyse`가 가능하다.

```powershell
.venv\Scripts\python.exe -m bench.summarize --tag <tag>
.venv\Scripts\python.exe -m bench.summarize --tag <tag> --markdown
```

자기 통제를 통과하지 못한 파일은 지우지 않고 그렇다는 라벨을 달아 남긴다. 측정
불가로 표시된 셀은 그날의 기계를 설명한다. 버리면 모든 시도가 성공한 것처럼 보인다.
