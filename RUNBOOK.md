# Runbook — re-running this project, phase by phase

Every result in `docs/` came from a command. This is those commands, in order,
with what each one asks and what a healthy answer looks like.

Read it like a lab notebook rather than a script. The phases build on each
other, but each one is independently runnable, and several need no accelerator
at all.

## What needs what

The project is deliberately layered so that most of it survives a change of
machine. Work down this table until you run out of hardware; everything above
the line you stop at still runs.

| tier | needs | covers |
|---|---|---|
| 0 | Python only | `tools/dxnn_inspect` (plus a `.dxnn` file), `repro/stats` |
| 1 | a C++17 compiler | all of W5's harnesses, `repro/cpp/04`–`07` |
| 2 | Python + OpenVINO wheel | W1, W2, W3 |
| 3 | + OpenCV and the OpenVINO C SDK | W4's runners, `repro/cpp/01`–`03` |

Tier 1 is the interesting one: the W5 harnesses link neither OpenVINO nor
OpenCV, which is both why they run anywhere and why their zero-allocation claim
means anything. See [`docs/findings-w5.md`](docs/findings-w5.md).

## Setup

### Python

Call the interpreter directly rather than activating the environment. On Windows
`activate` needs a relaxed execution policy, and on any machine an environment
activated earlier in `PATH` can silently win.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m bench.env_fingerprint
```

On Linux or macOS substitute `.venv/bin/python` throughout.

`env_fingerprint` prints the CPU, the GPUs, the RAM and the package versions.
Record it before quoting any number: every measured result in this repository is
about the machine that produced it.

### C++

No Microsoft toolchain is required. The build uses OpenVINO's **C** API, whose
ABI is stable, so GCC can link the MSVC-built import library.

```bash
# MSYS2 UCRT64
pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-cmake \
          mingw-w64-ucrt-x86_64-ninja mingw-w64-ucrt-x86_64-opencv

cmake -S cpp -B cpp/build -G Ninja
cmake --build cpp/build
```

OpenCV is optional. Without it the build still produces `probe` and every W5
target, and prints a warning naming what it skipped.

**Put the toolchain runtime on `PATH` before running anything.** Without it the
executables exit 127 with no message — the same silent failure this project
records for OpenVINO's own missing DLLs.

```bash
export PATH="/c/msys64/ucrt64/bin:$PATH"     # Git Bash
$env:PATH = "C:\msys64\ucrt64\bin;$env:PATH" # PowerShell
```

### Assets

```powershell
.venv\Scripts\python.exe -m pip install torch torchvision   # export only
.venv\Scripts\python.exe -m bench.prepare_models --models resnet50 --sizes 224
.venv\Scripts\python.exe -m bench.get_clips
```

`get_clips` fetches the Big Buck Bunny grid at three resolutions. Three clips at
one content, so resolution varies and content does not — `read` time is a stage
this project measures, and a different video would change it for reasons that
have nothing to do with the pipeline.

---

## W1 — where does a frame spend its time?

One frame at a time, every stage timed separately.

```powershell
.venv\Scripts\python.exe -m bench.run_sync --model models/resnet50_224.onnx `
    --backend cpu --source clips/bbb_360_2MB.mp4 --frames 300 --warmup 150 --tag w1-360
```

Then the full grid — three backends by three resolutions, five runs per cell,
with a cool-down and a busy-CPU gate between runs:

```powershell
.venv\Scripts\python.exe -m bench.run_matrix --tag gram --repeat 5 --cooldown 60 --max-busy-pct 25
.venv\Scripts\python.exe -m bench.summarize --tag gram
```

**What a healthy answer looks like.** `summarize` prints two tables and the
run-to-run coefficient of variation. `inference` should vary across backends and
stay flat across clips; `read` the other way round. It also prints the worst CV
it saw — if that is large, the grid is describing the machine and not the
backends, and the honest move is to say so rather than quote the means.

Write-up: [`docs/findings-w1.md`](docs/findings-w1.md).

---

## W2 — what does overlap buy, and where?

```powershell
.venv\Scripts\python.exe -m bench.run_async --model models/resnet50_224.onnx `
    --backend openvino --ov-device NPU --source clips/bbb_360_2MB.mp4 `
    --frames 600 --inflight 4 --tag w2-npu

.venv\Scripts\python.exe -m bench.run_ab --model models/resnet50_224.onnx `
    --backend openvino --ov-device NPU --tag w2-ab
```

`run_ab` is the one to trust: it runs sequential and overlapped as a *pair*, in
alternating order, so a machine that drifts during the session cannot be
mistaken for a speedup.

In-flight depth:

```powershell
.venv\Scripts\python.exe -m bench.run_sweep --depths 1,2,4,8,16 --tag w2-sweep
```

**Watch the order violations, not just the throughput.** Overlap that produces
frames out of order has not produced a speedup, it has produced a different
program. The runner reports both.

Write-up: [`docs/findings-w2.md`](docs/findings-w2.md).

---

## W3 — are the answers still correct?

```powershell
.venv\Scripts\python.exe -m bench.verify_equivalence --model models/resnet50_224.onnx `
    --source clips/bbb_360_2MB.mp4 --frames 200

.venv\Scripts\python.exe -m bench.compare_backends --model models/resnet50_224.onnx
```

`verify_equivalence` carries its own self-test: it can inject a deliberate
mismatch and confirm it is caught, so a pass means the check works rather than
that the check ran.

`bench/read_rate.py` answers the question W2 left open: whether `read` gets
cheaper when it is called more often, which is the standing explanation for the
one number in this repository that was never defended.

```
.venv\Scripts\python.exe -m bench.read_rate --compare-threads 16,1 --pin 6,7,8,9,10,11
.venv\Scripts\python.exe -m bench.read_rate --source clips/bbb_360_2MB.mp4 --gaps 0,5,10,19,30
```

**Pin the process or the result is noise.** On a hybrid CPU an unpinned control
of pure integer work moved 24% and 49% between the halves of a pair, which is
W1 finding 4 arriving in a different experiment. Pick a group of same-type
cores; the tool prints the control's spread and refuses to quote a ratio above
15%.

Write-up: [`docs/findings-w3.md`](docs/findings-w3.md).

---

## W4 — what does explicit C++ concurrency change?

```bash
cpp/build/probe.exe --model models/resnet50_224.onnx      # toolchain sanity
cpp/build/run_sync_cpp.exe  --model models/resnet50_224.onnx \
    --source clips/bbb_360_2MB.mp4 --frames 300 --tag w4-sync
cpp/build/run_async_cpp.exe --model models/resnet50_224.onnx \
    --source clips/bbb_360_2MB.mp4 --frames 600 --inflight 4 --tag w4-async
```

The comparisons, which are the point:

```powershell
.venv\Scripts\python.exe -m bench.compare_impl --mode sync   --tag w4-impl
.venv\Scripts\python.exe -m bench.compare_variants --variant fused --tag w4-fused
.venv\Scripts\python.exe -m bench.compare_variants --reanalyse results/w4-fused__*.json
```

`--reanalyse` re-reads saved pairs and applies the current criteria with **zero
machine time**. That is what let W4 discover, after the fact, that two of its
conclusions did not survive a corrected control.

**Expect some of these to refuse to answer.** `compare_variants` judges its
control by variance rather than by level, and on a laptop that has been running
all day it will report the cell as unmeasurable. That is the designed behaviour,
not a failure.

Write-up: [`docs/findings-w4.md`](docs/findings-w4.md).

---

## W5 — what must the pipeline own, and what must it refuse?

None of this needs an accelerator. All of it runs on a laptop with no GPU.

### W5-0 — read a compiled model's memory budget

```bash
python -m tools.dxnn_inspect --self-test
python -m tools.dxnn_inspect path/to/model.dxnn
python -m tools.dxnn_inspect path/to/model.dxnn --npu-memory 128M
python -m tools.dxnn_inspect path/to/model.dxnn --nfh-args
```

Reproduces DX-RT's own admission arithmetic from the model file. With
`--npu-memory` it shows the buffer count being walked down to fit, exactly as
the runtime would, and prints the warning the runtime would log.

Every run ends with a pass/fail count and exits non-zero if a check fails. The
strongest one compares the container's own declared total against the
reconstructed single-buffer footprint; agreement to the byte is the evidence
that the cost model is right.

You need a `.dxnn`. Compile one with DX-COM, or use a released one.

### W5-1 — ownership

```bash
cpp/build/w5_ownership.exe --frames 20000 --inflight 8 --workers 4
cpp/build/w5_ownership.exe --frames 1000 --interrupt-at 437
```

Expect zero steady-state allocations, every lease returned, and
`accepted == completed == emitted`.

Then break it on purpose — each check has an injection, and the harness exits
zero only when *exactly* the predicted check fails:

```bash
cpp/build/w5_ownership.exe --inject leak
cpp/build/w5_ownership.exe --inject alloc
cpp/build/w5_ownership.exe --inject early-release
```

`early-release` is the one to look at. It breaks 1 to 2 percent of frames and
nothing else: right count, right order, no leak, another frame's data.

And the claim the compiler checks, which must **fail** to build:

```bash
cmake -S cpp -B cpp/build -DEIRL_NEGATIVE_COMPILE=ON
cmake --build cpp/build --target negative_direction   # must error
cmake -S cpp -B cpp/build -DEIRL_NEGATIVE_COMPILE=OFF
```

### W5-2 — the scheduler and its two bounds

```bash
cpp/build/w5_scheduler.exe --frames 2000 --buffers 8 --queue-capacity 2
cpp/build/w5_scheduler.exe --sweep --frames 4000
```

The sweep varies buffers, queue depths and worker counts and reports both
bounds. Read the table with the distinction in mind: the ceiling column bounds
*leases*, and the in-flight column can legitimately sit one or two above it.

### W5-3 — the format conversion

```bash
cpp/build/w5_format.exe --verify
cpp/build/w5_format.exe --counts --frames 200
cpp/build/w5_format.exe --bench --pairs 10 --reps 30

# drive it with a real model's geometry
cpp/build/w5_format.exe --bench $(python -m tools.dxnn_inspect model.dxnn --nfh-args)
```

`--verify` needs no hardware and no reference output: the round trip is its own
oracle. `--bench` prints UNMEASURABLE and declines to quote a ratio when its
control moves more than 15 percent between pairs, which on a busy laptop is
most of the large geometries.

### W5-3 part two -- the model decides the buffers

The size a buffer must be is not a constant in the code. It comes from the
compiled model, once:

```bash
python -m tools.dxnn_inspect model.dxnn --contract contracts/model.json
cpp/build/w5_contract.exe --contract contracts/model.json --npu-memory 512M
```

Then break each property on purpose. Every one of these must refuse **before a
single stage runs**, and each prints `work executed  0 stages`:

```bash
cpp/build/w5_contract.exe --contract contracts/model.json --inject hardcoded-size
cpp/build/w5_contract.exe --contract contracts/model.json --inject bad-allocated
cpp/build/w5_contract.exe --contract contracts/model.json --inject bad-padding
cpp/build/w5_contract.exe --contract contracts/model.json --inject overflow
```

`hardcoded-size` is the one to look at. It sizes the pool from 150528, the
dense tensor, which is what the code had before. The model needs 157696, and
the difference is the padding. A run that starts anyway would have the device
reading 7168 bytes past the end of every buffer, and nothing at run time would
say so.

### W5 tooling -- models without a device

Two tools take a compiled model apart on a machine that has no NPU.

```
python -m tools.model_budget path/to/*.dxnn --memories 512M,1G,2G
python -m tools.dxnn_cpu_subgraph model.dxnn --run
```

`model_budget` answers the deployment question for a shelf of models at once:
given a device with N bytes, which ones fit and how many inferences can be in
flight. Every figure comes from the containers and from DX-RT's arithmetic.

`dxnn_cpu_subgraph` extracts the subgraphs the compiler left on the host and
runs them. A classifier usually has none. A detector carries its decode-and-NMS
tail as an ordinary ONNX model inside the container, and that half executes here
with no device present.

### W5-4 -- backpressure

Needs a source with its own clock, or the policies are indistinguishable:

```bash
cpp/build/w5_backpressure.exe --compare --frames 240 --repeat 3
cpp/build/w5_backpressure.exe --policy block --frames 200
cpp/build/w5_backpressure.exe --policy drop-oldest --frames 200
```

Read the **age at output** column, not the throughput one. Age is measured from
when a frame was due, so it counts the time a frame spent waiting to be
captured as well as the time it spent in the pipeline. That is the difference
between the policies, and pipeline latency alone hides it.

Write-up: [`docs/findings-w5.md`](docs/findings-w5.md), plan in
[`docs/w5-plan.md`](docs/w5-plan.md).

---

## The reproductions

Every defect found while building this is a runnable program with a broken mode
and a fixed mode. Exit status means "behaved as that mode predicts", so a broken
mode that comes out clean is itself reported rather than rounded to a pass.

```bash
python repro/run_all.py                  # everything runnable here
python repro/run_all.py --skip-openvino  # the portable subset
python repro/run_all.py --device NPU     # the OpenVINO ones on the NPU

cpp/build/repro/repro_05_reorder_off.exe --mode broken
cpp/build/repro/repro_05_reorder_off.exe --mode fixed
```

Three outcomes, not two:

- **ok** — broken reproduced the defect and fixed did not.
- **NOT REPRO** — the broken mode came out clean here. A fact about this
  machine, not a licence to drop the fix.
- **REGRESSION** — the fixed mode failed. The one to act on.

---

## A ten-minute path

If you want to see the shape of the project without downloading a model or a
video:

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

That is the ownership model, the concurrency bounds, the format rules, the
reproductions and the container arithmetic — none of which needs an accelerator,
a model file or a video.

---

## Reading the results afterwards

Results land in `results/` as JSON with one schema, which is what lets Python
and C++ runs sit in the same table and what makes `--reanalyse` possible.

```powershell
.venv\Scripts\python.exe -m bench.summarize --tag <tag>
.venv\Scripts\python.exe -m bench.summarize --tag <tag> --markdown
```

Files whose contents did not survive their own control are kept with a label
saying so rather than deleted. A cell marked unmeasurable describes the machine
that day, and throwing it away would leave the impression that every attempt
succeeded.
