# Setting up on the 550XBE

From a clean Samsung NT550XBE to a benchmark you can trust. Written 2026-09-14
against the gram; every command here was run except the ones that need hardware
this machine does not have, and those are marked.

Target: **Windows 11, i7-8565U (4C/8T, 15 W), 8 GB, UHD 620, GeForce MX150
(GP108, sm_61, 2 GB)**.

Done in order this takes an evening, most of it downloads and cooldowns. The
order matters: the harness is proved on CPU before any accelerator is
introduced, so that a failure later has only one possible cause.

---

## 0. The one thing that will waste your evening

**Every onnxruntime flavour installs the same `onnxruntime` package.** `pip
install onnxruntime-directml` on top of `onnxruntime-openvino` does not give you
both — it gives you a broken mixture, or one that loads and silently serves CPU
under an accelerator's label. The gram hit the silent version of this once
already: `openvino==2026.3.1` loaded, the provider failed with Windows error 127,
and ONNX Runtime fell back to CPU without saying so.

On the gram it never mattered, because `onnxruntime-openvino` alone served the
whole grid — CPU, iGPU and NPU are all Intel. **On this machine no single wheel
serves both the MX150 and the UHD 620.** So:

| venv | wheel | serves |
|---|---|---|
| `.venv-dml` | `onnxruntime-directml` | `cpu`, `dml` (MX150) |
| `.venv-cuda` | `onnxruntime-gpu` | `cpu`, `cuda` (MX150) |
| `.venv-ov` | `onnxruntime-openvino` + `openvino` | `cpu`, `openvino:CPU`, `openvino:GPU` (UHD 620) |

One venv per flavour, and run the matrix once per venv with `--backends`. Cells
are separate files under the same `--tag`, already-present cells are skipped,
and `summarize` merges them. `run_matrix` launches each cell with
`sys.executable`, so whichever venv you start it from is the one every cell in
that invocation uses.

This does mean a grid assembled across venvs varies the engine along with the
device — the same confound `findings-w1.md` already records as unresolved
between the ORT CPU row and the OpenVINO rows. It is not new, but it is now
structural rather than incidental, and it belongs in the write-up.

**Hold the onnxruntime version still across the venvs** as far as the wheels
allow, and record it per venv when they do not. The engine already varies by
construction; letting the engine *version* vary on top of it adds a second
uncontrolled difference to the same comparison for no gain. The
`onnxruntime-openvino` flavour tends to lag the others, so it is usually the one
that sets the version everything else should match.

---

## 1. Clone

```powershell
git clone https://github.com/SangjunRyu/DEEPX-poc.git edge-inference-runtime-lab
cd edge-inference-runtime-lab
```

Python 3.13.14 produced every existing result. Any 3.11+ will run the code; stay
on 3.13 if the wheels you need publish for it, and record what you used either
way — `env_fingerprint` does that automatically.

## 2. CPU first, before any accelerator

```powershell
python -m venv .venv-cpu
.venv-cpu\Scripts\python.exe -m pip install --upgrade pip
.venv-cpu\Scripts\python.exe -m pip install numpy opencv-python psutil onnxruntime
.venv-cpu\Scripts\python.exe -m bench.env_fingerprint
```

Call the venv interpreter by path rather than activating it. `activate` needs a
relaxed PowerShell execution policy, and an environment activated earlier in
`PATH` can silently win.

Measured on the 550XBE, 2026-09-15:

```json
{
  "platform": "Windows-11-10.0.26200-SP0",
  "python": "3.13.14",
  "cpu_name": "Intel(R) Core(TM) i7-8565U CPU @ 1.80GHz",
  "cpu_count_logical": 8,
  "gpus_nvidia_smi": ["GeForce MX150, 2048 MiB, 451.67"],
  "ram_total_gb": 7.9,
  "cpu_freq_mhz_at_start": 1792.0,
  "onnxruntime_version": "1.30.0",
  "onnxruntime_available_providers": ["AzureExecutionProvider", "CPUExecutionProvider"],
  "opencv_version": "5.0.0"
}
```

Three things to read off it.

`AzureExecutionProvider` alongside `CPUExecutionProvider` is what the plain
`onnxruntime` wheel ships and means nothing is wrong. No accelerator provider is
present yet, which is correct at this step.

**The GPU driver is 451.67, from 2020, and it blocks the CUDA path outright.**
See §5b before doing anything about it — it is the shipped OEM driver and
updating it is a precondition, not a tuning step.

`onnxruntime` is 1.30.0 here against 1.24.1 on the gram. That is fine — no
cross-machine absolute comparison is meaningful anyway — but see §0 about
holding it still *within* this machine.

## 3. Assets

Neither the model nor the clips are in git (`models/*.onnx` and `*.mp4` are
ignored). Both are regenerated, not restored.

```powershell
.venv-cpu\Scripts\python.exe -m pip install torch torchvision onnx --index-url https://download.pytorch.org/whl/cpu
.venv-cpu\Scripts\python.exe -m bench.prepare_models --models resnet50 --sizes 224
.venv-cpu\Scripts\python.exe -m bench.get_clips
```

**Use the CPU index URL.** The default `pip install torch` on Windows pulls the
CUDA build, which is several GB, and torch is needed here only to export the
ONNX file once. On a machine with 8 GB of RAM and a laptop SSD that is worth
avoiding. Uninstall it afterwards if space is tight; the 102 MB `.onnx` is the
only thing it leaves behind that matters.

`get_clips` fetches three Big Buck Bunny H.264 clips (360p/720p/1080p, 10 s,
300 frames each, CC-BY Blender Foundation). Keep `--frames` at or below 300 so
the source never loops, unless you are using `--duration-s`, which loops
deliberately and records `restarts`.

## 4. Smoke test

```powershell
.venv-cpu\Scripts\python.exe -m bench.run_sync --model models/resnet50_224.onnx `
    --backend cpu --source clips/bbb_360_2MB.mp4 --frames 40 --warmup 10 `
    --no-idle-gate --tag smoke
```

What a healthy run looks like — this is the same command on the gram:

```
 stage              mean      p50      p95      max    share
 read              0.71     0.59     1.74     3.58      4.6%
 preprocess        0.94     0.92     1.28     1.43      6.1%
 inference        13.50    12.61    19.99    25.26     87.9%
 postprocess       0.19     0.19     0.27     0.29      1.3%
 throughput       64.71 FPS
 bottleneck    inference   overlap ceiling 1.14x
```

Do **not** read the 550XBE's own numbers as better or worse than these. Two
machines are two instruments; the only cross-machine statement this project
makes is about ratios and shapes. Use it as a sanity check on the order of
magnitude: inference in the low tens of milliseconds is the right ballpark for
ResNet50 at 224 on a laptop CPU, and seconds means something is wrong.

`unaccounted` near zero and `order violations 0` mean the harness itself is
sound.

---

## 5. Find an off-host backend

This is the question the project's continuation rests on. W2's result is that
overlap pays only when inference runs somewhere with its own memory and its own
clock. On this machine the MX150 is the only candidate. Try in this order — the
order is by installation friction, not by preference.

### 5a. DirectML (fewest moving parts)

```powershell
python -m venv .venv-dml
.venv-dml\Scripts\python.exe -m pip install numpy opencv-python psutil onnxruntime-directml
.venv-dml\Scripts\python.exe -m bench.env_fingerprint
```

Look for `DmlExecutionProvider` in the provider list, then:

```powershell
.venv-dml\Scripts\python.exe -m bench.run_sync --model models/resnet50_224.onnx `
    --backend dml --source clips/bbb_360_2MB.mp4 --frames 200 --warmup 50 `
    --no-idle-gate --tag dmlcheck
```

`run_sync` aborts rather than saving a CPU number under a `dml` tag, so a run
that completes and writes a file is a run that actually reached the GPU.

DirectML needs no CUDA and no cuDNN — the GPU driver and DX12 are the whole
dependency, and the MX150 has both even on the 2020 driver the machine shipped
with. That is why this is the first thing to try: it is the one accelerator path
that does not depend on the driver update §5b requires. Update the driver
anyway, but a DML row can be measured before you do. `backends.py` turns off the memory-pattern
planner and forces sequential execution for this backend, which the DirectML EP
requires.

**One thing to verify rather than assume:** `session.run_async`, which the W2
pipeline is built on, is exercised here for the first time on a non-OpenVINO,
non-CPU provider. If `run_async` misbehaves under DML, that is a result to
record in the findings, not a blocker to work around — W1 is fully measurable
without it.

### 5b. CUDA (the one worth having)

**Update the GPU driver first. The shipped one cannot run this path at all.**

The 550XBE came with 451.67, released mid-2020. A driver exposes a maximum CUDA
runtime version, and 451.67 tops out at CUDA 11.0 — the CUDA 12.x runtime that
current `onnxruntime-gpu` wheels are built against needs a 527-or-newer driver
on Windows. No combination of pip packages works around that; the ceiling is in
the driver.

So the order is: install the newest NVIDIA driver that still lists the MX150
among its supported products, reboot, confirm with
`nvidia-smi` that both the driver version and the reported CUDA version moved,
and only then build the venv. Pascal support is in its final branch, so whatever
works is also roughly the last thing that will — pin it and record it in the
findings the way [`frozen-baseline.md`](frozen-baseline.md) records the gram's
NPU driver.

```powershell
nvidia-smi        # driver version, and the CUDA version it can host
python -m venv .venv-cuda
.venv-cuda\Scripts\python.exe -m pip install numpy opencv-python psutil onnxruntime-gpu
.venv-cuda\Scripts\python.exe -m bench.env_fingerprint
```

`CUDAExecutionProvider` appears or it does not. If it does not:

- **CUDA 13 dropped Pascal**, so the runtime must be 12.x, and the
  `onnxruntime-gpu` wheel must be built against the same CUDA major you have
  installed. Check ONNX Runtime's current CUDA/cuDNN requirement table for the
  version you install rather than assuming — the pairing is exact, and getting
  it wrong is the error-127 failure mode again.
- `onnxruntime-gpu` does not bundle the CUDA runtime or cuDNN. They come from
  the CUDA Toolkit or from the `nvidia-*-cu12` pip packages, and on Windows the
  DLLs have to be findable at import time.

A 2 GB card is not a constraint here. ResNet50 at 224, batch 1, FP32 is about
100 MB of weights and the pipeline holds a few dozen decoded frames; the MX150's
memory is the least of this backend's problems.

CUDA is worth the extra effort even though DirectML is easier, because two of
the open threads need it: CUDA events are how the inference timer gets split
into submit / device / completion the way `dx_rt` separates `get_latency()` from
`get_npu_inference_time()`, and Nsight is how a claim about where time goes
stops being an inference from stage totals. Neither is available through DML.

**If neither 5a nor 5b comes up**, the device axis is closed on this hardware.
Say so in the README and move to W4, the C++ port — see
[`README.md`](README.md). Do not quietly substitute the UHD 620 for it.

### 5c. The UHD 620, for the Intel row

```powershell
python -m venv .venv-ov
.venv-ov\Scripts\python.exe -m pip install numpy opencv-python psutil
.venv-ov\Scripts\python.exe -m pip install onnxruntime-openvino==1.24.1 "openvino==2025.4.1"
.venv-ov\Scripts\python.exe -c "import openvino; print(openvino.Core().available_devices)"
```

The last line is the cheap test: `GPU` in that list or not. Gen9.5 is old enough
that support in a 2025 OpenVINO release should be checked rather than assumed,
and it needs an Intel graphics driver for Whiskey Lake, which is on Intel's
legacy branch.

The version pair is not a preference. `onnxruntime-openvino` loads
`openvino.dll` from the separately installed `openvino` wheel and the ABI must
match; `openvino==2026.3.1` against this ORT build is the silent-fallback case.

This backend is **not** a substitute for the MX150. Gen9.5 shares memory and
media engine with video decode, and W1 finding 3 already measured that collision
on a far stronger iGPU. Expect it to be the bottleneck rather than the
accelerator — which makes it a useful row, just not the off-host one.

---

## 6. Characterise the instrument before comparing anything

`docs/protocol.md` §"The instrument is a laptop" describes **the gram**, not
laptops in general: ~15% run-to-run noise on absolutes, ratios about 8x more
stable, 60 s of cooldown where 10 s was not enough, an idle floor near 19%. A
15 W part in a thinner chassis throttles on its own schedule. Until this section
is redone, no ratio measured here has a detection floor attached to it, and the
protocol's first rule is that a claim smaller than the noise is not a claim.

**Idle floor**, which sets `--max-busy-pct`:

```powershell
.venv-cpu\Scripts\python.exe -c "from bench import noise; print(noise.busy_percent(5.0))"
```

Run it with your normal desktop open, because that is the condition you will
actually measure in. The gram idled near 19%, so its 15% default could never be
met and the gate recorded `settled: false` instead of pretending. Set the flag
above your measured floor, not below it.

**Noise and cooldown**, one cell, five repeats:

```powershell
.venv-cpu\Scripts\python.exe -m bench.run_sync --model models/resnet50_224.onnx `
    --backend cpu --source clips/bbb_1080_10MB.mp4 --duration-s 10 `
    --frames 100000 --warmup 100 --repeat 5 --cooldown 60 `
    --max-busy-pct <your floor + margin> --intra-threads 4 --tag noise550
```

Prefer `--duration-s` over a frame count wherever runs of different speeds are
compared: a fixed count measures for a different length of time at every speed,
so the fast points get less thermal exposure. On the 360p NPU sweep, moving to
10 s per point cut run-to-run CV from as much as 10.7% to 0.2-2.7%. Set
`--frames` high enough that the deadline is what ends the run.

Then read three things off the five runs: the throughput CV, whether the last
repeat is the slowest (heat accumulating between repeats, so raise `--cooldown`
until it is not), and the clock line — the gram fell 2000 → 1500 MHz within a
run in four runs out of five. Put the result in `docs/protocol.md` beside the
gram's table rather than replacing it.

**Pin `--intra-threads`.** On 4C/8T, 4 is the natural setting. W2 found that
unpinned runs are not just noisier but give a different answer, and that
finding's explanation — the pool landing on different core types of a hybrid
part — makes a falsifiable prediction here: **the 8565U has one kind of core, so
unpinned runs should not be bimodal.** If they are, `findings-w1.md` finding 4
is wrong. That is worth one deliberate unpinned run.

---

## 7. The first real grid

Once you know which backends exist and what the noise is:

```powershell
# whichever accelerator venv came up, plus the CPU row from the same wheel
.venv-dml\Scripts\python.exe -m bench.run_matrix --tag 550xbe `
    --backends cpu dml --repeat 5 --cooldown 60 --between-cells 60 `
    --max-busy-pct <your floor + margin> --intra-threads 4

# the Intel row, from the other venv, same tag
.venv-ov\Scripts\python.exe -m bench.run_matrix --tag 550xbe `
    --backends openvino:CPU openvino:GPU --repeat 5 --cooldown 60 `
    --between-cells 60 --max-busy-pct <your floor + margin>

.venv-cpu\Scripts\python.exe -m bench.summarize --tag 550xbe
```

`--dry-run` prints the plan and a time estimate first; use it, because the full
grid is hours and the estimate is roughly right.

**Tag by machine, not by phase.** `gram__*` is what makes the existing results
legible as belonging to hardware that no longer exists, and `550xbe__*` does the
same job going forward. Cells already on disk are skipped, so the grid can be
filled in sittings and resumed after an interrupt.

Note that `openvino:CPU` row: it is one of the four things
[`README.md`](README.md) asks for before the gram goes back, because it
separates a change of engine from a change of device. If it did not get run
there, running it here at least separates them on *this* machine.

---

## 8. When something looks wrong

| symptom | what it usually is |
|---|---|
| a run aborts with "requested X but running on CPUExecutionProvider" | working as designed. The provider did not load; fix the install rather than passing `--allow-fallback` |
| provider loads, numbers look like CPU | the silent fallback. Check `fell_back_to_cpu` and `actual_providers` in the saved JSON — every result stores them |
| Windows error 127 on import or session create | ABI mismatch between `onnxruntime-*` and its runtime (`openvino`, CUDA, cuDNN). Match the versions, do not upgrade one side |
| `CUDAExecutionProvider` never appears, whatever you install | check the driver before the wheels. `nvidia-smi` reports the highest CUDA version the driver can host, and the 550XBE shipped with 451.67, which caps at 11.0 |
| `settled: false` in the result | the idle gate gave up. The run is marked contaminated on purpose; close things or raise `--max-busy-pct` to your real floor |
| `drift` above 1.05x | not steady state. A throttling laptop is two machines and their average describes neither. Longer cooldown, or shorter runs |
| last repeat is always slowest | heat accumulating between repeats, not within one. Raise `--cooldown` |
| `read` dominates at 1080p | expected, and it is finding 3's territory. It also means an in-flight sweep there will find no knee — a limit that never binds cannot shape anything |
| out of memory | check what else is open before blaming the harness. ResNet50 FP32 is ~100 MB of weights and the pipeline holds a few dozen decoded frames at most |

## 9. What to write down

Every result file already embeds `meta.env`, `meta.backend` and the idle gate,
so the machine, the provider and the contamination state travel with the number.
What does not travel, and belongs in `docs/` as you go:

- the 550XBE's own noise table, beside the gram's in `docs/protocol.md`
- which backends came up and which did not, with the reason — a backend that
  failed to load is a recorded fact about this machine, not a gap
- whether unpinned CPU runs are bimodal here (finding 4's prediction)
- the driver and wheel versions that worked, the way
  [`frozen-baseline.md`](frozen-baseline.md) records the gram's
