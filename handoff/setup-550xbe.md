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
uncontrolled difference to the same comparison for no gain.

On this machine the version is not a free choice: the newest `onnxruntime-gpu`
is a CUDA 13 build and CUDA 13 does not support Pascal, so the CUDA venv has to
go back to a CUDA 12 build and everything else should follow it. §5b works out
which version that is. Install nothing unpinned until that is settled.

---

## 1. Clone

```powershell
git clone https://github.com/SangjunRyu/edge-inference-runtime-lab.git
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

And a warning about this field in general: **a provider in that list is not a
provider that works.** `get_available_providers()` reports what was compiled
into the wheel, not what can create a session — the provider DLL and its CUDA or
cuDNN dependencies are loaded later, and that is where the failure lands. On
this machine `CUDAExecutionProvider` appeared here and still could not run a
single frame (§5b). Treat the fingerprint as a statement about the wheel, and
`run_sync` completing as the statement about the machine.

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

Two things must both be true before this works, and the default install gets
both wrong on this machine. **Do not `pip install onnxruntime-gpu` unpinned.**

#### The constraint

The MX150 is Pascal, and **CUDA 13 dropped Pascal**. Current `onnxruntime-gpu`
wheels on PyPI are CUDA 13 builds, so the newest wheel is the one version of the
package that categorically cannot run on this GPU. Measured here, 2026-09-15,
with `onnxruntime-gpu` 1.30.0 installed and the CUDA EP visible in
`env_fingerprint`:

```
Error loading "...\onnxruntime\capi\onnxruntime_providers_cuda.dll"
which depends on "cublasLt64_13.dll" which is missing. (Error 126)
Failed to create CUDAExecutionProvider. Require cuDNN 9.* and CUDA 13.*
!! requested 'cuda' but the session is running on ['CPUExecutionProvider']
   Aborting so a CPU measurement is not saved under the 'cudacheck' tag.
```

**The number in `cublasLt64_NN.dll` is the CUDA major the wheel needs.** `_13`
means a CUDA 13 build and no Pascal support; `_12` is the one to get. That rule
outlives any version table, so use it rather than trusting a release note.

#### The target

| | |
|---|---|
| GPU | MX150, Pascal, sm_61 |
| CUDA | **12.x** — the last major that supports Pascal |
| cuDNN | 9.x |
| `onnxruntime-gpu` | the newest release that still links `cublasLt64_12` |
| driver | newest that still lists Pascal among supported products |

As of 2026-09 the ONNX Runtime side of that is expected to be **1.26.x**, with a
cp313 Windows wheel, so Python 3.13.14 does not have to move. Confirm before
committing to a multi-gigabyte download rather than taking the number on faith —
install the candidate and read the error, which names the CUDA major it wants.

Whatever version it lands on, **pin the other venvs to it too.** CUDA on Pascal
is the most constrained of the three flavours, so it is the one that sets the
version for the whole grid; §0 asks for the engine version to be held still and
this is what holds it. Among the CUDA 12 candidates, note that the gram ran
1.24.1 — matching it removes one more difference when comparing shapes across
the two machines.

#### The order

The driver comes first, because the rest cannot be tested without it. The 550XBE
shipped with **451.67**, from mid-2020, which tops out at CUDA 11.0 — below the
target, so nothing else can be evaluated until it moves. Install the newest
NVIDIA driver that still supports Pascal, reboot, and confirm with `nvidia-smi`
that both the driver version and the CUDA version it reports have changed. That
branch is the last one Pascal will get, so whatever works is roughly the final
answer: record it the way [`frozen-baseline.md`](frozen-baseline.md) records the
gram's NPU driver.

```powershell
nvidia-smi        # driver version, and the highest CUDA it can host

python -m venv .venv-cuda
.venv-cuda\Scripts\python.exe -m pip install numpy opencv-python psutil
.venv-cuda\Scripts\python.exe -m pip install "onnxruntime-gpu[cuda,cudnn]==1.26.0"
.venv-cuda\Scripts\python.exe -c "import onnxruntime as ort; ort.preload_dlls(); ort.print_debug_info()"
```

The `[cuda,cudnn]` extra pulls the CUDA and cuDNN runtimes as pip packages, so
the CUDA Toolkit does not have to be installed first. They land in
`site-packages/nvidia/`, which is not on `PATH`; `ort.preload_dlls()` is what
makes the loader find them, and `backends.py` calls it before creating a CUDA
session for exactly this reason. `print_debug_info()` prints the CUDA and cuDNN
versions ONNX Runtime actually resolved — check them against the table above
before running anything.

If the extras route does not resolve, fall back to a system CUDA 12.x Toolkit
plus cuDNN 9 on `PATH`. The `preload_dlls()` call is harmless either way.

Then the real test, which is a session creation and not a provider listing:

```powershell
.venv-cuda\Scripts\python.exe -m bench.run_sync --model models/resnet50_224.onnx `
    --backend cuda --source clips/bbb_360_2MB.mp4 --frames 200 --warmup 50 `
    --no-idle-gate --tag cudacheck
```

#### The second failure, and the worse one

With the driver updated and the CUDA 12 wheel in place, the session created on
CUDA and then this happened on the first convolution:

```
Could not locate cudnn_engines_tensor_ir64_9.dll.
CUDNN_BACKEND_TENSOR_DESCRIPTOR cudnnFinalize failed ... CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED
Non-zero status code returned while running Conv node. Name:'/conv1/Conv'
EP Error: ... using ['CUDAExecutionProvider', 'CPUExecutionProvider']
Falling back to ['CPUExecutionProvider'] and retrying.
```

and then the benchmark **completed and saved**, headed
`cuda -> CUDAExecutionProvider`, reporting 15.99 FPS — which is the CPU number.

Two separate defects, and the second matters more than the first.

**cuDNN loads sublibraries lazily.** `preload_dlls()` LoadLibrary's a fixed
list — in this ONNX Runtime it is seven cuDNN DLLs, and
`cudnn_engines_tensor_ir64_9.dll` is not among them. cuDNN reaches for it at the
first Conv, long after session creation, and by then only the Windows DLL search
path is consulted, where `site-packages/nvidia/*/bin` does not appear.
`backends.py` now calls `os.add_dll_directory()` on each of those directories
and keeps the handles alive for the process, so the directory stays searchable
rather than being preloaded once. Check the file is actually there before
reinstalling anything:

```powershell
Get-ChildItem .venv-cuda\Lib\site-packages\nvidia\cudnn\bin\cudnn_engines_tensor_ir64_9.dll
```

If it exists, this is a search-path problem and not a CUDA version problem. Do
not change the CUDA version to fix it.

**ONNX Runtime's Python wrapper swaps the session to CPU and retries, inside
`run()`.** Not at session creation — the retry lives in `InferenceSession.run()`
and `run_async()`, which catch the EP failure, call
`set_providers(["CPUExecutionProvider"])`, disable further fallback and re-issue
the call. The harness had already checked `get_providers()` at construction and
seen CUDA, so nothing downstream ever noticed. That is how a dead accelerator
produced a complete, plausible, correctly-ordered benchmark under the wrong
name.

`backends.py` now calls `disable_fallback()` for every non-CPU backend, and
re-reads the providers once after the first inference. Note that this is a
different thing from listing `CPUExecutionProvider` behind the accelerator:
that is graph partitioning, where an op with no accelerator kernel runs on the
host, and it stays.

**Delete any result produced before this fix.** It is a CPU measurement wearing
an accelerator's label, which is worse than no measurement:

```powershell
Remove-Item .\results\cudacheck__cuda__resnet50_224__bbb_360_2MB.json
```

After the fix there are exactly two acceptable outcomes from the `cudacheck`
command, and a third that must not occur:

| | |
|---|---|
| ✅ | real GPU numbers, `saved -> results\cudacheck__cuda__...` |
| ✅ | a CUDA or cuDNN exception, the run aborts, nothing is saved |
| ❌ | CPU numbers saved under the `cuda` tag |

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
| Windows error 126 loading `onnxruntime_providers_cuda.dll` | the CUDA runtime it links is absent. Read the DLL it names: `cublasLt64_13` means a CUDA 13 wheel, which Pascal cannot use at all — downgrade the wheel rather than install CUDA 13. `cublasLt64_12` missing means the right wheel with no runtime: `preload_dlls()`, the `[cuda,cudnn]` extra, or CUDA on `PATH` |
| `CUDAExecutionProvider` never appears, whatever you install | check the driver before the wheels. `nvidia-smi` reports the highest CUDA version the driver can host, and the 550XBE shipped with 451.67, which caps at 11.0 |
| a provider is listed but no run ever succeeds | listing is about the wheel, running is about the machine. See §2 |
| `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED` on the first Conv | cuDNN loading a sublibrary lazily from a directory that is not on the DLL search path. `backends.py` handles it with `os.add_dll_directory`; if it persists, check the DLL is present before changing any version |
| "Falling back to ['CPUExecutionProvider'] and retrying." | the accelerator died mid-run and ONNX Runtime silently continued on CPU. Should now be impossible — `disable_fallback()` is set for every non-CPU backend. If you see it, the result is void |
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
