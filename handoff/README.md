# Handoff — the instrument changes

Written 2026-09-14. Every number in `results/` was measured on a loaner laptop
that goes back the week of 2026-09-21. Work continues on a different machine
with a different set of accelerators.

This directory exists so the first session on the new machine starts from a plan
instead of from a re-discovery. **Nothing in `docs/` is edited to suit the new
hardware.** Those are findings about an instrument that existed, and they stay
as measured; what is no longer reproducible is listed in
[`frozen-baseline.md`](frozen-baseline.md) rather than deleted.

## The instrument

| | leaving — LG gram 15Z90T | arriving — Samsung NT550XBE |
|---|---|---|
| CPU | Core Ultra 7 255H, 16 logical, hybrid P/E/LP-E | Core i7-8565U, 4C/8T, uniform, 15 W |
| RAM | 31.5 GB | 8 GB |
| iGPU | Arc 140T (Xe2), XMX | UHD 620 (Gen9.5) |
| NPU | Intel AI Boost | **none** |
| dGPU | none | **GeForce MX150** — GP108, sm_61, 2 GB GDDR5 |

Two of those lines decide the project.

**The NPU is gone.** It was the only backend on which the W2 result could be
demonstrated at all: the CPU rows *disproved* the naive expectation, and the Arc
iGPU was never run through `run_ab`. Every number supporting "overlap pays only
when inference is off the host" comes from a device that will not exist here.

**A discrete accelerator arrives.** The MX150 is slow — roughly 1.1 TFLOPS FP32,
no tensor cores, and FP16 at 1/64 rate so there is no point reaching for it. But
speed is not what W2 depends on. It depends on inference running somewhere with
its own memory and its own clock, reached across a bus. The MX150 has that
property and the Arc iGPU did not.

So the machine is weaker and the project is not obviously worse off. W1 already
showed the interesting regime is the one where the accelerator is fast enough
that the *host* becomes the ceiling; on this machine that regime is harder to
reach, which makes the opposite regime — inference-bound, overlap worth nothing
— measurable in its own right rather than only as the CPU control.

## What the project needs from a machine

One off-host backend. In priority order, to be settled in the first session:

1. **`cuda`** — `onnxruntime-gpu` against a CUDA 12.x runtime. CUDA 13 dropped
   Pascal, so 12.x is the ceiling and the wheel must match it. Verify with
   `python -m bench.env_fingerprint`; `backends.py` already treats a missing
   `CUDAExecutionProvider` as a normal outcome to be recorded, not a failure.
2. **DirectML EP** — `onnxruntime-directml`, DX12, which the MX150 supports. It
   works where the CUDA EP does not, at the cost of changing the engine as well
   as the device — the same confound W1 already records as unresolved between
   the ORT CPU row and the OpenVINO rows. Windows only.
3. **`openvino --ov-device GPU`** on the UHD 620. Off-host in name only: Gen9.5
   shares memory and media engine with decode, and W1 finding 3 measured that
   collision on a far stronger iGPU. Expect it to be the bottleneck rather than
   the accelerator. Confirm OpenVINO 2025.4 still supports Gen9.5 before
   planning around it.

If none of the three comes up, the device axis is closed on this hardware. The
honest response is to say so in the README and move to W4 — the C++ port, whose
question is ownership and explicit concurrency, not device throughput.

## Phase by phase

**W1 — must be re-run from scratch.** The grid becomes cpu x {cuda | dml} x
openvino/GPU. One prediction is already on the table: **finding 4 should
disappear.** The bimodal CPU cells were attributed to an unpinned thread pool
landing on different core types of a hybrid part; the 8565U has one kind of
core, so if the bimodality survives on it, the attribution in `findings-w1.md`
is wrong and needs revisiting. That is worth running early — it tests a
published claim rather than adding a cell.

**W2 — the thesis is portable, the evidence is not.** `run_ab` and its pairing
discipline carry over unchanged. What has to be re-established is a device on
which async wins at all. Re-run the CPU control first: on 4C/8T the "concurrency
is subdivided, not added" mechanism predicts a *larger* per-request penalty at
`--intra-threads 4` than the ~4.5x measured on 16 logical cores, because there
is less parallelism to subdivide.

**W2 sweep — `run_sweep` carries over; the knees do not.** Both stored curves are
NPU. The shape of the finding is the transferable part: the knee sits where the
slowest stage is, so at 1080p it was decode and depth bought 1.19x, while at
360p it was inference and depth bought 1.73x. On the MX150, inference is slow,
so the prediction is a 1080p curve that saturates at depth 1-2 and a 360p curve
that keeps climbing further than the NPU's did.

**W3 — unblocked and cheap.** Correctness under overlap needs no fast hardware.
`--stall-every` forces out-of-order completion deliberately, so the reorder
buffer can be exercised on CPU alone, at any speed, on any machine.

**W4 — hardware-independent.** The C++ port is about ownership, lifetime and
explicit concurrency. An 8 GB machine is a perfectly good place to write it, and
it is the phase worth the most against the roles this repository is aimed at.

## Before the gram goes back

Ordered by what cannot be recovered afterwards.

1. **Re-run both sweeps under the current `run_sweep.py`**, with `--duration-s`
   set. The committed curves came from an earlier version: the 360p file carries
   a stale `knee: 16` from the superseded CV rule, and neither used the
   wall-clock deadline. This is the only NPU data that will ever exist.
2. **Run the missing `openvino --ov-device CPU` row.** `findings-w1.md` lists it
   under "Not established": the `cpu` row is ONNX Runtime's own kernels while the
   accelerator rows are the OpenVINO provider, so every CPU-vs-NPU comparison in
   this repository mixes a change of engine with a change of device. One row
   separates them, and after next week the other half of the comparison is gone.
3. **Run `run_ab` on the Arc iGPU.** W2 covers CPU and NPU only. The iGPU is the
   one backend whose async behaviour W1 says is not predictable from either,
   because it contends with decode.
4. **Write up the sweep as `docs/findings-w2-sweep.md`.** The commit message
   carries the result; the docs set does not yet.

Items 2 and 3 are one grid cell each. Item 1 is the one that matters.

## First session on the new machine

```powershell
git clone https://github.com/SangjunRyu/DEEPX-poc.git edge-inference-runtime-lab
cd edge-inference-runtime-lab
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m bench.env_fingerprint        # settle the backend question first
```

`models/*.onnx` and `clips/*.mp4` are gitignored and do not come with the clone.
Regenerate both:

```powershell
.venv\Scripts\python.exe -m pip install torch torchvision onnx
.venv\Scripts\python.exe -m bench.prepare_models --models resnet50 --sizes 224
.venv\Scripts\python.exe -m bench.get_clips
```

Torch is needed for the export and for nothing else. On a machine with 8 GB of
RAM and a laptop SSD it is worth uninstalling afterwards — the ONNX file is
102 MB and the wheels that produced it are an order of magnitude larger.

Then, before any comparison is run:

**Re-measure the instrument.** `docs/protocol.md` §"The instrument is a laptop"
is not a general statement about laptops, it is a characterisation of one
specific machine: ~15% run-to-run noise on absolutes, ratios roughly 8x more
stable, 60 s of cooldown needed where 10 s was not enough, an idle floor near
19%. A 15 W part in a thinner chassis will throttle differently and idle
differently. Reproduce that table with `--repeat 5` on one cell, set
`--max-busy-pct` from the measured idle floor rather than from the default, and
find the cooldown at which the last repeat stops being the slowest. Until that
is done, no ratio measured on this machine has a detection floor attached to it,
and the protocol's first rule is that a claim smaller than the noise is not a
claim.

## Also carried over

- [`frozen-baseline.md`](frozen-baseline.md) — the exact environment the existing
  results were produced in, and which of them can never be regenerated.
- [`open-threads.md`](open-threads.md) — every open question in the repository,
  each marked portable or dead.
