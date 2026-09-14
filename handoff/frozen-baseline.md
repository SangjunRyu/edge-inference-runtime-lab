# Frozen baseline

What produced everything currently in `results/`, and which of it can never be
produced again. Recorded 2026-09-14, before the machine was returned.

## The environment

From `python -m bench.env_fingerprint`, and stored verbatim in
[`gram-env.json`](gram-env.json). Every result file already embeds its own copy
under `meta.env`; this is the one to compare a future machine against.

```json
{
  "platform": "Windows-11-10.0.26200-SP0",
  "python": "3.13.14",
  "cpu_name": "Intel(R) Core(TM) Ultra 7 255H",
  "cpu_count_logical": 16,
  "gpus_nvidia_smi": [],
  "ram_total_gb": 31.5,
  "cpu_freq_mhz_at_start": 2000.0,
  "onnxruntime_version": "1.24.1",
  "onnxruntime_available_providers": [
    "OpenVINOExecutionProvider",
    "CPUExecutionProvider"
  ],
  "opencv_version": "5.0.0"
}
```

The full package list is in [`gram-pip-freeze.txt`](gram-pip-freeze.txt). The
ones that mattered:

```
onnxruntime-openvino==1.24.1
openvino==2025.4.1
openvino-telemetry==2025.2.0
numpy==2.3.5
opencv-python==5.0.0.93
psutil==7.2.2
torch==2.14.0          # export only
torchvision==0.29.0    # export only
onnx==1.22.0           # export only
```

Drivers, recorded when the NPU first worked at all:

| | version |
|---|---|
| Arc graphics | 32.0.101.6554 |
| Intel AI Boost NPU | 32.0.100.4841 (2026-07-24) |

The shipped NPU driver, `32.0.100.3159`, compiled the model and then died at
execution with `ZE_RESULT_ERROR_DEVICE_LOST` in
`zeCommandQueueExecuteCommandLists`. Worth remembering as a class of failure,
not as a fact about this machine: on an accelerator, "the graph compiled" and
"the graph runs" are separate claims.

The `onnxruntime-openvino` / `openvino` pairing is an ABI match, not a version
preference. `openvino==2026.3.1` loads, the provider then fails with Windows
error 127 (missing procedure), and ONNX Runtime falls back to CPU **silently** —
which would publish a CPU number under an NPU label. This is the reason
`fell_back_to_cpu` is stored with every result, and the reason every runner
aborts on an unrequested fallback unless `--allow-fallback` is passed.

## What cannot be regenerated

| result files | device | status |
|---|---|---|
| `gram__openvino-npu__*` (3) | Intel AI Boost NPU | **gone with the machine** |
| `gram-npu-1080_*`, `noisetest_*`, `npucheck_*` | NPU | gone |
| `ab-npu__ab4__openvino-npu__*` | NPU | gone |
| `sweep-npu__*`, `sweep-npu360__*` | NPU | gone |
| `gram__openvino-gpu__*` (3) | Arc 140T (Xe2) | gone; the UHD 620 is not a substitute |
| `gram__cpu__*`, `ab-pinned__*`, `w1*`, `ceilcheck_*`, `videocheck_*`, `smoke_*` | CPU | reproducible in kind, not in value — different CPU |

Nothing here should be deleted or re-labelled later. A result whose hardware no
longer exists is still a measurement; it stops being a *comparison* the moment
it is placed next to a number from a different machine, which is what the `gram`
tag in the filenames is for. **Keep tagging by machine.** The next tag should
name the 550XBE, not the phase.

## Known defects in the stored results

**`sweep-npu360__*.json` carries `"knee": 16`, which is wrong.** It was written
by an earlier `run_sweep.py` whose knee rule compared the gain against a raw CV.
The current rule compares two means against their combined standard error at
2 sigma, and under it the same stored points give **knee = 4**:

```
  1 -> 2   throughput  +51.2%  (needs > 14.2%)   gain
  2 -> 4   throughput  +14.2%  (needs > 12.4%)   gain
  4 -> 8   throughput   -0.2%  (needs >  2.0%)   NOISE
  8 -> 16  throughput   +3.2%  (needs >  3.7%)   NOISE
```

`sweep-npu__*.json` stores `"knee": 2`, which both rules agree on.

**Neither sweep used `--duration-s`.** Both ran a fixed 300 frames per point, so
the fast depths got less wall-clock time and less thermal exposure than the slow
ones — a bias pointing at the exact comparison the sweep exists to make. The
deadline option was added afterwards.

**Neither sweep records `source` or `seconds_measured` per point.** Those fields
postdate the runs. The source is recoverable from the filename and from
`meta.args.source`.

The stored `points` are unaffected by any of this; only the derived `knee` on
the 360p file and the cross-depth fairness are. Re-running both under the
current script is item 1 of the pre-return list in [`README.md`](README.md).

## The instrument's own numbers

These belong to the gram and must not be carried over to the new machine as
assumptions. From `docs/protocol.md`:

| | gram 15Z90T |
|---|---|
| throughput CV, 5 runs, 60 s cooldown | 9.7% |
| E2E p50 CV | 10.1% |
| overlap-ceiling CV | 2.2% |
| CV at 10 s cooldown | 14.9% |
| CPU clock within one run | 2000 → 1500 MHz, 4 runs of 5 |
| idle floor (system CPU busy) | ~19% |

The working rule that came out of them — **absolutes carry ~15% noise, ratios are
roughly 8x more stable, so every headline is a ratio** — is a conclusion about
measuring on a thermally limited laptop and is expected to hold. The numbers
themselves are not. A 15 W part will produce its own table.
