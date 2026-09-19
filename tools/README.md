# tools

Instrumentation for W5. Not part of the benchmark harness in [`bench/`](../bench):
nothing here measures anything, so nothing here depends on the machine.

## `dxnn_inspect`

Reads a compiled DEEPX model (`.dxnn`) and reproduces DX-RT's memory budget for
it, with no NPU present.

```
python -m tools.dxnn_inspect model.dxnn
python -m tools.dxnn_inspect model.dxnn --npu-memory 512M
python -m tools.dxnn_inspect model.dxnn --buffers 12 --json
python -m tools.dxnn_inspect --self-test
```

It exists because of the question W5-4 asks of this project's own runtime — how
many inferences may be in flight, and who decides the ceiling. DX-RT answers it
with arithmetic rather than a tuning knob, and every input to that arithmetic
is stored in the model file.

### What it prints

```
container            DXNN v8            target M1A_4K  (4096 MACs)  chip M1A
task npu_0           1852 layers        9225 commands
  input  input.1   UINT8    [1,224,224,3]  PRE_IM2COL  align 64
    logical                        150,528 B
    allocated                      157,696 B
    padding                          7,168 B
  output 1164      FLOAT32  [1,1000]  ALIGNED  align 64
    logical                          4,000 B
    allocated                        4,096 B
    output_all_size             56,143,872 B  charged per buffer

fixed cost        = 17,253,056 B
per-buffer cost   = 56,301,568 B

requested buffers        6  (DXRT_TASK_MAX_LOAD_DEFAULT)
device memory            134,217,728 B
maximum fitting count    2
  runtime would warn: Buffer count reduced from 6 to 2
```

### Three results worth stating separately

**The per-buffer cost is 56,301,568 bytes against a 4,096-byte visible output.**
Almost all of it is `output_all_size`. The name is the file's own; what the
region *contains* — activations only, or activations plus scratch and encoded
intermediates — has not been traced and is not claimed here. What is established
is that it dominates the footprint and that the runtime charges it once per
buffer, so inflight depth is bounded by device memory rather than chosen.

**The alignment unit counts elements, not bytes.** This was found by the
self-test rejecting an earlier assumption. A 1000-wide float32 output aligned as
bytes would occupy `align(4000, 64) = 4032`; the file says 4096, which is
`align(1000, 64) x 4`. The same rule explains the input once the layout says
which axes form the row: `PRE_IM2COL` folds the last two, giving
`224 x align(224x3, 64) = 224 x 704 = 157,696`. Reading the unit as bytes gets
the right order of magnitude and the wrong number — the kind of error that
survives review, which is why it is now a check rather than a sentence.

**The file's own declared total validates the reconstruction.** `rmap_info`
carries a `size` field the cost model never reads. For densenet121 it is
73,554,624, and `fixed + per-buffer x 1` is 73,554,624. An independent number
agreeing to the byte is the strongest evidence available without a device.

### Checks

Every run ends with a pass/fail count and exits non-zero if any check fails.

| check | what it catches |
|---|---|
| sections stay inside the file and do not overlap | a misread container base offset |
| `RMAP` / `WEIGHT` regions equal their compiled section sizes | metadata disagreeing with the payload |
| `INPUT` region equals the sum of the input tensors | a missed tensor |
| element-aligned prediction equals the declared allocation | a wrong padding rule |
| declared total equals the single-buffer footprint | a wrong cost model |
| memory regions sum to the declared total | a missed region |

An unrecognised layout is reported as unpredicted rather than passed silently.

### Where the numbers come from

Read from `dx_rt`, cited rather than copied. No DEEPX source is vendored into
this repository; every `dx_rt` file states that redistribution is prohibited.

| behaviour | source |
|---|---|
| section data begins at byte 8192 | `lib/parsers/v8_model_parser.cpp`, `offset = sizeInfo = 8192` |
| fixed and per-buffer cost | `lib/task_data.cpp:453-454` |
| walking the buffer count down to fit | `InferenceEngine::calculateMaxBufferCount` |
| what feeds that walk | `InferenceEngine::adjustBufferCountForNpuMemory` |
| default buffer count 6, limit 100 | `lib/include/dxrt/common.h:82-84` |
| the count is overridable by environment | `GetTaskMaxLoad`, `lib/configuration.cpp:568` |

`calculateMaxBufferCount` is reproduced as written, including one asymmetry: the
walk-down stops at 1 whether or not 1 fits, and the exception that follows is
raised on a single buffer rather than on the adjusted count. A tidier
reimplementation would no longer predict the runtime.

### Limits

Container version 8 only; v6 and v7 exist and are parsed differently by
`dx_rt`. Device memory is the one number not in the file — the published C++ API
documents an example device reporting `LPDDR4 4200 MHz, 1.98 GiB`, but the real
figure should come from a datasheet or from `--npu-memory` on a machine that has
one. Models whose graph is split across NPU and CPU will have CPU subgraphs that
this tool counts but does not cost, because `adjustBufferCountForNpuMemory`
skips tasks with no register-map info.
