# W5 — what must a pipeline own, and what must it refuse?

W1 asked where a frame spends its time, W2 what changes when stages overlap,
W3 whether the answers stay correct, W4 what happens when the pipeline is
written in C++ with explicit concurrency. Each of those took the pipeline as
given and measured it.

W5 is the first phase that changes the shape of the thing rather than measuring
it, and the question is narrower than "build a runtime":

> A pipeline that cannot refuse work has no upper bound on its memory or its
> latency. **What does it have to own to be able to refuse, and what does
> refusing cost?**

The artefact is an ownership-aware bounded runtime. The result is a table of
what bounding buys and what it costs, which is a different thing and is the
part worth defending.

## What changed in this revision

W5 now has a second half. Every phase is paired with the same question answered
in **DX-RT**, DEEPX's production NPU runtime, whose source is readable at
`dx-all-suite/dx-runtime/dx_rt`.

```
W5-0  tooling            dxnn_inspector
W5-1  ownership          my slot pool        <->  DX-RT FixedSizeBuffer / bufferCount
W5-2  scheduling         my queues/workers   <->  DX-RT DevicePool / task layer
W5-3  memory & format    my host buffers     <->  DXNN metadata + NPU Format Handler
W5-4  backpressure       my inflight limit   <->  DX-RT bufferCount as a memory budget
W5-5  cascade            optional
```

**The pairing has one rule, and the whole revision depends on it.** Reading
DX-RT is never a deliverable by itself. Each reading must terminate in one of
two things: a number my program computes, or an assertion my program checks. A
paragraph describing what DX-RT does is not a result. This repository spent
four phases learning that a claim without a checkable output is a claim that
survives being wrong.

The reason to pair at all is that the comparison only exists if my
implementation exists. Replacing W5 with a study of DX-RT would delete the
thing being compared.

## Why this phase and not another round of tuning

Three reasons, and the second is the strongest.

**It generalises what W4 already built.** The slot pool, the rule that a
completion callback must not block, the reorder buffer that exists because
postprocessing runs concurrently — those are not OpenVINO details. They are the
skeleton of any bounded pipeline, currently welded to one. W4's three runtime
defects were all ownership and lifetime defects, and
[`repro/`](../repro/README.md) now holds each as a runnable program.

**Its headline results are exact.** W4 spent most of its effort fighting a
machine that would not hold still, and lost: **five consecutive paired runs** in
the closing session failed their own control stage, the last at a coefficient of
variation of 74% with not one admissible pair in six. A phase whose main
deliverables are *counts* rather than *rates* does not depend on that machine at
all.

| deliverable | kind | needs a steady machine? |
|---|---|---|
| allocations per frame in steady state | counted | no |
| host copies per frame, bytes moved | counted / derived | no |
| leases outstanding at shutdown | asserted | no |
| `accepted == completed == emitted`, no duplicates | asserted | no |
| peak resident frames under a bounded policy | counted | no |
| frames dropped under each policy | counted | no |
| memory budget and buffer-count ceiling from a `.dxnn` | computed | no |
| encode/decode round trip is the identity | asserted | no |
| throughput, p50/p95 latency | measured | **yes** |

Seven of nine rows are facts about a program. **The counted rows must be
labelled as counted**, everywhere, or this advantage turns into the same
overclaiming the repository has spent four phases removing.

**It needs no accelerator, and now not even an NPU.** The DX-RT half was chosen
against the same constraint: every anchor below is either a pure function of
numbers stored in a `.dxnn`, or a host-side function from bytes to bytes. A
DX-M1 would add measurement, not understanding.

## W5-0 — tooling: `tools/dxnn_inspector`

A `.dxnn` is self-describing. The container is `DXNN`, a little-endian version
word, an 8192-byte JSON header, then binary sections at byte offsets. Parsers
for three container versions are readable at
`dx_rt/lib/parsers/v6|v7|v8_model_parser.cpp`.

The inspector is a program, not a note. Given a `.dxnn` it prints the layout and
the memory budget; given a device memory size it reproduces DX-RT's own
admission arithmetic.

```
$ dxnn-inspect models/densenet121_224x224.dxnn --npu-memory 512M

container            DXNN v8            target M1A_4K (mac 4096)
graph                1852 layers        9225 commands

input   input.1      UINT8   [1,224,224,3]  PRE_IM2COL  align 64
        logical                150,528 B
        encoded region         157,696 B     (+7,168 row padding)
output  1164         FLOAT32 [1,1000]       ALIGNED     align 64
        logical                  4,000 B
        allocated                4,096 B
        output_all_size     56,143,872 B

fixed cost        rmap 8,856,000 + weight 8,397,056 + ppu 0  =  17,253,056 B
per-buffer cost   align64(157,696) + 56,143,872              =  56,301,568 B

requested buffers        6
maximum fitting count    5
```

**Success:** for a given `.dxnn` the inspector's fixed and per-buffer costs
equal the expression at `dx_rt/lib/task_data.cpp:454`, and `--npu-memory`
reproduces the clamp at `InferenceEngine::calculateMaxBufferCount`. Both are
checkable by reading; neither needs a device.

Two entries in that output exist to keep a distinction the rest of the plan
depends on, and they are spelled out under *Claim discipline* below.

The inspector is infrastructure for W5-1, W5-3 and W5-4, which is why it comes
first and why it lives in `tools/` rather than being a phase of its own.

**Done.** `python -m tools.dxnn_inspect` is in the repository with 10 checks and
a self-test; see [`tools/README.md`](../tools/README.md). Two things came out of
building it that were not in this plan when it was written.

*The success criterion got a better test than the one proposed here.* Comparing
against `task_data.cpp:454` only proves the formula was copied correctly.
`rmap_info` also carries a `size` field the cost model never reads, and for
densenet121 it is 73,554,624 — exactly `fixed + per-buffer x 1`. An independent
number in the file agreeing to the byte is much stronger than agreeing with the
source the formula came from, so that is now the check that matters.

*The self-test rejected an assumption this document had already stated as fact.*
The claim that the 4,096-byte output region is the 4,000-byte tensor aligned to
64 is false — that arithmetic gives 4,032. The unit counts elements, and the
correction is recorded under *Claim discipline*. The lesson is the one W4 kept
finding: the assertion was cheap, and it was the only thing between a plausible
sentence and a wrong one.

## The correction that matters most: a file source cannot show backpressure

The backpressure experiment is the best idea in this plan and it **does not
work as described**. Written down before it is built, because it would
otherwise be discovered after a day of work.

With a video file read as fast as the pipeline can consume it, `Block` and
`DropOldest` are indistinguishable. Under `Block` the reader simply waits and
the pipeline self-paces; there is no external clock, so nothing ever becomes
late and a drop policy has nothing to drop. The proposal's own example table —
31 FPS, 94 resident frames, 0 dropped under an unbounded queue — is not
reachable from a file either: an unbounded queue in front of a source with no
clock grows until the process dies. It does not settle at 94.

**Freshness only exists when the source has its own clock.** So W5-4 has a
prerequisite:

```
--source-fps N      the reader emits frames on a wall-clock schedule and
                    does not wait for the pipeline
```

A source paced at 60 FPS in front of a runtime that sustains 30 is what makes
the three policies differ, and it is what a camera actually is. Everything
interesting follows from the source refusing to slow down.

## Three smaller corrections

**A lease cannot be moved out from under a running request.** OpenVINO holds a
pointer to the input tensor's memory for the whole inference, and
`repro/cpp/01` proves the same contract for the callback descriptor: the runtime
reads it at *completion* time, not at set time. So the ownership rule is not
"the lease travels". It is:

> The input lease is **pinned to the slot** from submit until the request
> completes. Only then does it move on.

Getting this wrong is the silent kind of wrong — the frame count stays right
and the values are another frame's. `repro/cpp/05` is the same failure mode one
stage later.

**"Zero allocations" has a boundary and the boundary has to be printed.** A
global `operator new` counter sees allocations made by our code. It does not
see OpenCV's allocator, OpenVINO's internals, or the driver. The honest claim is
"zero allocations *in the runtime's own code* in steady state", with the things
outside the fence named — the same treatment `unaccounted` gets in the stage
table.

**Bytes moved is derived, not measured, and should be cross-checked.** An
NCHW float32 tensor at 224x224x3 is 602,112 bytes. Three eliminated
intermediates is ~1.8 MB per frame of host traffic, which at 200 FPS is
~360 MB/s — large enough that it must show up in the preprocess stage time.
W4 already measured that path (`--fused`, preprocess B/A around 0.6), so the
arithmetic and the timing are two independent views of one change and should be
reported together. If they disagree, that is the finding.

The caution about the phrase "zero-copy" is correct and should be kept. What can
be claimed is *which boundary* the copies were removed up to; what happens
between the host buffer and the device is the driver's business and is not
observable from here.

## Order of work

Each phase has a success criterion that can be checked, not admired, and a
DX-RT anchor that must produce a number or an assertion.

### W5-1 — ownership

`FrameContext`, move-only `BufferLease`, and pools sized at startup.

**Success:** a run of 1000 frames reports zero allocations from the runtime's
own code after warm-up, and every lease is back in its pool at shutdown. Both
are assertions in the program, not numbers in a report.

The move-only choice earns its keep here rather than aesthetically: with
`shared_ptr` a leaked buffer is invisible, because the count merely stays above
zero and nothing complains. With move-only leases and a pool that asserts a full
return, **shutdown becomes a checkable invariant**.

**DX-RT anchor.** `FixedSizeBuffer(size, buffer_count, allocType, direction)` at
`lib/include/dxrt/fixed_size_buffer.h:37-43` is the same decision made in a
shipping runtime: buffers are a fixed set allocated up front with a direction,
not an allocation per request. The anchor must terminate in my pool carrying the
same two properties as assertions — fixed count, and a direction that is not
interchangeable.

**Done.** `cpp/src/w5_ownership.cpp`, reported in
[`docs/findings-w5.md`](findings-w5.md). 20,000 frames with zero steady-state
allocations and every lease returned, across fifteen configurations of buffer
count and worker count. The direction property came out stronger than DX-RT's:
`Direction` is a template parameter here, so the mismatch is a compile error
rather than a run-time condition, and `cpp/src/negative_direction.cpp` exists to
fail to build.

Two things the plan did not anticipate. **The allocation injection was
optimised away** on its first attempt — a compiler may delete a `new`/`delete`
pair it can prove unobservable, so the check reported "not as predicted" until
the pointer was made to escape. An injection that cannot fail tests nothing, and
only the prediction machinery caught it. **Early release broke 1.3–2.3% of
frames and nothing else** — right count, right order, no leak, no allocation,
another frame's data. Every count-based check passed. That is the argument for
the value check, in one number.

### W5-2 — the scheduler

`read -> preprocess -> infer -> postprocess -> output` as generic stages, with
worker counts and queue capacities on the command line: `--pre-workers`,
`--post-workers`, `--inflight`, `--queue-capacity`.

**Success:** the existing runners are reproducible as configurations of it, and
`bench/compare_variants.py` pairs old against new with the control holding. If
the generic version cannot reproduce W4's numbers, the abstraction cost
something and the amount is the result.

Do not build work stealing. Bounded queues, a worker pool and explicit state per
frame are enough, and the interesting problems are all in the ownership rather
than in the dispatch.

**DX-RT anchor.** `lib/device_pool/` splits device selection (`device_pool.cpp`,
`device_dispatcher.cpp`) from submission (`std_device_task_layer.cpp`,
`acc_device_task_layer.cpp`), with `nfh_layer.cpp` between them. The question to
answer with a number is **where the inflight ceiling comes from** — a queue
capacity, a worker count, or a buffer count — because W5-4 asserts the answer.

**Done, in part.** `cpp/include/scheduler.hpp` and `cpp/src/w5_scheduler.cpp`.
The anchor question is answered and it is arithmetic, not a measurement:
`ceiling = min(buffers, 1 + 3*queue + pre + infer + post)`, checked across
fourteen configurations where thirteen reached it and none exceeded it. Neither
knob is the answer alone — more buffers buy nothing once the structure runs out
of places, and deeper queues buy nothing the pool will not fund. `RingReorder`
replaces the map-based one, sized from the ceiling rather than guessed.

The first run failed its own ceiling check at 9 against 8 buffers, and the
design was not the problem: the counter was bracketed around the frame instead
of the lease. Kept as `repro/cpp/06_inflight_overcount`.

**Still open:** the paired comparison against W4's OpenVINO runners. The
structural half is done and portable; reproducing W4's numbers through the
generic scheduler needs the accelerator and a machine that holds still, and W4
established this one does not. Not claimed until measured.

### W5-3 — memory movement and format

Instrument four preprocessing paths — baseline, fused, preallocated, tensor view
over the same buffer — for allocations, copies and bytes moved.

**Success:** a table whose rows are counted, with the boundary stated, and whose
bytes-per-frame column is consistent with the preprocess timings W4 already has.

**DX-RT anchor, and this one is executable.** `lib/npu_format_handler.cpp` is
~50 KB of host-side code converting between a host tensor and the NPU's aligned
layout. It needs no NPU: it is bytes in, bytes out. Three facts are already
verified by reading:

- the fallback transpose at `:433` copies one element at a time with `memcpy`
  into a strided destination — no blocking, no SIMD;
- the accelerated paths at `:456-460` exist only under `USE_IPP` (Intel IPP) or
  `USE_NEON` (ARM), so a plain x86 build compiles neither;
- `USE_NPU_FORMAT_CONVERSION_ACCELERATION` is **OFF** by default
  (`cmake/dxrt.cfg.cmake:19`) and, when on, is gated again at run time.

So W5-3 gains a second experiment that is entirely portable: drive
encode/decode with the shapes and `align_unit` the inspector reads from a real
`.dxnn`, assert `decode(encode(x)) == x`, then measure. Correctness needs no
hardware because the round trip is its own oracle, and any improvement is
defended by the same paired discipline as everything else.

Licence: analyse and re-implement. Do not vendor DEEPX sources into this
repository. See *Claim discipline*.

**Done, and the measured half came out the wrong way round.**
`cpp/include/nfh.hpp` re-implements the padding and transpose rules;
`tools/dxnn_inspect.py --nfh-args` supplies the geometry so there is one parser.
29 correctness checks pass with no hardware, because the round trip is its own
oracle. The counted table is in [`findings-w5.md`](findings-w5.md): fusing
removes 315,392 bytes of host traffic per frame.

The expectation this phase was built on — that DX-RT's scalar fallback leaves
performance on the table — **is false at the size that matters.** At the
densenet121 input geometry the tensor is 157 KB, fits in cache, and the
textbook blocked transpose is **1.5x slower**. Blocking wins by 1.6x to 4x from
roughly 600 KB upward. So the claim worth making is not "the fallback is slow"
but "the fallback is slow above about a megabyte, and a classifier's input is
not". Four larger cells failed their control and are recorded as unmeasurable
rather than dropped.

### W5-4 — backpressure as a policy

`Block`, `DropOldest`, `DropNewest`, against a source paced with `--source-fps`
and an overload induced by slowing the sink.

| policy | FPS | p95 E2E | dropped | peak resident |
|---|---|---|---|---|
| unbounded | | | | |
| bounded, block | | | | |
| bounded, drop-oldest | | | | |

**Success:** the table filled in, with throughput measured under the
repository's paired discipline and the rest counted. The conclusion it is set up
to reach — that all three sustain the same throughput, so a larger queue buys
nothing but stale frames and resident memory — is a genuine systems result.

**DX-RT anchor, and this is the sharpest one.** In DX-RT the inflight limit is
not a tuning knob. It is a memory budget. At `lib/task_data.cpp:453-454`:

```
_outputMemSize = max(0, _npuModel.output_all_size);
_memUsage      = rmapSize + weightSize
               + align64(_encodedInputSize) * _bufferCount
               + _outputMemSize            * _bufferCount;
```

and `InferenceEngine::adjustBufferCountForNpuMemory` feeds exactly those two
aggregates into `calculateMaxBufferCount`, which walks the requested count down
until it fits the smallest bound device's memory, warns that it did, and throws
`InsufficientMemoryException` if even one buffer does not fit.

Every input to that arithmetic is in the `.dxnn`. For
`densenet121_224x224.dxnn`, compiled locally with DX-Compiler 2.4.0-rc.4:

| term | value |
|---|---|
| fixed cost (rmap + weight + ppu) | 17,253,056 B |
| per-buffer cost | 56,301,568 B |
| of which `output_all_size` | 56,143,872 B (99.7%) |
| final output tensor, allocated | 4,096 B |

**The per-buffer cost is ~13,700x the model's visible output.** So the question
W5-4 asks of my own runtime — what does raising the inflight depth cost, and who
decides the ceiling — is the same question a shipping NPU runtime answers with
an exception. W2's in-flight sweep found the throughput side of this; DX-RT
supplies the memory side.

**Success for the anchor:** the inspector computes both aggregates for an
arbitrary `.dxnn` and reproduces the clamp, and my runtime's `--inflight`
ceiling is derived from a stated per-slot cost rather than chosen.

### W5-5 — cascade

Optional, and only if W5-1 to W5-4 hold. A synthetic secondary stage is enough
to prove the scheduler routes and gathers correctly; a real second model is a
different project.

## Claim discipline for the DX-RT half

Reading a production codebase makes overclaiming easy, so the precision rules
are fixed here rather than negotiated per sentence.

**Name a field the way the code names it.** `output_all_size` is 56,143,872
bytes and the runtime charges it per buffer. What it *contains* — activations
only, or activations plus scratch and encoded intermediates — has not been
traced through the rmap structure. So the claim is "`output_all_size` dominates
the per-buffer footprint and is far larger than the user-visible output", not
"it is the activation arena". The second sentence may well be true; it is not
yet established.

**Keep three sizes apart.** They are different numbers and conflating them is
exactly the error the analysis exists to avoid.

| level | densenet121 output |
|---|---|
| logical tensor, 1000 x float32 | 4,000 B |
| allocated region | 4,096 B |
| NPU output region charged per buffer | 56,143,872 B |

**The alignment unit counts elements, not bytes**, and W5-0's self-test is what
established that by rejecting the opposite assumption. Aligning the byte count
gives `align(4000, 64) = 4032`; the file says 4096, which is
`align(1000, 64) x 4`. The input follows the same rule once the layout names the
row: `PRE_IM2COL` folds the last two axes, so
`224 x align(224x3, 64) = 224 x 704 = 157,696` against 150,528 logical. Both
close exactly, which is what makes them citable — and the byte reading would
have been cited just as confidently while being wrong.

**Say which part of the driver is DEEPX's.** The Linux driver tree contains
`dw-edma-*.c`, which follows the upstream DesignWare eDMA driver; the
DEEPX-specific paths are `dx_cdev*`, `modules/rt/` and the device control and
recovery code. The defensible sentence is "traced the boundary between the
DEEPX-specific device-control path and the underlying DMA infrastructure", not
"analysed DEEPX's DMA implementation".

**Licence.** Every `dx_rt` source file carries a header stating the software is
DEEPX property, supplied to customers of a DEEPX NPU, and that redistribution is
prohibited. Read it, cite it by file and line, re-implement the idea. Do not
copy the code into this repository, and quote only the minimum needed to make a
point.

**Anchors verified so far.** Each was opened and read; none is cited from
memory.

| claim | location |
|---|---|
| per-buffer memory formula | `lib/task_data.cpp:453-454` |
| buffer count clamped to device memory | `lib/inference_engine.cpp`, `adjustBufferCountForNpuMemory` / `calculateMaxBufferCount` |
| fixed-count buffers with a direction | `lib/include/dxrt/fixed_size_buffer.h:37-43` |
| naive fallback transpose | `lib/npu_format_handler.cpp:433` |
| SIMD only via IPP or NEON | `lib/npu_format_handler.cpp:456-460` |
| format acceleration off by default | `cmake/dxrt.cfg.cmake:19` |
| CPU subgraphs are ONNX run from memory | `lib/cpu_handle.cpp:236` |
| driver adapter is the device seam | `lib/driver_adapter/driver_adapter_factory.cpp` |
| device-open failure check disabled | `lib/driver_adapter/windows_driver_adapter.cpp`, `#if 0` around the `INVALID_HANDLE_VALUE` guard |
| `output_all_size` is `uint32_t` | `lib/include/dxrt/driver.h:475` |
| kernel file operations | `modules/pci_deepx/dx_cdev_ctrl.c:399-400` |

The disabled guard in the Windows adapter is worth its own entry in `repro/`.
Opening a device path that does not exist returns `INVALID_HANDLE_VALUE` and the
constructor proceeds anyway — the same class of silent failure as OpenVINO's
error 127 and ONNX Runtime's quiet fall back to CPU, both of which this
repository already reproduces.

## Shutdown is a deliverable, not a detail

Define it, then assert it:

```
stop accepting -> drain in flight -> postprocess what remains
-> return every lease -> join -> destroy
```

```
submitted   1000
accepted     437     (interrupted at frame 437)
completed    437
emitted      437
leaked         0
duplicated     0
```

This is not hypothetical. W2 lost 4 frames of 200 to a marker that overtook work
still in flight, the C++ port reproduced it, and
`repro/cpp/04_sentinel_overtakes` reproduces it on demand today at 198 of 200 —
with no accelerator and no runtime, which is why it will still reproduce on the
MX150.

> **To verify before citing:** the claim that a DX-APP release note records
> fixing exactly this — the async runner not waiting for submitted frames at
> shutdown, losing tail frames from saved video — is plausible and would be a
> strong external validation of that reproduction. It has not been checked
> against the actual release notes. An earlier suggestion in this project cited
> a DEEPX repository that does not exist, so nothing in this paragraph should be
> repeated until someone has read the note.

## What W5 must not break

**The JSON schema.** Every tool in `bench/` reads one shape, which is what lets
C++ and Python results sit in the same table and what made `--reanalyse`
possible. A runtime that emits something else costs a second analysis path and
the ability to compare against everything measured so far.

**Equivalence.** Buffer reuse and view binding are *precisely* the class of
change that produces the right number of frames with the wrong values in them.
`bench/verify_equivalence.py` must pass on every W5 path before any of its
numbers are quoted, and the order checks must run in the same session that
reports the throughput.

## What W5 is not

No lock-free queues, no custom allocator, no SIMD intrinsics **in my runtime**.
Each is low-level work that would look like systems engineering while answering
none of the questions above; the honest name for that is decoration. The one
exception is W5-3's format work, where the absence of a SIMD path in DX-RT's own
fallback is the finding rather than the decoration.

It is also not a study of DEEPX. The anchors are design input. If a phase's only
output is a description of someone else's code, that phase failed.

## Where it leaves the project

```
W1  where does a frame spend its time?
W2  what does overlap buy, and where?
W3  are the answers still correct?
W4  what does explicit concurrency and ownership change?
W5  what must the pipeline own, and what must it refuse?
```

W4's closing result was that Python's orchestration costs 25% sequentially and
4% overlapped, because overlap moves host work off the critical path. W5 asks
the next question that result implies: once host work is off the critical path,
**the thing that binds is no longer speed but admission** — how much work the
pipeline is willing to hold, and what it does when more arrives.

DX-RT answers that question by refusing to allocate. The sentence W5 is built to
earn is not "I read DEEPX's runtime". It is: *the ownership and backpressure
problems I hit while building this are bounded in a shipping NPU runtime by a
memory budget I can compute from a model file, and here is the program that
computes it.*
