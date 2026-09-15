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

## Why this phase and not another round of tuning

Two reasons, and the second is the stronger one.

**It generalises what W4 already built.** The slot pool, the rule that a
completion callback must not block, the reorder buffer that exists because
postprocessing runs concurrently — those are not OpenVINO details. They are the
skeleton of any bounded pipeline, currently welded to one. W4's three runtime
defects were all ownership and lifetime defects, and
[`repro/`](../repro/README.md) now holds each as a runnable program.

**Its headline results are exact.** This is the decisive argument and it is
worth being blunt about. W4 spent most of its effort fighting a machine that
would not hold still, and lost: three consecutive paired runs in the closing
session failed their own control stage, one at a coefficient of variation of
49%. A phase whose main deliverables are *counts* rather than *rates* does not
depend on that machine at all.

| deliverable | kind | needs a steady machine? |
|---|---|---|
| allocations per frame in steady state | counted | no |
| host copies per frame, bytes moved | counted / derived | no |
| leases outstanding at shutdown | asserted | no |
| `accepted == completed == emitted`, no duplicates | asserted | no |
| peak resident frames under a bounded policy | counted | no |
| frames dropped under each policy | counted | no |
| throughput, p50/p95 latency | measured | **yes** |

Five of seven rows are facts about the program. That is the right work for a
laptop that cannot hold still, and it is the reason to do W5 before any further
tuning. **The counted rows must be labelled as counted**, everywhere, or this
advantage turns into the same overclaiming the repository has spent four phases
removing.

**It is portable.** Nothing above needs the NPU. W5-1 through W5-4 run on the
MX150 unchanged, because the subject is CPU-side runtime architecture rather
than accelerator performance.

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

With that in place the experiment is the strongest single thing in W5:

| policy | FPS | p95 E2E | dropped | peak resident |
|---|---|---|---|---|
| unbounded | | | | |
| bounded, block | | | | |
| bounded, drop-oldest | | | | |

and the conclusion it is set up to reach — that all three sustain the same
throughput, so a larger queue buys nothing but stale frames and resident memory
— is a genuine systems result. Note that the throughput column is the *measured*
one, so it is the column that needs the paired discipline the rest of the
repository uses; dropped frames and peak residency are counts and do not.

## Three smaller corrections

**A lease cannot be moved out from under a running request.** The plan draws
buffers moving down the pipeline, `acquire -> preprocess -> inference ->
postprocess -> release`. But OpenVINO holds a pointer to the input tensor's
memory for the whole inference, and `repro/cpp/01` now proves the same contract
for the callback descriptor: the runtime reads it at *completion* time, not at
set time. So the ownership rule is not "the lease travels". It is:

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
table. A claim of zero allocations that quietly excludes the two largest
libraries in the process is worse than no claim.

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

Each phase has a success criterion that can be checked, not admired.

### W5-1 — ownership

`FrameContext`, move-only `BufferLease`, and pools sized at startup.

**Success:** a run of 1000 frames reports zero allocations from the runtime's
own code after warm-up, and every lease is back in its pool at shutdown. Both
are assertions in the program, not numbers in a report.

The move-only choice earns its keep here rather than aesthetically: with
`shared_ptr` a leaked buffer is invisible, because the count merely stays above
zero and nothing complains. With move-only leases and a pool that asserts a full
return, **shutdown becomes a checkable invariant**.

### W5-2 — the scheduler

`read -> preprocess -> infer -> postprocess -> output` as generic stages, with
worker counts and queue capacities on the command line: `--pre-workers`,
`--post-workers`, `--inflight`, `--queue-capacity`.

**Success:** the existing runners are reproducible as configurations of it, and
`bench/compare_variants.py` pairs old against new with the control holding. If
the generic version cannot reproduce W4's numbers, the abstraction cost
something and the amount is the result.

Do not build work stealing. Bounded queues, a worker pool and explicit state
per frame are enough, and the interesting problems are all in the ownership
rather than in the dispatch.

### W5-3 — memory movement

Instrument four preprocessing paths — baseline, fused, preallocated, tensor
view over the same buffer — for allocations, copies and bytes moved.

**Success:** a table whose rows are counted, with the boundary stated, and
whose bytes-per-frame column is consistent with the preprocess timings W4
already has.

### W5-4 — backpressure as a policy

`Block`, `DropOldest`, `DropNewest`, against a source paced with `--source-fps`
and an overload induced by slowing the sink.

**Success:** the table above, filled in, with throughput measured under the
repository's paired discipline and the rest counted.

### W5-5 — cascade

Optional, and only if W5-1 to W5-4 hold. A synthetic secondary stage is enough
to prove the scheduler routes and gathers correctly; a real second model is a
different project.

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

This is not hypothetical. W2 lost 4 frames of 200 to a marker that overtook
work still in flight, the C++ port reproduced it, and
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
the ability to compare against everything measured so far. This was the single
highest-leverage decision in W4 phase 1 and it is the same decision here.

**Equivalence.** Buffer reuse and view binding are *precisely* the class of
change that produces the right number of frames with the wrong values in them.
`bench/verify_equivalence.py` must pass on every W5 path before any of its
numbers are quoted, and the order checks must run in the same session that
reports the throughput — the discipline W3 established and W2 enforces.

## What W5 is not

No lock-free queues, no custom allocator, no SIMD intrinsics. Each is
low-level work that would look like systems engineering while answering none of
the questions above; the honest name for that is decoration. The subjects here
are ownership, lifetime, bounded resources, backpressure, scheduling, copy
boundaries, shutdown correctness and measurement, and there is more than enough
in those.

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
