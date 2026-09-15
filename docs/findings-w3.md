# W3 — does higher throughput still produce correct output?

W3 was planned as its own phase. It did not stay one: the moment the async
pipeline existed it was obvious that ordering had to be checked in the same run
that reported the speedup, not afterwards, so the checks were built into W2 and
`run_ab` was made to refuse a speedup figure for any configuration that
violated them.

This document records what W3 set out to ask, what the W2 work actually
answered, and — more usefully — the part of it that is still not answered.

## What W3 set out to ask

Throughput and correctness are not independent in a pipeline. Three distinct
ways an overlapped pipeline can be faster and wrong:

1. **Order.** Completions do not preserve submission order, so frames can be
   emitted shuffled. The count stays right, which is what makes it dangerous.
2. **Loss.** Frames submitted but never collected — the pipeline drops work at
   the edges, usually at shutdown, and the count comes out *nearly* right.
3. **Value.** The same frame produces a different answer than it would have
   sequentially, through a race on a shared buffer or a record paired with the
   wrong payload. Order and count both look perfect.

The framing came from DEEPX's own code. `frame_reorder.hpp` exists because
their async runner hit (1), and the comment on it names the property that makes
this class of bug hard: *"Frame count stays correct, so it is a silent
defect."*

## 1. Order — answered

`OrderChecker` verifies the emitted sequence is 0, 1, 2, … and is run on
**both** modes. The sequential pipeline cannot emit out of order, so its zero is
what makes the async zero mean something; it was verified at 80 frames, 0
violations.

With reordering disabled, on CPU at 360p with 4 in flight:

| reorder | cap | order violations | q_reorder mean | throughput |
|---|---|---|---|---|
| off | -- | **138 / 200** | 0.06 ms | 108.0 |
| on | 8 | 1 | 6.97 ms | 101.8 |
| on | 64 | **0** | 7.84 ms | 101.4 |

69% of frames out of order with the count exactly right. Restoring order costs
about 7 ms of mean latency and no measurable throughput.

The `cap` is a liveness guard, not a tuning knob: past it the buffer releases
out of order rather than stall forever on a gap that may never fill. A cap of 8
against a measured maximum displacement of 13 fired that path three times and
left real violations behind, so the default became
`4 x (inflight + queue_size)`. Displacement exceeds the in-flight limit because
a gap must also survive the output queue and the postprocess worker.

Every result in `docs/findings-w2.md` was produced with ordering verified.

## 2. Loss — found once, fixed, never systematically tested

The bug was real and it was ours. The end-of-stream sentinel was pushed as soon
as submission finished, so it overtook requests still in flight and their
completions arrived after the consumer had shut down. **196 frames of 200**, and
nothing looked wrong. The fix reacquires every in-flight permit before the
sentinel goes out, which proves every completion callback has run.

`OrderChecker` reports `emitted`, so a loss of this kind is visible after the
fact — that is how it was caught. But there is no test that deliberately
stresses shutdown, and no check that `emitted == frames submitted` is asserted
rather than eyeballed.

## 3. Value — now answered, and passing

`bench/verify_equivalence.py` runs the same clip through both pipelines and
compares top-k per frame index. It reads results out of the **real** runners --
`run_sync.measure_once` and `run_async.measure_once`, through a collector they
now accept -- because a second implementation written for the test could not
contain the bug the test is looking for.

CPU, 360p, 300 frames, top-5, tolerance 1e-4:

| in flight | compared | missing | extra | top-1 mismatches | top-k order | max prob delta |
|---|---|---|---|---|---|---|
| 1 | 300 | 0 | 0 | 0 | 0 | 0.00e+00 |
| 4 | 300 | 0 | 0 | 0 | 0 | 0.00e+00 |
| 16 | 300 | 0 | 0 | 0 | 0 | 0.00e+00 |

Bit-identical, not merely within tolerance. That is the expected result today:
`preprocess` allocates a fresh array per frame, so there is no shared buffer for
a later frame to mutate. The check exists for what comes next.

Three things make the pass mean something rather than just being green.

**A control.** The sequential pipeline is compared against itself first. If it
disagreed with itself nothing below would mean anything, and the run stops
there.

**Several depths.** A race needs concurrency to appear, so depth 1 passing says
nothing about depth 16.

**A self-test, because a check that has never failed is not known to work.**
`--self-test` corrupts the known-good result three ways -- the three shapes this
class of bug takes -- and confirms each is caught before the real comparison
runs:

```
  a frame dropped                            caught
  results of frames 0 and 4 swapped          caught
  one probability moved past tolerance       caught
```

**And the check reports its own power.** A swap between two frames that predict
the same classes produces no difference at all, so on a clip where every frame
looks alike this could verify values while being blind to pairing. It measures
that and prints it: on this clip, **176 distinct top-k sets across 300 frames,
228 of 299 adjacent pairs differing (76%)**. On a static scene the same run
would pass while covering much less, and it now says so instead of letting the
green imply more than it checked.

## Where W3 lives now

- Ordering evidence and the reorder trade-off: `docs/findings-w2.md`
- `bench/reorder.py` — `ReorderBuffer`, `OrderChecker`
- `bench/verify_equivalence.py` — value equivalence, its self-test, its
  sensitivity figure
- `run_ab` refuses to report a speedup when order was violated; `run_sweep`
  excludes such depths from the curve

## Still to do

- **A shutdown stress test** that asserts `emitted == submitted` rather than
  relying on someone noticing 196 instead of 200.
- **Run the equivalence check on an accelerator.** It has only been run on CPU;
  a device with its own memory and an asynchronous submission path has more
  ways to pair a result with the wrong frame.
- **Carry it into the C++ port.** Buffer reuse and a fused preprocessing pass
  are precisely the optimisations that would introduce the bug this is the only
  thing here that detects, so it should run against the C++ pipeline from the
  first build that produces predictions.
