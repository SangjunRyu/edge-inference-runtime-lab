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

## 3. Value — **not answered**

This is the gap, and it is the one W3 was really for.

Nothing in the repository checks that the async pipeline produces the *same
answers* as the sequential one. `OrderChecker` verifies indices. It does not
look at the classification result at all, so every failure mode that keeps the
indices tidy is invisible:

- a record paired with another frame's logits in the reorder buffer
- the preprocessed tensor being mutated by the next frame before the runtime
  has finished reading it — the input buffer is handed to `run_async` and the
  worker moves straight on
- an output tensor reused across concurrent requests

Any of these would show perfect order, a perfect count, and wrong predictions.

**The test is cheap and needs no accelerator.** Run the same clip through both
modes, record top-k per frame index, and compare:

- indices must match exactly
- for a fixed backend, class indices should match exactly; probabilities may
  differ in the last bits if the runtime schedules reductions differently, so
  compare with a tolerance and report how many frames differ at all
- run it at several in-flight depths, since a race needs concurrency to appear

Until that exists, every correctness claim here is about *ordering* only, and
the W2 tables should be read that way.

## Where W3 lives now

- Ordering evidence and the reorder trade-off: `docs/findings-w2.md`
- `bench/reorder.py` — `ReorderBuffer`, `OrderChecker`
- `run_ab` refuses to report a speedup when order was violated; `run_sweep`
  excludes such depths from the curve

## Still to do

- **The value-equivalence check above.** Portable: CPU only, any machine.
- **A shutdown stress test** that asserts `emitted == submitted` rather than
  relying on someone noticing 196 instead of 200.
- Both should carry into the C++ port as-is. A C++ pipeline that reorders or
  drops frames is no more acceptable than a Python one, and the port is exactly
  where a buffer-reuse optimisation could introduce the value bug that nothing
  currently detects.
