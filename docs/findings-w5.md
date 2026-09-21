# W5 findings -- what the pipeline owns

W5 asks what a pipeline has to own in order to refuse work. W5-1 is the
ownership layer: a fixed pool, a move-only lease, and a shutdown that is checked
rather than hoped for.

Everything below is counted or asserted. Nothing is a rate, so nothing depends
on the machine holding still -- which is the reason this phase was scheduled
after W4 lost five consecutive paired runs to a laptop that would not.

## What it is

```
cpp/include/ownership.hpp     Direction, BufferLease<D>, BufferPool<D>, FrameContext
cpp/include/ring_queue.hpp    a bounded queue that allocates once
cpp/include/alloc_counter.hpp global operator new/delete, counted
cpp/src/w5_ownership.cpp      the harness and its four checks
cpp/src/negative_direction.cpp a target that must fail to build
```

Run it:

```
cmake --build cpp/build --target w5_ownership
cpp/build/w5_ownership.exe --frames 1000 --inflight 4 --workers 2
```

The MSYS2 UCRT64 runtime must be on `PATH`, as for the other C++ targets. Without
it the process exits 127 with no message, which is the same silent failure
this repository already records for OpenVINO's error 127 -- a missing dependency
looks identical to a program that decided not to run.

## The measurement, and where its fence is

The claim is "zero allocations in steady state". W4's plan wrote down why that
claim is worthless without a stated boundary: a global counter sees whatever is
linked into the process, so a number that quietly excludes OpenCV and OpenVINO
is worse than no number.

**Here the boundary is the link line.** `w5_ownership` links neither OpenVINO
nor OpenCV -- only threads. With nothing else in the binary, every allocation
the counter sees was asked for by this repository's own code, so the honest
claim and the measured number are the same number. It is also why the target
builds and runs on the MX150 unchanged.

Counting is armed after a warm-up rather than at process start, because thread
creation, the pools and the first frames all allocate by design and including
them would be measuring start-up.

## Results

20,000 frames, 8 buffers per pool, 4 workers:

```
shutdown       submitted 20000  accepted 20000  completed 20000  emitted 20000

  ok    every input lease returned          0 outstanding, 20000 acquired, 20000 released
  ok    every output lease returned         0 outstanding, 20000 acquired, 20000 released
  ok    no buffer returned twice            0 double returns
  ok    accepted == completed == emitted    20000 / 20000 / 20000
  ok    no frame processed twice            0 duplicated
  ok    every frame carries its own values  0 of 20000 wrong
  ok    zero allocations in steady state    0 allocations, 0 bytes
```

Every combination of `--inflight` in {1, 2, 4, 8, 16} and `--workers` in
{1, 2, 4} passes with zero steady-state allocations -- fifteen configurations,
and the count is zero rather than small in all of them.

Interrupting the source mid-stream drains rather than truncates:

```
$ w5_ownership --frames 1000 --interrupt-at 437
shutdown       submitted 1000  accepted 437  completed 437  emitted 437
               every lease returned, no duplicates
```

This is the sequence W2 got wrong, when an end-of-stream marker overtook work
still in flight and four frames of 200 disappeared.

## The checks were made to fail on purpose

A check that has never failed has not been shown to work. Each one therefore has
an injection, and the harness exits zero when *exactly* the predicted check
fails -- the same convention `repro/` uses for broken and fixed modes.

| injection | predicted to break | observed |
|---|---|---|
| `leak` | every output lease returned | 1 outstanding at shutdown |
| `alloc` | zero allocations in steady state | 939 allocations, 60,096 bytes |
| `early-release` | every frame carries its own values | 13-23 of 1000 wrong |

### The allocation injection did not work the first time

Its first version put the pointer in a `volatile` local and freed it
immediately. The counter stayed at **zero**, and the run reported "NOT as
predicted".

Nothing was wrong with the counter. Since C++14 a compiler may delete a
`new`/`delete` pair outright when the allocation is not observable, and
`volatile` on the *variable* does not make the allocation observable. At `-O3`
GCC removed it. Publishing the address to a global atomic makes the pointer
escape, and the count went to 939.

That number is itself a small check. 936 frames follow the warm-up, and the
extra three are frames submitted before it that were still in flight when
counting armed -- the injection tests whether counting is armed, not whether the
frame is new, so the arithmetic closes.

**The lesson is about instruments, not about allocation.** W4 spent its last
session discovering that three of its measurements had been measuring
themselves. An injection that is silently optimised away is the same failure in
a cheaper form, and the only reason it was caught is that the harness predicts
which check should fail and complains when the prediction misses.

### Early release is the one worth the phase

Releasing the input lease before the work reading it has finished is legal C++,
costs nothing, raises no error, and produced **13 to 23 wrong frames out of
1000** across eight runs. Between 1.3% and 2.3%.

Every other check passed in those runs. The counts were right. The right number
of frames came out, each exactly once, in the right order, with no leak and no
allocation -- carrying another frame's data.

A test that counted frames would have passed. That is the entire argument for
checking values, and it is the same finding `repro/cpp/05` records one stage
later in the pipeline.

The rule the pool exists to enforce follows from `repro/cpp/01`, which proved
that a runtime handed a pointer reads it at completion time rather than at
hand-off time:

> The input lease is pinned to its slot from submit until the work reading it is
> done. Only then does it move on.

## The DX-RT anchor

`FixedSizeBuffer(size, buffer_count, allocType, direction)` in
`dx_rt/lib/include/dxrt/fixed_size_buffer.h:37-43` makes two decisions a
shipping NPU runtime makes: the count is fixed up front, and the buffer carries
its direction rather than the caller remembering it.

Both are reproduced here, and the second is stronger than the original. In
DX-RT direction is a constructor argument, so a mismatch is a run-time
condition. Here `Direction` is a template parameter, so `InputLease` and
`OutputLease` are different types and a mismatch cannot be compiled.

That claim is about the type system, so the compiler is the thing that checks
it. `cpp/src/negative_direction.cpp` exists to fail to build:

```
$ cmake -DEIRL_NEGATIVE_COMPILE=ON cpp/build && cmake --build cpp/build --target negative_direction
error: invalid initialization of reference of type 'eirl::OutputLease&'
       {aka 'eirl::BufferLease<eirl::Direction::Output>&'} from expression of
       type 'eirl::InputLease' {aka 'eirl::BufferLease<eirl::Direction::Input>'}
negative_direction.cpp:23
```

The build fails, on the intended line. A run-time check could not have made this
claim; it could only have reported the mistake after it was made.

## Why the queue was rewritten

`SafeQueue` is built on `std::queue`, which is a `deque`, which allocates and
frees blocks as it grows and shrinks. Invisible when the question is throughput,
fatal when the question is whether the pipeline allocates -- the answer would
have been "yes, in the queue", for reasons unrelated to the design under test.

`RingQueue` holds a vector sized at construction. Items move into and out of
fixed cells, and a popped cell keeps a moved-from value, which for
`FrameContext` means an empty lease and therefore nothing to release. Shutdown
semantics are unchanged, including the one that matters: an empty optional means
closed *and drained*, never merely closed.

## What this does not claim

The buffer pool is not an allocator and does not try to be faster than one. It
hands out slices of a single block taken at construction; the only claim is that
it does not call `malloc` afterwards, not that its bookkeeping beats one.

Throughput is not reported. The harness does synthetic work on purpose, so its
elapsed time describes this laptop rather than any design. W5-2 is where the
generic scheduler has to reproduce W4's numbers, and that comparison is measured
under the repository's paired discipline.

The double-return check refuses a second release rather than detecting one after
the fact. A pool that accepted it would corrupt its own free stack silently, and
the counters that make every other check possible would stop meaning anything.


# W5-2 -- the scheduler, and the two bounds that decide concurrency

W4's runners each hard-coded one shape. `cpp/include/scheduler.hpp` is the same
shape with the numbers pulled out: worker counts per stage, per-queue depths,
buffer counts, and stages supplied by the caller.

```
cpp/build/w5_scheduler.exe --frames 2000 --buffers 8 --queue-capacity 2
cpp/build/w5_scheduler.exe --sweep --frames 4000
```

## The anchor question, answered with arithmetic

DX-RT separates choosing a device from submitting to it (`lib/device_pool/`),
which leaves three plausible answers to what limits concurrency: the queue
depth, the worker count, or the buffer count. The plan asked for a number.

There are two bounds, and the smaller wins.

```
C_structural = S + sum(Qi) + sum(Wj)
C_resource   = the input pool
ceiling      = min of the two
```

`C_structural` is one frame being prepared by the source, whatever fits in each
bounded queue, and one per worker at each stage. The three in "three queues" is
a fact about this topology and nothing else: a pipeline with different queues,
or with different capacities per queue, changes the sum and not the form. The
code takes the queue depths separately for that reason.

`C_resource` is the input pool **because of this scheduler's lifetime model**,
in which a frame holds its input lease from the reader to the end of
postprocessing. The output pool is not in the bound: an output lease is held
only from the start of inference, so it constrains a segment of the path rather
than the whole of it. If every in-flight frame held both, the bound would be
`min(B_in, B_out, C_structural)`. The formula is a property of the lifetimes,
not a law, and the header says so beside it.

## The sweep

16 configurations, 4000 frames each:

```
  buf  queues   pre/inf/post   struct  resrc  ceil   leases  inflight  binds
    1   2/ 2/2    1/ 2/1         11      1     1         1       2   buffers
    2   2/ 2/2    1/ 2/1         11      2     2         2       3   buffers
    4   2/ 2/2    1/ 2/1         11      4     4         4       5   buffers
    8   2/ 2/2    1/ 2/1         11      8     8         8       9   buffers
   16   2/ 2/2    1/ 2/1         11     16    11        11      11   structure
   32   2/ 2/2    1/ 2/1         11     32    11        11      11   structure
   32   1/ 1/1    1/ 2/1          8     32     8         8       8   structure
   32   4/ 4/4    1/ 2/1         17     32    17        17      17   structure
   32   8/ 8/8    1/ 2/1         29     32    29        29      29   structure
   32   8/ 1/1    1/ 2/1         15     32    15        15      15   structure
   32   1/ 8/1    1/ 2/1         15     32    15        15      15   structure
   32   1/ 1/1    1/ 1/1          7     32     7         7       7   structure
   32   1/ 1/1    2/ 4/2         12     32    12        12      12   structure
   32   1/ 1/1    4/ 8/4         20     32    20        16      18   structure
    3   8/ 8/8    2/ 4/2         33      3     3         3       4   buffers
   64   1/ 1/1    1/ 1/1          7     64     7         7       7   structure
```

Fifteen of sixteen reached the ceiling; none exceeded it.

**That second clause is the claim.** A ceiling is an upper bound, not a target:
whether a run reaches it depends on the scheduling that day and on the workload
being long enough, so "15 of 16" is not an accuracy figure and the formula is
not 94% correct. The assertion is that nothing ever went above it, and that held
in all sixteen.

**Neither knob is the answer alone, and each half kills an intuition.** Rows 5
and 6 raise the pool from 8 to 32 and buy nothing: the structure has eleven
places and that is that. Row 15 has 33 places and three buffers and gets three.
Rows 8 and 9 show queue depth buying concurrency *only* while the pool has spare
buffers to fund it, and rows 10 and 11 show that it does not matter which queue
gets the depth -- the sum is what counts.

Which is why W5-4 matters. On a real device the buffer side is the externally
constrained one: DX-RT derives the count from device memory
(`lib/task_data.cpp:453-454`), and `tools/dxnn_inspect.py` computes it for a
compiled model. The knob that binds in production is the one nobody chose.

## Three quantities, not one

The `leases` and `inflight` columns differ, deliberately, and the first four
rows differ by exactly one. Reporting them separately is the correction that
this phase mostly consisted of.

| quantity | what it is | its bound |
|---|---|---|
| input leases | buffers checked out of the pool | `min(resource, structural)` |
| output leases | the same for the output pool | its own pool |
| pipeline in-flight | admitted, not yet finished | `structural` only |

Pipeline occupancy runs above lease occupancy because a frame keeps its place in
a postprocess worker for a moment after giving its buffers back. Adding these
into one "in-flight" number is what went wrong three separate times below.

## Three failures, one shape

W5-2's real content turned out to be the bounds themselves, and every failure in
it was a version of the same mistake: **a capacity argument applied to something
other than the quantity it was derived from.** All three are runnable.

### 1. The instrument measured across two events

The first run reported `peak in-flight 9` against a pool of 8 and failed its own
check. Every other check passed.

The counter was bracketed around the frame rather than the lease, and came down
after postprocessing had already returned the buffer. In that window another
thread could take the returned buffer and count itself: two frames counted, one
buffer held. Two atomics are two events, and a high-water mark taken across them
belongs to neither.

Fixed by recording the mark inside the pool, under the pool's own lock, at the
instant the buffer leaves the free stack. Kept as
**`repro/cpp/06_inflight_overcount`** -- twelve threads against four buffers,
where the outside counter peaks at 10 or 11 and the pool's own peak stays at 4.

That reproduction needed a correction of its own. Its first version used four
threads against eight buffers and reported NOT REPRODUCED: with more buffers
than threads nobody ever waits, so there is no instant at which a returned
buffer is taken again. The defect needs contention, which is now in the
program's header rather than in someone's memory.

### 2. The bound was right and the quantity was wrong

With the counter fixed, the peak was still 9 against a ceiling of 8 -- and this
time 9 was a value the counter genuinely held, linearized under one lock.

`min(resource, structural)` had been written down as a bound on "frames in
flight". It is not one. It bounds frames *holding a buffer*, and the derivation
does not cover a frame that has released early or has not acquired yet. Split
into two invariants over two quantities, both hold and neither is the other.

Kept as **`repro/cpp/07_bound_wrong_metric`**.

### 3. The same error sized a data structure, and it hung

The corrected sweep then stopped producing output. No crash, no error, a live
thread and a few megabytes resident -- indistinguishable from a slow machine
until you check the clock.

`RingReorder` had been sized from the ceiling, on the reasoning that an
unemitted frame still holds a lease. It does not: a completion reaches the
reorder *after* its buffers have gone back, so the indices outstanding there are
bounded by pipeline occupancy plus the emit path. The ring was too small.

On its own that would only have cost ordering. What made it a hang was the write
path: on an arrival past the window it flushed exactly one entry and wrote
anyway. One flush is not enough when the gap is wide, so `index % cap` still
aliased an occupied slot; the write destroyed a pending entry while the held
counter recorded it as new. From then on the counter and the flags disagreed,
and `while (held_ > 0) flush_lowest();` had no reason to terminate.

Two fixes for two mistakes: size the ring from the bound that applies to it, and
flush until the index fits. `drain()` is now a bounded walk over the window
rather than a loop that trusts a counter.

Kept as **`repro/cpp/08_reorder_window_overflow`**, single-threaded and
deterministic. Broken mode loses 4 of 16 entries at a window of 4; fixed mode
loses none at 16. The reproduction's drain is budgeted, so it *reports*
non-termination instead of demonstrating it -- a test that hangs is not a test.

**The rule all three share:** a capacity is only valid for the lifetime it was
derived from. Write the lifetime beside the formula, or the formula will outlive
it. That is now how `scheduler.hpp` states both bounds.

## What is not claimed

The paired comparison against W4's OpenVINO runners has not been run. The
structural half of W5-2 is done and portable; reproducing W4's numbers through
the generic scheduler needs the accelerator and a machine that holds still, and
W4 established this one does not.

# W5-3 -- what the host pays to hand a tensor to an NPU

`cpp/include/nfh.hpp` re-implements the padding and transpose rules from the
container, not from DEEPX's source. `tools/dxnn_inspect.py --nfh-args` prints
the geometry of a model that was actually compiled, so there is one parser and
the C++ is driven by it:

```
$ python -m tools.dxnn_inspect densenet121_224x224.dxnn --nfh-args
--rows 224 --row-elems 672 --align 64 --elem-bytes 1
```

## Correctness needs no hardware, because the round trip is its own oracle

`--verify` passes 29 checks: decode of encode is the identity over six
geometries including awkward ones, padding is zeroed so an encoding is a
function of its input alone, the blocked transpose is bitwise identical to the
naive one over five shapes, transposing twice returns the original, and the
fused encode-and-transpose equals the two passes it replaces. It also reproduces
the container's declared 157,696 bytes from 224 rows of 672 elements at unit 64.

## Counts, which do not depend on the machine

200 frames per path, densenet121 input geometry:

```
  path                          allocs/frame   passes   bytes touched/frame
  separate, per-frame buffers             2        2                637952
  separate, preallocated                  0        2                637952
  fused encode+transpose                  0        1                322560
```

Fusing removes 315,392 bytes of host traffic per frame. Counted and derived, not
measured -- and W4 already found one case where a real reduction in host work
bought exactly nothing once the pipeline overlapped, so whether this shows up in
a stage time is a separate question with its own answer.

## The measured half, and the result that came out the wrong way round

`dx_rt/lib/npu_format_handler.cpp:433` transposes one element at a time with
`memcpy` into a strided destination. No blocking, no vectorisation, and the
accelerated paths at `:456-460` compile only under Intel IPP or ARM NEON, with
`USE_NPU_FORMAT_CONVERSION_ACCELERATION` off by default. The obvious conclusion
is that the fallback leaves performance on the table.

`--bench` pairs blocked against naive with the order alternated inside each
pair, and a `memcpy` of the same bytes as the control. Ratios are blocked over
naive, so below 1.0 means blocking won.

| geometry | bytes | blocked/naive | control CV |
|---|---|---|---|
| 224 x 672 x 1 B | 157,696 | **1.52** | 2.6% |
| 224 x 672 x 4 B | 630,784 | 0.53 | 3.3% |
| 512 x 512 x 4 B | 1,048,576 | 0.61 | 6.9% |
| 1024 x 1024 x 1 B | 1,048,576 | 0.25 | 5.1% |
| 1024 x 1024 x 4 B | 4,194,304 | 0.37 | 3.7% |

**At the geometry the model actually uses, blocking is 1.5x slower.** The whole
tensor is 157 KB; it sits in cache, there is no pressure to relieve, and the
tiling loops are pure overhead. From roughly 600 KB upward blocking wins by 1.6x
to 4x, and the crossover follows total size rather than element width: at one
megabyte the 1-byte and 4-byte cases both favour blocking.

So the conclusion is the opposite of the one reading the source suggests.
**DX-RT's scalar fallback is a reasonable choice at the size its own model
inputs are**, and the textbook improvement is a regression there. The defensible
claim is not "the fallback is slow" but "the fallback is slow above about a
megabyte, and a classifier's input is not".

Four cells could not be measured and are recorded as such rather than dropped:

| geometry | bytes | control CV |
|---|---|---|
| 256 x 256 x 4 B | 262,144 | 23.7% |
| 2048 x 2048 x 1 B | 4,194,304 | 28.2% |
| 2048 x 2048 x 4 B | 16,777,216 | 35.3% |
| 4096 x 4096 x 4 B | 67,108,864 | 15.3% |

The control is unaffected by either variant, so its spread across pairs measures
the laptop. Above the 15% bar the tool prints UNMEASURABLE and declines to quote
a ratio. Two of those cells are the large ones where blocking should matter
most, so the interesting end of the curve is the end this machine cannot
currently measure, which is the same conclusion W4 reached about its own last
afternoon.

## What W5-3 does not claim

No SIMD. The plan rules it out as decoration, and the point here is that access
*order* is what costs, which a portable implementation can demonstrate and
defend. A vectorised transpose would win more and explain less.

Nothing is copied from DEEPX. Every `dx_rt` file states that the software is
supplied to customers of a DEEPX NPU and may not be redistributed, so what is
reproduced is behaviour, cited by file and line, and checked against the
container's own declared sizes.

# W5-3 part two -- the compiled model decides the buffers

The format work above measured conversions. It did not fix the thing underneath
them: **the buffer size was still a literal.** `150528` sat in a default
argument, correct for densenet121's input if you forget that the device reads a
padded tensor, and the padded one is `157696`.

A pool built from the logical size is 7,168 bytes short of what the hardware
reads. The allocation succeeds. The frames flow. The counts match. It is the
same class of defect the rest of W5 keeps finding, arriving through a constant
rather than through a bound.

## One source of truth

```
python -m tools.dxnn_inspect model.dxnn --contract contracts/model.json
cpp/build/w5_contract.exe --contract contracts/model.json --npu-memory 512M
```

The contract carries the dtype, shape, layout, alignment unit, logical and
allocated sizes per tensor, the per-direction buffer sizes, and the fixed and
per-buffer costs. Everything the pipeline needs in order to size and check
itself, produced once from the model the compiler wrote.

The loader does not take it on trust. `contract.cpp` re-derives the padded size
from the layout rule and compares it with what the file declares, using its own
table rather than sharing the inspector's -- two derivations that agree are
evidence, and one implementation would only ever agree with itself.

For densenet121 at 512 MiB of device memory:

```
  input   input.1 UINT8 [1,224,224,3] PRE_IM2COL align 64
          logical 150528 B, allocated 157696 B
  output  1164 FLOAT32 [1,1000] ALIGNED align 64
          logical 4000 B, allocated 4096 B
  buffers   input 157696 B, output 56143872 B
  costs     fixed 17253056 B, per buffer 56301568 B

buffers        6 of 6 requested, limited by 536870912 B of device memory
host pools     input 157696 B x 6
work executed  2000 stages, 0 rejected, all checks pass
```

The buffer count is not a flag. It is W5-2's resource bound, arriving from the
model file and the device's memory instead of from a command line.

## Refusing before the work, not after

Four injections, and the line that matters is the same in all four:

| injection | what it does | refused by |
|---|---|---|
| `hardcoded-size` | sizes the pool from `150528` | the start-up admission check |
| `bad-allocated` | allocation below the logical size | contract validation, on load |
| `bad-padding` | padding computed on bytes, not elements | contract validation, on load |
| `overflow` | a device one byte short of a single buffer | the capacity calculation |

```
  FAIL  the input pool matches the contract
        input buffer is 150528 B; the model requires 157696 B
        (too small, the device would read past it)

work executed  0 stages -- refused before the pipeline started
```

**Zero stages, in every case.** That is the property, not the message. After
inference the only evidence left is an answer that is wrong and looks right;
`repro/cpp/05` and W5-1's early-release injection are both records of how
convincing that looks.

`bad-padding` is worth its own line. It sets the allocation to
`align(150528 bytes, 64)`, which is `150528` again, because the logical size
already happens to be a multiple of 64. That is the byte-versus-element mistake
the inspector's self-test caught in W5-0, arriving from the other direction --
and it is refused here because the loader predicts `157696` from the layout and
says so.

# W5-4 -- what a bounded pipeline does when it cannot keep up

## The prerequisite, restated

The plan wrote this down before the code existed: with a file read as fast as
the pipeline consumes it, `Block` and `DropOldest` are indistinguishable.
Nothing is ever late, so a drop policy has nothing to drop.

So the source here keeps its own clock. It emits at `--source-fps` and waits for
nothing except under `block`, where waiting *is* the policy. That is what a
camera is.

The second half of the prerequisite is the measurement. **Age at output, not
pipeline latency.** Age is counted from when a frame was due, so it includes the
time a frame spent waiting to be captured. A policy that quietly captures late
has excellent pipeline latency and stale output, and only the first number shows
up if you measure the obvious thing.

## The result

A 60 FPS source, one worker at 25 ms per frame, so 40 FPS of capacity against 60
offered. Queue 4, pool 8, three repeats with the policy order rotated:

```
  policy             FPS     CV   p95 age   max age   dropped  resident
  unbounded        39.99   0.0%     208.8     212.0        73         7
  block            39.97   0.1%    1920.8    2026.2         0         4
  drop-oldest      39.96   0.1%     100.5     121.4        76         4
  drop-newest      39.99   0.0%     133.8     134.2        76         4
```

**All four sustain the same rate, to within 0.1%.** The queue was never what
limited throughput; the worker was. So a deeper queue buys nothing except older
frames and more resident memory -- which is the systems result this experiment
was built to produce, and it came out as predicted.

The throughput column is the measured one, and for once the spread is tiny. That
is not this laptop behaving: the service time is a spin loop of a fixed
duration, so the rate is arithmetic. Every other column is counted.

**Blocking costs 19x in staleness and hides it.** `block` drops nothing, which
looks like the safe choice, and delivers frames that are on average almost two
seconds old. The source itself ends up 1.5 seconds behind the wall clock,
because under `block` falling behind is how the pipeline applies backpressure to
a source that has nowhere to put it.

**Drop-oldest beats drop-newest at the same drop count.** 100 ms against 134 ms
of p95 age, 76 frames discarded either way. When the queue is full the freshest
frame is the one worth keeping, and discarding the arrival instead means the
consumer is served from a backlog it already could not clear.

## The prediction that did not hold, and why it is the better finding

The plan said an unbounded queue in front of a clocked source grows until the
process dies. It does not. It settled at **7 resident and dropped 73 frames.**

There was no unbounded queue, because there cannot be one. A frame needs a
buffer before it can be queued, and the pool has eight. **The pool is admission
control whether or not anybody designed it that way**, and removing the queue's
limit only moves the refusal one step earlier, to `try_acquire` returning
nothing.

That is W5-1, W5-2 and W5-4 closing on the same point from three directions. A
pipeline that owns its buffers already has a bound; the queue policy decides
*which* frames survive it, not how many. And it is why DX-RT deriving
`bufferCount` from device memory is not a detail of its configuration but the
place its admission control actually lives.

## What is not claimed

The service time is synthetic. This says what the policies do to freshness and
residency under overload, and says nothing about any real model's throughput.

Latency is reported as age at output rather than as a stage breakdown, because
the stages here are one spin loop. W1's decomposition is the tool for the other
question and this is not it.
