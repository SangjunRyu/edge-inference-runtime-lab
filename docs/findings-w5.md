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


# W5-2 -- the scheduler, and which knob bounds concurrency

W4's runners each hard-coded one shape. `cpp/include/scheduler.hpp` is the same
shape with the numbers pulled out: worker counts per stage, queue depths, buffer
count, and stages supplied by the caller.

```
cpp/build/w5_scheduler.exe --frames 2000 --buffers 8 --queue-capacity 2
cpp/build/w5_scheduler.exe --sweep
```

## The anchor question, answered with arithmetic

DX-RT separates choosing a device from submitting to it (`lib/device_pool/`),
which leaves three plausible answers to what actually limits concurrency: the
queue depth, the worker count, or the buffer count. The plan asked for the
answer as a number.

It is not a measurement. A frame holds its input lease from the moment the
reader takes one until postprocessing gives it back, so it cannot outnumber the
pool; and it has to be standing somewhere, so it cannot outnumber the places
there are to stand.

```
ceiling = min(buffers, 1 + 3*queue_capacity + pre + infer + post)
```

`--sweep` checks that against what the pipeline does, at 4000 frames per cell:

```
  buf  queue  pre/inf/post   places  ceiling   peak   binds
    1      2   1/ 2/1          11        1      1   buffers
    2      2   1/ 2/1          11        2      2   buffers
    4      2   1/ 2/1          11        4      4   buffers
    8      2   1/ 2/1          11        8      8   buffers
   16      2   1/ 2/1          11       11     11   structure
   32      2   1/ 2/1          11       11     11   structure
   32      1   1/ 2/1           8        8      8   structure
   32      4   1/ 2/1          17       17     17   structure
   32      8   1/ 2/1          29       29     29   structure
   32      1   1/ 1/1           7        7      7   structure
   32      1   2/ 4/2          12       12     12   structure
   32      1   4/ 8/4          20       20     19   structure
    3      8   2/ 4/2          33        3      3   buffers
   64      1   1/ 1/1           7        7      7   structure
```

Thirteen of fourteen reached the ceiling; none exceeded it. Reaching it is a
property of the run, not exceeding it is a property of the design, and only the
second is asserted.

**Neither knob is the answer on its own, and each half kills an intuition.**
Rows 5 and 6 raise the pool from 8 to 32 and buy nothing: the structure has
eleven places and that is that. Row 13 has 33 places and three buffers and gets
three. Rows 7 to 9 show queue depth buying concurrency *only* while the pool has
spare buffers to fund it.

Which is why W5-4 matters. On a real device the buffer side is the one that is
externally constrained: DX-RT derives the count from device memory
(`lib/task_data.cpp:453-454`), and `tools/dxnn_inspect.py` computes it for a
compiled model. The knob that binds in production is the one nobody chose.

## The reorder buffer stopped needing dynamic memory

`ReorderBuffer` uses `std::map` and allocates per completion, which is invisible
next to an inference and fatal to a steady-state claim. `RingReorder` replaces
it with a fixed ring, and the capacity is not a guess: an unemitted frame still
holds a lease, so the number of indices outstanding at once cannot exceed the
ceiling. A ring of that size can hold every gap that can physically exist.
Overflow would mean an ownership rule had stopped holding, so it is counted and
reported rather than resized away. Across the sweep it never overflowed, and
held at most 4 of 9 in the default configuration.

## The first run failed, and the design was not the problem

```
ceiling        8 = min(buffers 8, places 11)   observed peak 9
  FAIL  peak in-flight within the computed ceiling
```

Nine frames in flight with eight buffers to hold them. Every other check passed.

The counter was bracketed around the *frame* rather than around the *lease*: it
came down after postprocessing had already returned the buffer, and in that
window another thread could take the buffer and count itself. Both frames
counted, one buffer held.

The bound was real. The instrument was not, and it pointed at the design.
Recording the high-water mark inside the pool, under the pool's own lock, at the
instant the buffer leaves the free stack, makes the count and the thing it
counts move together.

**Kept as `repro/cpp/06_inflight_overcount`**, where four buffers and twelve
threads make the outside counter peak at 10 or 11 while the pool's own peak
stays at exactly 4.

That reproduction needed a correction of its own. Its first version used four
threads against eight buffers and reported NOT REPRODUCED -- with more buffers
than threads nobody ever waits, so there is no instant at which a returned
buffer is taken again. The defect needs contention to exist, which is now in the
program's header rather than in someone's memory.

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
