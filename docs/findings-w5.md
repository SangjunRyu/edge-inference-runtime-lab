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
