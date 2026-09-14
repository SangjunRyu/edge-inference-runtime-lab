# W2 findings

Overlapped pipeline against the sequential one. Every comparison is a **paired**
measurement: sync and async alternate inside one process, sharing one inference
session, and the ratio within each pair is what is reported. Two absolute
numbers taken minutes apart cannot support a speedup claim on a machine whose
throughput moves 10-30% by itself.

**Every run in every configuration below emitted all frames in submission
order.** Order is verified on both modes, and `run_ab` refuses to report a
speedup for any configuration that violated it.

Raw data: `results/ab-*.json`. Regenerate with `python -m bench.run_ab`.

## The result depends entirely on where inference runs

| configuration | sync FPS | async FPS | **throughput** | **latency (E2E p50)** | ratio CV |
|---|---|---|---|---|---|
| CPU, 4 threads pinned | 38-49 | 31-40 | **0.85x** | 12.9x worse | 6.2% |
| CPU, threads unpinned | 42-78 | 85-106 | 1.37x | 7.9x worse | 23.8% |
| **NPU (OpenVINO), 1080p** | 52-58 | 161-176 | **3.09x** | **0.83x better** | 6.5% |

On the NPU async wins on both axes. On the CPU with a pinned thread pool it
**loses**, and the loss is measured tightly enough (6.2% ratio CV) to be a
result rather than noise.

## Why CPU async loses: concurrency is subdivided, not added

`session.run_async` executes on ONNX Runtime's intra-op thread pool -- the same
pool that performs the computation. In-flight requests therefore do not acquire
new parallelism, they split the parallelism the sequential path already had.

| threads | sync inference p50 | async inference p50 | per-request cost |
|---|---|---|---|
| 4 pinned | 19-24 ms | 87-121 ms | **~4.5x** |
| unpinned (16 logical) | 10-17 ms | 33-41 ms | **~3x** |

With four requests in flight and four threads, each request gets roughly one
thread instead of four, so four concurrent inferences take about as long as
four sequential ones and the pipeline pays the coordination on top. The 1.37x
seen with the pool unpinned is not a pipeline effect either: with sixteen
logical cores there is parallelism the single-request path was not using, and
async happens to spread across it.

On a CPU there is no separate device. "Host" and "accelerator" are the same
silicon, so there is nothing for the host stages to overlap *with*.

## Why NPU async wins: 1/sum becomes 1/max

Pair 2 of the NPU run, stage p50 in ms:

```
             sync    async
read          9.75    5.05
preprocess    1.53    1.45
inference     8.06    6.16
postprocess   0.18    0.12
q_read           -    0.04
q_inflight       -    0.00
q_output         -    0.09
q_reorder        -    0.05
E2E p50      19.37   14.56
throughput   52.6    176.3  FPS
resident      1.00     3.13  frames (Little's Law)
```

Throughput follows the textbook prediction. Sequential is one frame per sum of
stages: 1/19.4 ms = 52 FPS, matching the 52.6 measured. Overlapped is one frame
per slowest stage: 1/6.16 ms = 162 FPS, against 176 measured. Little's Law
confirms the mechanism independently -- 1.00 frames resident in sync, 3.13 in
async.

Queue waits are essentially zero (0.00-0.09 ms). The NPU keeps up with the host,
so nothing backs up. This is the regime W1 predicted: once inference moves off
the host, the host stages become the ceiling and overlapping them pays.

## The latency improvement is not an overlap effect

Overlap cannot reduce a frame's own service time, so the 0.83x latency figure
needs a different explanation, and the stage table gives it: **the service times
themselves fell.** `read` went from 9.75 ms to 5.05 ms and `inference` from 8.06
to 6.16 ms for identical work.

The likely cause for `read` is that in sequential mode `VideoCapture.read()` is
called only once every ~19 ms, so the decoder's internal read-ahead cannot run
ahead; in the pipeline a dedicated thread calls it back to back and each call
returns sooner. Each worker also stays on one kind of work rather than cycling
between decode, tensor math and inference.

This is a hypothesis with a clear test that has not been run: measure `read`
alone at different call rates. Until then the honest statement is that async
lowered per-stage service time for reasons outside the pipeline structure, and
the latency win should not be attributed to overlap.

## Order costs latency, not throughput

CPU, 360p, inflight 4, 200 frames:

| reorder | cap | order violations | q_reorder mean | throughput |
|---|---|---|---|---|
| off | -- | **138 / 200** | 0.06 ms | 108.0 |
| on | 8 | 1 | 6.97 ms | 101.8 |
| on | 64 | **0** | 7.84 ms | 101.4 |

With reordering disabled, 69% of frames are emitted out of submission order
while the frame count is exactly right. That is the defect DEEPX's
`FrameReorderBuffer` exists to prevent, reproduced and quantified: counting
frames would not notice it.

Restoring order costs about 7 ms of mean latency (p95 ~33 ms) and no measurable
throughput.

The `cap` is a liveness guard, not a tuning knob: past it the buffer releases
out of order rather than stall forever on a gap that may never fill. A cap of 8
against a measured maximum displacement of 13 fired that path three times and
left real violations behind, so the default is now
`4 x (inflight + queue_size)`. Displacement exceeds the in-flight limit because
a gap must also survive the output queue and the postprocess worker.

## Two bugs the instrumentation caught

- **`read` reported 0.00 ms.** The timer started after the generator had
  already decoded. The same mistake as in W1: time the pull, not the loop body.
- **Four frames of 200 were silently dropped.** The end-of-stream sentinel was
  pushed as soon as submission finished, so it overtook requests still in
  flight and their completions arrived after the consumer had shut down. The
  count came out at 196 and nothing looked wrong. Fixed by reacquiring every
  in-flight permit before the sentinel goes out, which proves every completion
  callback has run.

Both belong to the failure class this project keeps meeting: the count is
almost right and nothing looks wrong.

## In-flight depth: throughput saturates, latency does not

NPU, 360p, queue sizes fixed at 4, **10 s measured per point**, 3 runs each:

| depth | FPS | CV | x vs 1 | E2E p50 | E2E p95 | q_inflight | inference p50 | resident |
|---|---|---|---|---|---|---|---|---|
| 1 | 226.7 | 2.7% | 1.00 | 28.1 | 45.9 | 2.35 | 3.52 | 6.96 |
| 2 | 363.3 | 1.8% | 1.60 | 21.1 | 25.8 | 1.45 | 5.07 | 7.95 |
| **4** | **383.8** | 0.3% | **1.69** | **25.8** | 28.7 | 1.05 | 10.14 | 10.08 |
| 8 | 383.6 | 0.5% | 1.69 | 38.5 | 47.9 | 1.24 | 20.39 | 15.12 |
| 16 | 386.8 | 0.2% | 1.71 | 74.1 | 104.1 | 1.24 | 36.27 | 29.65 |

```
 1 -> 2    throughput +60.3%  (needs >3.7%)   latency -25.1%   gain
 2 -> 4    throughput  +5.6%  (needs >2.1%)   latency +22.5%   gain
 4 -> 8    throughput  -0.0%  (needs >2.0%)   latency +49.4%   NOISE
 8 -> 16   throughput  +0.8%  (needs >2.0%)   latency +92.2%   NOISE
```

**The knee is 4.** Going on to 16 buys +0.8% throughput for **2.87x the
latency**. Depth is not a free dial: past saturation the extra frames are
queueing, not being worked on.

Two independent confirmations of the mechanism:

* `inference` p50 grows almost linearly with depth (3.52 -> 36.27 ms). That is
  not the model getting slower; it is each request waiting its turn inside the
  runtime. The stage timer spans submit to completion, so queueing inside ONNX
  Runtime lands in it -- the same reason `get_latency()` and
  `get_npu_inference_time()` are separate calls in DEEPX's SDK.
* `resident` tracks depth (6.96 -> 29.65) exactly as Little's Law requires.

`q_inflight` stays near 1 ms from depth 2 upward: the semaphore stops binding
almost immediately, so beyond the knee the limit is not what holds frames back
-- the slowest stage is.

**Depth 1 is not a sequential baseline.** Its E2E is the *worst* of the whole
sweep (28.1 ms) because the queues still hold four frames each while only one
request may be outstanding, so frames sit waiting for the single slot
(`q_inflight` 2.35 ms, resident 6.96). For a true sequential comparison use
`run_ab`, not depth 1.

### The same sweep at 1080p finds no knee at all

| depth | FPS | E2E p50 | q_inflight | resident |
|---|---|---|---|---|
| 1 | 160.0 | 40.2 | 3.42 | 6.80 |
| 2 | 183.6 | 16.5 | 0.00 | 2.79 |
| 4 | 187.9 | 14.9 | 0.00 | 2.87 |
| 8 | 190.9 | 17.8 | 0.00 | 3.26 |
| 16 | 178.0 | 13.8 | 0.00 | 2.85 |

Only 1 -> 2 clears the noise. From depth 2 upward `q_inflight` is **0.00 ms**
and `resident` sits near 2.8 whatever the limit says: fewer than two requests
are ever actually outstanding, because 1080p decode cannot feed them. The
pipeline is producer-bound, and an in-flight limit that never binds cannot
shape anything.

So the textbook curve requires the limit to bind. Whether depth is worth tuning
is a property of the workload, not of the pipeline.

## Measurement duration changed the answer's quality

The first version of this sweep used a fixed frame count (300 frames, 220
counted), which at these rates is **0.55-0.98 s of measurement per run** -- and
a different duration at every depth, so the fast points got less thermal
exposure than the slow ones.

Switching to a fixed 10 s per point cut the run-to-run CVs from
5.9 / 10.7 / 0.4 / 0.6 / 3.2% to **2.7 / 1.8 / 0.3 / 0.5 / 0.2%**. The clip is
300 frames, so a 10 s point replays it 12 times; `restarts` is recorded with
every result.

Looping is the standard approach and does not weaken the measurement: the
decode work is performed in full every time, and only the file read is served
from the page cache, which removes disk noise rather than adding any. MLPerf's
loadgen works the same way -- replay the sample set, but enforce a minimum
duration.

## Not established

- **Why per-stage service time falls in async mode** (see above). The decoder
  read-ahead hypothesis is untested.
- **That looping is equivalent to a single long pass.** A 10-minute source
  played once would settle it; not run.
- **GPU.** Only CPU and NPU have been compared this way. The Arc iGPU shares
  silicon with video decode (W1 finding 3), so its async behaviour is not
  predictable from either.
- **The CPU numbers use ONNX Runtime's own CPU provider**, not OpenVINO's, so
  the CPU-versus-NPU rows differ in engine as well as device.
