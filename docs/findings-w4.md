# W4 findings -- the C++ port

Phases and dependencies: [`w4-plan.md`](w4-plan.md), [`w4-setup.md`](w4-setup.md).

## Phase 0 -- the toolchain, cross-checked

Built with MSYS2 UCRT64 GCC 16.2, no Microsoft software involved, against
OpenVINO's **C** API because MinGW and MSVC share no C++ ABI. The probe loads
the model, enumerates devices and runs one inference:

```
OpenVINO 2025.4.1-20426-82bbf0292c5
devices: CPU GPU NPU
input    [1, 3, 224, 224]  ->  1000 logits
top-5    754:0.1191 905:0.0660 753:0.0574 904:0.0365 921:0.0264
```

The same synthetic input through the Python OpenVINO API and
`bench.pipeline.postprocess` gives those five values exactly. **The C API path
and the C++ softmax agreed with Python before a single stage was timed.** The
GPU returns 754:0.1202 against the NPU's 0.1191, the same 4-6e-3 spread
between accelerators that W3 measured.

## Phase 1 -- the sequential pipeline

NPU, 360p, 10 s per run, paired and alternating in one session
(`bench/compare_impl.py`):

| stage | cpp ms | python ms | cpp/python |
|---|---|---|---|
| read | 0.511 | 0.676 | 0.76 |
| preprocess | 1.095 | 1.093 | **1.00** |
| inference | 6.980 | 7.088 | **0.98** |
| postprocess | 0.016 | 0.148 | **0.11** |

**The cross-validation is the `inference` row.** It is the same model on the
same silicon and neither implementation performs it, so the two must agree
there or one of them is mistiming and nothing else means anything. They agree
to 2%.

> **Corrected later.** That 2% was one admissible pair, and it was a
> coincidence. Six counterbalanced pairs put the ratio at **0.86 with a CV of
> 3.2%**: the two do not dispatch the inference the same way, Python going
> through ONNX Runtime's OpenVINO provider and C++ through OpenVINO directly.
> The row is still the control, but what a control has to hold is its spread.
> See [the re-measurement](#the-sequential-comparison-re-measured----and-three-faults-in-the-instrument).

The rest is what the port was expected to show and roughly the size expected:

* **`postprocess` 0.11x** -- a softmax and a top-5 over 1000 floats. NumPy
  allocates several temporaries to do it; the C++ does one pass and a partial
  sort. Nine times faster, and 0.13 ms saved, which is the honest way to state
  it: large as a ratio, small as a number.
* **`read` 0.76x** -- the same OpenCV `VideoCapture` either way. The difference
  is the Python call overhead around it.
* **`preprocess` 1.00x** -- identical, because both call the same OpenCV
  primitives in the same order. That is the baseline phase 3's fused path has
  to beat, and it is a fair one.

> **Corrected later.** Also one pair. Six counterbalanced pairs put it at
> **0.66 (CV 2%)**, and a five-pair run at 0.35. The C++ preprocessing has not
> changed a line since phase 0 -- `git diff` confirms it -- yet the same code
> measures 0.24 ms on a cool machine and 1.02 ms on a warm one. The 1.00x was
> both implementations caught at the warm end of that range on the same
> afternoon.

Throughput is **not** reported. Three of four pairs drifted outside 0.95-1.05x
and were excluded, leaving one usable pair, and one pair has no spread: its CV
is 0.0% and every steadiness test passes trivially. The tool says "an
observation, not a measurement" rather than printing 1.05x as though it were
supported.

### A check that was wrong, and what replaced it

The first version cross-validated on stage **shares** and flagged a 6.8
percentage-point difference as an instrumentation fault. That was backwards.
Shares are normalised by E2E, so an implementation that genuinely speeds up the
host stages necessarily shifts every share -- the check would report any real
improvement as a defect.

The criterion is now the absolute time of the stage neither implementation
performs. Shares are still printed, because they are informative, but a
difference in them is not evidence of anything.

## Phase 2 -- the overlapped pipeline

NPU, 360p, 10 s, in-flight 4:

```
 preprocess     1.22 ms        q_read      0.14 ms
 inference     21.28 ms        q_inflight  5.29 ms   (18.5%)
 postprocess    0.04 ms        q_output    0.05 ms
                               q_reorder   0.04 ms
 E2E           28.64 ms        throughput  169.72 FPS
 resident       4.86 frames (Little's Law, in-flight 4 plus queues)
 order         emitted 1789  violations 0  IN SUBMISSION ORDER
```

Against the C++ sequential run on the same device and clip (~142 FPS), overlap
buys throughput and costs latency, the same shape W2 measured in Python.
`inference` rises from ~7 ms to 21 ms because four requests share one NPU --
the per-request cost of concurrency, visible here as it was there.

### Completions arrive in order, so the reorder buffer never fires

`out_of_order_arrivals 0, max_displacement 0`. The Python pipeline saw 152 of
200 arrive out of order on the same clip.

The structures differ. In C++ the completion callback does nothing but stamp
the time and hand the slot to one postprocess worker, which drains its queue in
FIFO order; OpenVINO completes the requests in submission order, so nothing
ever overtakes. In Python the postprocessing ran off ONNX Runtime's completion
path and genuinely raced.

**So the reorder machinery here is correct but untested by this workload.** The
Python side has `--stall-every` to inject an asymmetry and force the case; the
C++ side does not yet, and until it does, "0 violations" says less than the same
words said in W2.

### Three bugs, and the one that matters

- **The callback descriptor was a stack local.**
  `ov_infer_request_set_callback` takes a `const ov_callback_t*` and keeps the
  pointer, so passing the address of a loop-local left it dangling the moment
  the constructor advanced. With one slot it worked by accident; with two or
  more the pipeline hung. Fixed by giving each slot its own descriptor. This is
  exactly the ownership hazard the RAII wrapper exists to surface, and it was
  found by a hang rather than by the wrapper -- worth remembering when phase 3
  starts reusing buffers.
- **The completion callback blocked on a bounded queue.** It runs on an
  OpenVINO thread; blocking there deadlocks the runtime against its own
  `wait`. The output queue is unbounded now and the callback does the least
  possible. Backpressure is not lost -- it lives in the slot pool.
- **`ov_infer_request_get_output_tensor` returns `REQUEST_BUSY` from the
  callback path.** The callback runs before the request leaves its running
  state, so the collector calls `ov_infer_request_wait` first -- which returns
  immediately, the work being done. The API makes the submit/collect split
  explicit whether or not the structure does.

A fourth, of my own diagnosis rather than the code: the first attempts to
narrow the hang reported failures that were actually exit 127, the executable
not finding the OpenCV DLLs. The Windows loader names only the first missing
library, so an OpenCV problem reads as an OpenVINO one. Two rounds of reasoning
went into a hang that had not happened.

## Phase 3 -- tuning, and what the measurements say

Measured at 1080p, where host work is comparable to inference and a host-side
win therefore raises the ceiling. Paired: the two configurations alternate in
one session and the ratio inside each pair is reported
(`bench/compare_variants.py`).

| candidate | stage it targets | B/A | per-pair CV | verdict |
|---|---|---|---|---|
| fused preprocess | `preprocess` | **0.61** | 45% | real, magnitude imprecise |
| input tensor reuse | `inference` | 0.98 | 25% | **not supported** |

**The fused preprocess is faster.** One pass over the resized image instead of
five -- colour swap, scale, normalise and transpose folded into the loop that
writes the NCHW planes. Two independent measurements put it at 0.61x and 0.40x
of the baseline, so the direction is not in doubt even though the machine was
too unsteady to pin the size.

**Reusing the input tensor is not.** Binding one `ov_tensor` over the caller's
buffer at startup, instead of wrapping the same buffer again every frame, moves
`inference` by 2% against a 25% run-to-run spread. That is not a result, it is
a number smaller than the noise around it. The wrap is a shape allocation and a
pointer -- cheap beside a 7 ms inference -- so this is what the hypothesis
deserved rather than a disappointment. The repository said measuring a
hypothesis dead is a result; here is one.

### The measurement that had to be thrown away first

The four variants were run back to back before any of this. That version said
`--reuse-tensor` made *preprocessing* 2.3x faster, which it cannot -- it does
not touch preprocessing -- and showed `inference` swinging 1.9x between
variants that none of the flags affect. It was drift, read as a result.

`compare_variants` therefore alternates the configurations and adds a
**control**: the stage neither side changes. `--fused` must leave `inference`
alone and `--reuse-tensor` must leave `read` alone; if the control moves more
than 10% the pairing has not cancelled the drift and the other rows are
describing the machine. Both runs passed, but `--fused` passed at 1.09 -- close
enough to the bar to be worth saying out loud rather than reporting as clean.

### Every variant still produces the same answers

`verify_equivalence` now accepts the C++ runners' `--dump-topk` output, so the
C++ build is checked by the same comparison, the same self-test and the same
sensitivity figure as the Python. Against the Python sequential reference on
300 frames:

| build | top-1 | top-k order | ties | max prob delta |
|---|---|---|---|---|
| cpp sync | 0 | 0 | 3 | 5.96e-08 |
| cpp + reuse-tensor | 0 | 0 | 3 | 5.96e-08 |
| cpp + fused | 0 | 0 | 3 | 5.96e-08 |
| cpp async, 8 in flight, 4 collectors | 0 | 0 | 3 | 5.96e-08 |

This is the net that had to exist before phase 3 touched a buffer, and it is
why the fused path can be trusted: it rewrote the arithmetic and the answers
did not move.

**A criterion was corrected on the way.** The first run reported all four as
DIFFERS over three frames' top-5 *ordering*, at a maximum probability
difference of 5.96e-08 -- float32 epsilon. Two classes with equal probabilities
are a tie, and which one `std::partial_sort` puts first need not match NumPy's
argsort; neither has a claim to being right about a 6e-08 gap. Ties are now
counted separately from mismatches, the same principle `compare_backends`
already applies to the accelerators.

## Phase 2 addendum: making the reorder path real

W2 recorded that completions arrived in submission order, so the reorder buffer
was correct but never exercised. Injecting a stall did not help: with one
collector draining its queue FIFO, delaying frame 20 delays frame 21 as well
and nothing can overtake **by construction**.

The number of collectors is now a flag, and it is the whole story:

| post-workers | reorder | order violations | out-of-order arrivals |
|---|---|---|---|
| 1 | on | 0 | **0** |
| 4 | on | **0** | 96 |
| 4 | off | **72 / 479** | -- |

One collector cannot reorder and needs no buffer. Four collectors postprocess
concurrently, 96 completions overtake a predecessor, and without the buffer 72
frames of 479 are emitted out of order with the count perfectly correct.

**That is the trade DX-APP makes.** It postprocesses off the completion path,
which is why `FrameReorderBuffer` exists: the buffer is the price of
postprocessing concurrently, not an accident of their design.

## The GIL, separated at last

W2 attributed the CPU async loss to ONNX Runtime subdividing its intra-op pool
between concurrent requests, and said plainly that Python's own orchestration
had never been separated from it. `compare_impl --mode async` runs both
overlapped pipelines alternately in one session, which separates them.

NPU, 360p, in flight 4, 8 s per run, 4 pairs, 90 s cooldown:

| stage | cpp ms | python ms | cpp/python |
|---|---|---|---|
| read | 0.376 | 0.341 | 1.10 |
| preprocess | 0.410 | 0.899 | **0.46** |
| inference | 15.716 | 16.893 | 0.93 |
| postprocess | 0.018 | 0.183 | **0.10** |

```
throughput  cpp/python  x1.04   (min 1.00, max 1.09, CV 5.8%)
```

**Python's orchestration costs 4%.** The host stages are two to ten times
faster in C++ -- preprocessing 2.2x, postprocessing 10x -- and it buys four
percent of throughput, because inference is 77% of the C++ frame and the device
does not care which language asked.

So **W2's attribution survives.** The CPU async loss was ONNX Runtime splitting
its thread pool, not the interpreter lock. That mattered to check: "Python is
slow" is the explanation everyone reaches for, and here it would have been
wrong.

The comparison is only about the language because the structures were matched:
the C++ side runs **one** collector, since Python's async runner has a single
postprocess worker and C++ with four would have measured a different design.

### Async mode has no control stage, and says so

Sync mode validates the timers on `inference` -- the same model on the same
silicon, performed by neither implementation, agreeing to 2%. Async mode has no
equivalent. `read` was tried and came out at 0.44x: the Python reader decodes
while three other Python threads contend for the interpreter lock, so the stage
that "neither implementation changes" is changed by exactly the effect under
test. The interpreter lock reaches every host stage; that is what makes it
worth measuring and what leaves nothing to hold still. The tool reports the
absence rather than picking a control it would then have to explain away.

## C++ sequential against C++ overlapped

Phase 2 quoted 142 FPS and 169 FPS from separate runs, which on this machine is
not a comparison. Paired, NPU, 360p, in flight 4, one collector:

```
throughput  async/sync  x1.32   (min 1.14, max 1.65, CV 17.4%)
```

Consistent with the Python result in the same configuration (1.44x at 360p on
the NPU), and with W1's overlap ceiling for that cell (1.39x).

`read` came out at 0.83x in the overlapped runner -- the same direction as the
9.75 to 5.05 ms drop W2 recorded in Python and left unexplained, though much
smaller here. It is a second data point for that open question rather than an
answer to it.

The control check does not apply to this pairing and is switched off by
`--control-stage none`. Comparing a sequential runner with an overlapped one
changes every host stage by construction: a dedicated reader thread decodes
continuously where an inline one decodes between inferences. A control that the
comparison itself moves reports drift that is not there, which is what the
first run of this did at `read` 0.83x.

## The sequential comparison, re-measured -- and three faults in the instrument

Phase 1 reported the C++/Python stage ratios from **one admissible pair**, and
declined to report throughput at all. Closing that was meant to be a tighter
number. It produced three defects in the measuring instead, and they are worth
more than the number was.

### The ratios were not formed inside the pairs

`compare_impl` divided a **median of C++ runs by a median of Python runs**. The
two halves of every ratio came from different moments on a machine that moves,
which is the one thing pairing exists to prevent. Throughput had been paired
correctly since W2; the stage table never was.

Re-reading a five-pair run under the corrected arithmetic:

| stage | cpp/python | per-pair range | CV |
|---|---|---|---|
| read | 0.55 | 0.39 - 0.79 | 39% |
| preprocess | 0.35 | 0.20 - 0.50 | 44% |
| **inference (control)** | 0.88 | **0.72 - 1.01** | 17% |
| postprocess | 0.09 | 0.04 - 0.11 | 43% |

**The control swings 40%** on the one stage neither implementation performs.
Aggregated first it reads 0.92 and passes; the fault is invisible to any check
applied after averaging, which is where the check was.

### C++ ran first in every pair

The per-pair control slid **1.01, 0.90, 0.86, 0.72, 0.83** down one session --
a shape no implementation can produce. Whatever warms up between the halves of
a pair was charged to Python every time, and *a paired design cannot cancel a
bias it applies uniformly*. The order now alternates.

**The hypothesis was wrong.** Six counterbalanced pairs:

```
cpp first    1.254  1.362  1.207   mean 1.274
python first 1.333  1.293  1.242   mean 1.289
```

1.2% apart against a 4.6% spread. The slide was session-long drift, not order.
The fix stays because the design was indefensible either way, but it bought
nothing, and that is the result.

### `inference` is not the control it was described as

Six counterbalanced pairs put it at **0.86 with a CV of 3.2%** -- and the
five-pair run agrees on the level. A ratio that reproduces to 3% is not a timer
fault and not drift. It is a constant, and its cause was already written at the
top of the tool: **Python reaches the NPU through ONNX Runtime's OpenVINO
provider, C++ through OpenVINO directly.** Different runtimes to the same
silicon.

So Phase 1's "they agree to 2%, so the timers are validated" was **one pair's
worth of coincidence**. The two paths differ by 10-16% on the inference stage,
reproducibly.

A control detects the machine *changing* between the halves of a pair, so what
it has to hold is its **spread**, not its level. The check is now on the CV, and
the level is reported as a finding: the OpenVINO EP costs 10-16% over the C API
on this path, which is a second implementation's worth of evidence about
something W2 could only assume.

### What the throughput ratio actually is

| run | pairs | throughput | control |
|---|---|---|---|
| counterbalanced | 6 | **1.28** (1.21 - 1.36, CV 4.6%) | 0.86, CV 3.2% |
| cpp-first | 5 | 1.47 (1.23 - 1.83, CV 21.5%) | 0.86, CV 16.8% |

The second's control moved 17%, so its spread is the machine. The first's holds
to 3%, but only **one of its six pairs** met the steady-state rule, because by
then the laptop had been measuring for hours -- so the 1.28 is quoted from all
six pairs with that said plainly rather than promoted to a headline.

**C++ sequential runs somewhere near 1.3x Python's throughput**, and the stage
table says why: the C++ frame is 4.60 ms against 6.14 ms, which predicts 1.34.

### The number that matters is the contrast with the overlapped case

```
sequential   cpp/python  x1.28      host work is on the critical path
overlapped   cpp/python  x1.04      host work is hidden behind the device
```

**Python's overhead costs ~25% sequentially and 4% overlapped.** It does not
shrink -- the host stages are just as much faster in C++ either way. Overlap
moves them off the critical path, and a cost that is not on the critical path
is not a cost. That is the same result W2 reached about the pipeline, arriving
from the other direction: *the reason to overlap is not that the host is slow,
it is that the host stops mattering.*

### Saved runs can be re-read

Every criterion above was corrected **after** the runs it applies to. All four
paired comparisons in this repository were re-read under the corrected ones
without spending another minute on the machine, because the pairs are saved in
full: `compare_impl --reanalyse results/....json`. The criteria here have been
corrected four times; this is the first time that cost nothing.

Re-reading also confirmed what **not** to change. A version that dropped the
steady-state rule entirely was tried, on the grounds that it rejected five of
six pairs while the control held; re-reading the GIL run under it surfaced two
pairs with throughput ratios of 0.37 and 0.65 beside the 1.04 from the pairs
that held. Those are broken runs, not warm ones. The rule stayed, and the
figures over every pair are now printed beneath the admissible ones so the cost
of admissibility is on the page instead of in a decision nobody can see.

## Closing W4: phase 3 re-read, and an experiment that answers itself

Two things were left open. `compare_variants` had the same two faults the
sequential comparison found -- it ran A first in every pair and judged its
control by level rather than spread -- so phase 3's numbers were read under
criteria since retracted. And `--fused` had only ever been measured in the
sequential runner.

Both are closed now. Neither closed the way it was expected to.

### Phase 3 does not survive the corrected criteria

The corrections were carried across and `--reanalyse` applied them to the pairs
already on disk, at no cost in machine time:

| run | what it claimed | control | control CV | verdict |
|---|---|---|---|---|
| `--fused` | preprocess 0.61x | `inference` | **35%** | fails |
| `--reuse-tensor` | inference 0.98x | `read` | **15%** | fails |

Under the old criterion -- is the control's *level* near 1.00? -- both passed,
at 1.09 and 0.98. Under the corrected one both controls move further than the
bar allows, which means the pairing did not cancel the machine and every row
carries that movement as well as the flag. **The level was fine and the spread
was not**, which is exactly the failure the sequential comparison was rebuilt
to catch after being fooled by it once.

The effect on the two conclusions is not symmetrical:

* **`--reuse-tensor` "not supported" survives, and strengthens.** It was
  already a 2% effect against a 25% spread. The control now adds that the run
  could not have detected a real effect either. An absence of evidence is
  exactly what a run of that quality is entitled to report.
* **`--fused` 0.61x loses its support.** The direction still rests on two
  independent measurements (0.61 and 0.40) and on the mechanism -- one pass
  over the resized image instead of five, which cannot be slower -- but the run
  that produced the number no longer clears the bar.

A counterbalanced re-measurement was run to replace it. **It failed harder**:
six pairs at 1080p, control at CV 49%, per-run throughput swinging from 107 to
370 FPS inside one session. After a day of measuring, the laptop could no
longer support a 1080p paired comparison. That is the honest description, and
it is preferable to a number quoted from it.

### The async question answers itself, exactly

`--fused` in the overlapped runner was meant to ask whether removing host work
from the critical path raises throughput. It turns out not to need a
measurement at all.

In a **sequential** runner a frame costs the *sum* of its stages, so any stage
that gets cheaper makes the frame cheaper. In a **pipelined** runner the
steady-state rate is set by the *slowest* stage; the others run concurrently
and their savings have nowhere to go. That is Amdahl's law in the form a
pipeline actually takes, and it turns the question into arithmetic on a table
that already exists.

Service time per frame at each stage -- the stage's own time divided by how
many servers it has -- on the NPU at 360p, four in flight, one collector, taken
from the GIL run whose per-stage CVs were 1-6%:

```
read          0.376 ms       one reader thread
preprocess    0.410 ms       one worker
inference    15.716 / 4  =   3.929 ms    four in flight
postprocess   0.018 ms       one collector
                             bottleneck: inference, ceiling 255 FPS
```

Preprocessing is **10% of the bottleneck**. Halving it leaves the bottleneck
exactly where it was:

> **best possible throughput gain: x1.0000.**

Not "about one", not "within noise". Preprocessing would have to grow **9.6x**
before halving it could buy a single frame per second.

At 1080p, where host work is much larger, the answer is the same for a more
interesting reason:

```
read          1.208 ms
preprocess    0.609 ms
inference     4.935 / 4  =   1.234 ms
```

Read and inference land **within 2% of each other**: the pipeline is
decode-bound and device-bound at once. Preprocessing reaches 49% of the
bottleneck here -- far closer than at 360p -- and still does not bind, so the
gain is again **x1.0000**. What would help at 1080p is a second decoder thread.
That is a different experiment, and a better one.

The conclusion is robust in the safe direction. Dividing inference by the
in-flight count assumes the device really delivers four-way concurrency; if it
delivers less, the inference service time is *larger* and preprocessing is even
further from binding. The ceiling is an upper bound the stage table permits,
not a prediction -- measured throughput sits below it -- but the *gain* is
what is being computed, and it does not depend on reaching the ceiling.

| cell | preprocess as a share of the bottleneck | best possible gain from 0.56x preprocess |
|---|---|---|
| sequential, 360p | 2.5% of the **sum** | x1.011 |
| sequential, 1080p | 9.0% of the **sum** | x1.041 |
| overlapped, 360p | 10% of the **max** | **x1.0000** |
| overlapped, 1080p | 49% of the **max** | **x1.0000** |

So the measurements run to answer this could not have answered it. Their
throughput ratios came back at CV 38% and 41%, for an effect whose true size is
**zero**. This experiment was not underpowered by a little.

### What that is worth

This is the sharpest thing W4 produced, and it is the GIL result arriving from
the other side.

> **A host-side optimisation that is real sequentially can be provably
> worthless once the pipeline overlaps -- and the stage table says so before
> anything is measured.**

Python's orchestration costs 25% sequentially and 4% overlapped. Fused
preprocessing is worth 4.1% sequentially and exactly 0% overlapped. Both are the
same fact: **overlap does not make host work cheaper, it makes host work stop
counting**, and a stage that is off the critical path has no cost to save.

It also changes what the fused path is *for*. It is not a throughput
optimisation in this pipeline and cannot be made into one by measuring it
better. It is a latency and memory-traffic optimisation -- it removes three
intermediate buffers and roughly 1.8 MB per frame of host memory traffic -- and
those are terms [W5](w5-plan.md) is built to report, where they are counted
rather than raced for.

The cheapest lesson is the practical one. **Compute the ceiling from the stage
table before spending an hour of machine time.** Both fused measurements cost
about an hour each; one line of arithmetic on data already sitting in
`results/` would have said in advance that one of them had nothing to find.

## Still to do

- **A machine that holds still.** Every pair in every phase-3 run, original
  and re-measured, left the 0.95-1.05 band. Stage ratios survive that far
  better than rates do, which is why they are what gets quoted.
- **A tight number for the fused preprocess**, which it still does not have.
  The direction is not in doubt; the size needs a steady machine, and the
  ceiling calculation above says to measure it as memory traffic and latency
  rather than throughput, because throughput cannot move.
- **A second decoder thread at 1080p**, which the same calculation says is the
  optimisation that would actually bind there.
- **The OpenVINO EP's 10-16%.** The control's level says ONNX Runtime's
  provider costs that much over the C API per inference on this path. It is
  reproducible across three runs and entirely unattributed: it could be the
  provider's binding, an extra copy, or the way either timer brackets the call.
  Portable to the MX150, and it needs no accelerator to chase.
