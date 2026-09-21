# Does `read` get cheaper when it is called more often?

W2 measured `read` at 9.75 ms in the sequential pipeline and 5.05 ms in the
overlapped one, for identical work. Overlap cannot shorten a frame's own service
time, so that pair of numbers has sat in [`findings-w2.md`](findings-w2.md)
unattributed, and with it the 0.83x end-to-end latency figure that leans on it.
[`handoff/open-threads.md`](../handoff/open-threads.md) called this the
highest-value portable thread in the project, because the test either attributes
the number or retracts it.

The standing hypothesis was decoder pipelining: FFmpeg decodes H.264 with
several threads, those threads overlap across *calls*, and a caller that returns
immediately keeps them busy while a caller that waits 19 ms pays a full
single-frame latency each time.

`bench/read_rate.py` tests it by **turning the proposed cause off**, which is
stronger than measuring the effect again. The decoder's thread count reaches
FFmpeg through `OPENCV_FFMPEG_CAPTURE_OPTIONS`, so the tool runs each setting in
its own process and compares.

## The answer, and its strength

**The measurement never held its control, across six attempts, so no ratio is
quoted.** What can be said is weaker than an attribution and stronger than
nothing:

> No configuration produced an effect near the size the hypothesis has to
> explain. W2's numbers require a **48%** drop. With the gap filled the way a
> sequential pipeline actually fills it, the largest effect seen was **14%**,
> and one setting showed **1%**. Even taking the uncontrolled figures at face
> value, call cadence does not account for W2's `read`.

So the attribution is withdrawn. `findings-w2.md`'s falling `read` time returns
to being **unexplained**, which is a different state from *explained by
read-ahead* and the honest one. The tool stays, and will answer the question on
a machine that can hold still.

One thing the sweep did establish, because it does not need the paired control:
the direction is not even consistent. At 360p back-to-back reads are clearly
cheaper than spaced ones (0.27 ms against 0.50 ms); at 1080p the two are within
noise of each other and the sign flips between runs. A mechanism that reverses
with resolution is not the mechanism behind a single 48% figure.

## Six attempts, and what each failure was

The interesting part of this phase is not the decoder. It is that a control
failed six times and each failure had a different, nameable cause.

| attempt | control | control CV | what it was |
|---|---|---|---|
| 1 | 64 MB memcpy, unpinned | 35-42% | the control competed with the decoder |
| 2 | integer loop, unpinned | 24-49% | hybrid-CPU core migration |
| 3 | integer loop, pinned to 0-5 | 39% | the wrong core group |
| 4 | integer loop, pinned to 6-11, short window | **5%** | passed, over 3 pairs |
| 5 | same, 8 pairs | 23-35% | the window got longer |
| 6 | warmed and medianed, 10 pairs | 21-28% | still the window |

### 1. The control competed with the thing under test

The first control copied 64 MB. Memory bandwidth is exactly what a video decoder
consumes, so the control was not measuring the machine, it was measuring the
experiment. **A control must be affected by the machine and by nothing else**,
and this repository's own definition says it detects the machine *changing*
between the halves of a pair. One that moves when the condition moves cannot do
that.

### 2. The hybrid CPU, arriving in a different experiment

Replacing it with a cache-resident integer loop, no memory traffic and no
library threads, still gave 24-49%. That is enormous for a deterministic
instruction count, and it is [W1 finding 4](findings-w1.md) reappearing: the
255H has performance, efficient and low-power-efficient cores, and an unpinned
thread lands on different ones between two measurements seconds apart.

W1 attributed the two-level CPU timings in the original grid to exactly this and
W2 confirmed it by pinning. Here it arrived uninvited, in a measurement that has
nothing to do with inference, which is the strongest support that claim has yet
had.

### 3. Not all core groups are equal

Pinning to CPUs 0-5 gave 39%; pinning to 6-11 gave 5%. Same machine, same
moment, same code. The tool takes the group as an argument rather than picking
one, because which group is quiet is a property of the machine and the day.

### 4-6. The window, and a power state

Pinned, the control passed at 5% over three pairs and degraded to 23-35% over
eight or ten. The degradation tracks the length of the window, which is what
[`protocol.md`](protocol.md) already records: the clock falls from 2000 to
1500 MHz within a single run, four runs out of five.

One part of that had a specific and fixable cause. The control was being timed
immediately after a condition that spends most of its time asleep, and **a core
that has been idle is in a low power state, so the first milliseconds of the
control measure the climb back out.** Spinning first, untimed, then timing, then
taking a median of three, moved it from 23-35% to 21-28%. Better, and not
enough.

## Why the control failed, as far as it can be named

Ordered by how much of the failure each one explains, with what the evidence is.

**Core migration on a hybrid part.** Demonstrated: pinning changed the control
from 24-49% to 5%. This is the largest single contributor and the only one that
was fully removed.

**Power-state recovery after an idle condition.** Demonstrated by the fix
helping. An experiment whose conditions differ in how much they *sleep* will
find its control differing in what P-state it starts from, and the control will
then report a difference the conditions did not cause.

**Frequency drift across the window.** Demonstrated by the degradation with
window length and already characterised in `protocol.md`. This is the residual
and it is the one that was not solved.

**Shared resources between the control and the decoder, even when the control
touches no memory.** This is the part worth stating carefully, because the first
fix only addressed half of it. Moving the control off memory bandwidth removed
the cache and the memory controller from the coupling. It did not remove the
**package power budget** or the **frequency domain**. FFmpeg decoding 1080p on
several cores raises package power, and on a laptop that pulls every core's
frequency down, including the pinned one running the control. The control is
therefore never fully independent of the condition on a thermally limited part;
it is only less dependent than a memcpy was.

**Background work.** `protocol.md` records this machine idling near 19% CPU,
which is why the idle gate's 15% default cannot be met without closing
applications. A single background wake inside a 20 ms sample is a large relative
error.

**The control's own length, which is a genuine tension.** Twenty milliseconds is
long enough for the timer and short enough that one descheduling dominates.
Making it longer steadies it and lengthens the window, and a longer window is
what item three says makes everything worse. There is no length that is right
for both.

### What an NPU would have done to this

It would have made the coupling worse rather than better, and the reason is
already measured in this repository.

On an integrated part the NPU is a separate block, but it is not a separate
machine: it shares the memory controller and the package power budget with the
cores. [W1 finding 3](findings-w1.md) measured the extreme version of this on
the iGPU, where inference and video decode are literally the same silicon: at
1080p the GPU's `read` doubled to 7.70 ms while the other three backends sat
between 3.86 and 4.78 ms. The NPU landed in between, which is what a separate
block that still shares memory looks like.

So an NPU-backed control would inherit that coupling plus the one
[W1 finding 2](findings-w1.md) identified separately: the stage timer around
`session.run()` spans copy, submit and **wait**, so an "NPU" measurement is
partly a host measurement and moves with whatever else the host is doing.

**This experiment used no accelerator precisely to remove that coupling, and it
still was not enough.** That is the useful thing the failure says: the shared
resource that defeated the control here is not bandwidth or cache, both of which
were engineered out. It is frequency, and frequency is shared by everything on
the package whether or not the work is.

## What would settle it

**Measure in cycles rather than in seconds.** Every cause above except core
migration acts through the clock. A read timed in cycles and a control timed in
cycles both shrink and grow together when the frequency moves, and the ratio
survives. That is one change to the instrument and it would make this machine
usable for this question.

**Or take the question to a machine that holds a clock.** The tool takes
`--pin`, `--compare-threads` and `--paired-gap-kind`; nothing about it is
specific to this laptop.

Until one of those happens, `findings-w2.md`'s falling `read` time is recorded
as unexplained, and the 0.83x latency figure it supports carries that caveat
rather than an attribution.

## Running it

```
.venv\Scripts\python.exe -m bench.read_rate --compare-threads 16,1 --pin 6,7,8,9,10,11
.venv\Scripts\python.exe -m bench.read_rate --source clips/bbb_360_2MB.mp4 --gaps 0,5,10,19,30
```

The paired test fills the gap with work by default, because that is what a
sequential pipeline puts there. Comparing against an *idle* gap measures the
CPU's power state as much as the decoder's behaviour, which is finding 4 above
turned into a default.

Raw data in `results/readrate-1080.json`, `results/readrate-1080-pinned.json` and
`results/readrate-1080-busy.json`, kept with their control failures rather than
discarded.
