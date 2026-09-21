# Reproductions

Every defect found while building this repository, as a program you can run.

The findings documents describe these defects in prose. Prose is not a
reproduction: it cannot tell you whether the bug is still there, whether a
rewrite reintroduced it, or whether the fix was ever necessary. These can.

## The contract

Each program takes `--mode broken` and `--mode fixed`, and **exits 0 when it
behaved the way that mode predicts**. So there are three outcomes, not two:

| status | meaning |
|---|---|
| `ok` | broken reproduced the defect, fixed did not |
| `NOT REPRO` | the broken mode came out clean. The condition is no longer being exercised here — **not** the same as "the code is safe", and not a reason to drop the fix |
| `REGRESSION` | the fixed path failed. This is the one to act on |

Nothing blocks forever: every wait is bounded and a timeout is reported as a
result rather than left for someone to notice.

```bash
python repro/run_all.py                  # everything runnable here
python repro/run_all.py --device NPU     # the OpenVINO ones on the NPU
python repro/run_all.py --skip-openvino  # the portable subset only
python repro/run_all.py -v               # with each program's own output
```

Current status on the gram (GCC 16.2, OpenVINO 2025.4.1, Core Ultra 7 255H):

```
 cpp/01_dangling_callback        NOT REPRO
 cpp/02_callback_blocks_queue    ok
 cpp/03_request_busy             NOT REPRO
 cpp/04_sentinel_overtakes       ok
 cpp/05_reorder_off              ok
 stats/01_unpaired_ratio         ok
 stats/02_order_bias             ok
 stats/03_single_pair            ok
```

## What is here

### `cpp/` — the runtime and threading defects

Built by the main `cpp/CMakeLists.txt`, so `cmake --build build` builds them
with everything else and they cannot drift away from the code they document.
None of them needs OpenCV: a defect that can only be reproduced once a video
decoder is installed is a defect nobody will reproduce.

| | needs | what it is |
|---|---|---|
| `01_dangling_callback` | OpenVINO, a model | `ov_infer_request_set_callback` keeps the pointer it is given, and the descriptor was a stack local |
| `02_callback_blocks_queue` | OpenVINO, a model | the completion callback blocked on a bounded queue and deadlocked the runtime against its own `wait` |
| `03_request_busy` | OpenVINO, a model | the callback fires before the request leaves its running state, so accessors return `REQUEST_BUSY` |
| `04_sentinel_overtakes` | **nothing** | end of stream announced while work was in flight: 198 frames of 200, and nothing looks wrong |
| `05_reorder_off` | **nothing** | concurrent postprocessing emits frames out of order with the count exactly right |
| `06_inflight_overcount` | **nothing** | a concurrency counter bracketed around the frame instead of the lease reports more frames in flight than there are buffers |
| `07_bound_wrong_metric` | **nothing** | `min(resource, structural)` asserted about pipeline occupancy, which it does not bound -- it bounds frames holding a buffer |
| `08_reorder_window_overflow` | **nothing** | a reorder ring sized from that same wrong bound overwrites pending entries and its drain never terminates |
| `09_constructor_cannot_refuse` | **nothing** | an absent device opens successfully, and the guard that was meant to catch it is both compiled out and incapable of helping if it were not |

**04 and 05 need no accelerator and no runtime.** They are the two most likely
to come back in a rewrite and they run anywhere, including the MX150.

### `stats/` — the measurement defects

These matter more than they look. On real hardware there is nothing to check an
answer against: if the instrument says 1.47x and the truth is 1.28x, nothing on
the machine will say so. So the machine is simulated, with the drift and the
bimodality W1 actually measured, and **ground truth is known by construction**.
Then "the estimator is wrong" is a fact rather than a suspicion.

| | what it is |
|---|---|
| `01_unpaired_ratio` | the paired comparison formed ratios *across* pairs: `median(A runs) / median(B runs)`. One number, no spread, so the control check had nothing to test and passed by default — on a machine that had moved 35% |
| `02_order_bias` | A ran first in every pair, so within-pair warming was charged to B every time. A paired design cannot cancel a bias it applies uniformly |
| `03_single_pair` | one surviving pair has a CV of exactly 0.0%, so every quality gate passes trivially. A gate that cannot fail is worse than no gate |

`02` is the useful one to understand. **On the real machine the bias turned out
to be absent** — six counterbalanced pairs gave 1.274 against 1.289 — which
left the experiment unable to distinguish "no bias" from "no sensitivity".
Here the bias is dialled in by hand, so the design can be shown to detect a
15% within-pair drift as a +14.8% error and the counterbalanced version to
recover 1.283 against a truth of 1.280. Without that, counterbalancing was
theatre.

### `loader/check_dlls.py`

Not a defect — a diagnostic that cost more time than any of the real ones.
Every executable failed with exit 127 naming a missing OpenVINO DLL; it was an
OpenCV problem. **The Windows loader resolves imports in order and stops at the
first failure**, so which library it names is an accident of link order rather
than a clue. This enumerates the import table and reports *all* of them.

```bash
python repro/loader/check_dlls.py                      # every exe under cpp/build
python repro/loader/check_dlls.py path/to/thing.exe
```

## The two that no longer reproduce, and why that is written down

`01` and `03` both came out clean on this build, and the table says so.

**`01` is undefined behaviour.** Whether a dangling stack pointer hangs,
crashes or silently works depends on the compiler, the optimiser and the stack
layout on the day — which is exactly what made the original bug expensive, and
exactly why a passing broken mode proves nothing.

So `01` has a third mode. `--mode probe` mutates a descriptor **the program
still owns** after handing it to the runtime, and observes which value the
callback arrives with. No undefined behaviour, so it answers the same way every
time:

```
args set BEFORE the mutation, callbacks received: 0
args set AFTER  the mutation, callbacks received: 1

The runtime read the descriptor at COMPLETION time, so it kept the pointer.
```

That is the diagnostic that turned a mysterious hang into a one-line fix, and
it is what still justifies the fix now that the hang has gone quiet.

**`03` was tried in six configurations** — {CPU, NPU} x {collector thread,
inside the callback} x {4, 16 requests} — and read the output cleanly in all
six. The `-8` was definitely observed during phase 2, so either the condition
is narrower than any of those, or it depended on the state the dangling
descriptor (01) left the request in, which was still present at the time. That
second possibility is untested and is the honest first guess.

The `wait` call stays in the runner either way. The API says accessors on a
running request are invalid; **code that relies on an undocumented grace period
is not code worth keeping**, and `wait` costs nothing there because the work is
already done.

## Where each one came from

| reproduction | written up in |
|---|---|
| 01, 02, 03 | [`docs/findings-w4.md`](../docs/findings-w4.md) — "Three bugs, and the one that matters" |
| 04 | [`docs/findings-w2.md`](../docs/findings-w2.md), and again in the C++ port |
| 05 | [`docs/findings-w4.md`](../docs/findings-w4.md) — "Phase 2 addendum: making the reorder path real" |
| 06, 07, 08 | [`docs/findings-w5.md`](../docs/findings-w5.md) — "Three failures, one shape" |
| 09 | [`docs/findings-w5.md`](../docs/findings-w5.md) — "W5 tooling" |
| stats/01, stats/02 | [`docs/findings-w4.md`](../docs/findings-w4.md) — "three faults in the instrument" |
| stats/03 | [`docs/findings-w2.md`](../docs/findings-w2.md), `bench/run_ab.py` |
| loader | [`docs/findings-w4.md`](../docs/findings-w4.md) — the fourth bug, "of my own diagnosis rather than the code" |
