# DX-APP: sync vs async

Reading notes. Paths relative to `DEEPX-AI/dx_app` @ main.

## The async pipeline is five workers, not three

`src/python_example/common/runner/async_runner.py`, module docstring:

```
read_worker -> preprocess_worker -> wait_worker -> postprocess_worker -> render_worker
```

Submission and collection are **different workers**. `_preprocess_worker` calls
`req_id = self.ie.run_async([input_tensor])` and moves on; `_wait_worker`
collects. Collapsing these two into one "inference worker" would block on
collection and stop being asynchronous.

Five `SafeQueue`s, all bounded: `read / reqid / output / render / display`.
DEEPX's own performance notes give the bound as `maxsize=4`.

In-flight depth is tracked explicitly — `_track_inflight_submit` on submit,
`_track_inflight_complete` on collect, accumulating a time-weighted integral
(`inflight_time_sum += inflight_current * dt`) plus `inflight_max`.

## Two metric types, not one

`src/python_example/common/utility/profiling.py`:

| | sync | async |
|---|---|---|
| class | `ProfilingMetrics` | `AsyncProfilingMetrics` |
| total | `total_time` | `end_to_end_time` |
| rate | `get_fps()` | `get_throughput()` |
| extra | — | `add_queue_size()` / `get_avg_queue_size()` |

Queue depth is a *metric*, and only on the async side. Latency and throughput
are modelled as different things at the type level.

`print_sync_performance_summary` emits **Read** as its own row, separate from
Preprocess.

## Shutdown ordering is deliberate

```python
def _set_stop(self, queues):
    self._stop_event.set()          # before the queues
    ...
```

`_enqueue` loops on `while not self._stop_event.is_set(): q.put(timeout=0.1)`,
and `SafeQueue.put()` returns `False` immediately once the queue is stopped.
Stopping the queues first and setting the event second would busy-spin the
workers at 100% CPU.

`SafeQueue.get()` returns `None` for both timeout and stopped, which is why
end-of-stream is a distinct `_SENTINEL = object()` — `None` could not
distinguish the two.

`_push_sentinel` waits up to 5s for room rather than evicting, because evicting
"silently dropped the tail frames of a stream"; only after the grace period does
it force room, so shutdown cannot deadlock either.

## The correctness problem: completion order

`src/cpp_example/common/utility/frame_reorder.hpp` (80 lines), verbatim:

> RunAsync completions do NOT preserve submission order — postprocess runs in
> the callback thread, so a heavier frame can finish after a lighter successor
> — so without this buffer a saved video comes out locally shuffled inside the
> in-flight window, which reads as jitter on playback. **Frame count stays
> correct, so it is a silent defect.**

`FrameReorderBuffer` keys a `std::map` on a monotonic submit index and emits in
order, force-flushing the lowest entry once more than `cap` items are held so a
never-arriving index cannot stall display forever. Strict ordering against
bounded memory and guaranteed progress — the same shape of trade-off that
`_push_sentinel`'s grace period makes.

## Open questions

- [ ] There is no `frame_reorder` on the Python side. How does `render_worker`
      handle completion order — or does the same silent defect remain there?
- [ ] `_set_stop` drains every queue and *then* calls `_push_sentinel`, whose
      documented contract is to not discard queued items. Abort path and normal
      end-of-stream path share one helper with different requirements —
      intended?

## What this project reproduces

| DX-APP design | W1 | W2 | Why |
|---|---|---|---|
| Read timed separately | yes | | adding it later invalidates W1 |
| sync/async metrics separated | yes | | `total` vs `end_to_end` named apart from the start |
| bounded queues | | yes | depth is the independent variable |
| submit/collect split | | yes | this is what makes it async |
| queue depth as a metric | | yes | DEEPX treats it as one |
| frame reorder | | yes | measure with it and without it |
| Factory / Interface / Visualizer | no | no | one model needs no abstraction |
