# Open threads

Everything this repository currently records as unresolved, collected from
`docs/findings-w1.md`, `docs/findings-w2.md`, `docs/protocol.md` and the
in-flight sweep, with one judgement added to each: **does it survive the change
of machine?**

- **portable** — answerable on the 550XBE, no accelerator required.
- **needs a device** — answerable only if `cuda` or DirectML comes up.
- **closing** — the hardware that could answer it goes back next week.

---

## closing — do these before the machine goes

### The CPU row mixes a change of engine with a change of device
`findings-w1.md`, "Not established". `cpu` is ONNX Runtime's own kernels;
`openvino/GPU` and `openvino/NPU` are the OpenVINO provider. Every CPU-vs-NPU
statement in this repository therefore carries an uncontrolled second variable.
One `openvino --ov-device CPU` row removes it. The row is cheap on either
machine, but the NPU and Arc rows it has to sit beside exist only on this one.

### The Arc iGPU has never been through `run_ab`
`findings-w2.md`, "Not established". W2 compares CPU and NPU. W1 finding 3 says
the iGPU is the one backend whose async behaviour cannot be predicted from
either, because inference and video decode are the same silicon. The UHD 620
will not answer this — it is a different generation with a much weaker media
path, so it would be a new question rather than this one.

### `read` across backends at 1080p
`findings-w1.md`, "Not established". Came out at 37% cross-axis against 24%
within-cell, under the 2x bar the report uses, so finding 3 rests on the 360p
and 720p columns plus the mechanism. Re-running it with more repeats on the Arc
is the only way to settle it as asked.

---

## portable — answerable on any machine

### Why per-stage service time falls in async mode
`findings-w2.md`. `read` went 9.75 → 5.05 ms and `inference` 8.06 → 6.16 ms for
identical work, which overlap cannot explain — overlap does not shorten a
frame's own service time. The standing hypothesis is decoder read-ahead: in
sequential mode `VideoCapture.read()` is called once every ~19 ms and the
decoder cannot run ahead, while a dedicated thread calls it back to back.

**The test needs no accelerator**: measure `read` alone at controlled call
rates. This is the highest-value portable thread in the list, because the 0.83x
latency figure in the W2 table is currently unattributed and the test either
attributes it or retracts it.

### Split the inference timer the way `dx_rt` does
The case for it is now made and published; the change itself is not. At 360p,
measured `inference` p50 climbs 3.52 → 36.27 ms from depth 1 to 16 while
throughput is flat (`findings-w2.md`, "In-flight depth"). Nothing on the device
got slower; queue wait moved inside the timed region, because the timer wraps
the call boundary and the call boundary is where requests wait. W1 finding 2
reached the same conclusion from source resolution at +74%; depth makes it 10x.

`dx_rt` exposes `get_latency()` and `get_npu_inference_time()` separately for
exactly this reason. Doing the same here means recording submit time, device
time and completion time as three fields instead of one — with CUDA events under
the CUDA EP, or with ONNX Runtime's own profiler under any EP. It is the single
change that would most improve what the harness can say, and it is a host-side
change.

### The GIL ceiling
`docs/protocol.md`, "Known confounds". The W2 pipeline runs its stages as
threads under one interpreter lock. NumPy and OpenCV release it for their heavy
calls; the orchestration around them does not. Expected to cap scaling before
the C++ port does, and explicitly recorded as something that must be measured
rather than assumed. On a 4-core machine the ceiling arrives sooner, which makes
it easier to see, not harder. This is also the measurement that motivates W4.

### W3 — correctness under overlap
`--stall-every` deliberately delays every Nth frame's postprocess so a lighter
successor overtakes it, which is how the reorder path gets exercised at all. W2
established the defect (138 of 200 frames out of order with reordering off, with
the frame count exactly right) and the cost of fixing it (~7 ms mean latency, no
throughput). What has not been done is varying the asymmetry and finding where
the reorder cap's liveness guard starts firing on purpose. None of it needs a
fast device.

### Does the bimodal CPU behaviour follow the hybrid CPU?
`findings-w1.md` finding 4 attributes the two-level CPU timings to an unpinned
thread pool landing on different core types of a hybrid part, and W2 supports it
— pinning removed the two-level behaviour and cut the paired-ratio CV from 23.8%
to 6.2%. The 8565U has one kind of core, so the attribution makes a falsifiable
prediction: **unpinned runs on the new machine should not be bimodal.** If they
are, the explanation in `findings-w1.md` is wrong. Cheap, early, and it tests a
claim already published rather than adding a cell.

---

## needs a device

### Is there an off-host backend at all
The first question of the first session. `cuda` on sm_61, else DirectML, else
the UHD 620. See [`README.md`](README.md) for the order and the fallback.

### The depth curve on a slow accelerator
`run_sweep` carries over; the stored knees do not — every curve is NPU. The
transferable claim is that **the in-flight limit only shapes the pipeline when it
binds**: at 360p it bound and depth bought 1.69x with a knee at 4, while at
1080p decode could never feed more than two outstanding requests and there was
no knee at all. On the MX150 inference is slow, so the prediction is that the
limit binds at both resolutions and 1080p acquires the knee it lacked here. A
curve that contradicts that is the more interesting outcome.

### The 1080p sweep has not been re-run at fixed duration
Carried from the pre-return list rather than resolved. If it is not done before
the machine goes back, `findings-w2.md`'s 1080p section keeps resting on
0.55-0.98 s of measurement per run, and the 360p re-run showed that duration was
worth 5-10x in CV. Nothing on the new machine can repair it.
