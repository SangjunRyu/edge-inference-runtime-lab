"""W3 -- does the overlapped pipeline produce the same answers?

`OrderChecker` verifies that frames come out as 0, 1, 2, ... It never looks at
the prediction, so every failure mode that keeps the indices tidy is invisible
to it:

* a record paired with another frame's logits in the reorder buffer
* the input tensor being mutated by the next frame before the runtime has
  finished reading it -- the buffer is handed to `run_async` and the worker
  moves straight on to the next frame
* an output tensor reused across concurrent requests

Each of those yields perfect order, a perfect count, and wrong answers.

This runs the same clip through both pipelines and compares top-k per frame
index. It reads results out of the **real** runners -- `run_sync.measure_once`
and `run_async.measure_once`, through a collector they now accept -- because a
second implementation written for the test could not contain the bug the test
is looking for.

Two controls make a pass mean something:

* **sync against sync.** If the sequential pipeline disagrees with itself the
  comparison below is meaningless, so it is checked first.
* **several depths.** A race needs concurrency to appear, so depth 1 passing
  says nothing about depth 16.

Expect this to pass today: `preprocess` allocates a fresh array per frame, so
there is no buffer to mutate. Its value is as the net under W4 -- buffer reuse
and a fused preprocessing pass are exactly the optimisations that would
introduce the bug nothing else here detects.

    python -m bench.verify_equivalence --model models/resnet50_224.onnx \\
        --backend cpu --source clips/bbb_360_2MB.mp4 --frames 300 \\
        --depths 1 2 4 8 16
"""
from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

from bench import env_fingerprint, run_async, run_sync
from bench.backends import OnnxBackend, available


def parse_args():
    p = argparse.ArgumentParser(description="W3 value-equivalence check")
    p.add_argument("--model", required=True)
    p.add_argument("--backend", default="cpu")
    p.add_argument("--ov-device", default=None)
    p.add_argument("--source", default=None)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--intra-threads", type=int, default=None)
    p.add_argument("--depths", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    p.add_argument("--queue-size", type=int, default=4)
    p.add_argument("--prob-tol", type=float, default=1e-4,
                   help="allowed absolute difference in a class probability")
    p.add_argument("--candidate-dump", nargs="+", default=None,
                   metavar="LABEL=FILE",
                   help="compare these top-k dumps against the Python "
                        "sequential reference instead of running the Python "
                        "async pipeline. Written by the C++ runners' "
                        "--dump-topk, so the C++ build is checked by the same "
                        "comparison, the same self-test and the same "
                        "sensitivity figure as the Python one.")
    p.add_argument("--self-test", action="store_true",
                   help="corrupt the collected results on purpose and confirm "
                        "the comparison notices; a check that has never failed "
                        "is not known to work")
    p.add_argument("--allow-fallback", action="store_true")
    p.add_argument("--tag", default="w3")
    p.add_argument("--out-dir", default="results")
    return p.parse_args()


def _args_for(args, inflight: int) -> Namespace:
    # Warm-up is zero and the frame count is fixed: this is a correctness run,
    # so every frame must be collected and both sides must see the same ones.
    return Namespace(
        source=args.source, frames=args.frames, warmup=0, warmup_s=0.0,
        duration_s=0.0, top_k=args.top_k, queue_size=args.queue_size,
        inflight=inflight, reorder="on", reorder_cap=None,
        stall_every=0, stall_ms=0.0, stall_timeout_s=30.0,
    )


def compare(ref: dict[int, list], got: dict[int, list], tol: float) -> dict:
    """Compare two index -> top-k maps."""
    missing = sorted(set(ref) - set(got))
    extra = sorted(set(got) - set(ref))
    shared = sorted(set(ref) & set(got))

    top1_mismatch, topk_mismatch, ties, max_delta = [], [], [], 0.0
    for i in shared:
        a, b = ref[i], got[i]
        if a is None or b is None:
            topk_mismatch.append(i)
            continue
        if a[0][0] != b[0][0]:
            top1_mismatch.append(i)
        for (_, pa), (_, pb) in zip(a, b):
            max_delta = max(max_delta, abs(pa - pb))
        if [c for c, _ in a] != [c for c, _ in b]:
            # A different order is not automatically a different answer. Two
            # classes whose probabilities are equal to within the tolerance are
            # a tie, and which one a sort puts first is an implementation
            # detail -- std::partial_sort and numpy's argsort need not agree,
            # and neither has a claim to being right about a 6e-08 gap.
            # Counted separately so the distinction stays visible instead of
            # being folded into either "identical" or "wrong".
            tied = all(abs(pa - pb) <= tol
                       for (_, pa), (_, pb) in zip(a, b))
            (ties if tied else topk_mismatch).append(i)

    return {
        "frames_compared": len(shared),
        "missing_from_candidate": missing[:20],
        "missing_count": len(missing),
        "unexpected_in_candidate": extra[:20],
        "unexpected_count": len(extra),
        "top1_mismatches": top1_mismatch[:20],
        "top1_mismatch_count": len(top1_mismatch),
        "topk_order_mismatches": topk_mismatch[:20],
        "topk_mismatch_count": len(topk_mismatch),
        "topk_order_ties": ties[:20],
        "topk_tie_count": len(ties),
        "max_prob_delta": max_delta,
        "prob_within_tolerance": max_delta <= tol,
        "identical": (not missing and not extra and not top1_mismatch
                      and not topk_mismatch and max_delta <= tol),
    }


def _as_map(pairs: list) -> dict[int, list]:
    return {i: t for i, t in pairs}


def load_dump(path: str) -> dict[int, list]:
    """One JSON object per frame, as the C++ runners write it."""
    out: dict[int, list] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out[int(d["idx"])] = [(int(c), float(p)) for c, p in d["topk"]]
    return out


def self_test(ref: dict[int, list], tol: float) -> bool:
    """Corrupt a known-good result three ways and confirm each is caught.

    The three faults are the three ways an overlapped pipeline can be fast and
    wrong while its order and its frame count both look perfect: a frame
    silently dropped, two frames' results swapped, and a value quietly off.
    """
    keys = sorted(ref)
    if len(keys) < 4:
        print("  not enough frames to self-test")
        return False

    cases = []

    dropped = dict(ref)
    del dropped[keys[len(keys) // 2]]
    cases.append(("a frame dropped", dropped,
                  lambda r: r["missing_count"] == 1))

    # Find any two frames that actually predict different classes. On a 30 fps
    # clip most neighbours are visually identical and predict the same top-k,
    # so swapping those two produces no difference at all -- see the
    # sensitivity figure the main run prints.
    pair = next(((a, b) for a in keys for b in keys
                 if a < b and ref[a] and ref[b]
                 and [c for c, _ in ref[a]] != [c for c, _ in ref[b]]), None)
    if pair:
        a, b = pair
        swapped = dict(ref)
        swapped[a], swapped[b] = ref[b], ref[a]
        cases.append((f"results of frames {a} and {b} swapped", swapped,
                      lambda r: r["topk_mismatch_count"] >= 2))
    else:
        print("  every frame predicts the same classes here, so a swap is")
        print("  undetectable and this check has no power against one")

    nudged = dict(ref)
    k = keys[-2]
    if ref[k]:
        cls, prob = ref[k][0]
        nudged[k] = [(cls, prob + 10 * tol)] + list(ref[k][1:])
        cases.append(("one probability moved past tolerance", nudged,
                      lambda r: not r["prob_within_tolerance"]))

    ok = True
    for name, candidate, detects in cases:
        r = compare(ref, candidate, tol)
        caught = detects(r) and not r["identical"]
        print(f"  {name:<42} {'caught' if caught else 'MISSED'}")
        ok = ok and caught
    return ok


def sensitivity(ref: dict[int, list]) -> dict:
    """How much power this clip gives the check.

    A swap between two frames that predict the same classes produces no
    difference at all, so on a clip where every frame looks alike the check can
    verify values but not pairing. This reports that honestly rather than
    letting a green result imply more than it covers.
    """
    keys = sorted(ref)
    tops = [ref[i][0][0] if ref[i] else None for i in keys]
    fulls = [tuple(c for c, _ in ref[i]) if ref[i] else None for i in keys]
    adjacent_differ = sum(1 for x, y in zip(fulls, fulls[1:]) if x != y)
    return {
        "frames": len(keys),
        "distinct_top1_classes": len(set(tops)),
        "distinct_topk_sets": len(set(fulls)),
        "adjacent_pairs_differing": adjacent_differ,
        "adjacent_pairs": max(0, len(keys) - 1),
    }


def main() -> int:
    args = parse_args()
    print(f"available backends: {available()}")
    backend = OnnxBackend(args.model, args.backend, args.intra_threads,
                          args.ov_device)
    if backend.fell_back and not args.allow_fallback:
        print(f"!! requested '{args.backend}' but running on "
              f"{backend.actual_providers}; aborting.")
        return 2
    height, width = backend.hw
    dev = f"/{backend.ov_device}" if backend.ov_device else ""
    print(f"{Path(args.model).name} | {args.backend}{dev} | "
          f"{args.frames} frames | top-{args.top_k}")

    # Control: the sequential pipeline against itself.
    print("\n--- control: sync vs sync ---")
    c1: list = []
    run_sync.measure_once(backend, _args_for(args, 1), height, width, c1)
    c2: list = []
    run_sync.measure_once(backend, _args_for(args, 1), height, width, c2)
    ref = _as_map(c1)
    control = compare(ref, _as_map(c2), args.prob_tol)
    print(f"  {len(ref)} frames, identical={control['identical']}, "
          f"max prob delta {control['max_prob_delta']:.2e}")
    if not control["identical"]:
        print("  !! the sequential pipeline does not agree with itself.")
        print("     Nothing below would mean anything; stopping.")
        return 1

    sens = sensitivity(ref)
    pct = (100 * sens["adjacent_pairs_differing"] / sens["adjacent_pairs"]
           if sens["adjacent_pairs"] else 0.0)
    print(f"  sensitivity: {sens['distinct_topk_sets']} distinct top-k sets "
          f"over {sens['frames']} frames; {sens['adjacent_pairs_differing']} of "
          f"{sens['adjacent_pairs']} adjacent pairs differ ({pct:.0f}%)")
    if pct < 20:
        print("  NOTE: most neighbouring frames predict the same classes, so a")
        print("        swap between them cannot be detected. This run verifies")
        print("        values, and pairing only where predictions differ.")

    results = {"control_sync_vs_sync": control, "sensitivity": sens,
               "depths": {}}
    failures = []

    if args.candidate_dump:
        # The C++ build is compared frame for frame against the Python
        # sequential reference. Nothing about the comparison changes: same
        # control, same self-test, same sensitivity figure.
        for spec in args.candidate_dump:
            label, _, path = spec.partition("=")
            if not path:
                label, path = Path(spec).stem, spec
            got = load_dump(path)
            r = compare(ref, got, args.prob_tol)
            results["depths"][label] = r
            verdict = "IDENTICAL" if r["identical"] else "DIFFERS"
            print(f"\n--- {label} vs python sync ---")
            print(f"  compared {r['frames_compared']}  missing {r['missing_count']}"
                  f"  extra {r['unexpected_count']}"
                  f"  top1 mismatches {r['top1_mismatch_count']}"
                  f"  top-k order {r['topk_mismatch_count']}"
                  f"  ties {r['topk_tie_count']}"
                  f"  max prob delta {r['max_prob_delta']:.2e}  -> {verdict}")
            if not r["identical"]:
                failures.append(label)
                for key in ("missing_from_candidate", "unexpected_in_candidate",
                            "top1_mismatches", "topk_order_mismatches"):
                    if r[key]:
                        print(f"    {key}: {r[key]}")

    if args.self_test:
        print("\n--- self-test: can this check detect a fault at all? ---")
        if not self_test(ref, args.prob_tol):
            print("  !! the comparison missed a fault it was given on purpose."
                  "\n     A pass below would prove nothing; stopping.")
            return 1
    for depth in ([] if args.candidate_dump else args.depths):
        print(f"\n--- async inflight={depth} vs sync ---")
        col: list = []
        run_async.measure_once(backend, _args_for(args, depth), height, width, col)
        r = compare(ref, _as_map(col), args.prob_tol)
        results["depths"][str(depth)] = r
        verdict = "IDENTICAL" if r["identical"] else "DIFFERS"
        print(f"  compared {r['frames_compared']}  missing {r['missing_count']}"
              f"  extra {r['unexpected_count']}"
              f"  top1 mismatches {r['top1_mismatch_count']}"
              f"  top-k order {r['topk_mismatch_count']}"
              f"  ties {r['topk_tie_count']}"
              f"  max prob delta {r['max_prob_delta']:.2e}  -> {verdict}")
        if not r["identical"]:
            failures.append(depth)
            for key in ("missing_from_candidate", "unexpected_in_candidate",
                        "top1_mismatches", "topk_order_mismatches"):
                if r[key]:
                    print(f"    {key}: {r[key]}")

    w = 70
    print("\n" + "=" * w)
    if failures:
        print(f" VALUE EQUIVALENCE FAILED at depths {failures}.")
        print(" The pipeline emitted frames in the right order and the right")
        print(" number of them, and got different answers. Order checking")
        print(" would not have caught this.")
    else:
        tested = (", ".join(results["depths"].keys()) if args.candidate_dump
                  else ", ".join(map(str, args.depths)))
        print(f" Value equivalence holds for: {tested}.")
        print(f" {len(ref)} frames, top-{args.top_k} classes identical,"
              f" probabilities within {args.prob_tol:g}.")
    print("=" * w)

    meta = {"tag": args.tag, "mode": "verify", "args": vars(args),
            "backend": backend.describe(), "env": env_fingerprint.collect()}
    dev_slug = f"-{backend.ov_device.lower()}" if backend.ov_device else ""
    src_slug = Path(args.source).stem if args.source else "synthetic"
    out = (Path(args.out_dir) /
           f"{args.tag}__verify__{args.backend}{dev_slug}"
           f"__{Path(args.model).stem}__{src_slug}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "results": results,
                               "failed_depths": failures}, indent=2),
                   encoding="utf-8")
    print(f"\nsaved -> {out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
