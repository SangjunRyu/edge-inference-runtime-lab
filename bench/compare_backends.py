"""Do different backends give the same answers?

A separate question from `verify_equivalence`, which asks whether the
*overlapped* pipeline agrees with the sequential one on **one** backend. This
asks whether the backends agree with **each other**, running the same
sequential pipeline on each.

It matters because the whole project rests on moving inference off the host,
and "the NPU is 3.3x faster" is only interesting if the NPU is also answering
the same question. The models here are FP32 -- no quantisation is involved --
so any disagreement is kernel selection and accumulation order, which is the
floor under any later accuracy work rather than a result about it.

Disagreement is reported against the **margin** it happened at. A backend that
flips a decision the reference won by 0.0003 has not disagreed about the image;
it has broken a tie differently, and saying "2 of 300 frames differ" without
that context overstates it considerably.

    python -m bench.compare_backends --model models/resnet50_224.onnx \\
        --source clips/bbb_360_2MB.mp4 --frames 300 \\
        --reference cpu --against openvino:CPU openvino:GPU openvino:NPU
"""
from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

from bench import env_fingerprint, run_sync
from bench.backends import OnnxBackend
from bench.verify_equivalence import _as_map, compare


def _spec(text: str) -> tuple[str, str | None]:
    """'openvino:NPU' -> ('openvino', 'NPU');  'cpu' -> ('cpu', None)."""
    backend, _, dev = text.partition(":")
    return backend, (dev or None)


def parse_args():
    p = argparse.ArgumentParser(description="cross-backend agreement")
    p.add_argument("--model", required=True)
    p.add_argument("--source", default=None)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--reference", default="cpu",
                   help="backend[:device] every other is compared against")
    p.add_argument("--against", nargs="+",
                   default=["openvino:CPU", "openvino:GPU", "openvino:NPU"])
    p.add_argument("--prob-tol", type=float, default=1e-4)
    p.add_argument("--tag", default="backends")
    p.add_argument("--out-dir", default="results")
    return p.parse_args()


def collect(model: str, spec: str, args) -> dict[int, list]:
    backend, dev = _spec(spec)
    b = OnnxBackend(model, backend, None, dev)
    if b.fell_back:
        raise RuntimeError(f"{spec} fell back to {b.actual_providers}")
    h, w = b.hw
    col: list = []
    run_sync.measure_once(
        b, Namespace(source=args.source, frames=args.frames, warmup=0,
                     warmup_s=0.0, duration_s=0.0, top_k=args.top_k),
        h, w, col)
    return _as_map(col)


def margins(ref: dict[int, list], got: dict[int, list]) -> list[dict]:
    """Every top-1 flip, with the margin the reference decided it by."""
    out = []
    for i in sorted(ref):
        a, b = ref.get(i), got.get(i)
        if not a or not b or a[0][0] == b[0][0]:
            continue
        margin = a[0][1] - a[1][1] if len(a) > 1 else float("nan")
        out.append({"frame": i, "reference_class": a[0][0],
                    "reference_p": a[0][1], "runner_up_class": a[1][0],
                    "runner_up_p": a[1][1], "margin": margin,
                    "candidate_class": b[0][0]})
    return out


def main() -> int:
    args = parse_args()
    print(f"{Path(args.model).name} | {args.frames} frames | "
          f"reference = {args.reference}")

    ref = collect(args.model, args.reference, args)
    print(f"  reference collected {len(ref)} frames")

    rows, details = [], {}
    for spec in args.against:
        if spec == args.reference:
            continue
        got = collect(args.model, spec, args)
        c = compare(ref, got, args.prob_tol)
        flips = margins(ref, got)
        details[spec] = {"compare": c, "top1_flips": flips}
        rows.append((spec, c, flips))
        print(f"  {spec} collected {len(got)} frames")

    w = 78
    print("\n" + "=" * w)
    print(f" AGREEMENT WITH {args.reference}")
    print("-" * w)
    print(f" {'backend':<16}{'top-1 differs':>15}{'top-k order':>13}"
          f"{'max prob delta':>17}")
    for spec, c, _ in rows:
        print(f" {spec:<16}{c['top1_mismatch_count']:>15}"
              f"{c['topk_mismatch_count']:>13}{c['max_prob_delta']:>17.2e}")
    print("-" * w)

    for spec, c, flips in rows:
        if not flips:
            continue
        print(f" {spec}: top-1 flipped on {[f['frame'] for f in flips]}")
        for f in flips:
            print(f"   frame {f['frame']}: reference chose {f['reference_class']}"
                  f" (p={f['reference_p']:.4f}) over {f['runner_up_class']}"
                  f" (p={f['runner_up_p']:.4f}), margin {f['margin']:.5f};"
                  f" this backend chose {f['candidate_class']}")
        worst = max(f["margin"] for f in flips)
        delta = c["max_prob_delta"]
        if worst < delta:
            print(f"   Every flip happened at a margin ({worst:.5f}) smaller"
                  f" than the numerical spread between the two"
                  f" implementations ({delta:.5f}).")
            print(f"   These are ties broken differently, not disagreements"
                  f" about the image.")
    print("=" * w)

    meta = {"tag": args.tag, "mode": "cross-backend", "args": vars(args),
            "env": env_fingerprint.collect()}
    out = (Path(args.out_dir) /
           f"{args.tag}__cross__{Path(args.model).stem}"
           f"__{Path(args.source).stem if args.source else 'synthetic'}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "reference": args.reference,
                               "results": details}, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
