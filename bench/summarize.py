"""Summarise a matrix run into Markdown tables.

Reads every result written under one ``--tag`` and prints the tables that go in
the README, plus the two checks the matrix exists to make:

* ``inference`` should vary across backends and stay flat across clips.
* ``read`` should vary across clips and stay flat across backends.

Spread is reported as a coefficient of variation so it can be compared against
the machine's own run-to-run noise, which is printed alongside. A difference
smaller than that noise is not a difference.

    python -m bench.summarize --tag gram
    python -m bench.summarize --tag gram --markdown >> README.md
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load(out_dir: str, tag: str) -> list[dict]:
    rows = []
    for p in sorted(Path(out_dir).glob(f"{tag}__*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        runs = d.get("runs") or ([d["result"]] if "result" in d else [])
        if not runs:
            continue
        meta, bk = d["meta"], d["meta"]["backend"]
        dev = bk.get("openvino_device")
        rows.append({
            "file": p.name,
            "backend": bk["requested_backend"] + (f"/{dev}" if dev else ""),
            "clip": Path(meta["args"].get("source") or "synthetic").stem,
            "n_runs": len(runs),
            # Median across runs: one bad run should not define a cell.
            "read": statistics.median(r["stages_ms"]["read"]["p50"] for r in runs),
            "preprocess": statistics.median(
                r["stages_ms"]["preprocess"]["p50"] for r in runs),
            "inference": statistics.median(
                r["stages_ms"]["inference"]["p50"] for r in runs),
            "postprocess": statistics.median(
                r["stages_ms"]["postprocess"]["p50"] for r in runs),
            "e2e": statistics.median(r["e2e_ms"]["p50"] for r in runs),
            "ceiling": statistics.median(r["overlap_ceiling_x"] for r in runs),
            "bottleneck": max(
                set(r["bottleneck_stage"] for r in runs),
                key=lambda s: sum(1 for r in runs if r["bottleneck_stage"] == s)),
            "fps_cv": (d.get("across_runs") or {}).get(
                "throughput_fps", {}).get("cv", 0.0),
            "ort": meta["env"].get("onnxruntime_version"),
        })
    return rows


def cv(vals: list[float]) -> float:
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return 0.0
    m = statistics.fmean(vals)
    return (statistics.stdev(vals) / m) if m else 0.0


def grid(rows: list[dict], metric: str) -> tuple[list[str], list[str], dict]:
    backends = sorted({r["backend"] for r in rows})
    clips = sorted({r["clip"] for r in rows},
                   key=lambda c: int("".join(ch for ch in c if ch.isdigit())[:4] or 0))
    cells = {(r["backend"], r["clip"]): r[metric] for r in rows}
    return backends, clips, cells


def table(rows: list[dict], metric: str, unit: str = "ms") -> str:
    backends, clips, cells = grid(rows, metric)
    out = [f"| {metric} ({unit}) | " + " | ".join(clips) + " | across clips |",
           "|---" * (len(clips) + 2) + "|"]
    for b in backends:
        vals = [cells.get((b, c)) for c in clips]
        shown = [f"{v:.2f}" if v is not None else "—" for v in vals]
        out.append(f"| **{b}** | " + " | ".join(shown) +
                   f" | CV {100 * cv([v for v in vals if v is not None]):.0f}% |")
    per_clip = []
    for c in clips:
        vals = [cells.get((b, c)) for b in backends]
        per_clip.append(f"CV {100 * cv([v for v in vals if v is not None]):.0f}%")
    out.append("| *across backends* | " + " | ".join(per_clip) + " | |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="matrix")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    rows = load(args.out_dir, args.tag)
    if not rows:
        print(f"no results matching {args.out_dir}/{args.tag}__*.json")
        return 1

    noise = max((r["fps_cv"] for r in rows), default=0.0)
    orts = {r["ort"] for r in rows}

    print(f"\n{len(rows)} cells, tag '{args.tag}'")
    print(f"worst run-to-run throughput CV across cells: {100 * noise:.1f}%")
    if len(orts) > 1:
        print(f"!! mixed ONNX Runtime versions {sorted(orts)} -- cells built "
              f"with different runtimes are not comparable")

    for metric, unit in (("inference", "ms"), ("read", "ms"),
                         ("preprocess", "ms"), ("e2e", "ms"),
                         ("ceiling", "x")):
        print(f"\n{table(rows, metric, unit)}")

    print("\n| cell | bottleneck | ceiling | runs |")
    print("|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["backend"], r["clip"])):
        print(f"| {r['backend']} x {r['clip']} | {r['bottleneck']} | "
              f"{r['ceiling']:.2f}x | {r['n_runs']} |")

    # The two expectations, stated as numbers.
    _, _, inf = grid(rows, "inference")
    _, _, rd = grid(rows, "read")
    backends, clips, _ = grid(rows, "inference")
    inf_across_clips = statistics.fmean(
        [cv([inf.get((b, c)) for c in clips]) for b in backends] or [0])
    rd_across_backends = statistics.fmean(
        [cv([rd.get((b, c)) for b in backends]) for c in clips] or [0])

    print("\n## Do the axes separate?\n")
    print(f"- `inference` across clips (want ~0, machine noise {100 * noise:.0f}%): "
          f"**{100 * inf_across_clips:.0f}%**")
    print(f"- `read` across backends (want ~0, machine noise {100 * noise:.0f}%): "
          f"**{100 * rd_across_backends:.0f}%**")
    for name, val in (("inference varies with the clip", inf_across_clips),
                      ("read varies with the backend", rd_across_backends)):
        if val > max(noise, 0.05) * 1.5:
            print(f"- ⚠ {name} by more than the noise floor — explain this "
                  f"before quoting the grid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
