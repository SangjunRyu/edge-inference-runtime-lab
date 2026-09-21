"""Which models fit on a device, and at what concurrency.

`dxnn_inspect` answers that for one model. This answers it for a shelf of them,
which is the question someone actually has when choosing what to deploy: given a
device with N bytes of memory, what can I run, and how many inferences can be in
flight while I run it?

Every number comes from the compiled models and from DX-RT's own arithmetic. No
NPU is involved, and nothing here is measured, so nothing here depends on the
machine it runs on.

    python -m tools.model_budget models/*.dxnn --npu-memory 1G
    python -m tools.model_budget path/to/*.dxnn --memories 512M,1G,2G
    python -m tools.model_budget path/to/*.dxnn --json

The interesting column is usually the last one. For a classifier the per-buffer
cost is dominated by `output_all_size` rather than by anything visible in the
model's signature, so two models with the same input shape can differ by an
order of magnitude in how many of them can be in flight at once.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

from tools.dxnn_inspect import (
    DEFAULT_BUFFERS,
    DxnnError,
    InsufficientMemory,
    budget,
    checks,
    load,
    max_buffer_count,
    parse_size,
)


def collect(paths: list[str]) -> list[dict]:
    rows = []
    for pattern in paths:
        matched = sorted(glob.glob(pattern)) or ([pattern] if Path(pattern).exists() else [])
        if not matched:
            rows.append({"path": pattern, "error": "no such file"})
            continue
        for p in matched:
            try:
                c = load(p)
            except (DxnnError, OSError, json.JSONDecodeError) as e:
                rows.append({"path": p, "error": str(e)})
                continue
            b = budget(c)
            results = checks(c)
            inputs = [t for task in c.tasks for t in task.info.get("inputs", [])]
            shape = inputs[0].get("shape") if inputs else None
            rows.append({
                "path": p,
                "name": Path(p).name,
                "target": c.target,
                "input_shape": shape,
                "cpu_subgraphs": len(c.cpu_models),
                "fixed_cost": b.fixed,
                "per_buffer_cost": b.per_buffer,
                "output_all_size": sum(t.output_mem_bytes for t in c.tasks),
                "single_buffer_total": b.fixed + b.per_buffer,
                "checks_passed": all(ok for ok, _ in results),
            })
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="tools.model_budget",
        description="Which compiled models fit on a device, and at what concurrency.")
    ap.add_argument("models", nargs="+", help=".dxnn paths or globs")
    ap.add_argument("--memories", default="512M,1G,2G",
                    help="device memory sizes to evaluate (binary suffixes)")
    ap.add_argument("--buffers", type=int, default=DEFAULT_BUFFERS,
                    help=f"requested buffer count (default {DEFAULT_BUFFERS})")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    rows = collect(args.models)
    good = [r for r in rows if "error" not in r]
    mems = [(m, parse_size(m)) for m in args.memories.split(",") if m.strip()]

    for r in good:
        r["fits"] = {}
        for label, m in mems:
            try:
                r["fits"][label] = max_buffer_count(m, r["fixed_cost"],
                                                    r["per_buffer_cost"], args.buffers)
            except InsufficientMemory:
                r["fits"][label] = 0

    if args.json:
        print(json.dumps({"requested_buffers": args.buffers,
                          "memories": [m for m, _ in mems],
                          "models": rows}, indent=2))
        return 0 if all(r.get("checks_passed", False) for r in good) else 1

    for r in rows:
        if "error" in r:
            print(f"  skipped {r['path']}: {r['error']}", file=sys.stderr)

    if not good:
        print("no readable models", file=sys.stderr)
        return 2

    print(f"requested buffer count {args.buffers}"
          f"   (DX-RT's default is {DEFAULT_BUFFERS})\n")

    name_w = max(len(r["name"]) for r in good)
    head = (f"  {'model':<{name_w}} {'input':>16} {'cpu':>4} {'fixed MB':>9}"
            f" {'per-buf MB':>11} {'out share':>10}")
    for label, _ in mems:
        head += f" {label:>7}"
    print(head)

    for r in sorted(good, key=lambda x: x["per_buffer_cost"]):
        shape = "x".join(str(d) for d in (r["input_shape"] or [])) or "?"
        share = r["output_all_size"] / r["per_buffer_cost"] if r["per_buffer_cost"] else 0
        line = (f"  {r['name']:<{name_w}} {shape:>16} {r['cpu_subgraphs']:>4}"
                f" {r['fixed_cost'] / 1e6:9.1f} {r['per_buffer_cost'] / 1e6:11.1f}"
                f" {share * 100:9.1f}%")
        for label, _ in mems:
            n = r["fits"][label]
            line += f" {('-' if n == 0 else str(n)):>7}"
        print(line)

    print("\n  cpu        = subgraphs the compiler left on the host")
    print("  out share  = how much of the per-buffer cost is output_all_size")
    print(f"  {'/'.join(m for m, _ in mems)} = buffers that fit, of the"
          f" {args.buffers} requested; '-' means not even one")

    # The point of the table, stated rather than left to be noticed.
    widest = max(good, key=lambda r: r["per_buffer_cost"])
    narrowest = min(good, key=lambda r: r["per_buffer_cost"])
    if widest is not narrowest and narrowest["per_buffer_cost"]:
        factor = widest["per_buffer_cost"] / narrowest["per_buffer_cost"]
        print(f"\n  {widest['name']} costs {factor:.1f}x more per buffer than"
              f" {narrowest['name']},")
        print("  which is what decides concurrency on a fixed device, not the input shape.")

    failed = [r for r in good if not r["checks_passed"]]
    if failed:
        print(f"\n  {len(failed)} model(s) failed their container checks:",
              ", ".join(r["name"] for r in failed))
    return 0 if not failed and len(good) == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
