"""Run the backend x resolution matrix.

The two axes exist to be separated. Across the grid:

* ``inference`` should respond to the **backend** and not to the clip, because
  the model always sees a 224x224 tensor whatever the source resolution.
* ``read`` and ``preprocess`` should respond to the **clip** and not to the
  backend, because decoding and resizing happen on the host either way.

Anything that violates those two expectations is the interesting result, and
neither can be shown without holding one axis still while moving the other.

Each cell runs in a fresh process so no session, allocator or driver state
carries between backends. Cells whose result file already exists are skipped,
so the grid can be filled in sittings and resumed after an interrupt.

    python -m bench.run_matrix --tag gram --repeat 5 --cooldown 60 \
        --max-busy-pct 25

    python -m bench.run_matrix --tag gram --dry-run     # show the plan only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# (backend, ov_device) -- ov_device is None for the plain ONNX Runtime CPU EP.
BACKENDS: list[tuple[str, str | None]] = [
    ("cpu", None),
    ("openvino", "GPU"),
    ("openvino", "NPU"),
]

CLIPS = ["clips/bbb_360_2MB.mp4", "clips/bbb_720_5MB.mp4", "clips/bbb_1080_10MB.mp4"]


def out_path(out_dir: str, tag: str, backend: str, dev: str | None,
             model: str, clip: str) -> Path:
    dev_slug = f"-{dev.lower()}" if dev else ""
    return (Path(out_dir) /
            f"{tag}__{backend}{dev_slug}__{Path(model).stem}"
            f"__{Path(clip).stem}.json")


def parse_args():
    p = argparse.ArgumentParser(description="backend x resolution matrix")
    p.add_argument("--model", default="models/resnet50_224.onnx")
    p.add_argument("--tag", default="matrix")
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--cooldown", type=float, default=60.0)
    p.add_argument("--max-busy-pct", type=float, default=25.0)
    p.add_argument("--intra-threads", type=int, default=None)
    p.add_argument("--between-cells", type=float, default=60.0,
                   help="seconds to rest between cells, on top of --cooldown")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--clips", nargs="+", default=CLIPS)
    p.add_argument("--force", action="store_true", help="re-run existing cells")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not Path(args.model).exists():
        print(f"model not found: {args.model}\n"
              f"  python -m bench.prepare_models --models resnet50 --sizes 224")
        return 1
    missing = [c for c in args.clips if not Path(c).exists()]
    if missing:
        print(f"clips not found: {missing}\n  python -m bench.get_clips")
        return 1

    cells = [(b, d, c) for b, d in BACKENDS for c in args.clips]
    todo = [x for x in cells
            if args.force or not out_path(args.out_dir, args.tag, x[0], x[1],
                                          args.model, x[2]).exists()]

    # One run is frames*E2E; repeats are separated by --cooldown.
    per_cell = args.repeat * args.frames * 0.015 + (args.repeat - 1) * args.cooldown
    total_min = (len(todo) * (per_cell + args.between_cells)) / 60

    print(f"matrix: {len(BACKENDS)} backends x {len(args.clips)} clips "
          f"= {len(cells)} cells")
    print(f"to run: {len(todo)}  (skipping {len(cells) - len(todo)} already present)")
    print(f"rough estimate: {total_min:.0f} min\n")
    for b, d, c in todo:
        print(f"  {b + ('/' + d if d else ''):<16}{Path(c).stem}")
    if args.dry_run or not todo:
        return 0

    print()
    failed: list[str] = []
    for i, (backend, dev, clip) in enumerate(todo, 1):
        label = f"{backend}{'/' + dev if dev else ''} x {Path(clip).stem}"
        print("#" * 70)
        print(f"# cell {i}/{len(todo)}: {label}")
        print("#" * 70)
        cmd = [sys.executable, "-m", "bench.run_sync",
               "--model", args.model, "--backend", backend,
               "--source", clip, "--frames", str(args.frames),
               "--warmup", str(args.warmup), "--repeat", str(args.repeat),
               "--cooldown", str(args.cooldown),
               "--max-busy-pct", str(args.max_busy_pct),
               "--tag", args.tag, "--out-dir", args.out_dir]
        if dev:
            cmd += ["--ov-device", dev]
        if args.intra_threads is not None:
            cmd += ["--intra-threads", str(args.intra_threads)]

        rc = subprocess.run(cmd).returncode
        if rc != 0:
            # Most likely the provider was unavailable and run_sync refused to
            # save a CPU number under this cell's tag. Keep going: a missing
            # cell is recoverable, a wrong one is not.
            print(f"!! cell failed (exit {rc}): {label}")
            failed.append(label)
        if i < len(todo) and args.between_cells > 0:
            print(f"   resting {args.between_cells:.0f}s before the next cell")
            time.sleep(args.between_cells)

    print("\n" + "=" * 70)
    print(f" matrix done: {len(todo) - len(failed)}/{len(todo)} cells written")
    if failed:
        print(" failed cells (no result saved, by design):")
        for f in failed:
            print(f"   - {f}")
    print(" summarise with:  python -m bench.summarize --tag " + args.tag)
    print("=" * 70)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
