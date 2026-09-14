"""Download the Big Buck Bunny test grid.

Why this clip:

* **Licence.** Big Buck Bunny is CC-BY 3.0 (Blender Foundation), so the input
  can be named and re-fetched by anyone reading the results. A benchmark whose
  input cannot be obtained is not reproducible.
* **Matched variants.** The same content is published at 360p / 720p / 1080p and
  at six bitrates each, all H.264. Resolution and bitrate become independent
  knobs over an otherwise identical input -- which is exactly the stage-balance
  axis this project sweeps. Changing clip *and* resolution at once would confound
  the two.
* **Content is irrelevant to timing.** A CNN has no data-dependent control flow,
  so what is on screen does not change inference time. Decode cost, resolution
  and bitrate do. The clip is chosen for licence and for the matched grid, not
  for realism.

Each file is 10s / 300 frames at 30fps. Keep ``--frames`` at or below 300 so the
source never loops; a loop costs a seek and a keyframe re-decode that shows up
as an outlier in ``read.max``.

    python -m bench.get_clips                    # default 3-resolution set
    python -m bench.get_clips --all              # full 3 x 6 grid
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

BASE = "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264"
RESOLUTIONS = ("360", "720", "1080")
SIZES = ("1MB", "2MB", "5MB", "10MB", "20MB", "30MB")

# One mid bitrate per resolution: isolates resolution with bitrate roughly held.
DEFAULT = [("360", "2MB"), ("720", "5MB"), ("1080", "10MB")]


def url_for(res: str, size: str) -> str:
    return f"{BASE}/{res}/Big_Buck_Bunny_{res}_10s_{size}.mp4"


def fetch(res: str, size: str, out_dir: Path) -> Path:
    out = out_dir / f"bbb_{res}_{size}.mp4"
    if out.exists():
        print(f"  exists  {out.name}")
        return out
    req = urllib.request.Request(url_for(res, size), headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=120) as r, open(out, "wb") as f:
        f.write(r.read())
    print(f"  got     {out.name}  ({out.stat().st_size / 1024**2:.1f} MB)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="fetch Big Buck Bunny test clips")
    ap.add_argument("--all", action="store_true", help="full resolution x bitrate grid")
    ap.add_argument("--out-dir", default="clips")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = [(r, s) for r in RESOLUTIONS for s in SIZES] if args.all else DEFAULT
    print(f"downloading {len(pairs)} clip(s) into {out_dir}/")
    for res, size in pairs:
        fetch(res, size, out_dir)

    print("\nBig Buck Bunny (c) Blender Foundation, CC-BY 3.0 "
          "-- https://peach.blender.org/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
