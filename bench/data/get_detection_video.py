"""Fetch the upstream OpenCV pedestrian sample and verify its identity."""
import argparse
import hashlib
from pathlib import Path
import urllib.request

URL = 'https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi'
SHA256 = '45cddc9490be69345cbdab64ca583be65987e864ca408038e648db99e10516cf'
ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'clips/real-video/vtest.avi')
    args = parser.parse_args()
    data = args.output.read_bytes() if args.output.exists() else urllib.request.urlopen(URL, timeout=60).read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256:
        raise RuntimeError(f'Source hash mismatch: {digest}; refusing to overwrite or accept this file')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists():
        args.output.write_bytes(data)
    print(f'{args.output}\nSHA256: {digest}\nSource: {URL}')


if __name__ == '__main__':
    main()
