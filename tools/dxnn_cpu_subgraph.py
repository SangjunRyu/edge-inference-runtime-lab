"""Run the part of a compiled DEEPX model that does not need the NPU.

A `.dxnn` is not all NPU. When the compiler meets an operation the device does
not implement it leaves that piece on the host, and stores it in the container
as an ordinary ONNX model. `dx_rt` loads it straight from memory into an ONNX
Runtime session (`lib/cpu_handle.cpp:236`), which means the host half of a
compiled model is runnable on any machine with `onnxruntime` installed.

Classifiers usually have none: densenet121 compiles entirely to the NPU and its
`cpu_models` is empty. Detectors do, because the decode-and-NMS tail is not
device work. A YOLO container carries a `cpu_0` subgraph of about 170 KB, and
that subgraph runs here.

This is the difference between reading a container and executing one. The NPU
half cannot be run without a DX-M1; the host half can, and it is where a
detector's output actually takes its final shape.

    python -m tools.dxnn_cpu_subgraph model.dxnn
    python -m tools.dxnn_cpu_subgraph model.dxnn --extract out/
    python -m tools.dxnn_cpu_subgraph model.dxnn --run

`--run` needs `onnxruntime`. Everything else needs only the standard library.
Nothing is copied from DEEPX: the subgraph is the user's own compiled model and
the container format is read from the file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tools.dxnn_inspect import HEADER_SIZE, DxnnError, load


def cpu_sections(container) -> dict:
    return container.header.get("data", {}).get("cpu_models", {}) or {}


def read_blob(path: Path, offset: int, size: int) -> bytes:
    with path.open("rb") as f:
        f.seek(HEADER_SIZE + offset)
        return f.read(size)


def graph_text(container) -> str | None:
    """The compiler's own description of how the pieces connect."""
    sec = container.header.get("data", {}).get("graph_info")
    if not sec:
        return None
    raw = read_blob(container.path, sec["offset"] - 12, sec["size"] + 12)
    return raw.decode("utf-8", errors="replace")


def describe_topology(container) -> list[str]:
    txt = graph_text(container)
    if not txt:
        return []
    start = txt.find("{")
    if start < 0:
        return []
    try:
        g = json.loads(txt[start: txt.rfind("}") + 1])
    except Exception:
        return []
    out = []
    order = g.get("toposort_order") or []
    out.append("order      " + " -> ".join(order))
    for node in g.get("graphs", []):
        ins = ", ".join(i.get("name", "?") for i in node.get("inputs", []))
        outs = ", ".join(o.get("name", "?") for o in node.get("outputs", []))
        out.append(f"  {node.get('name','?'):<8} on {node.get('device','?'):<4}"
                   f"  in [{ins}]  out [{outs}]")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="tools.dxnn_cpu_subgraph",
        description="Extract and run the host-side subgraphs of a compiled model.")
    ap.add_argument("model", help="path to a .dxnn")
    ap.add_argument("--extract", metavar="DIR", help="write each subgraph as .onnx")
    ap.add_argument("--run", action="store_true",
                    help="load with onnxruntime and execute on synthetic input")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    try:
        container = load(args.model)
    except (DxnnError, OSError, json.JSONDecodeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    secs = cpu_sections(container)
    report: dict = {"model": container.path.name, "target": container.target,
                    "cpu_subgraphs": len(secs), "subgraphs": []}

    if not args.json:
        print(f"model      {container.path.name}   target {container.target}")
        for line in describe_topology(container):
            print(line)
        print()
        if not secs:
            print("no host subgraphs: the compiler placed the whole graph on the NPU,")
            print("so there is nothing here that runs without a device.")

    if not secs:
        if args.json:
            print(json.dumps(report, indent=2))
        return 0

    for name, sec in secs.items():
        blob = read_blob(container.path, sec["offset"], sec["size"])
        entry = {"name": name, "bytes": len(blob),
                 "looks_like_onnx": b"onnx" in blob[:64]}

        if args.extract:
            outdir = Path(args.extract)
            outdir.mkdir(parents=True, exist_ok=True)
            dst = outdir / f"{container.path.stem}_{name}.onnx"
            dst.write_bytes(blob)
            entry["extracted_to"] = str(dst)

        if args.run:
            try:
                import numpy as np
                import onnxruntime as ort
            except ImportError as e:
                print(f"error: --run needs onnxruntime and numpy ({e})", file=sys.stderr)
                return 2
            try:
                so = ort.SessionOptions()
                so.log_severity_level = 3
                # From memory, the way dx_rt does it: the bytes in the container
                # are the model, and writing them to disk first would only add a
                # way for the two to differ.
                sess = ort.InferenceSession(blob, so, providers=["CPUExecutionProvider"])
            except Exception as e:
                entry["error"] = f"session: {e}"
                report["subgraphs"].append(entry)
                continue

            feeds = {}
            ok = True
            for i in sess.get_inputs():
                shape = [d if isinstance(d, int) and d > 0 else 1 for d in i.shape]
                dt = {"tensor(float)": np.float32, "tensor(float16)": np.float16,
                      "tensor(int64)": np.int64, "tensor(int32)": np.int32,
                      "tensor(uint8)": np.uint8}.get(i.type)
                if dt is None:
                    entry["error"] = f"unhandled input type {i.type}"
                    ok = False
                    break
                feeds[i.name] = np.zeros(shape, dtype=dt)
            if not ok:
                report["subgraphs"].append(entry)
                continue

            entry["inputs"] = [{"name": i.name, "type": i.type, "shape": i.shape}
                               for i in sess.get_inputs()]
            try:
                outs = sess.run(None, feeds)
                entry["outputs"] = [{"name": o.name, "shape": list(a.shape),
                                     "dtype": str(a.dtype)}
                                    for o, a in zip(sess.get_outputs(), outs)]
                entry["ran"] = True
            except Exception as e:
                entry["error"] = f"run: {e}"
                entry["ran"] = False

        report["subgraphs"].append(entry)

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    for e in report["subgraphs"]:
        print(f"subgraph   {e['name']}   {e['bytes']:,} B"
              f"   {'ONNX' if e['looks_like_onnx'] else 'unrecognised'}")
        if "extracted_to" in e:
            print(f"  written  {e['extracted_to']}")
        if "error" in e:
            print(f"  FAILED   {e['error']}")
        for i in e.get("inputs", []):
            print(f"  in       {i['name']:<24} {i['type']:<16} {i['shape']}")
        for o in e.get("outputs", []):
            print(f"  out      {o['name']:<24} {o['dtype']:<16} {o['shape']}")
        if e.get("ran"):
            print("  ran      yes, on the CPU execution provider, with no device present")

    failed = [e for e in report["subgraphs"] if "error" in e]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
