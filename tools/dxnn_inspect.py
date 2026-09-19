"""Read a ``.dxnn`` and reproduce DX-RT's memory budget for it.

A compiled DEEPX model is self-describing: the container carries the tensor
layouts, the alignment unit, the compiled section sizes and the NPU memory
regions the runtime will charge for. That is enough to answer, with no NPU
present, the question W5-4 asks of this project's own runtime:

    how many inferences can be in flight at once, and who decides?

In DX-RT the answer is not a tuning knob, it is a memory budget. This tool
reproduces that arithmetic. Every number it prints is either read from the file
or computed from numbers read from the file; nothing here is measured, so
nothing here depends on the machine.

The arithmetic mirrors these places in ``dx_rt`` (read, not guessed):

* section data begins at byte 8192 -- ``lib/parsers/v8_model_parser.cpp``,
  where ``offset = sizeInfo = 8192`` after an 8-byte magic-and-version prefix;
* per-buffer and fixed cost -- ``lib/task_data.cpp:453-454``;
* the walk-down that fits a requested buffer count into device memory --
  ``InferenceEngine::adjustBufferCountForNpuMemory`` and
  ``InferenceEngine::calculateMaxBufferCount`` in ``lib/inference_engine.cpp``;
* the default buffer count of 6 -- ``DXRT_TASK_MAX_LOAD_DEFAULT`` in
  ``lib/include/dxrt/common.h:82``, overridable by ``DXRT_TASK_MAX_LOAD`` and
  capped at 100.

Usage::

    python -m tools.dxnn_inspect model.dxnn
    python -m tools.dxnn_inspect model.dxnn --npu-memory 512M
    python -m tools.dxnn_inspect model.dxnn --buffers 12 --json
    python -m tools.dxnn_inspect --self-test

No DEEPX source is vendored here. The file format is read from the file.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

MAGIC = b"DXNN"

#: Section byte offsets are relative to this, not to the end of the JSON text.
#: ``v8_model_parser.cpp`` sets ``offset = sizeInfo = 8192`` and then indexes
#: ``buffer + offset + section.offset``. The JSON header itself is the 8184
#: bytes between the 8-byte prefix and this boundary.
HEADER_SIZE = 8192

SUPPORTED_VERSIONS = (8,)

#: ``DXRT_TASK_MAX_LOAD_DEFAULT`` / ``DXRT_TASK_MAX_LOAD_LIMIT``.
DEFAULT_BUFFERS = 6
MAX_BUFFERS = 100

DTYPE_BYTES = {
    "INT8": 1, "UINT8": 1,
    "INT16": 2, "UINT16": 2, "FLOAT16": 2, "BFLOAT16": 2,
    "INT32": 4, "UINT32": 4, "FLOAT32": 4,
    "INT64": 8, "UINT64": 8, "FLOAT64": 8,
}


class DxnnError(Exception):
    """The file is not a container this tool can read."""


class InsufficientMemory(Exception):
    """Mirrors ``InsufficientMemoryException``: one buffer does not fit."""


def align(value: int, unit: int = 64) -> int:
    """``data_align`` in dx_rt: round up to a whole number of ``unit`` bytes."""
    if unit <= 0:
        raise ValueError("alignment unit must be positive")
    return -(-value // unit) * unit


# --------------------------------------------------------------------------
# container
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Section:
    """One addressable run of bytes inside the container."""
    name: str
    kind: str
    offset: int
    size: int

    @property
    def start(self) -> int:
        return HEADER_SIZE + self.offset

    @property
    def end(self) -> int:
        return self.start + self.size


@dataclass
class NpuTask:
    """One NPU subgraph: its compiled sections and its register-map info."""
    name: str
    sections: dict[str, Section]
    info: dict

    @property
    def rmap_bytes(self) -> int:
        return self.sections["rmap"].size

    @property
    def weight_bytes(self) -> int:
        return self.sections["weight"].size

    @property
    def ppu_bytes(self) -> int:
        sec = self.sections.get("ppu")
        return sec.size if sec else 0

    def region(self, name: str) -> dict | None:
        for m in self.info.get("memory", []):
            if m.get("name") == name:
                return m
        return None

    def region_size(self, name: str) -> int:
        r = self.region(name)
        return int(r["size"]) if r else 0

    @property
    def encoded_input_bytes(self) -> int:
        """Sum of the per-input encoded regions, as dx_rt sums them."""
        return sum(int(i["memory"]["size"]) for i in self.info.get("inputs", []))

    @property
    def output_mem_bytes(self) -> int:
        """``output_all_size``: the region charged once per buffer.

        Named the way the file and the runtime name it. What it contains --
        activations only, or activations plus scratch and encoded
        intermediates -- is not established by this tool and is not claimed.
        """
        return self.region_size("OUTPUT")


@dataclass
class Container:
    path: Path
    file_size: int
    version: int
    header: dict
    target: str
    tasks: list[NpuTask]
    sections: list[Section] = field(default_factory=list)

    @property
    def cpu_models(self) -> dict:
        return self.header.get("data", {}).get("cpu_models", {}) or {}

    @property
    def hw(self) -> dict:
        return self.header.get("hw_config", {}) or {}


def _section(name: str, raw: dict) -> Section:
    return Section(name, raw.get("type", "bytes"), int(raw["offset"]), int(raw["size"]))


def _walk_sections(node: dict, prefix: str, out: list[Section]) -> None:
    for key, val in node.items():
        if not isinstance(val, dict):
            continue
        label = f"{prefix}{key}"
        if "offset" in val and "size" in val:
            out.append(_section(label, val))
        else:
            _walk_sections(val, f"{label}/", out)


def load(path: str | Path, target: str | None = None) -> Container:
    p = Path(path)
    blob_size = p.stat().st_size
    with p.open("rb") as f:
        prefix = f.read(8)
        if len(prefix) < 8 or prefix[:4] != MAGIC:
            raise DxnnError(f"{p.name}: not a DXNN container (magic {prefix[:4]!r})")
        version = struct.unpack("<I", prefix[4:8])[0]
        if version not in SUPPORTED_VERSIONS:
            raise DxnnError(
                f"{p.name}: container version {version} is not supported "
                f"(this tool reads {', '.join(map(str, SUPPORTED_VERSIONS))})"
            )
        raw = f.read(HEADER_SIZE - 8)
        text = raw.split(b"\x00", 1)[0].decode("utf-8")
        header = json.loads(text)

        data = header.get("data", {})
        compiled = data.get("compiled_data", {}) or {}
        if not compiled:
            raise DxnnError(f"{p.name}: no compiled_data section")
        targets = sorted(compiled)
        if target is None:
            if len(targets) > 1:
                raise DxnnError(
                    f"{p.name}: several targets {targets}; pass --target"
                )
            target = targets[0]
        elif target not in compiled:
            raise DxnnError(f"{p.name}: no target {target!r}; have {targets}")

        tasks: list[NpuTask] = []
        for task_name, secs in compiled[target].items():
            built = {k: _section(f"{task_name}/{k}", v) for k, v in secs.items()}
            if "rmap_info" not in built:
                raise DxnnError(f"{p.name}: task {task_name} has no rmap_info")
            ri = built["rmap_info"]
            f.seek(ri.start)
            tasks.append(NpuTask(task_name, built, json.loads(f.read(ri.size))))

    sections: list[Section] = []
    _walk_sections(data, "", sections)
    return Container(p, blob_size, version, header, target, tasks, sections)


# --------------------------------------------------------------------------
# the budget, exactly as dx_rt computes it
# --------------------------------------------------------------------------

@dataclass
class Budget:
    fixed: int
    per_buffer: int
    per_task: list[dict]

    def usage(self, buffers: int) -> int:
        """``_memUsage`` at ``task_data.cpp:454`` for a given buffer count."""
        return self.fixed + self.per_buffer * buffers


def budget(container: Container) -> Budget:
    fixed = 0
    per_buffer = 0
    rows = []
    for t in container.tasks:
        task_fixed = t.rmap_bytes + t.weight_bytes + t.ppu_bytes
        encoded_in = align(t.encoded_input_bytes, 64)
        out = t.output_mem_bytes
        fixed += task_fixed
        per_buffer += encoded_in + out
        rows.append({
            "task": t.name,
            "rmap": t.rmap_bytes,
            "weight": t.weight_bytes,
            "ppu": t.ppu_bytes,
            "encoded_input": t.encoded_input_bytes,
            "encoded_input_aligned": encoded_in,
            "output_all_size": out,
        })
    return Budget(fixed, per_buffer, rows)


def max_buffer_count(npu_memory: int, fixed: int, per_buffer: int, requested: int) -> int:
    """Reproduce ``InferenceEngine::calculateMaxBufferCount``.

    The walk-down stops at 1 whether or not 1 fits; the shortfall is only
    raised afterwards, and the check that raises it uses one buffer rather than
    the adjusted count. Reproduced as written, including that asymmetry --
    a reimplementation that "fixes" it would no longer predict the runtime.
    """
    adjusted = requested
    while adjusted > 1:
        if fixed + per_buffer * adjusted <= npu_memory:
            break
        adjusted -= 1
    if fixed + per_buffer > npu_memory:
        raise InsufficientMemory(
            f"NPU memory insufficient for model: required {fixed + per_buffer} bytes "
            f"(fixed {fixed} + per-buffer {per_buffer}), available {npu_memory} bytes"
        )
    return adjusted


# --------------------------------------------------------------------------
# checks -- things the file asserts about itself
# --------------------------------------------------------------------------

def checks(container: Container) -> list[tuple[bool, str]]:
    out: list[tuple[bool, str]] = []

    spans = sorted(((s.start, s.end, s.name) for s in container.sections if s.size > 0))
    for start, end, name in spans:
        if end > container.file_size:
            out.append((False, f"section {name} ends at {end} past end of file {container.file_size}"))
    overlaps = [
        f"{a[2]} and {b[2]}"
        for a, b in zip(spans, spans[1:]) if b[0] < a[1]
    ]
    out.append((
        not overlaps,
        "sections do not overlap" if not overlaps else f"sections overlap: {'; '.join(overlaps)}",
    ))

    for t in container.tasks:
        for region, actual in (("RMAP", t.rmap_bytes), ("WEIGHT", t.weight_bytes)):
            declared = t.region_size(region)
            out.append((
                declared == actual,
                f"{t.name}: {region} region {declared:,} == compiled section {actual:,}",
            ))
        declared_in = t.region_size("INPUT")
        out.append((
            declared_in == t.encoded_input_bytes,
            f"{t.name}: INPUT region {declared_in:,} == sum of input tensors {t.encoded_input_bytes:,}",
        ))

        # The strongest check available, because it uses a number the cost
        # model never reads. rmap_info declares a total ``size``; dx_rt's
        # footprint for a single buffer is fixed + per-buffer once. If the two
        # agree, the cost model has been reconstructed correctly.
        declared_total = int(t.info.get("size") or 0)
        single = (t.rmap_bytes + t.weight_bytes + t.ppu_bytes
                  + align(t.encoded_input_bytes, 64) + t.output_mem_bytes)
        if declared_total:
            out.append((
                declared_total == single,
                f"{t.name}: declared total {declared_total:,} == "
                f"single-buffer footprint {single:,}",
            ))
        regions = sum(int(m.get("size", 0)) for m in t.info.get("memory", []))
        out.append((
            regions == declared_total,
            f"{t.name}: memory regions sum to {regions:,} == declared total {declared_total:,}",
        ))
        for kind in ("inputs", "outputs"):
            for tensor in t.info.get(kind, []):
                alloc = int(tensor["memory"]["size"])
                logical = logical_bytes(tensor)
                name = tensor.get("name", "?")
                layout = tensor.get("layout", "?")
                if logical is not None:
                    out.append((
                        alloc >= logical,
                        f"{t.name}: {kind[:-1]} {name} allocation {alloc:,} "
                        f"covers logical {logical:,}",
                    ))
                pred = predicted_allocation(tensor)
                if pred is None:
                    out.append((
                        True,
                        f"{t.name}: {kind[:-1]} {name} layout {layout} has no "
                        f"stated padding rule -- allocation {alloc:,} not predicted",
                    ))
                else:
                    out.append((
                        pred == alloc,
                        f"{t.name}: {kind[:-1]} {name} element-aligned prediction "
                        f"{pred:,} == declared {alloc:,}  ({layout})",
                    ))
    return out


def logical_bytes(tensor: dict) -> int | None:
    """Element count times element width: the tensor's own size, before padding."""
    width = DTYPE_BYTES.get(str(tensor.get("dtype", "")).upper())
    shape = tensor.get("shape")
    if width is None or not shape:
        return None
    n = 1
    for d in shape:
        n *= int(d)
    return n * width


#: How many trailing axes each layout folds into the row it aligns.
#: ``ALIGNED`` pads the last axis; ``PRE_IM2COL`` pads the flattened im2col row,
#: which is the last two axes together.
LAYOUT_ROW_AXES = {"ALIGNED": 1, "PRE_IM2COL": 2}


def predicted_allocation(tensor: dict) -> int | None:
    """Predict a tensor's allocated region from its shape, layout and align unit.

    The rule, derived from the file and checked against it: **the alignment
    unit counts elements, not bytes.** Fold the trailing axes the layout names
    into one row, round that row up to a whole number of ``align_unit``
    elements, then multiply by the element width.

    It matters that this is elements. A 1000-wide float32 output aligned as
    *bytes* would occupy ``align(4000, 64) = 4032``; the file says 4096, which
    is ``align(1000, 64) * 4``. Reading it as bytes gets the right order of
    magnitude and the wrong number, which is the kind of error that survives
    review. Returns ``None`` for a layout with no stated rule rather than
    guessing one.
    """
    width = DTYPE_BYTES.get(str(tensor.get("dtype", "")).upper())
    shape = tensor.get("shape")
    unit = int(tensor.get("align_unit") or 0)
    axes = LAYOUT_ROW_AXES.get(str(tensor.get("layout", "")).upper())
    if width is None or not shape or unit <= 0 or axes is None or len(shape) < axes:
        return None
    row = 1
    for d in shape[-axes:]:
        row *= int(d)
    outer = 1
    for d in shape[:-axes]:
        outer *= int(d)
    return outer * align(row, unit) * width


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def _n(v: int) -> str:
    return f"{v:,}"


def parse_size(text: str) -> int:
    """``512M`` and ``1.98G`` are binary; ``512MB`` is decimal. Bare digits are bytes."""
    s = text.strip().upper().replace("IB", "I")
    factors = [
        ("KB", 1000), ("MB", 1000**2), ("GB", 1000**3), ("TB", 1000**4),
        ("KI", 1024), ("MI", 1024**2), ("GI", 1024**3), ("TI", 1024**4),
        ("K", 1024), ("M", 1024**2), ("G", 1024**3), ("T", 1024**4),
    ]
    for suffix, mult in factors:
        if s.endswith(suffix):
            return int(float(s[: -len(suffix)]) * mult)
    return int(float(s))


def render(container: Container, buffers: int, npu_memory: int | None) -> str:
    b = budget(container)
    hw = container.hw
    lines: list[str] = []

    macs = {t.info.get("npu", {}).get("mac") for t in container.tasks}
    mac = macs.pop() if len(macs) == 1 else None
    lines.append(
        f"container            DXNN v{container.version}"
        f"            target {container.target}"
        + (f"  ({mac} MACs)" if mac else "")
        + (f"  chip {hw['chip_version']}" if hw.get("chip_version") else "")
    )
    lines.append(f"file                 {container.path.name}  ({_n(container.file_size)} B)")
    if container.cpu_models:
        lines.append(f"cpu subgraphs        {len(container.cpu_models)} (ONNX, run on the host)")
    else:
        lines.append("cpu subgraphs        none -- the whole graph compiled to the NPU")

    for t in container.tasks:
        counts = t.info.get("counts", {})
        lines.append("")
        lines.append(
            f"task {t.name}           {counts.get('layer', '?')} layers"
            f"        {counts.get('cmd', '?')} commands"
        )
        for kind in ("inputs", "outputs"):
            for tensor in t.info.get(kind, []):
                shape = ",".join(str(d) for d in tensor.get("shape", []))
                logical = logical_bytes(tensor)
                alloc = int(tensor["memory"]["size"])
                lines.append(
                    f"  {kind[:-1]:<7}{tensor.get('name', '?'):<10}"
                    f"{tensor.get('dtype', '?'):<9}[{shape}]"
                    f"  {tensor.get('layout', '?')}  align {tensor.get('align_unit', '?')}"
                )
                if logical is not None:
                    lines.append(f"    {'logical':<22}{_n(logical):>16} B")
                lines.append(f"    {'allocated':<22}{_n(alloc):>16} B")
                if logical is not None and alloc != logical:
                    lines.append(f"    {'padding':<22}{_n(alloc - logical):>16} B")
        lines.append(f"    {'output_all_size':<22}{_n(t.output_mem_bytes):>16} B  charged per buffer")

    lines.append("")
    lines.append(
        f"fixed cost        rmap {_n(b.per_task[0]['rmap'])}"
        f" + weight {_n(b.per_task[0]['weight'])}"
        f" + ppu {_n(b.per_task[0]['ppu'])}"
        if len(b.per_task) == 1 else "fixed cost"
    )
    lines.append(f"                  = {_n(b.fixed)} B")
    lines.append(
        f"per-buffer cost   align64(encoded input) + output_all_size"
        f"  = {_n(b.per_buffer)} B"
    )

    lines.append("")
    lines.append(f"requested buffers        {buffers}"
                 + ("  (DXRT_TASK_MAX_LOAD_DEFAULT)" if buffers == DEFAULT_BUFFERS else ""))
    lines.append(f"memory at that count     {_n(b.usage(buffers))} B")
    if npu_memory is not None:
        lines.append(f"device memory            {_n(npu_memory)} B")
        try:
            fitted = max_buffer_count(npu_memory, b.fixed, b.per_buffer, buffers)
        except InsufficientMemory as e:
            lines.append(f"maximum fitting count    none -- {e}")
        else:
            lines.append(f"maximum fitting count    {fitted}")
            if fitted < buffers:
                lines.append(
                    f"  runtime would warn: Buffer count reduced from {buffers} to {fitted}"
                )
            head = [n for n in range(1, buffers + 1)]
            for n in head:
                total = b.usage(n)
                mark = "fits" if total <= npu_memory else "over"
                lines.append(f"    x{n:<3}{_n(total):>18} B   {mark}")

    results = checks(container)
    failed = [msg for ok, msg in results if not ok]
    lines.append("")
    lines.append(f"checks                   {len(results) - len(failed)}/{len(results)} pass")
    for ok, msg in results:
        if not ok:
            lines.append(f"  FAIL  {msg}")
    return "\n".join(lines)


def to_json(container: Container, buffers: int, npu_memory: int | None) -> dict:
    b = budget(container)
    results = checks(container)
    out: dict = {
        "file": str(container.path),
        "file_size": container.file_size,
        "container_version": container.version,
        "target": container.target,
        "hw_config": container.hw,
        "cpu_subgraphs": len(container.cpu_models),
        "tasks": [],
        "fixed_cost": b.fixed,
        "per_buffer_cost": b.per_buffer,
        "per_task": b.per_task,
        "requested_buffers": buffers,
        "memory_at_requested": b.usage(buffers),
        "checks": [{"ok": ok, "what": msg} for ok, msg in results],
        "checks_passed": all(ok for ok, _ in results),
    }
    for t in container.tasks:
        out["tasks"].append({
            "name": t.name,
            "counts": t.info.get("counts", {}),
            "inputs": [_tensor_json(x) for x in t.info.get("inputs", [])],
            "outputs": [_tensor_json(x) for x in t.info.get("outputs", [])],
            "output_all_size": t.output_mem_bytes,
        })
    if npu_memory is not None:
        out["npu_memory"] = npu_memory
        try:
            out["max_fitting_buffers"] = max_buffer_count(
                npu_memory, b.fixed, b.per_buffer, buffers)
            out["clamped"] = out["max_fitting_buffers"] < buffers
        except InsufficientMemory as e:
            out["max_fitting_buffers"] = None
            out["clamped"] = True
            out["error"] = str(e)
    return out


def _tensor_json(t: dict) -> dict:
    return {
        "name": t.get("name"),
        "dtype": t.get("dtype"),
        "shape": t.get("shape"),
        "layout": t.get("layout"),
        "align_unit": t.get("align_unit"),
        "logical_bytes": logical_bytes(t),
        "allocated_bytes": int(t["memory"]["size"]),
    }


# --------------------------------------------------------------------------
# self-test: the arithmetic, with no file involved
# --------------------------------------------------------------------------

def self_test() -> int:
    failures: list[str] = []

    def check(cond: bool, what: str) -> None:
        if not cond:
            failures.append(what)

    check(align(0) == 0, "align(0) is 0")
    check(align(1) == 64, "align(1) rounds to 64")
    check(align(64) == 64, "align(64) is unchanged")
    check(align(65) == 128, "align(65) rounds to 128")
    check(align(157696) == 157696, "the densenet input region is already aligned")

    # The unit counts elements, not bytes. Aligning the byte count instead
    # gives 4032 for a 1000-wide float32 output; the file says 4096.
    check(align(4000) == 4032, "aligning bytes gives 4032, which is the wrong answer")
    check(align(1000) * 4 == 4096, "aligning elements gives 4096, which the file confirms")

    out_tensor = {"dtype": "FLOAT32", "shape": [1, 1000],
                  "layout": "ALIGNED", "align_unit": 64}
    check(predicted_allocation(out_tensor) == 4096, "ALIGNED pads the last axis")
    in_tensor = {"dtype": "UINT8", "shape": [1, 224, 224, 3],
                 "layout": "PRE_IM2COL", "align_unit": 64}
    check(predicted_allocation(in_tensor) == 157696, "PRE_IM2COL pads the im2col row")
    check(predicted_allocation({**in_tensor, "layout": "SOMETHING_NEW"}) is None,
          "an unknown layout predicts nothing rather than guessing")

    fixed, per = 17_253_056, 56_301_568
    check(max_buffer_count(1 << 40, fixed, per, 6) == 6, "a large device keeps the request")
    check(max_buffer_count(fixed + per * 3, fixed, per, 6) == 3, "a tight device walks down to 3")
    check(max_buffer_count(fixed + per, fixed, per, 6) == 1, "room for one buffer yields one")

    try:
        max_buffer_count(fixed + per - 1, fixed, per, 6)
    except InsufficientMemory:
        pass
    else:
        failures.append("one byte short of a single buffer must raise")

    # Monotonic in device memory: more memory never fits fewer buffers.
    prev = 0
    for mem in range(fixed + per, fixed + per * 8, per // 4):
        n = max_buffer_count(mem, fixed, per, 8)
        check(n >= prev, "buffer count is monotonic in device memory")
        prev = n

    # The requested count is a ceiling, never raised.
    check(max_buffer_count(1 << 40, fixed, per, 2) == 2, "a big device does not raise the request")

    check(parse_size("512M") == 512 * 1024**2, "512M is binary")
    check(parse_size("512MB") == 512 * 1000**2, "512MB is decimal")
    check(parse_size("2G") == 2 * 1024**3, "2G is binary")
    check(parse_size("4096") == 4096, "a bare number is bytes")

    if failures:
        for f in failures:
            print(f"FAIL  {f}")
        print(f"\n{len(failures)} failed")
        return 1
    print("self-test: all arithmetic checks pass")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="dxnn-inspect",
        description="Read a .dxnn and reproduce DX-RT's memory budget for it.",
    )
    ap.add_argument("model", nargs="?", help="path to a .dxnn file")
    ap.add_argument("--npu-memory", help="device memory, e.g. 512M, 1.98G, or bytes")
    ap.add_argument("--buffers", type=int, default=DEFAULT_BUFFERS,
                    help=f"requested buffer count (default {DEFAULT_BUFFERS})")
    ap.add_argument("--target", help="compiled target, when the container has several")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--self-test", action="store_true", help="check the arithmetic and exit")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    if not args.model:
        ap.error("a .dxnn path is required (or --self-test)")
    if not 1 <= args.buffers <= MAX_BUFFERS:
        ap.error(f"--buffers must be between 1 and {MAX_BUFFERS}")

    try:
        container = load(args.model, args.target)
    except (DxnnError, OSError, json.JSONDecodeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    mem = parse_size(args.npu_memory) if args.npu_memory else None
    if args.json:
        print(json.dumps(to_json(container, args.buffers, mem), indent=2))
    else:
        print(render(container, args.buffers, mem))

    return 0 if all(ok for ok, _ in checks(container)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
