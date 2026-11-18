"""Environment fingerprint.

Every measurement run records the machine it ran on. Without this a number in
results/ is not reproducible and not comparable across runs.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone


def _try(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _cpu_name() -> str | None:
    if platform.system() == "Windows":
        # wmic is absent on recent Windows 11 builds, so try CIM first.
        out = _try(lambda: subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor).Name"],
            capture_output=True, text=True, timeout=20).stdout)
        if out and out.strip():
            return out.strip().splitlines()[0].strip()
        out = _try(lambda: subprocess.run(
            ["wmic", "cpu", "get", "name"], capture_output=True, text=True, timeout=10
        ).stdout)
        if out:
            lines = [l.strip() for l in out.splitlines() if l.strip()]
            if len(lines) > 1:
                return lines[1]
    return platform.processor() or None


def _gpu_names() -> list[str]:
    out = _try(lambda: subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
         "--format=csv,noheader"], capture_output=True, text=True, timeout=10
    ).stdout)
    if not out:
        return []
    return [l.strip() for l in out.splitlines() if l.strip()]


def collect() -> dict:
    fp: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_name": _cpu_name(),
        "cpu_count_logical": _try(lambda: __import__("os").cpu_count()),
        "gpus_nvidia_smi": _gpu_names(),
    }

    try:
        import psutil  # optional
        fp["ram_total_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
        freq = psutil.cpu_freq()
        fp["cpu_freq_mhz_at_start"] = round(freq.current, 0) if freq else None
    except Exception:
        fp["ram_total_gb"] = None
        fp["cpu_freq_mhz_at_start"] = None

    try:
        import onnxruntime as ort
        fp["onnxruntime_version"] = ort.__version__
        fp["onnxruntime_available_providers"] = ort.get_available_providers()
    except Exception as exc:
        fp["onnxruntime_version"] = None
        fp["onnxruntime_import_error"] = repr(exc)

    try:
        import cv2
        fp["opencv_version"] = cv2.__version__
    except Exception:
        fp["opencv_version"] = None

    return fp


if __name__ == "__main__":
    print(json.dumps(collect(), indent=2))
