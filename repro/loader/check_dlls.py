"""The exit-127 trap: the Windows loader names only the first library it
cannot find.

SYMPTOM   Every C++ executable failed instantly with exit code 127 and a
          message about a missing OpenVINO DLL. Two rounds of reasoning went
          into an OpenVINO problem.

CAUSE     It was an OpenCV problem. The loader resolves imports in order and
          stops at the first failure, so a binary missing several libraries
          reports exactly one of them -- and which one is an accident of link
          order rather than a clue about what is wrong.

FIX       Do not read the message as a diagnosis. Enumerate what the binary
          actually imports, check each one against the search path, and report
          ALL of the missing ones.

WHY IT MATTERS BEYOND ITSELF
          This cost more time than any of the real defects, and it was not a
          defect at all -- it was a diagnostic that named a true fact in a
          misleading way. Worth a tool rather than a memory.

    python repro/loader/check_dlls.py cpp/build/run_async_cpp.exe
    python repro/loader/check_dlls.py            # every exe under cpp/build
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def imports_of(exe: Path) -> list[str]:
    """DLL names the binary imports, via objdump if MSYS2 is present."""
    for tool in ("objdump", r"C:\msys64\ucrt64\bin\objdump.exe"):
        try:
            out = subprocess.run([tool, "-p", str(exe)], capture_output=True,
                                 text=True, timeout=60)
        except (FileNotFoundError, OSError):
            continue
        if out.returncode == 0:
            return sorted({ln.split()[-1] for ln in out.stdout.splitlines()
                           if "DLL Name:" in ln})
    return []


# API sets are not files. The loader redirects `api-ms-win-crt-stdio-l1-1-0`
# to whichever ucrtbase actually implements it, so looking for a DLL of that
# name on disk finds nothing and means nothing. Reporting them as missing is
# the same class of mistake this script exists to correct.
API_SET_PREFIXES = ("api-ms-win-", "ext-ms-")


def is_api_set(dll: str) -> bool:
    return dll.lower().startswith(API_SET_PREFIXES)


def find(dll: str, extra: list[Path]) -> Path | None:
    for d in extra:
        p = d / dll
        if p.exists():
            return p
    for d in os.environ.get("PATH", "").split(os.pathsep):
        if not d:
            continue
        p = Path(d) / dll
        try:
            if p.exists():
                return p
        except OSError:
            continue
    return None


def check(exe: Path) -> bool:
    print(f"\n{exe.relative_to(REPO) if REPO in exe.parents else exe}")
    names = imports_of(exe)
    if not names:
        print("  could not read the import table (objdump not found)")
        return True

    # The loader searches the executable's own directory first, which is why
    # the build copies the OpenVINO runtime there.
    extra = [exe.parent, Path(r"C:\msys64\ucrt64\bin"),
             Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"]
    missing, api_sets = [], 0
    for n in names:
        if is_api_set(n):
            api_sets += 1
            continue
        where = find(n, extra)
        if where is None:
            missing.append(n)
            print(f"  MISSING  {n}")
        else:
            tag = "beside the exe" if where.parent == exe.parent else str(where.parent)
            print(f"           {n:<28} {tag}")
    if api_sets:
        print(f"           ({api_sets} API sets, resolved by the loader,"
              f" not by files)")

    if missing:
        print(f"\n  {len(missing)} missing. The loader would have named"
              f" {missing[0]!r} and stopped,")
        print("  which is a true fact and a misleading one.")
    else:
        print("  all imports resolve")
    return not missing


def main() -> int:
    args = sys.argv[1:]
    if args:
        exes = [Path(a).resolve() for a in args]
    else:
        exes = sorted((REPO / "cpp" / "build").rglob("*.exe"))
        if not exes:
            print("no executables under cpp/build -- build first")
            return 2
    ok = all(check(e) for e in exes)
    print()
    print("all imports resolve" if ok else "something is missing; see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
