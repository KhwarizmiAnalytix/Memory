#!/usr/bin/env python3
"""Fail the allocator CI leg unless configure selected the requested backend.

Reads the CMake summary in configure.log and the generated build files
(Ninja or Visual Studio) so a flag that silently falls back to the default
cannot pass the job.
"""

from __future__ import annotations

import os
import pathlib
import sys

# Label -> value that must appear in the CMake summary, independent of padding.
SUMMARY = {
    "tbb": {
        "Mimalloc": "OFF",
        "Mimalloc stats": "OFF",
        "Tbb": "ON",
        "GPU backend": "none",
    },
    "native": {
        "Mimalloc": "OFF",
        "Mimalloc stats": "OFF",
        "Tbb": "OFF",
        "GPU backend": "none",
    },
    "mimalloc-stats": {
        "Mimalloc": "ON",
        "Mimalloc stats": "ON",
        "Tbb": "OFF",
        "GPU backend": "none",
    },
    "numa": {
        "Mimalloc": "ON",
        "Mimalloc stats": "OFF",
        "Tbb": "OFF",
        "Numa": "ON",
        "GPU backend": "none",
    },
    "memkind": {
        "Mimalloc": "ON",
        "Mimalloc stats": "OFF",
        "Tbb": "OFF",
        "Memkind": "ON",
        "GPU backend": "none",
    },
    "metal": {
        "Mimalloc": "ON",
        "Mimalloc stats": "OFF",
        "Tbb": "OFF",
        "GPU backend": "metal",
    },
    "profiler": {
        "Mimalloc": "ON",
        "Mimalloc stats": "OFF",
        "Tbb": "OFF",
        "Profiler": "ON",
        "GPU backend": "none",
    },
}

DEFINES = {
    "tbb": ("MEMORY_HAS_MIMALLOC=0", "MEMORY_HAS_TBB=1", "MEMORY_HAS_MIMALLOC_STATS=0"),
    "native": (
        "MEMORY_HAS_MIMALLOC=0",
        "MEMORY_HAS_TBB=0",
        "MEMORY_HAS_MIMALLOC_STATS=0",
    ),
    "mimalloc-stats": (
        "MEMORY_HAS_MIMALLOC=1",
        "MEMORY_HAS_TBB=0",
        "MEMORY_HAS_MIMALLOC_STATS=1",
    ),
    "numa": ("MEMORY_HAS_MIMALLOC=1", "MEMORY_HAS_NUMA=1", "MEMORY_HAS_TBB=0"),
    "memkind": ("MEMORY_HAS_MIMALLOC=1", "MEMORY_HAS_MEMKIND=1", "MEMORY_HAS_TBB=0"),
    "metal": (
        "MEMORY_HAS_MIMALLOC=1",
        "MEMORY_HAS_METAL=1",
        "MEMORY_HAS_CUDA=0",
        "MEMORY_HAS_HIP=0",
        "MEMORY_HAS_TBB=0",
    ),
    "profiler": (
        "MEMORY_HAS_MIMALLOC=1",
        "MEMORY_HAS_PROFILER=1",
        "MEMORY_HAS_TBB=0",
        "MEMORY_HAS_MIMALLOC_STATS=0",
    ),
}


def build_text(build_dir: pathlib.Path) -> str:
    """Return the generated build rules that carry compile definitions."""
    ninja = build_dir / "build.ninja"
    if ninja.is_file():
        return ninja.read_text(encoding="utf-8", errors="replace")
    chunks = []
    for path in build_dir.rglob("*"):
        if path.suffix.lower() in {".vcxproj", ".rsp"} or path.name == "flags.make":
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
    if not chunks:
        sys.exit(f"no build.ninja or vcxproj under {build_dir}")
    return "\n".join(chunks)


def main() -> int:
    allocator = os.environ.get("ALLOCATOR", "")
    if allocator not in SUMMARY:
        sys.exit(f"unknown allocator {allocator!r}")
    log_path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "configure.log")
    build_dir = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "build")
    log = log_path.read_text(encoding="utf-8", errors="replace")
    rules = build_text(build_dir)

    summary = {}
    for line in log.splitlines():
        label, separator, value = line.partition(":")
        if separator:
            summary[label.strip()] = value.strip()

    for label, value in SUMMARY[allocator].items():
        if summary.get(label) != value:
            sys.exit(f"configure summary missing '{label}: {value}'")

    if allocator == "tbb":
        if "Intel TBB memory allocator configuration complete" not in log:
            sys.exit("TBB allocator was not configured")
        if "TBB::tbbmalloc target is not available" in log:
            sys.exit("TBB::tbbmalloc was not found")
    if allocator == "numa":
        if "NUMA support enabled" not in log:
            sys.exit("NUMA was requested but not enabled")
        if "NUMA library not found" in log:
            sys.exit("libnuma was not found")
    if allocator == "metal" and "Metal framework:" not in log:
        sys.exit("Metal framework was not found")
    if (
        allocator == "profiler"
        and "Memory: using Profiler::Profiler (find_package)" not in log
    ):
        sys.exit("Profiler was not found via find_package")

    for define in DEFINES[allocator]:
        if define not in rules:
            sys.exit(f"build rules missing {define}")

    if allocator == "mimalloc-stats":
        if "MI_STATS=1" not in rules:
            sys.exit("mimalloc was not compiled with MI_STATS=1")
        if "MI_STATS=0" in rules:
            sys.exit("mimalloc still has MI_STATS=0")
    elif "MI_STATS=1" in rules:
        sys.exit("MI_STATS=1 was set on a non-stats build")

    print(f"verified {allocator}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
