#!/usr/bin/env python3
"""Build and run a consumer using only the relocated Bazel distribution."""

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def library_env(lib_dir: Path, base: dict[str, str] | None = None) -> dict[str, str]:
    """Point the dynamic loader at a relocated shared-library directory.

    Windows searches PATH for DLLs. Linux uses LD_LIBRARY_PATH and macOS uses
    DYLD_LIBRARY_PATH; putting the directory on PATH does not load those libraries.
    """
    env = dict(os.environ if base is None else base)
    directory = str(lib_dir)
    if sys.platform == "win32":
        name = "PATH"
    elif sys.platform == "darwin":
        name = "DYLD_LIBRARY_PATH"
    else:
        name = "LD_LIBRARY_PATH"
    env[name] = directory + os.pathsep + env.get(name, "")
    return env


def verify(package: Path, shared: bool) -> None:
    with tempfile.TemporaryDirectory(prefix="memory-consumer-") as temporary:
        root = Path(temporary)
        prefix = root / "relocated prefix"
        with tarfile.open(package) as archive:
            archive.extractall(prefix, filter="data")
        libraries = list((prefix / "lib").glob("*"))
        suffixes = {library.suffix for library in libraries}
        if shared and not suffixes.intersection({".dll", ".dylib", ".so"}):
            raise RuntimeError("Shared distribution contains no shared library")
        if not shared and not suffixes.intersection({".a", ".lib"}):
            raise RuntimeError("Static distribution contains no archive")
        source = root / "consumer"
        source.mkdir()
        (source / "main.cpp").write_text(
            '#include "memory.h"\n'
            "#ifdef MEMORY_BUILDING_DLL\n#error DLL producer macro leaked to consumer\n#endif\n"
            "int main() { auto data = memory::allocate<int>(16); "
            "for (int i = 0; i < 16; ++i) data.data()[i] = i; "
            "return data.data()[15] == 15 ? 0 : 1; }\n"
        )
        (source / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.24)\n"
            "project(MemoryInstalledConsumer LANGUAGES CXX)\n"
            "find_package(Memory CONFIG REQUIRED)\n"
            "add_executable(consumer main.cpp)\n"
            "target_compile_features(consumer PRIVATE cxx_std_20)\n"
            "target_link_libraries(consumer PRIVATE Memory::Memory)\n"
            "enable_testing()\nadd_test(NAME consumer COMMAND consumer)\n"
        )
        build = root / "build"
        command = [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build),
            "-DCMAKE_PREFIX_PATH=" + str(prefix),
        ]
        if shutil.which("ninja"):
            command += ["-G", "Ninja"]
        subprocess.run(command, check=True)
        subprocess.run(
            ["cmake", "--build", str(build), "--config", "Release", "--parallel", "2"],
            check=True,
        )
        subprocess.run(
            ["ctest", "--test-dir", str(build), "-C", "Release", "--output-on-failure"],
            env=library_env(prefix / "lib"),
            check=True,
        )
        print("Relocated Memory package consumer passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--shared", action="store_true")
    args = parser.parse_args()
    verify(args.package.resolve(), args.shared)


if __name__ == "__main__":
    main()
