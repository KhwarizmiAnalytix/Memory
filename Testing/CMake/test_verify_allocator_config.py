"""Regression checks for allocator CI verification of CMake output."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CMAKE = shutil.which("cmake")
TBB_MARKER = "-- Intel TBB memory allocator configuration complete\n"
TBB_DEFINES = (
    "MEMORY_HAS_MIMALLOC=0",
    "MEMORY_HAS_TBB=1",
    "MEMORY_HAS_MIMALLOC_STATS=0",
)


class AllocatorVerificationTests(unittest.TestCase):
    """Accept summary formatting changes while rejecting incorrect backends."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory(prefix="allocator-verification-")
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)

    def verify(
        self, summary: str, defines: tuple[str, ...] = TBB_DEFINES
    ) -> subprocess.CompletedProcess[str]:
        """Exercise the same command-line entry point as allocator CI."""
        log = self.path / "configure.log"
        log.write_text(summary, encoding="utf-8")
        build = self.path / "build"
        build.mkdir(exist_ok=True)
        (build / "build.ninja").write_text(" ".join(defines), encoding="utf-8")
        return subprocess.run(
            [
                sys.executable,
                str(ROOT / "Scripts/verify_allocator_config.py"),
                str(log),
                str(build),
            ],
            env={**os.environ, "ALLOCATOR": "tbb"},
            capture_output=True,
            text=True,
            check=False,
        )

    def summary(self, padding: int = 21) -> str:
        """Represent both the original and current summary alignment."""
        values = {
            "Mimalloc": "OFF",
            "Mimalloc stats": "OFF",
            "Tbb": "ON",
            "GPU backend": "none",
        }
        return TBB_MARKER + "\n".join(
            f"  {label:<{padding}}: {value}" for label, value in values.items()
        )

    @unittest.skipUnless(CMAKE, "cmake is required")
    def test_current_cmake_summary(self) -> None:
        """Read the actual producer so alignment changes cannot break CI again."""
        assert CMAKE is not None
        script = self.path / "CMakeLists.txt"
        script.write_text(
            f"""
cmake_minimum_required(VERSION 3.24)
project(AllocatorSummary NONE)
set(MEMORY_STANDALONE ON)
set(MEMORY_ENABLE_MIMALLOC OFF)
set(MEMORY_ENABLE_MIMALLOC_STATS OFF)
set(MEMORY_ENABLE_TBB ON)
set(MEMORY_GPU_BACKEND none)
set(MEMORY_SOURCE_DIR "${{CMAKE_CURRENT_SOURCE_DIR}}")
function(memory_print_lto_linker_summary)
endfunction()
include("{(ROOT / "Cmake/summary.cmake").as_posix()}")
memory_print_configuration_summary()
""",
            encoding="utf-8",
        )
        result = subprocess.run(
            [CMAKE, "-S", str(self.path), "-B", str(self.path / "summary-build")],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        verified = self.verify(TBB_MARKER + result.stdout + result.stderr)
        self.assertEqual(verified.returncode, 0, verified.stdout + verified.stderr)
        self.assertIn("verified tbb", verified.stdout)

    def test_summary_spacing(self) -> None:
        for padding in (0, 20, 21, 30):
            with self.subTest(padding=padding):
                result = self.verify(self.summary(padding))
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_incorrect_or_missing_summary_value(self) -> None:
        for line in ("Tbb: OFF", "Tbb: ON_EXTRA", "Other Tbb: ON", ""):
            with self.subTest(line=line):
                summary = self.summary(0).replace("  Tbb: ON", line)
                result = self.verify(summary)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("configure summary missing 'Tbb: ON'", result.stderr)

    def test_missing_compile_definition(self) -> None:
        result = self.verify(self.summary(), TBB_DEFINES[:1] + TBB_DEFINES[2:])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("build rules missing MEMORY_HAS_TBB=1", result.stderr)

    def test_missing_tbb_configuration(self) -> None:
        result = self.verify(self.summary().replace(TBB_MARKER, ""))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("TBB allocator was not configured", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
