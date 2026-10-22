"""Regression checks for compiler requirements and CMake host integration."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT = "Memory"
PREFIX = PROJECT.upper()
ROOT = Path(__file__).resolve().parents[2]
CHECKS = ROOT / "Cmake/checks.cmake"
CMAKE = shutil.which("cmake")
STANDARDS = (17, 20, 23)
FLOORS = {
    17: {"GNU": "9.1", "Clang": "9.0", "AppleClang": "11.0", "MSVC": "19.14"},
    20: {"GNU": "10.0", "Clang": "11.0", "AppleClang": "13.0", "MSVC": "19.29"},
    23: {"GNU": "11.0", "Clang": "17.0", "AppleClang": "15.0", "MSVC": "19.34"},
}


@unittest.skipUnless(CMAKE, "cmake is required")
class ConfigurationTests(unittest.TestCase):
    """Exercise failure paths without installing obsolete compilers."""

    def run_script(self, directory, content):
        """Run an isolated CMake script and capture its diagnostic output."""
        script = Path(directory) / "check.cmake"
        script.write_text(content, encoding="utf-8")
        return subprocess.run(
            [CMAKE, "-P", str(script)],
            capture_output=True,
            text=True,
            check=False,
        )

    def compiler_script(self, compiler, version, standard):
        """Stub platform discovery while exercising the real version checks."""
        return f"""
cmake_minimum_required(VERSION 3.24)
# Compiler validation must run even with an old build directory's cache.
set(TEMP_CHECKS_CONFIGURED TRUE)
set(TEMP_VALIDATION_COMPLETED stale)
set(CMAKE_CXX_COMPILER_ID "{compiler}")
set(CMAKE_CXX_COMPILER_VERSION "{version}")
set({PREFIX}_CXX_STANDARD {standard})
# Platform checks are unrelated to the simulated compiler version.
function(find_package)
endfunction()
set({PREFIX}_HAVE_WINDOWS_H TRUE)
set({PREFIX}_HAVE_UNISTD_H TRUE)
set({PREFIX}_HAVE_PTHREAD_H TRUE)
include("{CHECKS.as_posix()}")
"""

    def test_compiler_floors(self):
        """Accept each floor and reject older compilers for every standard."""
        with tempfile.TemporaryDirectory(prefix="cmake-compiler-") as directory:
            for standard in STANDARDS:
                for compiler, floor in FLOORS[standard].items():
                    with self.subTest(standard=standard, compiler=compiler):
                        result = self.run_script(
                            directory, self.compiler_script(compiler, floor, standard)
                        )
                        self.assertEqual(
                            result.returncode, 0, result.stdout + result.stderr
                        )
                        major, minor = map(int, floor.split("."))
                        older = f"{major}.{minor - 1}" if minor else f"{major - 1}.99"
                        result = self.run_script(
                            directory, self.compiler_script(compiler, older, standard)
                        )
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn(f"required for C++{standard}", result.stderr)

    def test_repeated_include_checks_new_standard(self):
        """A previous successful include must not bypass a higher requirement."""
        with tempfile.TemporaryDirectory(prefix="cmake-recheck-") as directory:
            content = self.compiler_script("GNU", "10.0", 20)
            content += f"""
set({PREFIX}_CXX_STANDARD 23)
include("{CHECKS.as_posix()}")
"""
            result = self.run_script(directory, content)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("required for C++23", result.stderr)

    def test_embedded_options_and_helpers(self):
        """Embedding preserves host options, probes, and helper functions."""
        with tempfile.TemporaryDirectory(prefix="cmake-host-") as directory:
            source = Path(directory)
            host = f"""
cmake_minimum_required(VERSION 3.24)
project(Host VERSION 1.0 LANGUAGES CXX C)
# Embedded hosts can supply the required Logging dependency themselves.
# Test Memory's helper ownership independently of vendored Logging's CMake.
add_library(Logging::Logging INTERFACE IMPORTED)
function(xsigma_target_apply_lto)
    set(HOST_HELPER_INTACT TRUE PARENT_SCOPE)
endfunction()
set({PREFIX}_ENABLE_TESTING OFF CACHE BOOL "")
set({PREFIX}_ENABLE_BENCHMARK OFF CACHE BOOL "")
set({PREFIX}_ENABLE_PROFILER OFF CACHE BOOL "")
set({PREFIX}_ENABLE_MIMALLOC OFF CACHE BOOL "")
set({PREFIX}_LTO_MODE off CACHE STRING "")
set({PREFIX}_CXX_STANDARD 20 CACHE STRING "")
set(BUILD_EXAMPLES ON CACHE BOOL "")
set(BUILD_BENCHMARKS ON CACHE BOOL "")
set(CMAKE_REQUIRED_FLAGS "-DHOST_CHECK_FLAG=1")
add_subdirectory("{ROOT.as_posix()}" module)
xsigma_target_apply_lto()
if(NOT HOST_HELPER_INTACT)
    message(FATAL_ERROR "Host LTO helper was overwritten")
endif()
if(NOT CMAKE_REQUIRED_FLAGS STREQUAL "-DHOST_CHECK_FLAG=1")
    message(FATAL_ERROR "Host probe flags were changed")
endif()
if(NOT BUILD_TESTING)
    message(FATAL_ERROR "Host BUILD_TESTING was changed")
endif()
if(NOT BUILD_EXAMPLES OR NOT BUILD_BENCHMARKS)
    message(FATAL_ERROR "Host example or benchmark options were changed")
endif()
get_target_property(features {PROJECT} INTERFACE_COMPILE_FEATURES)
if(NOT "cxx_std_20" IN_LIST features)
    message(FATAL_ERROR "Public C++ requirement was lost")
endif()
if(TARGET {PROJECT}CxxTests)
    message(FATAL_ERROR "Disabled tests were configured")
endif()
"""
            (source / "CMakeLists.txt").write_text(host, encoding="utf-8")
            build = source / "build"
            command = [CMAKE, "-S", str(source), "-B", str(build), "-DBUILD_TESTING=ON"]
            result = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            # Reconfigure the same cache with an explicit project opt-in and host veto.
            host = host.replace("if(NOT BUILD_TESTING)", "if(BUILD_TESTING)")
            (source / "CMakeLists.txt").write_text(host, encoding="utf-8")
            result = subprocess.run(
                command[:-1] + ["-DBUILD_TESTING=OFF", f"-D{PREFIX}_ENABLE_TESTING=ON"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_bundled_logging_preserves_host_options(self):
        """Restore global options even when bundled Logging forces cache entries."""
        with tempfile.TemporaryDirectory(prefix="cmake-logging-") as directory:
            source = Path(directory)
            host = f"""
cmake_minimum_required(VERSION 3.24)
project(Host VERSION 1.0 LANGUAGES CXX C)
set(MEMORY_ENABLE_TESTING OFF CACHE BOOL "")
set(MEMORY_ENABLE_BENCHMARK OFF CACHE BOOL "")
set(MEMORY_ENABLE_MIMALLOC OFF CACHE BOOL "")
set(MEMORY_ENABLE_PROFILER OFF CACHE BOOL "")
set(MEMORY_LTO_MODE off CACHE STRING "")
add_subdirectory("{ROOT.as_posix()}" module)
foreach(option IN ITEMS BUILD_TESTING BUILD_EXAMPLES BUILD_BENCHMARKS)
    get_property(value CACHE ${{option}} PROPERTY VALUE)
    if(NOT value)
        message(FATAL_ERROR "Host ${{option}} cache was changed")
    endif()
endforeach()
"""
            (source / "CMakeLists.txt").write_text(host, encoding="utf-8")
            command = [
                CMAKE,
                "-S",
                str(source),
                "-B",
                str(source / "build"),
                "-DBUILD_TESTING=ON",
                "-DBUILD_EXAMPLES=ON",
                "-DBUILD_BENCHMARKS=ON",
            ]
            for _ in range(2):
                result = subprocess.run(
                    command, capture_output=True, text=True, check=False
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
