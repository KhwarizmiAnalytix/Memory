# Review fixes — 2026-11-05

Fixes for a code review of the staged change set (five functional issues and one
test portability issue). All six findings are addressed; the changes were made on
top of the staged set and are not yet committed.

## Fixes

### 1. [P1] Installed static CMake packages had unresolved allocator dependencies

A static `libMemory.a` does not absorb its `PRIVATE` dependencies: CMake exports
them as `$<LINK_ONLY:...>`, and the installed `MemoryTargets.cmake` referenced
`Mimalloc::mimalloc`, which no installed package provided. A consumer's
`find_package(Memory)` failed at configure time (reproduced before fixing).

- [CMakeLists.txt](../../CMakeLists.txt): the local wrapper targets
  (`Mimalloc::mimalloc`, `Tbb::tbbmalloc`, `Numa::numa`) are linked through
  `$<BUILD_INTERFACE:...>`, so the export never names them. The underlying
  mimalloc target name and the TBB/NUMA selection are captured as
  `MEMORY_PACKAGE_*` values for the package config. `FindNuma.cmake` is installed
  next to the config for static NUMA builds.
- [Cmake/MemoryConfig.cmake.in](../../Cmake/MemoryConfig.cmake.in): for static
  packages only, recreates the dependency targets and attaches them `LINK_ONLY`,
  following the existing Logging/Profiler re-attach pattern:
  - mimalloc: `find_dependency(mimalloc CONFIG HINTS <prefix>)`, then links the
    same target the build used — `mimalloc-static` when vendored (its own install
    rules ship `lib/cmake/mimalloc-<ver>`), or the external `mimalloc` target.
  - TBB: `find_dependency(TBB COMPONENTS tbbmalloc)`.
  - NUMA: the installed `FindNuma.cmake`.
  - CUDA/HIP: `find_dependency(CUDAToolkit)` / `find_dependency(hip)`; `CUDA::*`
    and `hip::host` are real package targets and stay in the export unchanged.

### 2. [P1] Default Release CUDA configuration failed because LTO defaults to `auto`

[Cmake/lto.cmake](../../Cmake/lto.cmake): with `MEMORY_GPU_BACKEND=cuda`, `auto`
now degrades to LTO off with a status message. An explicitly requested
`thin`/`full`/`ipo` still fails with a fatal error.

### 3. [P2] `data_view(retained_ptr const&)` was unreachable through its headers

The definition lived in `data_ptr.h`, so a consumer including only
`common/retained_ptr.h` and `common/data_view.h` got an undefined-inline warning
and a link failure. It moved to
[include/common/data_view.h](../../include/common/data_view.h), where
`retained_ptr<T>` is already complete via `allocator.h` → `retained_ptr.h`.

### 4. [P2] Named CLI options dropped dotted compiler and generator selections

[Scripts/helpers/cli.py](../../Scripts/helpers/cli.py) gained
`dotted_compiler()`, `dotted_generator()` and `VS_GENERATORS`, the single
translation for tokens such as `gcc-13`, `clang-18`, `/opt/bin/g++`, `ninja`,
`xcode` and `vs22`. Both named-option dispatch and the legacy pipeline in
[Scripts/setup.py](../../Scripts/setup.py) now use them, so the two modes cannot
disagree. The last dotted choice wins, matching legacy in-order processing; a
dotted `vsNN` with no compiler lets Visual Studio choose, as before.

Behavior note: in named-option mode, `config.vs22` on a non-Windows host now
requests the Visual Studio generator (CMake fails loudly) rather than silently
falling back to Ninja. The legacy mode still honors `vsNN` only on Windows.

Regression test: `test_named_options_preserve_dotted_compiler_and_generator` in
[Testing/CMake/test_setup_cli.py](../../Testing/CMake/test_setup_cli.py).

### 5. [P2] The CUDA allocator required C++20 in a supported C++17 configuration

[src/gpu/cuda_caching_allocator.cpp](../../src/gpu/cuda_caching_allocator.cpp):
`std::bit_cast<void*>(addr)` reverted to `reinterpret_cast` with its original
`NOLINT(performance-no-int-to-ptr)`, and the `<bit>` include was removed. This
matches the inverse conversion in `cu_vm_free`.

### 6. [P3] Bazel helper test failed on macOS because of `/tmp` resolution

[Testing/CMake/test_bazel_setup.py](../../Testing/CMake/test_bazel_setup.py):
the test uses a temporary directory and builds the expected
`--output_user_root` with the same `Path.resolve()` normalization as the helper
(macOS resolves `/tmp` to `/private/tmp`).

## Validation

| Check | Result |
|---|---|
| CMake build + CTest (`setup.py config.build.test`) | 9/9 passed |
| Bazel build + tests (`setup_bazel.py config.build.test`) | 6/6 passed |
| Python tests under `Testing/CMake` | 38 passed, 1 skipped (previous failure fixed) |
| Static package, external mimalloc: consumer configures, links, runs | Passed |
| Static package, vendored mimalloc, including a relocated install prefix | Passed |
| Consumer including only `retained_ptr.h` + `data_view.h` | Builds without warnings |
| Isolated LTO project with the CUDA backend | `auto` → off; `thin` → error; `off` → off |
| clang-format, ruff, flake8, mypy, `git diff --check` on touched files | Clean |

The static package checks used a scratch consumer that allocates with
`make_retained<float>` and views the result through `data_view`, built against a
separately installed static Logging package.

## Not verified and known issues

- **lintrunner is not usable in this environment.** lintrunner 0.12.7 fails with
  `No such file or directory (os error 2)` before any adapter runs, so the
  underlying tools were run directly. `cmake-format --check CMakeLists.txt`
  already failed on two hunks unrelated to these fixes.
- **Static ThinLTO packages do not link on macOS (pre-existing).** With the
  default `auto` LTO, a consumer of the static package fails with an undefined
  `validate_stream_handle`. The archive contains Homebrew LLVM bitcode but is
  created with `/usr/bin/ar`, which cannot index it (the
  `ranlib: could not get target triple from bitcode file` warnings). The package
  fix was therefore validated with `MEMORY_LTO_MODE=off`. Using `llvm-ar` with
  Homebrew clang is the likely remedy; untested.
- **Untested here:** the TBB, NUMA, CUDA and HIP static-package paths (not
  available on this macOS host), a real CUDA SDK compile under C++17, and
  Windows builds.
