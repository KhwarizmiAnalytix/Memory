# Memory (`Library/Memory`)

**Memory** layer: unique `data_ptr<T>` + non-owning `data_view<T>`;
`allocator<T>` (CPU via **mimalloc** / **TBB** / platform aligned malloc;
CUDA/HIP/Metal via the process-wide **caching allocator**). Optional **NUMA**,
**memkind**. Always links **Logging**. Full design and status:
[`Docs/memory_design.md`](../../Docs/memory_design.md).

## Status (August 2026)

**Done.** Unique `data_ptr` (copy always clones) + non-owning `data_view`
(window on a `data_ptr`, or `borrow()` for foreign memory). PyTorch-style
512 B / 2–20 MiB GPU segment cache on CUDA, HIP, and Metal (expandable VM
segments, mutex dropped around driver malloc, OOM flush+retry). Process-wide
`empty_cache` / `memory_allocated` / `memory_reserved` / `set_memory_fraction`
/ `reset_peak_memory_stats`. Tensor copy always clones; wrap / `t()` / `slice()`
borrow. Sized ctors take `device_index` + stream; `assign_async` records dest
and expression sources. Metal + CMake/Bazel tests green on macOS.

**Still open.** CUDA/HIP runtime verification (no `TestHip*.cpp`); CUDA graphs
/ MemPool; pinned host cache; AllocConf knobs; OOM snapshot / alloc history;
`cudaMallocAsync` backend; views do not refcount owners; Metal `record_stream`
is a no-op (device 0, no fp64); tensor defaults still GPU 0 / default stream;
`empty_cache` is not re-exported from Vectorization. Do not call `empty_cache`
on the allocate/free hot path.

Details: [`Docs/memory_design.md`](../../Docs/memory_design.md) §10.

## Layout

- `CMakeLists.txt` — `MEMORY_ENABLE_*`, `MEMORY_CXX_STANDARD`.
- `BUILD.bazel` — `//Library/Memory:Memory`; GPU sources selected via Bazel defines.
- `Cmake/` — `cuda.cmake`, `hip`, `tbb_memory.cmake`, `numa.cmake`, etc.
- `common/` (`data_ptr.h`, `data_view.h`, device/NUMA), `helper/`, `gpu/`
  (CUDA/HIP caching allocator, Metal caching allocator + bind helpers),
  `profiler/` (unified cache stats).
- `Testing/Cxx/` — tests and benchmark binaries when enabled.

---

## CMake options

### Feature and toolchain

| CMake variable | Default | Summary |
|----------------|---------|---------|
| `MEMORY_ENABLE_LTO` | OFF | Link-time optimization |
| `MEMORY_ENABLE_COVERAGE` | OFF | Coverage instrumentation |
| `MEMORY_ENABLE_TESTING` | ON | Test subtree |
| `MEMORY_ENABLE_EXAMPLES` | OFF | Examples |
| `MEMORY_ENABLE_GTEST` | ON | GoogleTest |
| `MEMORY_ENABLE_BENCHMARK` | ON | Google Benchmark |
| `MEMORY_ENABLE_MIMALLOC` | ON | mimalloc |
| `MEMORY_ENABLE_MIMALLOC_STATS` | OFF | Compile mimalloc statistics (`MI_STAT=1` on the vendored `mimalloc-static`; release builds otherwise compile counters out). Enables `cpu::memory_allocator::{has_stats, stats_print, process_info}` and runtime reporting via `MIMALLOC_SHOW_STATS=1` (dump at process exit) / `MIMALLOC_VERBOSE=1`. setup.py: `--mimalloc_stats` (a `--` flag, not a dotted token — `_` is a token delimiter). Requires `MEMORY_ENABLE_MIMALLOC` |
| `MEMORY_ENABLE_MEMKIND` | OFF | memkind (Linux); forced OFF on non-Linux in CMake |
| `MEMORY_ENABLE_LIBTORCH` | From `VECTORIZATION_ENABLE_LIBTORCH` (setup.py `torch` token), else OFF | LibTorch CPU-allocation comparison benchmark (`BenchmarkPyTorchComparison`); requires LibTorch in `CMAKE_PREFIX_PATH` |
| `MEMORY_ENABLE_ICECC` / `MEMORY_ENABLE_CACHE` / `MEMORY_ENABLE_CLANGTIDY` / `MEMORY_ENABLE_FIX` / `MEMORY_ENABLE_IWYU` / `MEMORY_ENABLE_SANITIZER` / `MEMORY_ENABLE_SPELL` / `MEMORY_ENABLE_VALGRIND` | see `CMakeLists.txt` | Tooling |

### Set by module finders / platform

| CMake variable | Summary |
|----------------|---------|
| `MEMORY_ENABLE_CUDA` | From `include(cuda)` — GPU CUDA path |
| `MEMORY_ENABLE_HIP` | From HIP finder |
| `MEMORY_ENABLE_TBB` | From `Cmake/tbb_memory.cmake` (TBB **allocator**, not Parallel’s SMP TBB) |
| `MEMORY_ENABLE_NUMA` | From `include(numa)`; forced OFF on non-Unix in CMake |

### `CACHE STRING`

| CMake variable | Default | Values |
|----------------|---------|--------|
| `MEMORY_CXX_STANDARD` | 20 | `11`–`23` |
| `MEMORY_SANITIZER_TYPE` | address | if sanitizer ON |
| `MEMORY_LINKER_CHOICE` | default | `default`, `lld`, `mold`, `gold`, `lld-link` |
| `MEMORY_CACHE_BACKEND` | none | `none`, `ccache`, `sccache`, `buildcache` |

Public `MEMORY_HAS_*` macros are documented at the top of `CMakeLists.txt`.

---

## Bazel flags

Starlark: [`bazel/memory.bzl`](../../bazel/memory.bzl). Rules: [`bazel/BUILD.bazel`](../../bazel/BUILD.bazel).

### `--define` keys

| Define | `config_setting` | Effect |
|--------|------------------|--------|
| `memory_enable_cuda` | `//bazel:enable_cuda` | `MEMORY_HAS_CUDA=1`; CUDA sources/deps in `BUILD.bazel` |
| `memory_enable_hip` | `//bazel:enable_hip` | `MEMORY_HAS_HIP=1`; HIP sources/deps |
| `memory_enable_tbb` | `//bazel:memory_enable_tbb` | `MEMORY_HAS_TBB=1` (TBB **allocator**) |
| `memory_enable_mimalloc` | default **`true`** in root `.bazelrc`; **`false`** → `//bazel:disable_mimalloc` (no `@mimalloc` dep) | `MEMORY_HAS_MIMALLOC` |
| `memory_enable_numa` | `//bazel:enable_numa` | `MEMORY_HAS_NUMA=1` |
| `memory_enable_memkind` | `//bazel:enable_memkind` | `MEMORY_HAS_MEMKIND=1` |
| `memory_enable_benchmark` | (defaults in `.bazelrc`; no `select` in `memory.bzl`) | Parity with CMake; benchmark targets always listed in `Testing/Cxx/BUILD.bazel` unless you add gating |

### Convenience configs

`build:cuda`, `build:hip`, `build:mimalloc`, `build:numa`, `build:memkind`, `build:tbb` (`parallel_backend=tbb` + `memory_enable_tbb`) — see root `.bazelrc`.

### CMake-only

`MEMORY_CXX_STANDARD` → fixed `c++20` in `memory.bzl`. LTO, coverage, sanitizers, clang-tidy, linker/cache, spell, Valgrind, Icecream, `MEMORY_ENABLE_LIBTORCH` — **CMake only**.
