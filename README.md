# Memory (`Library/Memory`)

**Memory** layer: CPU allocators, **GPU** memory (CUDA / HIP), **mimalloc**, optional **TBB** scalable allocator, **NUMA**, **memkind**, allocation statistics, and integration with **Logging** when enabled.

## Layout

- `CMakeLists.txt` — `MEMORY_ENABLE_*`, `MEMORY_GPU_ALLOC`, `MEMORY_CXX_STANDARD`.
- `BUILD.bazel` — `//Library/Memory:Memory`; GPU sources selected via Bazel defines.
- `Cmake/` — `cuda.cmake`, `hip`, `tbb_memory.cmake`, `numa.cmake`, etc.
- `cpu/`, `gpu/`, `backend/`, `common/` — implementation trees.
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
| `MEMORY_ENABLE_LOGGING` | ON | Link Logging; `MEMORY_HAS_LOGGING` |
| `MEMORY_ENABLE_MIMALLOC` | ON | mimalloc |
| `MEMORY_ENABLE_MEMKIND` | OFF | memkind (Linux); forced OFF on non-Linux in CMake |
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
| `MEMORY_GPU_ALLOC` | `POOL_ASYNC` | `SYNC`, `ASYNC`, `POOL_ASYNC` |
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
| `memory_gpu_alloc` | `//bazel:gpu_alloc_sync` / `gpu_alloc_async` / `gpu_alloc_pool_async` | Values: `sync`, `async`, `pool_async` |
| `memory_enable_allocation_stats` | `//bazel:enable_allocation_stats` | `MEMORY_HAS_ALLOCATION_STATS=1` |
| `memory_enable_benchmark` | (defaults in `.bazelrc`; no `select` in `memory.bzl`) | Parity with CMake; benchmark targets always listed in `Testing/Cxx/BUILD.bazel` unless you add gating |

### Convenience configs

`build:cuda`, `build:hip`, `build:mimalloc`, `build:numa`, `build:memkind`, `build:tbb` (`parallel_backend=tbb` + `memory_enable_tbb`), `build:gpu_alloc_*` — see root `.bazelrc`.

### CMake-only

`MEMORY_CXX_STANDARD` → fixed `c++20` in `memory.bzl`. LTO, coverage, sanitizers, clang-tidy, linker/cache, spell, Valgrind, Icecream — **CMake only**.
