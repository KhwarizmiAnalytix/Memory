# Memory

Allocation paths used by `data_ptr` / `data_view` and GPU memory management.
See root `/CLAUDE.md` for general coding/testing/build rules — this file only
covers what's specific to this library. Design narrative, done/open list:
`Docs/memory_design.md` §10.

**Done (2026-08):** unique `data_ptr` + `data_view`; CUDA/HIP/Metal segment
cache (expandable segments, mutex dropped around malloc, process-wide
`empty_cache` / stats / fraction); tensor device_index + stream;
`assign_async` records expression sources. Tensor copy always clones.

**Open:** CUDA/HIP runtime tests; graphs/MemPool; pinned host cache; AllocConf;
OOM stack capture; `cudaMallocAsync`; view does not refcount owner; Metal async /
device 0 / no fp64; tensor defaults GPU 0; `empty_cache` not on Vectorization.

## What lives here (and why)

After the allocator consolidation, the library intentionally keeps these
allocation paths:

- `allocator<T>` (`allocator.h`) — the path `common/data_ptr.h` uses. CPU
  allocations call `helper/memory_allocator.h` (`cpu::memory_allocator`,
  a thin wrapper over mimalloc / TBB / platform aligned malloc) directly;
  there is no virtual allocator interface anymore. GPU allocate/free go
  through `gpu::caching_allocator_for_device`. Static helpers mirror
  `torch.cuda.memory`: `empty_cache`, `memory_allocated` /
  `max_memory_allocated`, `memory_reserved` / `max_memory_reserved`,
  `set_memory_fraction`, `reset_peak_memory_stats`.
- `data_ptr<T>` (`common/data_ptr.h`) — unique owner. Copy always deep-clones.
  Stores `device_index_` and `stream_`. `view()` returns a `data_view`.
- `data_view<T>` (`common/data_view.h`) — non-owning window over a `data_ptr`
  (or `borrow()` for foreign memory). Does not keep the owner alive.
- GPU allocations (CUDA, HIP, or Metal — compile-time exclusive) go through
  `gpu/caching_allocator.h` → `gpu::caching_allocator_for_device(device_index)`:
  - CUDA/HIP: `cuda_caching_allocator` — PyTorch-style segment cache with
    stream-aware reuse, optional expandable VM segments (off by default),
    mutex dropped around driver
    malloc. HIP uses the same Impl via `gpu/gpu_runtime.h`.
  - Metal: `metal_caching_allocator` — same size classes on shared
    `MTLBuffer`s / heaps (`record_stream` is a no-op for sync dispatch).
    Kernel bind helpers live in `metal_buffer_allocator.{h,mm}`.
- Shared size-class policy: `gpu/caching_allocator_config.h`.

`allocator<T>` dispatches GPU allocate/free through `is_active_gpu_device()`
so CUDA/HIP/Metal share one call site (Metal still rejects `double`).

Do **not** call `empty_cache` on the allocate/free hot path. Do **not**
reintroduce the deleted BFC/pool/retry/tracking backends, `process_state`,
the `Allocator` interface, `gpu_memory_*` helpers, or `visualization/`
without a measured need.

## GPU feature-guard macro: `MEMORY_HAS_CUDA` / `MEMORY_HAS_HIP` / `MEMORY_HAS_METAL`

All GPU-conditional code in `gpu/` must be guarded with `MEMORY_HAS_CUDA` /
`MEMORY_HAS_HIP` / `MEMORY_HAS_METAL`, defined by CMake from the selected
`--gpu_backend=`. **Not** `PROJECT_HAS_CUDA`/`PROJECT_HAS_HIP` — those
symbols don't exist anywhere in this repo, so code guarded by them compiles
out silently and the GPU path never actually runs. This exact bug hit 13
test files here before being fixed in commit `f15cf987`; if you touch a
`#if` guard in `gpu/` or `Testing/Cxx/TestGpu*.cpp`, double-check it's
`MEMORY_HAS_*` before assuming the branch is live.

## `try`/`catch` is allowed in `gpu/`, by exception

Root `/CLAUDE.md` bans `try`/`catch` in new application code by default,
but GPU code legitimately catches `std::exception` around calls into the
CUDA/HIP runtime, which throws on driver-level failures. This is an
intentional boundary around a third-party API, not a lapse — don't "clean
it up" to return-value-only error handling as a drive-by change, and match
this pattern (catch at the CUDA/HIP call boundary, translate to the
project's own error/result type immediately) if you add new GPU runtime
calls. Note `cuda_caching_allocator` itself throws
(`std::bad_alloc`/`std::invalid_argument`/`std::logic_error`) as part of
its API contract; callers going through `allocator<T>` inherit that
behavior on the allocation path.
