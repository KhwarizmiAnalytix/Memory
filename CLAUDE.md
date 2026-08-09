# Memory

Allocation paths used by `data_ptr` and GPU memory management. See root
`/CLAUDE.md` for general coding/testing/build rules — this file only covers
what's specific to this library.

## What lives here (and why)

After the allocator consolidation, the library intentionally keeps these
allocation paths:

- `allocator<T>` (`allocator.h`) — the path `common/data_ptr.h` uses. CPU
  allocations call `helper/memory_allocator.h` (`cpu::memory_allocator`,
  a thin wrapper over mimalloc / TBB / platform aligned malloc) directly;
  there is no virtual allocator interface anymore.
- GPU allocations (CUDA, HIP, or Metal — compile-time exclusive) go through
  `gpu/caching_allocator.h` → `gpu::caching_allocator_for_device(device_index)`:
  - CUDA/HIP: `cuda_caching_allocator` — PyTorch-style segment cache with
    stream-aware reuse. HIP uses the same Impl via `gpu/gpu_runtime.h`
    (CUDA API spellings → hip*).
  - Metal: `metal_caching_allocator` — same segment-cache model on shared
    `MTLBuffer`s (`record_stream` is a no-op for sync dispatch). Kernel bind
    helpers live in `metal_buffer_allocator.{h,mm}` (`mtl_buffer_handle` /
    `mtl_buffer_offset` only).
- Shared size-class policy: `gpu/caching_allocator_config.h`.

`allocator<T>` dispatches GPU allocate/free through `is_active_gpu_device()`
so CUDA/HIP/Metal share one call site (Metal still rejects `double`).

The removed machinery (BFC/pool/retry/tracking backends, `process_state`,
the `Allocator` interface, `gpu_memory_*` helpers, `visualization/`) was
deleted deliberately — don't reintroduce it without a measured need.

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
