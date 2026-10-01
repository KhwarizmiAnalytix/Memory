# Memory

Repository-specific engineering guidance for allocation, ownership and GPU memory
management. See root `/CLAUDE.md` for general coding/testing/build rules.

[`Docs/memory_runtime_implementation_plan.md`](Docs/memory_runtime_implementation_plan.md)
is the only design document: goals, target architecture (§4), contracts (§5),
hot-path performance rules (§6), phased tasks (§7), validation manifest (§8) and
status (§9). Historical review, churn and benchmark evidence is condensed in its
appendices. Update it when implementation changes; do not add other design docs,
Done/Next tables or phase specifications here or in `Docs/`.

Use the plan's distinction between implemented, shim-tested and hardware-accepted
behavior. Performance changes must cite a Phase 0/8 measurement and keep the §6.1
hot-path invariant tests passing. When proposing designs, note the PyTorch/Eigen
parallel as §2 does.

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
- `data_ptr<T>` (`common/data_ptr.h`) — unique owner (move-only). Explicit
  `clone()` method for deep-copy. Stores `device_index_` and `stream_`.
  `view()` returns a `data_view`.
- `data_view<T>` (`common/data_view.h`) — non-owning window over a `data_ptr`
  (or `borrow()` for foreign memory). Does not keep the owner alive.
- `pinned_buffer<T>` (`common/pinned_buffer.h`) — move-only pinned host buffer
  for CUDA/HIP transfers, backed by `cpu::pinned_memory_allocator`. Registered
  streams must outlive buffer destruction; recycling waits for their completion.
  Unsupported on CPU-only/Metal builds. Separate pinned-memory stats/cache limit.
- GPU allocations (CUDA, HIP, or Metal — compile-time exclusive) go through
  `gpu/caching_allocator.h` → `gpu::caching_allocator_for_device(device_index)`:
  - CUDA/HIP: `cuda_caching_allocator` — PyTorch-style segment cache with
    stream-aware reuse, optional expandable VM segments (off by default),
    mutex dropped around driver
    malloc. HIP uses the same Impl via `gpu/gpu_runtime.h`.
  - Metal: `metal_caching_allocator` — same size classes on shared
    `MTLBuffer`s / heaps with explicit completion-token bookkeeping; actual
    command-buffer integration and acceptance are tracked in the canonical plan.
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

## Benchmark Documentation

- Measurement rules: plan §6.7; workloads and tuning tasks: plan Phase 0 and Phase 8.
- Historical CUDA results (Debug, bounded iterations): plan Appendix C; raw data in
  `Docs/cuda_baseline_2026-09-28.json`. Not acceptance evidence.
- `Testing/Cxx/BenchmarkCudaCachingAllocator.cpp` — 14 scenarios: cold/warm, sizes,
  cache efficiency, round-robin streams. `BenchmarkCudaCachingAllocatorChurn.cpp` —
  churn reproduction (plan Appendix B).
