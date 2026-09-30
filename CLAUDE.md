# Memory

Allocation paths used by `data_ptr` / `data_view` and GPU memory management.
See root `/CLAUDE.md` for general coding/testing/build rules — this file only
covers what's specific to this library. Design narrative, done/open list:
`Docs/memory_design.md` §10.

**Done (2026-08):** unique `data_ptr` + `data_view`; CUDA/HIP/Metal segment
cache (expandable segments, mutex dropped around malloc, process-wide
`empty_cache` / stats / fraction); tensor device_index + stream;
`assign_async` records expression sources. Tensor copy always clones.
CUDA caching allocator benchmarks (2026-09-29): direct malloc comparison,
cold/warm path analysis, fragmentation resilience, multi-stream scaling.

**Done (2026-09-30):** Phase 0–3 specifications AND CPU-side implementations complete:
- Phase 0: README corrections + validation manifest template ✓
- Phase 1: Failure safety + error semantics (spec ✓, CPU tests ✓)
- Phase 2: copy_sync() completion semantics (spec ✓, impl ✓, tests ✓)
- Phase 3: Storage identity + adoption + retained operations (spec ✓, impl ✓, tests ✓)
  - `allocation_id` type with process-wide uniqueness ✓
  - `allocate_adopted()` factory for foreign memory ✓
  - `copy_async_retained()` for retained transfers ✓
  - `retained_operation_service` API skeleton ✓
- All 206 tests passing (182 existing + 24 Phase 3 adoption/service)
See `Docs/phase0_1_2_3_summary.md` for overview; commit d23b18b for Phase 3 impl.

**Done (2026-09-30, Phase 2 GPU no-hardware fixes):**
- CopyRuntime shim: Testing/CopyRuntime/ for testing GPU code without hardware ✓
- copy_token: failed-state reporting, device_guard on queries/sync ✓
- allocator.h: peer-copy device_guard, copy_async_retained retention ✓
- retained_operation_service: condition_variable backpressure, failed tracking ✓
- All 11 CopyRuntime tests passing; no regression in 206 main tests ✓
See commit bc8f808 for service backpressure; 0adb77d for Phase 2 GPU fixes.

**Done (2026-09-30, Phase 1.1 churn-crash fix):**
- Root cause: `get_free_block_locked` constructed stack-local `cache_block key` whose
  `stream_uses` `std::set` head-node was allocated from a corrupted heap ✓
- Fix: `block_search_key` + transparent `cache_block_comparator` eliminates `cache_block`
  construction entirely from the lower_bound hot path; also nulled `src->prev`/`src->next`
  before `delete src` in `try_merge_locked` ✓
- Validated: 10-rep × 6-size churn benchmark clean; 255/255 tests pass ✓
- Finding 8 in `cpu_gpu_memory_review.md` closed ✓
- Full write-up + validation: `Docs/phase1_churn_diagnosis.md`

**Next (non-GPU, 2026-10):**
- Phase 3.5: retained_operation_service background polling thread
- Phase 2 GPU: device context activation + failed-state reporting (tests passing)
- Phase 1 diagnostics: cleanup hooks, quarantine counters (optional)

**Open (GPU hardware required):** 
- Phase 1 churn fix validation (Linux ASan); Phase 2 event-based completion; Phase 2 multi-stream ordering
- Optional: graphs/MemPool; AllocConf; `cudaMallocAsync`; OOM stack capture
- Known constraints: view does not refcount owner (by design, Phase 3 documents);
  Metal async / device 0 / no fp64; tensor defaults GPU 0; `empty_cache` not on Vectorization

## Implementation roadmap and design specifications

See the detailed phase plan in `Docs/memory_runtime_implementation_plan.md` for the complete architecture.

**Phases 0–3 CPU-independent work (2026-09-30):**

- `Docs/validation_manifest_template.md` — Use for every test run to record compiler, platform, GPU backend, hardware, and results
- `Docs/phase1_2_token_error_spec.md` — Token state machine (pending/complete/failed), error types, and API contract
- `Docs/phase2_copy_completion_spec.md` — `copy_sync()` semantics, operation-specific events (not stream queries), device context
- `Docs/phase3_storage_identity_spec.md` — `allocation_id`, adoption contract, retained ownership, borrowed pointer limits
- `Docs/phase0_1_2_3_summary.md` — Summary, implementation sequence, and what's ready to start

**Implementation progress (non-GPU first):**
1. ✅ Phase 2 CPU-side: completion_state enum, copy_sync(), token state queries (DONE)
2. ✅ Phase 1 non-churn: failure safety tests, overflow detection (DONE)
3. ✅ Phase 3 design+CPU: allocation_id, allocate_adopted(), copy_async_retained() (DONE)
4. Phase 4: retained_operation_service background polling thread (next)
5. Phase 1 GPU: churn diagnosis (needs CUDA/HIP debugger on self-hosted runners)
6. Phase 2 GPU: device context validation, event-based completion
7. Phases 5–8 depend on Phases 1–4

---

## CPU Memory Allocation — Fragmentation & Backend Characteristics

### Fragmentation Behavior (2026-09)

CPU allocators exhibit different fragmentation profiles under stress:

- **malloc (unaligned baseline)**: Coalesces adjacent free blocks across size
  classes (glibc malloc behavior). Lower fragmentation under many-small-allocate
  + selective-free patterns because segregated-list allocators cannot coalesce
  across size boundaries.
- **aligned_malloc / posix_memalign**: Segregated by alignment. Alignment requests
  > default force elevated heap overhead; fragmentation increases with alignment
  diversity.
- **mimalloc**: Eager per-thread local heaps reduce lock contention. Supports
  fast deallocation via segment reclamation. Fragmentation depends on thread
  affinity and deallocation order; "use after free"-like leaks are possible if
  pointers move between threads.
- **TBB scalable_malloc**: Partitioned heap by CPU. Cache-friendly for scalable
  workloads. Fragmentation grows with non-local access patterns (allocation
  on CPU 0, deallocation on CPU 1).

### Benchmark Alignment Fix (P0, 2026-09-29)

Fixed `memory_interface_api` wrapper in BenchmarkCPUMemoryAllocators.cpp to
forward alignment parameter to `cpu::memory_allocator::allocate()`. Previous
implementation silently ignored alignment, causing unaligned benchmarks for the
STL-style facade while other backends (mimalloc, TBB) received correct alignment.
This masked alignment-specific performance characteristics and produced unfair
comparisons.

### Tuning Parameters for Mimalloc (P1 Investigation)

Key environment variables for profiling:

- `MIMALLOC_SHOW_STATS=1` — dump counters at exit (also available via
  `memory::cpu::memory_allocator::stats_print()`)
- `MIMALLOC_EAGER_REGION_DELAY=<ms>` — delay before regions are reclaimed
  (default 100ms; set 0 for immediate reuse)
- `MIMALLOC_RESET_DELAY=<ms>` — when to decommit pages (default 0)
- `MIMALLOC_LARGE_OS_PAGES=1` — use huge pages (Linux/Windows; may require
  elevated privileges)
- `MIMALLOC_HEAP_DESTROY_DELAY=<ms>` — delay before heap cleanup on thread exit
- `MIMALLOC_VERBOSE=1` — enable verbose output during initialization

### Fragmentation Telemetry (P1 Roadmap)

Add to profiler:

- `fragmentation_ratio = (reserved - allocated) / reserved` per allocator
- `peak_memory_reserved` tracking (similar to GPU `max_memory_reserved`)
- Per-size-class allocation/deallocation counters (mimalloc via `mi_stats_*`)
- Thread-local heap migration events (mimalloc/TBB)

### Future Work (P2)

- **NUMA-aware allocation**: For large blocks (>100MB), detect NUMA topology and
  allocate on local node via `numa_alloc_local()` when available.
- **Memory pooling**: Pre-allocate fixed-size pools for predictable allocation
  patterns (e.g., model weights, activations).
- **Adaptive backend selection**: Route allocations to mimalloc (low contention,
  many threads), TBB (NUMA locality), or platform malloc (single-threaded)

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

## Benchmark Documentation

Performance analysis and baseline measurements:

- `Docs/cpu_gpu_memory_review.md` — CPU/GPU memory allocation comparison, Order 0 baseline recordings
- `Docs/cuda_benchmark_analysis.md` — CUDA caching allocator vs direct malloc: cold/warm paths,
  multi-stream scaling, fragmentation resilience, throughput analysis
- `Testing/Cxx/BenchmarkCudaCachingAllocator.cpp` — Benchmark suite: 14 test scenarios covering
  allocation patterns, cache efficiency, and contention under concurrent streams
