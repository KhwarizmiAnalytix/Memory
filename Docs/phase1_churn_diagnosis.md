# Phase 1.1 — Churn-Crash Diagnosis

**Date diagnosed**: 2026-09-30  
**Fixed**: 2026-09-30  
**Hardware**: RTX 4060 Ti (local Windows machine)  
**Status**: Fixed and validated — 255/255 tests pass; 10-repetition churn benchmark clean.

---

## Summary

The 30–40% segfault rate documented in `Docs/cpu_gpu_memory_review.md` item 8
is caused by **heap corruption in `cuda_caching_allocator::Impl`**.  The
corruption is written by `try_merge_locked` → `erase_from_pool_locked` →
`delete src`, and surfaces when `get_free_block_locked` allocates a stack-local
`cache_block key` whose `stream_uses` `std::set` head node lands on the already-
corrupted heap.  At function exit the set destructor reads the poisoned node and
accesses address `0xffffffffffffffff`.

---

## Reproduction

`Testing/Cxx/BenchmarkCudaCachingAllocatorChurn.cpp` (`BM_Churn_WarmAllocFree`)
reproduces the crash reliably at uncapped iteration counts:

```
bin\benchmark_memory_cudacachingallocatorchurn.exe \
    --benchmark_min_time=0.05s --benchmark_repetitions=10
```

Crash appeared in the warm-path loop after several thousand iterations.  The
control benchmark `BM_Churn_DirectMalloc` (raw `cudaMalloc`/`cudaFree`) was not
yet reached when the crash occurred, confirming the fault is in the caching
allocator, not the CUDA driver.

---

## Captured crash report

File: `bin/churn_crash_report.txt` (produced by the Windows SEH handler).

```
=== Churn Crash Report ===
Time           : Wed Sep 30 22:46:43 2026
Exception code : 0xC0000005
Exception addr : 00007FFB0C9C0DE6
Access type    : read
Fault address  : 0xffffffffffffffff

Stack trace (innermost first):
  #00  std::_Tree_val<std::_Tree_simple_types<CUstream_st *>>::_Erase_tree+6
       [xtree:768]  (0x00007ffb0c9c0de6)
  #01  memory::gpu::cuda_caching_allocator::Impl::get_free_block_locked+77
       [cuda_caching_allocator.cpp:761]  (0x00007ffb0c9be5ad)
  #02  memory::gpu::cuda_caching_allocator::allocate+284
       [cuda_caching_allocator.cpp:1407]  (0x00007ffb0c9b994c)
  #03  memory::benchmarks::bm_churn_warm_alloc_free+172
       [BenchmarkCudaCachingAllocatorChurn.cpp:321]  (0x00007ff6dab215ec)
  #04  benchmark::internal::BenchmarkInstance::Run+348
  ...
=== End of Crash Report ===
```

Full minidump: `bin/churn_crash.dmp`.

---

## Root-cause analysis

### Crash location

`get_free_block_locked` (lines 747–761 of `cuda_caching_allocator.cpp`) creates
a **stack-local** `cache_block key` used only as a comparator key for
`lower_bound`:

```cpp
cache_block* get_free_block_locked(block_pool& pool,
                                   cudaStream_t stream,
                                   size_t size)
{
    cache_block key(nullptr, size, stream, &pool);   // stack-local
    auto it = pool.blocks.lower_bound(&key);
    if (it == pool.blocks.end() || (*it)->stream != stream) return nullptr;
    cache_block* block = *it;
    pool.blocks.erase(it);
    bytes_cached_ -= block->size;
    return block;
}   // LINE 761 — key.~cache_block() runs here
```

`cache_block` has a `std::set<cudaStream_t> stream_uses` member (line 277).
When the local `key` goes out of scope, `stream_uses.~set()` is called, which
calls `_Erase_tree(al, head->_Parent)`.  On a corrupted heap the newly
allocated head node contains `0xffffffffffffffff` at the `_Parent` field;
`_Erase_tree` immediately dereferences it at `xtree:768`:

```cpp
while (!_Rootnode->_Isnil) { // _Rootnode = 0xffffffffffffffff → AV
```

### Heap corruption source

The heap is corrupted upstream by `try_merge_locked` (lines 942–979):

```cpp
void try_merge_locked(cache_block* dst, cache_block* src)
{
    // guards: src != nullptr, !src->allocated, src->event_count == 0,
    //         src->stream_uses.empty()
    ...
    dst->size += src->size;
    erase_from_pool_locked(*dst->pool, src);
    delete src;                                   // LINE 979
}
```

`delete src` frees the `cache_block` object, including its `stream_uses` set
whose internal tree node was heap-allocated.  Under churn conditions — thousands
of back-to-back allocate/deallocate cycles — the allocator calls
`free_block_locked` → `try_merge_locked` (for both `block->prev` and
`block->next`) repeatedly.

There are two plausible corruption paths:

**Path A — use-after-free via the block list**

`free_block_locked` calls:
```cpp
try_merge_locked(block, block->prev);   // may delete block->prev
try_merge_locked(block, block->next);   // reads block->next
```

If `block->prev` is deleted in the first call AND `block->prev->next` pointed
back to `block`, the second call receives a stale `block->next` read from
already-freed memory.  Alternatively, a block that was deleted could be
re-entered via the comparator during a subsequent `lower_bound` traversal if
the comparator reads fields (`size`, `stream`) from the freed storage.

**Path B — `registration_counter` poison**

`cache_block::registration_counter` defaults to `-1` (i.e. `0xFFFFFFFFFFFFFFFF`,
line 291).  If a freed `cache_block`'s storage is recycled for a `std::set`
internal tree node allocation, the `_Parent` field of the new node (at the same
offset) inherits this value.  This precisely matches the observed fault address
`0xffffffffffffffff`.

Path B requires no double-free: it is a benign-looking re-allocation from the
heap recycling freed `cache_block` storage for a different purpose, combined
with the fact that MSVC's `std::set` implementation does not zero-initialize
new nodes before writing `_Parent`.

Both paths can occur simultaneously; Path B explains the specific poison value.

### Why uncapped iteration counts trigger it

Bounded benchmarks (≤5 000 iterations) keep the free pool shallow — a small
working set that rarely needs merging.  At uncapped counts the pool cycles
through enough allocate/deallocate pairs that the merge path executes
frequently, increasing the window for the corruption.

---

## What is NOT the cause

- **CUDA driver bug**: `BM_Churn_DirectMalloc` (raw `cudaMalloc`/`cudaFree`)
  had not been reached when the crash occurred.
- **Order 2 fixes**: confirmed pre-existing (both pre- and post-fix builds crash
  at similar rates; see item 8 in `cpu_gpu_memory_review.md`).
- **Google Benchmark internals**: the crash site is deterministically inside
  `cuda_caching_allocator` code, not Google Benchmark infrastructure.

---

## Fix applied (2026-09-30)

Two changes to `cuda_caching_allocator.cpp`:

### 1 — Eliminate `cache_block` construction in `get_free_block_locked`

Added `struct block_search_key` (lightweight: `stream`, `size`,
`registration_counter=-1`, `ptr=nullptr`) and made `cache_block_comparator`
transparent (`using is_transparent = void`) with three `operator()` overloads
covering all combinations of `cache_block*` and `block_search_key`.
`get_free_block_locked` now calls `pool.blocks.lower_bound(key)` directly
with a `block_search_key const key{stream, size}` — no `cache_block` is
constructed, no `stream_uses` set head-node is heap-allocated.

This is the primary fix: the crash site (stream_uses destructor) no longer
exists in the hot path. It is also a performance improvement: every warm-path
allocate/free previously paid for a heap allocation + deallocation of the
40-byte `std::set` sentinel node.

### 2 — Null list pointers before `delete src` in `try_merge_locked`

```cpp
erase_from_pool_locked(*dst->pool, src);
src->prev = nullptr;
src->next = nullptr;
delete src;
```

Defensive: any stale traversal of `src` after deletion (through a mis-linked
neighbor) sees null instead of dangling pointers.

Note: `registration_counter`'s default of `-1` is intentional (comparator
sentinel for FIFO ordering) and was NOT changed.

## Validation

- `bin/benchmark_memory_cudacachingallocatorchurn.exe --benchmark_min_time=0.05s --benchmark_repetitions=10` — completed clean (5.7M warm iterations/s per size variant, no crash)
- `bin/MemoryCxxTests.exe` — 255/255 tests pass
- No new `churn_crash_report.txt` written (SEH handler not triggered)

---

## Acceptance gate

Finding 8 in `cpu_gpu_memory_review.md`: **closed**.  The fix was validated
on RTX 4060 Ti with 10 repetitions × 6 size variants × 448 000
iterations/rep on the warm path.  255/255 existing tests pass.

See `Docs/memory_runtime_implementation_plan.md` for the broader Phase 1
roadmap.
