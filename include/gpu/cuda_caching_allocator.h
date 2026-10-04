/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * This file is part of XSigma and is licensed under a dual-license model:
 *
 *   - Open-source License (GPLv3):
 *       Free for personal, academic, and research use under the terms of
 *       the GNU General Public License v3.0 or later.
 *
 *   - Commercial License:
 *       A commercial license is required for proprietary, closed-source,
 *       or SaaS usage. Contact us to obtain a commercial agreement.
 *
 * Contact: licensing@xsigma.co.uk
 * Website: https://www.xsigma.co.uk
 */

#pragma once

#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "common/device.h"
#include "common/memory_macros.h"
#include "profiler/gpu_memory_snapshot.h"
#include "profiler/unified_memory_stats.h"

#include "common/memory_export.h"

namespace memory
{
namespace gpu
{
/**
 * @brief CUDA/HIP caching allocator with PyTorch CUDACachingAllocator semantics
 *
 * Behaviorally ports the core of PyTorch's CUDACachingAllocator
 * (c10/cuda/CUDACachingAllocator.cpp). Under MEMORY_HAS_HIP the same logic
 * runs on hipMalloc/hipEvent* via gpu/gpu_runtime.h (CUDA API spellings).
 *
 * - Requests rounded to 512-byte multiples; small (<= 1 MiB) requests are
 *   packed into 2 MiB segments, 1-10 MiB requests into 20 MiB segments, and
 *   larger requests rounded up to 2 MiB multiples - one driver malloc per segment
 * - Oversized cached blocks are split on reuse and the remainder returned to
 *   the pool; freed blocks coalesce with free neighbors
 * - Free pools are scoped per allocation stream; blocks are never reused on a
 *   different stream than the one they were allocated on
 * - Cross-stream uses are tracked via record_stream() (PyTorch recordStream
 *   semantics) or the deallocate stream hint; reuse is deferred with CUDA/HIP
 *   events until the recorded streams catch up
 * - On driver malloc failure the entire cache is flushed (pending events
 *   synchronized, whole cached segments released) and the allocation retried
 *   once before throwing std::bad_alloc
 *
 * XSigma extensions beyond upstream:
 * - Optional max_cached_bytes cap with largest-first trimming of releasable
 *   (whole-segment) cached blocks on deallocate; the default is unlimited,
 *   matching PyTorch
 *
 * @note Metal uses metal_caching_allocator (same size classes, sync dispatch).
 */
class MEMORY_VISIBILITY cuda_caching_allocator
{
public:
    // Opaque: a cudaStream_t / hipStream_t converts to it implicitly; the
    // implementation converts back, so this header needs no vendor runtime header.
    using stream_type = void*;

    /**
     * @brief Construct a CUDA caching allocator
     * @param device CUDA device index (default: 0)
     * @param max_cached_bytes Maximum bytes to cache (default: unlimited)
     * @throws std::runtime_error if device is invalid
     */
    MEMORY_API explicit cuda_caching_allocator(
        int device = 0, size_t max_cached_bytes = std::numeric_limits<size_t>::max());

    /**
     * @brief Destructor - releases all cached memory
     */
    MEMORY_API ~cuda_caching_allocator();

    /**
     * @brief Allocate GPU memory with caching
     *
     * All returned pointers are at least 256-byte aligned: blocks live at
     * 512-byte-rounded offsets within cudaMalloc segments, and the driver
     * guarantees segment bases are 256-byte aligned.
     *
     * @param size Number of bytes to allocate
     * @param stream CUDA stream for stream-aware caching (optional)
     * @return Pointer to allocated memory
     * @throws std::bad_alloc if allocation fails
     * @throws std::invalid_argument if size is zero
     */
    MEMORY_API void* allocate(size_t size, stream_type stream = nullptr);

    /**
     * @brief Deallocate GPU memory (may cache for reuse)
     * @param ptr Pointer to memory to deallocate
     * @param size Size of memory block (unused; kept for interface compatibility)
     * @param stream Stream hint: a stream other than the allocation stream is
     *        treated as a cross-stream use (recordStream semantics) and reuse
     *        is deferred until that stream's pending work completes
     * @throws std::invalid_argument if ptr is not owned by this allocator
     * @throws std::logic_error if double free detected
     */
    MEMORY_API void deallocate(void* ptr, size_t size, stream_type stream = nullptr);

    /**
     * @brief Deallocate GPU memory by looking up the allocation stream
     *
     * Looks up the stream from the cache_block (the allocation stream recorded
     * at allocate time) and deallocates. Suitable for use as a deleter function
     * pointer when the stream is not available to the caller (e.g. in a
     * storage_handle deleter).
     *
     * @param ptr Pointer to memory to deallocate
     * @param nbytes Size of memory block (unused; kept for deleter compatibility)
     * @throws std::invalid_argument if ptr is not owned by this allocator
     * @throws std::logic_error if double free detected
     */
    MEMORY_API void deallocate_with_stream_lookup(void* ptr, size_t nbytes) noexcept;

    /**
     * @brief Record a cross-stream use of a live allocation (PyTorch recordStream)
     *
     * Declares that the memory is (or will be) used on @p stream. When the
     * allocation is later freed, its reuse is deferred with a CUDA event until
     * all recorded streams' pending work has completed. Uses on the allocation
     * stream itself need no recording and are ignored.
     *
     * @param ptr Live allocation previously returned by allocate()
     * @param stream Stream on which the memory is used
     * @throws std::runtime_error if ptr is not a live allocation of this allocator
     */
    MEMORY_API void record_stream(void* ptr, stream_type stream);

    /// True when @p ptr is the base address of a live allocation from this cache
    /// (not an interior pointer, a freed block or foreign memory). Takes the
    /// device lock; used to validate copy endpoints before anything is submitted.
    MEMORY_API bool owns_live_allocation(void const* ptr) const;

    /**
     * @brief Clear all cached memory immediately
     * @note This will synchronize with all pending CUDA operations
     */
    MEMORY_API void empty_cache();

    /**
     * @brief Set maximum bytes to cache
     * @param bytes Maximum cache size (0 = no caching)
     */
    MEMORY_API void set_max_cached_bytes(size_t bytes);

    /**
     * @brief Get maximum cache size
     * @return Maximum bytes that can be cached
     */
    MEMORY_API size_t max_cached_bytes() const;

    /**
     * @brief Opt into cuMemMap / hipMem* segment backing (default: off)
     *
     * When false (default), segments use cudaMalloc / hipMalloc. The previous
     * always-on VM path reserved exactly one segment's worth of VA per cache
     * miss and never expanded it — a net loss versus driver malloc. Enable
     * only when experimenting with a real expandable-VA implementation.
     */
    MEMORY_API void set_expandable_segments(bool enabled);

    MEMORY_API bool expandable_segments() const;

    /**
     * @brief Cap reserved device memory as a fraction of device capacity
     *
     * Matches torch.cuda.set_per_process_memory_fraction. @p fraction is in
     * (0, 1]. Subsequent driver allocations that would push reserved bytes past
     * fraction * device_total_memory() flush the cache and then fail with
     * std::bad_alloc if still over the cap.
     */
    MEMORY_API void set_memory_fraction(double fraction);

    MEMORY_API double memory_fraction() const;

    /**
     * @brief Reset peak allocated/reserved/cached counters to the live values
     *
     * Matches torch.cuda.reset_peak_memory_stats.
     */
    MEMORY_API void reset_peak_stats();

    /**
     * @brief Device capacity in bytes (cudaMemGetInfo total / HIP equivalent)
     */
    MEMORY_API size_t device_total_memory() const;

    /**
     * @brief Callback invoked when an allocation cannot be served from the cache
     *
     * Free-memory callbacks run between the first cache miss and the cudaMalloc
     * fallback (upstream trigger_free_memory_callbacks). If any callback returns
     * true (it freed memory), the cache is retried once before the driver call.
     * Callbacks run while the allocator lock is held; the lock is recursive, so a
     * callback may safely deallocate or empty_cache() on this same allocator.
     */
    using free_memory_callback = std::function<bool()>;

    /**
     * @brief Register a free-memory callback (upstream FreeCudaMemoryCallbacksRegistry)
     * @param callback Returns true if it freed device memory
     */
    MEMORY_API void add_free_memory_callback(const free_memory_callback& callback);

    /**
     * @brief Remove all registered free-memory callbacks
     */
    MEMORY_API void clear_free_memory_callbacks();

    /**
     * @brief Get comprehensive allocation statistics
     * @return Statistics structure with performance metrics
     */
    MEMORY_API unified_cache_stats stats() const;

    // O(1) lock-free reads of the four basic counters (plan §6.1, P3.5).
    // Each is a single relaxed atomic load — no mutex acquired.  For a
    // consistent full snapshot (bytes_cached, cache_blocks, inactive_split),
    // use stats() instead.
    MEMORY_API size_t bytes_allocated_now()      const noexcept;
    MEMORY_API size_t peak_bytes_allocated_now() const noexcept;
    MEMORY_API size_t bytes_reserved_now()       const noexcept;
    MEMORY_API size_t peak_bytes_reserved_now()  const noexcept;
    // Bytes sitting in the free pools, immediately reusable (blocks awaiting
    // cross-stream events are not counted): O(1), no lock.
    MEMORY_API size_t bytes_cached_now()         const noexcept;
    MEMORY_API size_t peak_bytes_cached_now()    const noexcept;

    /**
     * @brief Enable or disable the allocation-history ring
     *        (`torch.cuda.memory._record_memory_history`).
     *
     * Independent of profiler memory-event reporting. When enabled, allocate / free /
     * segment / OOM actions are stored up to @p max_entries (oldest dropped).
     * @p max_entries 0 keeps the previous cap (default 100000).
     */
    MEMORY_API void record_memory_history(
        bool enabled, size_t max_entries = kDefaultMemoryHistoryEntries);

    /**
     * @brief Segment/block map plus history ring (`torch.cuda.memory._snapshot`).
     *
     * Does not capture C++/Python stacks. When history is enabled, a
     * `gpu_memory_trace_action::snapshot` entry is appended first.
     */
    MEMORY_API gpu_memory_snapshot snapshot();

    /**
     * @brief Get device index this allocator manages
     * @return CUDA device index
     */
    MEMORY_API int device() const;

    // Non-copyable but movable
    cuda_caching_allocator(const cuda_caching_allocator&)                       = delete;
    cuda_caching_allocator&            operator=(const cuda_caching_allocator&) = delete;
    MEMORY_API                         cuda_caching_allocator(cuda_caching_allocator&&) noexcept;
    MEMORY_API cuda_caching_allocator& operator=(cuda_caching_allocator&&) noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
/// Maximum device indices the process-wide registry supports (plan §6.3).
inline constexpr int kMaxDevices = 16;

/**
 * @brief Returns the process-wide caching allocator for a CUDA/HIP device.
 *
 * Lock-free on the warm path: a non-null atomic load returns immediately with
 * no mutex acquired.  The first call per device uses std::call_once to create
 * the allocator exactly once; subsequent calls are pure load+branch (§6.3,
 * P3.1: 0 registry locks on warm allocate).
 *
 * The allocator is a process-lifetime singleton (intentional leak, §6.3) so
 * it outlives any static-storage destructor that may still hold a reference.
 *
 * @param device_index CUDA/HIP device index in [0, kMaxDevices).
 * @return Reference to the shared caching allocator for the device.
 * @throws std::out_of_range if device_index is out of [0, kMaxDevices).
 */
MEMORY_API cuda_caching_allocator& caching_allocator_for_device(int device_index);

/**
 * @brief Flush cached segments for all initialized CUDA/HIP devices.
 *
 * Calls empty_cache() on every per-device allocator that has been initialized,
 * in device-index order (0 → kMaxDevices-1), releasing unreferenced backing
 * memory back to the driver.  Does not destroy the allocators (they remain
 * valid for the process lifetime).  Safe to call at any point; a no-op if no
 * device has been initialized yet.
 */
MEMORY_API void shutdown();
#endif

}  // namespace gpu
}  // namespace memory
