/*
 * Quarisma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * This file is part of Quarisma and is licensed under a dual-license model:
 *
 *   - Open-source License (GPLv3):
 *       Free for personal, academic, and research use under the terms of
 *       the GNU General Public License v3.0 or later.
 *
 *   - Commercial License:
 *       A commercial license is required for proprietary, closed-source,
 *       or SaaS usage. Contact us to obtain a commercial agreement.
 *
 * Contact: licensing@quarisma.co.uk
 * Website: https://www.quarisma.co.uk
 */

#pragma once

#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <string>

#include "common/device.h"
#include "common/memory_macros.h"
#include "profiler/unified_memory_stats.h"

#if MEMORY_HAS_CUDA
#include <cuda_runtime_api.h>

#include "common/memory_export.h"
#endif

namespace memory
{
namespace gpu
{
/**
 * @brief CUDA caching allocator with PyTorch CUDACachingAllocator semantics
 *
 * Behaviorally ports the core of PyTorch's CUDACachingAllocator
 * (c10/cuda/CUDACachingAllocator.cpp):
 * - Requests rounded to 512-byte multiples; small (<= 1 MiB) requests are
 *   packed into 2 MiB segments, 1-10 MiB requests into 20 MiB segments, and
 *   larger requests rounded up to 2 MiB multiples - one cudaMalloc per segment
 * - Oversized cached blocks are split on reuse and the remainder returned to
 *   the pool; freed blocks coalesce with free neighbors
 * - Free pools are scoped per allocation stream; blocks are never reused on a
 *   different stream than the one they were allocated on
 * - Cross-stream uses are tracked via record_stream() (PyTorch recordStream
 *   semantics) or the deallocate stream hint; reuse is deferred with CUDA
 *   events until the recorded streams catch up
 * - On cudaMalloc failure the entire cache is flushed (pending events
 *   synchronized, whole cached segments released) and the allocation retried
 *   once before throwing std::bad_alloc
 *
 * XSigma extensions beyond upstream:
 * - Optional max_cached_bytes cap with largest-first trimming of releasable
 *   (whole-segment) cached blocks on deallocate; the default is unlimited,
 *   matching PyTorch
 *
 * Features:
 * - Stream-aware memory caching with CUDA events
 * - Configurable cache size limits
 * - Comprehensive performance statistics
 * - Thread-safe operations
 * - Exception-safe RAII design
 * - Integration with Memory device management
 *
 * @note This allocator is designed for CUDA devices only
 */
class MEMORY_VISIBILITY cuda_caching_allocator
{
public:
#if MEMORY_HAS_CUDA
    using stream_type = cudaStream_t;
#else
    using stream_type = void*;
#endif

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
    MEMORY_API void add_free_memory_callback(free_memory_callback callback);

    /**
     * @brief Remove all registered free-memory callbacks
     */
    MEMORY_API void clear_free_memory_callbacks();

    /**
     * @brief Get comprehensive allocation statistics
     * @return Statistics structure with performance metrics
     */
    MEMORY_API unified_cache_stats stats() const;

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

#if MEMORY_HAS_CUDA
/**
 * @brief Returns the process-wide caching allocator for a CUDA device.
 *
 * Lazily creates one cuda_caching_allocator per device index and returns the
 * shared instance. The registry lives inside the Memory library so that all
 * translation units (and all libraries linking Memory) share the same
 * per-device allocators.
 *
 * @param device_index CUDA device index (must be valid for the host)
 * @return Reference to the shared caching allocator for the device
 *
 * **Thread Safety**: Thread-safe; creation is serialized internally
 */
MEMORY_API cuda_caching_allocator& caching_allocator_for_device(int device_index);
#endif

/**
 * @brief Template wrapper for type-safe CUDA caching allocator
 *
 * Provides a template interface compatible with Memory's GPU allocator patterns
 * while leveraging the high-performance caching allocator underneath.
 *
 * @tparam T Element type
 * @tparam alignment Memory alignment requirement (default: 256ULL bytes)
 */
template <typename T, std::size_t alignment = 256ULL>
class cuda_caching_allocator_template
{
public:
    using value_type      = T;
    using pointer         = T*;
    using const_pointer   = const T*;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using stream_type     = cuda_caching_allocator::stream_type;

    static constexpr size_type scalar_size     = sizeof(value_type);
    static constexpr size_type alignment_bytes = alignment;

    /**
     * @brief Construct template allocator
     * @param device CUDA device index
     * @param max_cached_bytes Maximum cache size
     */
    explicit cuda_caching_allocator_template(
        int device = 0, size_t max_cached_bytes = std::numeric_limits<size_t>::max())
        : allocator_(device, max_cached_bytes)
    {
    }

    /**
     * @brief Allocate aligned memory for elements
     * @param count Number of elements to allocate
     * @param stream CUDA stream (optional)
     * @return Pointer to allocated memory
     */
    pointer allocate(size_type count, stream_type stream = nullptr)
    {
        size_t bytes         = count * sizeof(T);
        size_t aligned_bytes = ((bytes + alignment - 1) / alignment) * alignment;
        void*  ptr           = allocator_.allocate(aligned_bytes, stream);
        return static_cast<pointer>(ptr);
    }

    /**
     * @brief Deallocate memory
     * @param ptr Pointer to deallocate
     * @param count Number of elements (for size calculation)
     * @param stream CUDA stream (optional)
     */
    void deallocate(pointer ptr, size_type count, stream_type stream = nullptr)
    {
        size_t bytes         = count * sizeof(T);
        size_t aligned_bytes = ((bytes + alignment - 1) / alignment) * alignment;
        allocator_.deallocate(ptr, aligned_bytes, stream);
    }

    /**
     * @brief Record a cross-stream use of a live allocation (PyTorch recordStream)
     */
    void record_stream(pointer ptr, stream_type stream) { allocator_.record_stream(ptr, stream); }

    /**
     * @brief Get underlying allocator statistics
     */
    unified_cache_stats stats() const { return allocator_.stats(); }

    /**
     * @brief Clear cache
     */
    void empty_cache() { allocator_.empty_cache(); }

    /**
     * @brief Get device index
     */
    int device() const { return allocator_.device(); }

private:
    cuda_caching_allocator allocator_;
};

}  // namespace gpu
}  // namespace memory
