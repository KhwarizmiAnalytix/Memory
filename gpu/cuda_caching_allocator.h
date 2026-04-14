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
#include <limits>
#include <memory>
#include <string>

#include "common/memory_macros.h"
#include "common/device.h"
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
 * @brief High-performance CUDA caching allocator for quantitative finance
 *
 * This allocator provides efficient memory management through intelligent caching
 * of GPU memory blocks. It's optimized for Monte Carlo simulations and PDE solvers
 * where frequent allocation/deallocation patterns can benefit from caching.
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
     * @param size Size of memory block (for validation)
     * @param stream CUDA stream for stream-aware caching (optional)
     * @throws std::invalid_argument if ptr is not owned by this allocator
     * @throws std::logic_error if double free detected
     */
    MEMORY_API void deallocate(void* ptr, size_t size, stream_type stream = nullptr);

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
    cuda_caching_allocator(const cuda_caching_allocator&)                         = delete;
    cuda_caching_allocator&              operator=(const cuda_caching_allocator&) = delete;
    MEMORY_API                         cuda_caching_allocator(cuda_caching_allocator&&) noexcept;
    MEMORY_API cuda_caching_allocator& operator=(cuda_caching_allocator&&) noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

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
