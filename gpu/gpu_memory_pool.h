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

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "common/memory_macros.h"
#include "common/device.h"

#if MEMORY_HAS_CUDA
#include <cuda_runtime.h>
#include "common/memory_export.h"
#endif

namespace memory
{
namespace gpu
{

/**
 * @brief Configuration parameters for GPU memory pools
 *
 * This structure defines the configuration parameters used to initialize
 * and manage GPU memory pools, including block sizes, pool limits, and
 * allocation strategies optimized for Monte Carlo simulations and PDE solvers.
 */
struct MEMORY_VISIBILITY gpu_memory_pool_config
{
    /** @brief Minimum block size in bytes (default: 1KB) */
    size_t min_block_size = 1024;

    /** @brief Maximum block size in bytes (default: 64MB) */
    size_t max_block_size = 64ULL * 1024ULL;

    /** @brief Block size growth factor (default: 2.0) */
    double block_growth_factor = 2.0;

    /** @brief Maximum total pool size in bytes (default: 1GB) */
    size_t max_pool_size = 1024ULL * 1024;

    /** @brief Maximum number of cached blocks per size class (default: 16) */
    size_t max_cached_blocks = 16;

    /** @brief Enable memory alignment for SIMD operations (default: true) */
    bool enable_alignment = true;

    /** @brief Memory alignment boundary in bytes (default: 256ULL for GPU coalescing) */
    size_t alignment_boundary = 256ULL;

    /** @brief Enable memory usage tracking (default: true) */
    bool enable_tracking = true;

    /** @brief Enable debug mode with additional checks (default: false) */
    bool debug_mode = false;
};

/**
 * @brief GPU memory pool statistics
 *
 * Contains statistics about memory pool usage including allocation counts,
 * cache performance, and memory utilization metrics.
 */
struct MEMORY_VISIBILITY gpu_memory_pool_statistics
{
    /** @brief Total number of allocations performed */
    size_t total_allocations = 0;

    /** @brief Total number of deallocations performed */
    size_t total_deallocations = 0;

    /** @brief Number of cache hits */
    size_t cache_hits = 0;

    /** @brief Number of cache misses */
    size_t cache_misses = 0;

    /** @brief Cache hit rate (0.0 to 1.0) */
    double cache_hit_rate = 0.0;

    /** @brief Total bytes allocated */
    size_t total_bytes_allocated = 0;

    /** @brief Total bytes deallocated */
    size_t total_bytes_deallocated = 0;

    /** @brief Current bytes in use */
    size_t current_bytes_in_use = 0;

    /** @brief Peak bytes in use */
    size_t peak_bytes_in_use = 0;

    /** @brief Bytes currently cached */
    size_t cached_memory = 0;

    /** @brief Number of active allocations */
    size_t active_allocations = 0;
};

/**
 * @brief Memory block metadata for tracking and management
 *
 * Contains metadata about allocated memory blocks including size,
 * device information, and usage statistics for profiling and debugging.
 */
struct MEMORY_VISIBILITY gpu_memory_block
{
    /** @brief Pointer to the allocated memory */
    void* ptr = nullptr;

    /** @brief Size of the allocated block in bytes */
    size_t size = 0;

    /** @brief Device where the memory is allocated */
    device_option device;

    /** @brief Timestamp when the block was allocated */
    std::chrono::high_resolution_clock::time_point allocation_time;

    /** @brief Number of times this block has been reused */
    std::atomic<size_t> reuse_count{0};

    /** @brief Whether the block is currently in use */
    std::atomic<bool> in_use{false};

    gpu_memory_block() : device(device_enum::CPU, 0) {}
    gpu_memory_block(void* p, size_t s, const device_option& dev)
        : ptr(p), size(s), device(dev), allocation_time(std::chrono::high_resolution_clock::now())
    {
    }

    // Delete copy operations due to atomic members
    gpu_memory_block(const gpu_memory_block&)            = delete;
    gpu_memory_block& operator=(const gpu_memory_block&) = delete;

    // Provide move operations
    gpu_memory_block(gpu_memory_block&& other) noexcept
        : ptr(other.ptr),
          size(other.size),
          device(other.device),
          allocation_time(other.allocation_time),
          reuse_count(other.reuse_count.load()),
          in_use(other.in_use.load())
    {
        other.ptr  = nullptr;
        other.size = 0;
        other.reuse_count.store(0);
        other.in_use.store(false);
    }

    gpu_memory_block& operator=(gpu_memory_block&& other) noexcept
    {
        if (this != &other)
        {
            ptr             = other.ptr;
            size            = other.size;
            device          = other.device;
            allocation_time = other.allocation_time;
            reuse_count.store(other.reuse_count.load());
            in_use.store(other.in_use.load());

            other.ptr  = nullptr;
            other.size = 0;
            other.reuse_count.store(0);
            other.in_use.store(false);
        }
        return *this;
    }
};

/**
 * @brief High-performance GPU memory pool manager
 *
 * Provides efficient memory pool management for GPU computations with support
 * for both CUDA and HIP backends. Optimized for Monte Carlo simulations
 * and PDE solvers with configurable block sizes, memory alignment, and
 * resource tracking capabilities.
 *
 * Key features:
 * - Configurable block size pools to minimize allocation overhead
 * - SIMD-aligned memory allocation for coalesced GPU access patterns
 * - Thread-safe operations with minimal locking overhead
 * - Memory usage tracking and leak detection
 * - Exception-safe resource management
 * - Support for both CUDA and HIP backends
 *
 * Mathematical foundation:
 * The pool uses a geometric progression for block sizes: S_i = S_0 * r^i
 * where S_0 is the minimum block size and r is the growth factor.
 * This ensures efficient memory utilization while minimizing fragmentation.
 *
 * @example
 * ```cpp
 * // Configure memory pool for Monte Carlo simulations
 * gpu_memory_pool_config config;
 * config.min_block_size = 4096;        // 4KB minimum
 * config.max_block_size = 128 * 1024ULL;  // 128MB maximum
 * config.block_growth_factor = 1.5;    // Moderate growth
 *
 * auto pool = gpu_memory_pool::create(config);
 *
 * // Allocate memory for simulation data
 * auto block = pool->allocate(1024ULL, device_enum::CUDA);
 * // ... use memory for computations ...
 * pool->deallocate(block);
 * ```
 */
class MEMORY_VISIBILITY gpu_memory_pool
{
public:
    /**
     * @brief Create a new GPU memory pool with specified configuration
     * @param config Configuration parameters for the memory pool
     * @return Unique pointer to the created memory pool
     */
    MEMORY_API static std::unique_ptr<gpu_memory_pool> create(
        const gpu_memory_pool_config& config);

    /**
     * @brief Virtual destructor for proper cleanup
     */
    MEMORY_API virtual ~gpu_memory_pool() = default;

    /**
     * @brief Allocate memory from the pool
     * @param size Size in bytes to allocate
     * @param device_type Target device type (CUDA or HIP)
     * @param device_index Device index (default: 0)
     * @return Memory block containing allocated memory and metadata
     * @throws std::bad_alloc if allocation fails
     * @throws std::invalid_argument if size is zero or device is invalid
     */
    MEMORY_API virtual gpu_memory_block allocate(
        size_t size, device_enum device_type, int device_index = 0) = 0;

    /**
     * @brief Deallocate memory back to the pool
     * @param block Memory block to deallocate
     * @throws std::invalid_argument if block is invalid
     */
    MEMORY_API virtual void deallocate(const gpu_memory_block& block) = 0;

    /**
     * @brief Get current memory usage statistics
     * @return Total allocated memory in bytes
     */
    MEMORY_API virtual size_t get_allocated_bytes() const = 0;

    /**
     * @brief Get peak memory usage since pool creation
     * @return Peak allocated memory in bytes
     */
    MEMORY_API virtual size_t get_peak_allocated_bytes() const = 0;

    /**
     * @brief Get number of active allocations
     * @return Number of currently allocated blocks
     */
    MEMORY_API virtual size_t get_active_allocations() const = 0;

    /**
     * @brief Clear all cached memory blocks
     * Forces immediate deallocation of all cached blocks to free GPU memory
     */
    MEMORY_API virtual void clear_cache() = 0;

    /**
     * @brief Get detailed memory usage report
     * @return String containing detailed memory statistics
     */
    MEMORY_API virtual std::string get_memory_report() const = 0;

    /**
     * @brief Get comprehensive memory pool statistics
     * @return Structure containing detailed statistics about pool usage
     */
    MEMORY_API virtual gpu_memory_pool_statistics get_statistics() const = 0;

protected:
    gpu_memory_pool() = default;
    MEMORY_DELETE_COPY_AND_MOVE(gpu_memory_pool);
};

}  // namespace gpu
}  // namespace memory
