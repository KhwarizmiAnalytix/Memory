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
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "common/memory_macros.h"
#include "common/memory_export.h"

namespace memory
{

// ============================================================================
// UNIFIED TIMING STATISTICS
// ============================================================================

/**
 * @brief Atomic timing statistics for memory operations (thread-safe)
 */
struct MEMORY_VISIBILITY atomic_timing_stats
{
    std::atomic<uint64_t> total_allocations{0};
    std::atomic<uint64_t> total_deallocations{0};
    std::atomic<uint64_t> total_alloc_time_us{0};
    std::atomic<uint64_t> total_dealloc_time_us{0};
    std::atomic<uint64_t> min_alloc_time_us{UINT64_MAX};
    std::atomic<uint64_t> max_alloc_time_us{0};
    std::atomic<uint64_t> min_dealloc_time_us{UINT64_MAX};
    std::atomic<uint64_t> max_dealloc_time_us{0};
    std::atomic<uint64_t> total_transfer_time_us{0};
    std::atomic<uint64_t> cuda_sync_time_us{0};

    // Default constructor
    atomic_timing_stats() = default;

    // Copy constructor
    MEMORY_API atomic_timing_stats(const atomic_timing_stats& other) noexcept;

    // Copy assignment operator
    MEMORY_API atomic_timing_stats& operator=(const atomic_timing_stats& other) noexcept;

    /**
     * @brief Reset all timing statistics to zero
     */
    MEMORY_API void reset() noexcept;

    /**
     * @brief Calculate average allocation time in microseconds
     * @return Average allocation time, or 0.0 if no allocations
     */
    MEMORY_API double average_alloc_time_us() const noexcept;

    /**
     * @brief Calculate average deallocation time in microseconds
     * @return Average deallocation time, or 0.0 if no deallocations
     */
    MEMORY_API double average_dealloc_time_us() const noexcept;
};

// ============================================================================
// UNIFIED RESOURCE STATISTICS
// ============================================================================

/**
 * @brief Unified resource statistics for memory allocators
 */
struct MEMORY_VISIBILITY unified_resource_stats
{
    std::atomic<int64_t> num_allocs{0};
    std::atomic<int64_t> num_deallocs{0};
    std::atomic<int64_t> bytes_in_use{0};
    std::atomic<int64_t> peak_bytes_in_use{0};
    std::atomic<int64_t> largest_alloc_size{0};
    std::atomic<int64_t> active_allocations{0};
    std::atomic<int64_t> total_bytes_allocated{0};
    std::atomic<int64_t> total_bytes_deallocated{0};
    std::atomic<int64_t> failed_allocations{0};
    std::atomic<int64_t> potential_leaks{0};
    std::atomic<int64_t> total_allocations{0};

    // Additional fields for BFC allocator compatibility
    std::atomic<int64_t> bytes_reserved{0};
    std::atomic<int64_t> peak_bytes_reserved{0};
    std::atomic<int64_t> bytes_reservable_limit{0};
    std::atomic<int64_t> largest_free_block_bytes{0};
    std::atomic<int64_t> pool_bytes{0};
    std::atomic<int64_t> peak_pool_bytes{0};
    std::atomic<int64_t> bytes_limit{0};

    // Default constructor
    unified_resource_stats() = default;

    // Copy constructor
    MEMORY_API unified_resource_stats(const unified_resource_stats& other) noexcept;

    // Copy assignment operator
    MEMORY_API unified_resource_stats& operator=(const unified_resource_stats& other) noexcept;

    /**
     * @brief Reset all resource statistics to zero
     */
    MEMORY_API void reset() noexcept;

    /**
     * @brief Calculate average allocation size in bytes
     * @return Average allocation size, or 0.0 if no allocations
     */
    MEMORY_API double average_allocation_size() const noexcept;

    /**
     * @brief Calculate memory efficiency ratio
     * @return Memory efficiency ratio (0.0 to 1.0+)
     */
    MEMORY_API double memory_efficiency() const noexcept;

    /**
     * @brief Calculate allocation success rate as percentage
     * @return Success rate percentage (0.0 to 100.0)
     */
    MEMORY_API double allocation_success_rate() const noexcept;

    /**
     * @brief Generate debug string representation of statistics
     * @return Formatted debug string
     */
    MEMORY_API std::string debug_string() const;
};

// ============================================================================
// UNIFIED CACHE STATISTICS
// ============================================================================

/**
 * @brief Unified cache statistics for caching allocators
 */
struct MEMORY_VISIBILITY unified_cache_stats
{
    std::atomic<size_t> cache_hits{0};
    std::atomic<size_t> cache_misses{0};
    std::atomic<size_t> bytes_cached{0};
    std::atomic<size_t> driver_allocations{0};
    std::atomic<size_t> driver_frees{0};
    std::atomic<size_t> cache_evictions{0};
    std::atomic<size_t> peak_bytes_cached{0};
    std::atomic<size_t> cache_blocks{0};
    std::atomic<size_t> successful_allocations{0};
    std::atomic<size_t> successful_frees{0};
    std::atomic<size_t> bytes_allocated{0};

    // Default constructor
    unified_cache_stats() = default;

    // Copy constructor
    MEMORY_API unified_cache_stats(const unified_cache_stats& other) noexcept;

    // Copy assignment operator
    MEMORY_API unified_cache_stats& operator=(const unified_cache_stats& other) noexcept;

    /**
     * @brief Reset all cache statistics to zero
     */
    MEMORY_API void reset() noexcept;

    /**
     * @brief Calculate cache hit rate as ratio
     * @return Cache hit rate (0.0 to 1.0)
     */
    MEMORY_API double cache_hit_rate() const noexcept;

    /**
     * @brief Calculate cache efficiency as percentage
     * @return Cache efficiency percentage (0.0 to 100.0)
     */
    MEMORY_API double cache_efficiency_percent() const noexcept;

    /**
     * @brief Calculate driver call reduction factor
     * @return Driver call reduction factor (1.0+)
     */
    MEMORY_API double driver_call_reduction() const noexcept;
};

// ============================================================================
// COMPREHENSIVE MEMORY STATISTICS
// ============================================================================

/**
 * @brief Comprehensive memory statistics combining all aspects
 */
struct MEMORY_VISIBILITY comprehensive_memory_stats
{
    unified_resource_stats resource_stats;
    atomic_timing_stats    timing_stats;
    unified_cache_stats    cache_stats;
    std::string            allocator_name;

    MEMORY_API comprehensive_memory_stats(std::string name = "Unknown");

    /**
     * @brief Calculate overall efficiency combining resource and cache efficiency
     * @return Overall efficiency ratio (0.0 to 1.0+)
     */
    MEMORY_API double overall_efficiency() const noexcept;

    /**
     * @brief Calculate operations per second based on timing statistics
     * @return Operations per second, or 0.0 if no timing data
     */
    MEMORY_API double operations_per_second() const noexcept;

    /**
     * @brief Generate comprehensive memory statistics report
     * @return Formatted report string
     */
    MEMORY_API std::string generate_report() const;
};

// ============================================================================
// MEMORY FRAGMENTATION METRICS
// ============================================================================

/**
 * @brief Memory fragmentation analysis metrics
 *
 * Provides comprehensive analysis of memory fragmentation including:
 * - External Fragmentation = Sum of all free blocks that are too small for requests
 * - Internal Fragmentation = Sum of (Allocated Size - Requested Size) for all allocations
 *
 * **Performance Impact**: O(1) for basic metrics, O(n) for detailed analysis
 */
struct MEMORY_VISIBILITY memory_fragmentation_metrics
{
    size_t total_free_blocks{0};         ///< Number of free memory blocks
    size_t largest_free_block{0};        ///< Size of largest contiguous free block
    size_t smallest_free_block{0};       ///< Size of smallest free block
    size_t average_free_block_size{0};   ///< Average size of free blocks
    double fragmentation_ratio{0.0};     ///< Overall fragmentation ratio [0.0, 1.0]
    double external_fragmentation{0.0};  ///< External fragmentation percentage
    double internal_fragmentation{0.0};  ///< Internal fragmentation percentage
    size_t wasted_bytes{0};              ///< Total bytes lost to fragmentation

    /**
     * @brief Default constructor initializing all metrics to zero.
     */
    memory_fragmentation_metrics() = default;

    /**
     * @brief Copy constructor
     */
    MEMORY_API memory_fragmentation_metrics(const memory_fragmentation_metrics& other) noexcept;

    /**
     * @brief Copy assignment operator
     */
    MEMORY_API memory_fragmentation_metrics& operator=(
        const memory_fragmentation_metrics& other) noexcept;

    /**
     * @brief Reset all fragmentation metrics to zero
     */
    MEMORY_API void reset() noexcept;

    /**
     * @brief Calculates comprehensive fragmentation analysis.
     *
     * @param total_allocated Total bytes currently allocated
     * @param total_requested Total bytes originally requested
     * @param free_blocks Vector of free block sizes
     * @return Computed fragmentation metrics
     *
     * **Complexity**: O(n) where n is number of free blocks
     * **Thread Safety**: Thread-safe for read-only operations
     */
    MEMORY_API static memory_fragmentation_metrics calculate(
        size_t                     total_allocated,
        size_t                     total_requested,
        const std::vector<size_t>& free_blocks) noexcept;

    /**
     * @brief Generate debug string representation of fragmentation metrics
     * @return Formatted debug string
     */
    MEMORY_API std::string debug_string() const;
};

// ============================================================================
// TYPE ALIASES FOR CLARITY
// ============================================================================

// Clear type aliases for different use cases
using allocator_stats         = unified_resource_stats;  ///< CPU allocator resource statistics
using allocation_timing_stats = atomic_timing_stats;     ///< CPU allocation timing statistics
using timing_stats_snapshot   = atomic_timing_stats;  ///< Timing statistics snapshot (non-atomic)
using gpu_timing_stats        = atomic_timing_stats;  ///< GPU timing statistics
using gpu_resource_statistics = unified_resource_stats;    ///< GPU resource statistics
using cuda_caching_allocator_stats = unified_cache_stats;  ///< CUDA cache statistics

}  // namespace memory
