/**
 * @file unified_memory_stats_simple.cpp
 * @brief Implementation of simplified unified memory statistics system
 *
 * @author Quarisma Development Team
 * @date 2024
 * @copyright Quarisma Ltd. All rights reserved.
 */

#include "profiler/unified_memory_stats.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <ios>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace memory
{

// ============================================================================
// UNIFIED RESOURCE STATISTICS IMPLEMENTATION
// ============================================================================

// ============================================================================
// ATOMIC TIMING STATISTICS IMPLEMENTATION
// ============================================================================

// Default constructor - already defaulted in header

// Copy constructor
atomic_timing_stats::atomic_timing_stats(const atomic_timing_stats& other) noexcept
    : total_allocations(other.total_allocations.load(std::memory_order_relaxed)),
      total_deallocations(other.total_deallocations.load(std::memory_order_relaxed)),
      total_alloc_time_us(other.total_alloc_time_us.load(std::memory_order_relaxed)),
      total_dealloc_time_us(other.total_dealloc_time_us.load(std::memory_order_relaxed)),
      min_alloc_time_us(other.min_alloc_time_us.load(std::memory_order_relaxed)),
      max_alloc_time_us(other.max_alloc_time_us.load(std::memory_order_relaxed)),
      min_dealloc_time_us(other.min_dealloc_time_us.load(std::memory_order_relaxed)),
      max_dealloc_time_us(other.max_dealloc_time_us.load(std::memory_order_relaxed)),
      total_transfer_time_us(other.total_transfer_time_us.load(std::memory_order_relaxed)),
      cuda_sync_time_us(other.cuda_sync_time_us.load(std::memory_order_relaxed))
{
}

// Copy assignment operator
atomic_timing_stats& atomic_timing_stats::operator=(const atomic_timing_stats& other) noexcept
{
    if (this != &other)
    {
        total_allocations.store(
            other.total_allocations.load(std::memory_order_relaxed), std::memory_order_relaxed);
        total_deallocations.store(
            other.total_deallocations.load(std::memory_order_relaxed), std::memory_order_relaxed);
        total_alloc_time_us.store(
            other.total_alloc_time_us.load(std::memory_order_relaxed), std::memory_order_relaxed);
        total_dealloc_time_us.store(
            other.total_dealloc_time_us.load(std::memory_order_relaxed), std::memory_order_relaxed);
        min_alloc_time_us.store(
            other.min_alloc_time_us.load(std::memory_order_relaxed), std::memory_order_relaxed);
        max_alloc_time_us.store(
            other.max_alloc_time_us.load(std::memory_order_relaxed), std::memory_order_relaxed);
        min_dealloc_time_us.store(
            other.min_dealloc_time_us.load(std::memory_order_relaxed), std::memory_order_relaxed);
        max_dealloc_time_us.store(
            other.max_dealloc_time_us.load(std::memory_order_relaxed), std::memory_order_relaxed);
        total_transfer_time_us.store(
            other.total_transfer_time_us.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        cuda_sync_time_us.store(
            other.cuda_sync_time_us.load(std::memory_order_relaxed), std::memory_order_relaxed);
    }
    return *this;
}

void atomic_timing_stats::reset() noexcept
{
    total_allocations.store(0, std::memory_order_relaxed);
    total_deallocations.store(0, std::memory_order_relaxed);
    total_alloc_time_us.store(0, std::memory_order_relaxed);
    total_dealloc_time_us.store(0, std::memory_order_relaxed);
    min_alloc_time_us.store(UINT64_MAX, std::memory_order_relaxed);
    max_alloc_time_us.store(0, std::memory_order_relaxed);
    min_dealloc_time_us.store(UINT64_MAX, std::memory_order_relaxed);
    max_dealloc_time_us.store(0, std::memory_order_relaxed);
    total_transfer_time_us.store(0, std::memory_order_relaxed);
    cuda_sync_time_us.store(0, std::memory_order_relaxed);
}

double atomic_timing_stats::average_alloc_time_us() const noexcept
{
    uint64_t const allocs = total_allocations.load(std::memory_order_relaxed);
    if (allocs == 0)
    {
        return 0.0;
    }
    return static_cast<double>(total_alloc_time_us.load(std::memory_order_relaxed)) / allocs;
}

double atomic_timing_stats::average_dealloc_time_us() const noexcept
{
    uint64_t const deallocs = total_deallocations.load(std::memory_order_relaxed);
    if (deallocs == 0)
    {
        return 0.0;
    }
    return static_cast<double>(total_dealloc_time_us.load(std::memory_order_relaxed)) / deallocs;
}

// ============================================================================
// UNIFIED RESOURCE STATISTICS IMPLEMENTATION
// ============================================================================

// Default constructor - already defaulted in header

// Copy constructor
unified_resource_stats::unified_resource_stats(const unified_resource_stats& other) noexcept
    : num_allocs(other.num_allocs.load(std::memory_order_relaxed)),
      num_deallocs(other.num_deallocs.load(std::memory_order_relaxed)),
      bytes_in_use(other.bytes_in_use.load(std::memory_order_relaxed)),
      peak_bytes_in_use(other.peak_bytes_in_use.load(std::memory_order_relaxed)),
      largest_alloc_size(other.largest_alloc_size.load(std::memory_order_relaxed)),
      active_allocations(other.active_allocations.load(std::memory_order_relaxed)),
      total_bytes_allocated(other.total_bytes_allocated.load(std::memory_order_relaxed)),
      total_bytes_deallocated(other.total_bytes_deallocated.load(std::memory_order_relaxed)),
      failed_allocations(other.failed_allocations.load(std::memory_order_relaxed)),
      potential_leaks(other.potential_leaks.load(std::memory_order_relaxed)),
      total_allocations(other.total_allocations.load(std::memory_order_relaxed)),
      bytes_reserved(other.bytes_reserved.load(std::memory_order_relaxed)),
      peak_bytes_reserved(other.peak_bytes_reserved.load(std::memory_order_relaxed)),
      bytes_reservable_limit(other.bytes_reservable_limit.load(std::memory_order_relaxed)),
      largest_free_block_bytes(other.largest_free_block_bytes.load(std::memory_order_relaxed)),
      pool_bytes(other.pool_bytes.load(std::memory_order_relaxed)),
      peak_pool_bytes(other.peak_pool_bytes.load(std::memory_order_relaxed)),
      bytes_limit(other.bytes_limit.load(std::memory_order_relaxed))
{
}

// Copy assignment operator
unified_resource_stats& unified_resource_stats::operator=(
    const unified_resource_stats& other) noexcept
{
    if (this != &other)
    {
        num_allocs.store(
            other.num_allocs.load(std::memory_order_relaxed), std::memory_order_relaxed);
        num_deallocs.store(
            other.num_deallocs.load(std::memory_order_relaxed), std::memory_order_relaxed);
        bytes_in_use.store(
            other.bytes_in_use.load(std::memory_order_relaxed), std::memory_order_relaxed);
        peak_bytes_in_use.store(
            other.peak_bytes_in_use.load(std::memory_order_relaxed), std::memory_order_relaxed);
        largest_alloc_size.store(
            other.largest_alloc_size.load(std::memory_order_relaxed), std::memory_order_relaxed);
        active_allocations.store(
            other.active_allocations.load(std::memory_order_relaxed), std::memory_order_relaxed);
        total_bytes_allocated.store(
            other.total_bytes_allocated.load(std::memory_order_relaxed), std::memory_order_relaxed);
        total_bytes_deallocated.store(
            other.total_bytes_deallocated.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        failed_allocations.store(
            other.failed_allocations.load(std::memory_order_relaxed), std::memory_order_relaxed);
        potential_leaks.store(
            other.potential_leaks.load(std::memory_order_relaxed), std::memory_order_relaxed);
        total_allocations.store(
            other.total_allocations.load(std::memory_order_relaxed), std::memory_order_relaxed);
        bytes_reserved.store(
            other.bytes_reserved.load(std::memory_order_relaxed), std::memory_order_relaxed);
        peak_bytes_reserved.store(
            other.peak_bytes_reserved.load(std::memory_order_relaxed), std::memory_order_relaxed);
        bytes_reservable_limit.store(
            other.bytes_reservable_limit.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        largest_free_block_bytes.store(
            other.largest_free_block_bytes.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        pool_bytes.store(
            other.pool_bytes.load(std::memory_order_relaxed), std::memory_order_relaxed);
        peak_pool_bytes.store(
            other.peak_pool_bytes.load(std::memory_order_relaxed), std::memory_order_relaxed);
        bytes_limit.store(
            other.bytes_limit.load(std::memory_order_relaxed), std::memory_order_relaxed);
    }
    return *this;
}

void unified_resource_stats::reset() noexcept
{
    num_allocs.store(0, std::memory_order_relaxed);
    num_deallocs.store(0, std::memory_order_relaxed);
    bytes_in_use.store(0, std::memory_order_relaxed);
    peak_bytes_in_use.store(0, std::memory_order_relaxed);
    largest_alloc_size.store(0, std::memory_order_relaxed);
    active_allocations.store(0, std::memory_order_relaxed);
    total_bytes_allocated.store(0, std::memory_order_relaxed);
    total_bytes_deallocated.store(0, std::memory_order_relaxed);
    failed_allocations.store(0, std::memory_order_relaxed);
    potential_leaks.store(0, std::memory_order_relaxed);
}

double unified_resource_stats::average_allocation_size() const noexcept
{
    int64_t const allocs = num_allocs.load(std::memory_order_relaxed);
    if (allocs == 0)
    {
        return 0.0;
    }
    return static_cast<double>(total_bytes_allocated.load(std::memory_order_relaxed)) / allocs;
}

double unified_resource_stats::memory_efficiency() const noexcept
{
    int64_t const peak = peak_bytes_in_use.load(std::memory_order_relaxed);
    if (peak == 0)
    {
        return 1.0;
    }
    return average_allocation_size() / peak;
}

double unified_resource_stats::allocation_success_rate() const noexcept
{
    int64_t const total_attempts = num_allocs.load(std::memory_order_relaxed) +
                                   failed_allocations.load(std::memory_order_relaxed);
    if (total_attempts == 0)
    {
        return 100.0;
    }
    return (static_cast<double>(num_allocs.load(std::memory_order_relaxed)) / total_attempts) *
           100.0;
}

std::string unified_resource_stats::debug_string() const
{
    std::ostringstream oss;

    int64_t const allocs     = num_allocs.load(std::memory_order_relaxed);
    int64_t const deallocs   = num_deallocs.load(std::memory_order_relaxed);
    int64_t const bytes_used = bytes_in_use.load(std::memory_order_relaxed);
    int64_t const peak_bytes = peak_bytes_in_use.load(std::memory_order_relaxed);
    int64_t const largest    = largest_alloc_size.load(std::memory_order_relaxed);
    int64_t const active     = active_allocations.load(std::memory_order_relaxed);

    oss << "unified_resource_stats: " << "allocs=" << allocs << ", deallocs=" << deallocs
        << ", active=" << active << ", in_use=" << (bytes_used / 1024.0 / 1024.0) << "MB"
        << ", peak=" << (peak_bytes / 1024.0 / 1024.0) << "MB"
        << ", largest=" << (largest / 1024.0 / 1024.0) << "MB" << ", efficiency=" << std::fixed
        << std::setprecision(1) << (memory_efficiency() * 100.0) << "%"
        << ", success_rate=" << std::fixed << std::setprecision(1) << allocation_success_rate()
        << "%";

    return oss.str();
}

// ============================================================================
// UNIFIED CACHE STATISTICS IMPLEMENTATION
// ============================================================================

// Default constructor - already defaulted in header

// Copy constructor
unified_cache_stats::unified_cache_stats(const unified_cache_stats& other) noexcept
    : cache_hits(other.cache_hits.load(std::memory_order_relaxed)),
      cache_misses(other.cache_misses.load(std::memory_order_relaxed)),
      bytes_cached(other.bytes_cached.load(std::memory_order_relaxed)),
      driver_allocations(other.driver_allocations.load(std::memory_order_relaxed)),
      driver_frees(other.driver_frees.load(std::memory_order_relaxed)),
      cache_evictions(other.cache_evictions.load(std::memory_order_relaxed)),
      peak_bytes_cached(other.peak_bytes_cached.load(std::memory_order_relaxed)),
      cache_blocks(other.cache_blocks.load(std::memory_order_relaxed)),
      successful_allocations(other.successful_allocations.load(std::memory_order_relaxed)),
      successful_frees(other.successful_frees.load(std::memory_order_relaxed)),
      bytes_allocated(other.bytes_allocated.load(std::memory_order_relaxed))
{
}

// Copy assignment operator
unified_cache_stats& unified_cache_stats::operator=(const unified_cache_stats& other) noexcept
{
    if (this != &other)
    {
        cache_hits.store(
            other.cache_hits.load(std::memory_order_relaxed), std::memory_order_relaxed);
        cache_misses.store(
            other.cache_misses.load(std::memory_order_relaxed), std::memory_order_relaxed);
        bytes_cached.store(
            other.bytes_cached.load(std::memory_order_relaxed), std::memory_order_relaxed);
        driver_allocations.store(
            other.driver_allocations.load(std::memory_order_relaxed), std::memory_order_relaxed);
        driver_frees.store(
            other.driver_frees.load(std::memory_order_relaxed), std::memory_order_relaxed);
        cache_evictions.store(
            other.cache_evictions.load(std::memory_order_relaxed), std::memory_order_relaxed);
        peak_bytes_cached.store(
            other.peak_bytes_cached.load(std::memory_order_relaxed), std::memory_order_relaxed);
        cache_blocks.store(
            other.cache_blocks.load(std::memory_order_relaxed), std::memory_order_relaxed);
        successful_allocations.store(
            other.successful_allocations.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        successful_frees.store(
            other.successful_frees.load(std::memory_order_relaxed), std::memory_order_relaxed);
        bytes_allocated.store(
            other.bytes_allocated.load(std::memory_order_relaxed), std::memory_order_relaxed);
    }
    return *this;
}

void unified_cache_stats::reset() noexcept
{
    cache_hits.store(0, std::memory_order_relaxed);
    cache_misses.store(0, std::memory_order_relaxed);
    bytes_cached.store(0, std::memory_order_relaxed);
    driver_allocations.store(0, std::memory_order_relaxed);
    driver_frees.store(0, std::memory_order_relaxed);
    cache_evictions.store(0, std::memory_order_relaxed);
    peak_bytes_cached.store(0, std::memory_order_relaxed);
    cache_blocks.store(0, std::memory_order_relaxed);
    successful_allocations.store(0, std::memory_order_relaxed);
    successful_frees.store(0, std::memory_order_relaxed);
    bytes_allocated.store(0, std::memory_order_relaxed);
}

double unified_cache_stats::cache_hit_rate() const noexcept
{
    size_t const hits   = cache_hits.load(std::memory_order_relaxed);
    size_t const misses = cache_misses.load(std::memory_order_relaxed);
    size_t const total  = hits + misses;
    return total > 0 ? static_cast<double>(hits) / total : 0.0;
}

double unified_cache_stats::cache_efficiency_percent() const noexcept
{
    size_t const hits   = cache_hits.load(std::memory_order_relaxed);
    size_t const misses = cache_misses.load(std::memory_order_relaxed);
    size_t const total  = hits + misses;
    if (total == 0)
    {
        return 0.0;
    }
    return (static_cast<double>(hits) / total) * 100.0;
}

double unified_cache_stats::driver_call_reduction() const noexcept
{
    size_t const hits              = cache_hits.load(std::memory_order_relaxed);
    size_t const driver_calls_free = driver_frees.load(std::memory_order_relaxed);
    size_t const driver_calls      = driver_allocations.load(std::memory_order_relaxed) +
                                driver_frees.load(std::memory_order_relaxed);
    if (driver_calls == 0)
    {
        return 1.0;
    }
    return static_cast<double>(hits + driver_calls_free) / driver_calls;
}

// ============================================================================
// COMPREHENSIVE MEMORY STATISTICS IMPLEMENTATION
// ============================================================================

// Constructor
comprehensive_memory_stats::comprehensive_memory_stats(std::string name)
    : allocator_name(std::move(name))
{
}

double comprehensive_memory_stats::overall_efficiency() const noexcept
{
    // Combine resource efficiency and cache efficiency
    double const resource_eff = resource_stats.memory_efficiency();
    double const cache_eff    = cache_stats.cache_hit_rate();
    return (resource_eff + cache_eff) / 2.0;
}

double comprehensive_memory_stats::operations_per_second() const noexcept
{
    uint64_t const total_ops     = timing_stats.total_allocations.load(std::memory_order_relaxed);
    uint64_t const total_time_us = timing_stats.total_alloc_time_us.load(std::memory_order_relaxed);
    if (total_time_us == 0)
    {
        return 0.0;
    }
    return (static_cast<double>(total_ops) * 1000000.0) / total_time_us;
}

std::string comprehensive_memory_stats::generate_report() const
{
    std::ostringstream oss;
    oss << "=== " << allocator_name << " Memory Statistics Report ===\n";
    oss << "Resource Stats: " << resource_stats.debug_string() << "\n";
    oss << "Cache Performance: Hit Rate " << std::fixed << std::setprecision(1)
        << (cache_stats.cache_hit_rate() * 100.0) << "%\n";
    oss << "CUDA Sync Time: " << timing_stats.cuda_sync_time_us.load() << " μs\n";
    oss << "Transfer Time: " << timing_stats.total_transfer_time_us.load() << " μs\n";
    oss << "Overall Efficiency: " << std::fixed << std::setprecision(1)
        << (overall_efficiency() * 100.0) << "%\n";
    return oss.str();
}

// ============================================================================
// MEMORY FRAGMENTATION METRICS IMPLEMENTATION
// ============================================================================

// Copy constructor
memory_fragmentation_metrics::memory_fragmentation_metrics(
    const memory_fragmentation_metrics& other) noexcept = default;

// Copy assignment operator
memory_fragmentation_metrics& memory_fragmentation_metrics::operator=(
    const memory_fragmentation_metrics& other) noexcept
{
    if (this != &other)
    {
        total_free_blocks       = other.total_free_blocks;
        largest_free_block      = other.largest_free_block;
        smallest_free_block     = other.smallest_free_block;
        average_free_block_size = other.average_free_block_size;
        fragmentation_ratio     = other.fragmentation_ratio;
        external_fragmentation  = other.external_fragmentation;
        internal_fragmentation  = other.internal_fragmentation;
        wasted_bytes            = other.wasted_bytes;
    }
    return *this;
}

void memory_fragmentation_metrics::reset() noexcept
{
    total_free_blocks       = 0;
    largest_free_block      = 0;
    smallest_free_block     = 0;
    average_free_block_size = 0;
    fragmentation_ratio     = 0.0;
    external_fragmentation  = 0.0;
    internal_fragmentation  = 0.0;
    wasted_bytes            = 0;
}

memory_fragmentation_metrics memory_fragmentation_metrics::calculate(
    size_t total_allocated, size_t total_requested, const std::vector<size_t>& free_blocks) noexcept
{
    memory_fragmentation_metrics metrics;

    if (free_blocks.empty() || total_allocated == 0)
    {
        return metrics;  // Return default-initialized metrics
    }

    // Calculate basic free block statistics
    metrics.total_free_blocks   = free_blocks.size();
    metrics.largest_free_block  = *std::max_element(free_blocks.begin(), free_blocks.end());
    metrics.smallest_free_block = *std::min_element(free_blocks.begin(), free_blocks.end());

    size_t const total_free = std::accumulate(free_blocks.begin(), free_blocks.end(), size_t{0});
    metrics.average_free_block_size = total_free / free_blocks.size();

    // Calculate fragmentation ratios
    if (total_allocated > 0)
    {
        metrics.fragmentation_ratio =
            static_cast<double>(total_allocated - metrics.largest_free_block) / total_allocated;
        metrics.external_fragmentation =
            static_cast<double>(total_free - metrics.largest_free_block) / total_allocated * 100.0;
    }

    if (total_requested > 0 && total_allocated >= total_requested)
    {
        metrics.internal_fragmentation =
            static_cast<double>(total_allocated - total_requested) / total_allocated * 100.0;
        metrics.wasted_bytes = total_allocated - total_requested;
    }

    return metrics;
}

std::string memory_fragmentation_metrics::debug_string() const
{
    std::ostringstream oss;
    oss << "memory_fragmentation_metrics: " << "free_blocks=" << total_free_blocks
        << ", largest=" << (largest_free_block / 1024.0 / 1024.0) << "MB"
        << ", smallest=" << (smallest_free_block / 1024.0 / 1024.0) << "MB"
        << ", average=" << (average_free_block_size / 1024.0 / 1024.0) << "MB"
        << ", frag_ratio=" << std::fixed << std::setprecision(3) << fragmentation_ratio
        << ", ext_frag=" << std::fixed << std::setprecision(1) << external_fragmentation << "%"
        << ", int_frag=" << std::fixed << std::setprecision(1) << internal_fragmentation << "%"
        << ", wasted=" << (wasted_bytes / 1024.0 / 1024.0) << "MB";
    return oss.str();
}

}  // namespace memory
