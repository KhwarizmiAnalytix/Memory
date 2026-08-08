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

#include "profiler/unified_memory_stats.h"

#include <cstddef>

namespace memory
{

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
      bytes_allocated(other.bytes_allocated.load(std::memory_order_relaxed)),
      bytes_reserved(other.bytes_reserved.load(std::memory_order_relaxed)),
      inactive_split_bytes(other.inactive_split_bytes.load(std::memory_order_relaxed)),
      num_alloc_retries(other.num_alloc_retries.load(std::memory_order_relaxed)),
      num_ooms(other.num_ooms.load(std::memory_order_relaxed)),
      num_sync_all_streams(other.num_sync_all_streams.load(std::memory_order_relaxed))
{
}

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
            other.driver_allocations.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        driver_frees.store(
            other.driver_frees.load(std::memory_order_relaxed), std::memory_order_relaxed);
        cache_evictions.store(
            other.cache_evictions.load(std::memory_order_relaxed), std::memory_order_relaxed);
        peak_bytes_cached.store(
            other.peak_bytes_cached.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        cache_blocks.store(
            other.cache_blocks.load(std::memory_order_relaxed), std::memory_order_relaxed);
        successful_allocations.store(
            other.successful_allocations.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        successful_frees.store(
            other.successful_frees.load(std::memory_order_relaxed), std::memory_order_relaxed);
        bytes_allocated.store(
            other.bytes_allocated.load(std::memory_order_relaxed), std::memory_order_relaxed);
        bytes_reserved.store(
            other.bytes_reserved.load(std::memory_order_relaxed), std::memory_order_relaxed);
        inactive_split_bytes.store(
            other.inactive_split_bytes.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        num_alloc_retries.store(
            other.num_alloc_retries.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        num_ooms.store(other.num_ooms.load(std::memory_order_relaxed), std::memory_order_relaxed);
        num_sync_all_streams.store(
            other.num_sync_all_streams.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
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
    bytes_reserved.store(0, std::memory_order_relaxed);
    inactive_split_bytes.store(0, std::memory_order_relaxed);
    num_alloc_retries.store(0, std::memory_order_relaxed);
    num_ooms.store(0, std::memory_order_relaxed);
    num_sync_all_streams.store(0, std::memory_order_relaxed);
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

}  // namespace memory
