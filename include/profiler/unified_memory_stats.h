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

#include <atomic>
#include <cstddef>

#include "common/memory_export.h"
#include "common/memory_macros.h"

namespace memory
{

/**
 * @brief Statistics for caching allocators (currently: cuda_caching_allocator)
 *
 * This is the GPU caching-allocator statistics surface of the Memory library.
 * CPU alloc/free/OOM events go through profiled_cpu_memory_reporter, compiled
 * into allocate/free only when MEMORY_HAS_PROFILER=1 and then only after
 * profiler::memory_profiling_active(). The CPU path (cpu::memory_allocator)
 * still carries no always-on counters — that would defeat its "thin dispatch
 * over mimalloc/TBB" design; use the benchmark suite
 * (BenchmarkCPUMemoryAllocators) for CPU performance data and
 * cpu::memory_allocator::usable_size() for per-block tooling. mimalloc's own
 * opt-in statistics (MEMORY_ENABLE_MIMALLOC_STATS) are exposed via
 * cpu::memory_allocator::{has_stats, stats_print, process_info}.
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
    std::atomic<size_t> peak_bytes_allocated{0};

    // PyTorch DeviceStats-style fields (cuda_caching_allocator): total segment
    // bytes held from the driver, bytes in free split-off remainders that
    // cannot be returned to the driver, OOM cache-flush retries, and
    // allocations that failed even after the flush-and-retry chain.
    std::atomic<size_t> bytes_reserved{0};
    std::atomic<size_t> peak_bytes_reserved{0};
    std::atomic<size_t> inactive_split_bytes{0};
    std::atomic<size_t> num_alloc_retries{0};
    std::atomic<size_t> num_ooms{0};
    // Number of synchronize-and-free-events passes (empty_cache / OOM flush)
    std::atomic<size_t> num_sync_all_streams{0};

    // Backing equation (plan §5.4, task 7.1), filled by cuda_caching_allocator::stats()
    // (a scan under the allocator lock, not a hot-path read):
    //   bytes_reserved == bytes_allocated + bytes_pending + bytes_cached
    //                     + bytes_quarantined + bytes_unaccounted
    // bytes_allocated is live capacity; bytes_cached is reusable capacity (free
    // blocks in the pools); bytes_pending is freed capacity still waiting for a
    // cross-stream event; bytes_quarantined is capacity withheld because a use
    // could not be proven complete or its metadata could not be cached.
    // bytes_unaccounted is the residue and must be 0: it is non-zero only when a
    // double fault lost quarantined block metadata (the memory is still held).
    std::atomic<size_t> bytes_pending{0};
    std::atomic<size_t> bytes_quarantined{0};
    std::atomic<size_t> bytes_unaccounted{0};

    // Waste definitions (task 7.4), same scan:
    //  - bytes_requested: sum of caller-requested sizes of live allocations.
    //    Internal waste = bytes_allocated - bytes_requested (rounding and
    //    unsplit remainders inside live blocks); see internal_waste_bytes().
    //  - inactive_split_bytes (above): capacity of free blocks that are part of
    //    a split segment (cannot be returned to the driver while a neighbour is live).
    //  - largest_cached_block: the largest single reusable block, the biggest
    //    request that a cache hit can serve.
    // Hit/miss denominators: cache_hits + cache_misses counts allocation
    // requests that reached the cache lookup (a request that then fails with
    // OOM is a miss; a zero-size request returns before the lookup and is in
    // neither). External fragmentation is not derived from these.
    std::atomic<size_t> bytes_requested{0};
    std::atomic<size_t> largest_cached_block{0};

    // Capacity lost to rounding inside live blocks. Valid on a stats() copy.
    size_t internal_waste_bytes() const noexcept
    {
        size_t const a = bytes_allocated.load(std::memory_order_relaxed);
        size_t const r = bytes_requested.load(std::memory_order_relaxed);
        return a > r ? a - r : 0;
    }

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
     * @brief Reset peak counters to the current allocated/reserved/cached values.
     *
     * Matches torch.cuda.reset_peak_memory_stats: live counters are unchanged.
     */
    MEMORY_API void reset_peaks() noexcept;

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

using cuda_caching_allocator_stats = unified_cache_stats;  ///< CUDA cache statistics

}  // namespace memory
