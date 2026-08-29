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
