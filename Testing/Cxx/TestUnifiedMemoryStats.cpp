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

#include "MemoryTest.h"
#include "profiler/unified_memory_stats.h"

using namespace memory;

// ============================================================================
// unified_cache_stats — the only statistics surface (cuda_caching_allocator)
// ============================================================================

MEMORYTEST(UnifiedCacheStats, CacheHitRateNoActivityIsZero)
{
    const unified_cache_stats stats;
    EXPECT_EQ(stats.cache_hit_rate(), 0.0);
    EXPECT_EQ(stats.cache_efficiency_percent(), 0.0);
    END_TEST();
}

MEMORYTEST(UnifiedCacheStats, CacheHitRateWithActivity)
{
    unified_cache_stats stats;
    stats.cache_hits.store(3);
    stats.cache_misses.store(1);
    EXPECT_EQ(stats.cache_hit_rate(), 0.75);
    EXPECT_EQ(stats.cache_efficiency_percent(), 75.0);
    END_TEST();
}

MEMORYTEST(UnifiedCacheStats, DriverCallReductionNoDriverCallsIsOne)
{
    const unified_cache_stats stats;
    EXPECT_EQ(stats.driver_call_reduction(), 1.0);
    END_TEST();
}

MEMORYTEST(UnifiedCacheStats, DriverCallReductionWithActivity)
{
    unified_cache_stats stats;
    stats.cache_hits.store(4);
    stats.driver_frees.store(1);
    stats.driver_allocations.store(1);
    // (hits + driver_frees) / (driver_allocations + driver_frees) = 5 / 2
    EXPECT_EQ(stats.driver_call_reduction(), 2.5);
    END_TEST();
}

MEMORYTEST(UnifiedCacheStats, CopyConstructorAndAssignment)
{
    unified_cache_stats source;
    source.cache_hits.store(6);
    source.num_ooms.store(2);
    source.num_sync_all_streams.store(1);

    const unified_cache_stats copy(source);
    EXPECT_EQ(copy.cache_hits.load(), 6U);
    EXPECT_EQ(copy.num_ooms.load(), 2U);
    EXPECT_EQ(copy.num_sync_all_streams.load(), 1U);

    unified_cache_stats assigned;
    assigned = source;
    EXPECT_EQ(assigned.cache_hits.load(), 6U);

    assigned = assigned;
    EXPECT_EQ(assigned.cache_hits.load(), 6U);
    END_TEST();
}

MEMORYTEST(UnifiedCacheStats, ResetPeaksKeepsLiveCounters)
{
    unified_cache_stats stats;
    stats.bytes_allocated.store(128);
    stats.peak_bytes_allocated.store(1024);
    stats.bytes_reserved.store(256);
    stats.peak_bytes_reserved.store(2048);
    stats.bytes_cached.store(64);
    stats.peak_bytes_cached.store(512);

    stats.reset_peaks();
    EXPECT_EQ(stats.bytes_allocated.load(), 128U);
    EXPECT_EQ(stats.peak_bytes_allocated.load(), 128U);
    EXPECT_EQ(stats.bytes_reserved.load(), 256U);
    EXPECT_EQ(stats.peak_bytes_reserved.load(), 256U);
    EXPECT_EQ(stats.bytes_cached.load(), 64U);
    EXPECT_EQ(stats.peak_bytes_cached.load(), 64U);
    END_TEST();
}

MEMORYTEST(UnifiedCacheStats, Reset)
{
    unified_cache_stats stats;
    stats.cache_hits.store(10);
    stats.bytes_cached.store(4096);
    stats.reset();

    EXPECT_EQ(stats.cache_hits.load(), 0U);
    EXPECT_EQ(stats.bytes_cached.load(), 0U);
    END_TEST();
}
