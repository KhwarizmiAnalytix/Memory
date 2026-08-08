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

#include <string>
#include <vector>

#include "MemoryTest.h"

using namespace memory;

// ============================================================================
// atomic_timing_stats
// ============================================================================

MEMORYTEST(AtomicTimingStats, DefaultConstructedIsZero)
{
    const atomic_timing_stats stats;
    EXPECT_EQ(stats.total_allocations.load(), 0U);
    EXPECT_EQ(stats.average_alloc_time_us(), 0.0);
    EXPECT_EQ(stats.average_dealloc_time_us(), 0.0);
    END_TEST();
}

MEMORYTEST(AtomicTimingStats, AverageAllocTime)
{
    atomic_timing_stats stats;
    stats.total_allocations.store(4);
    stats.total_alloc_time_us.store(40);
    EXPECT_EQ(stats.average_alloc_time_us(), 10.0);
    END_TEST();
}

MEMORYTEST(AtomicTimingStats, AverageDeallocTime)
{
    atomic_timing_stats stats;
    stats.total_deallocations.store(5);
    stats.total_dealloc_time_us.store(50);
    EXPECT_EQ(stats.average_dealloc_time_us(), 10.0);
    END_TEST();
}

MEMORYTEST(AtomicTimingStats, CopyConstructor)
{
    atomic_timing_stats original;
    original.total_allocations.store(7);
    original.max_alloc_time_us.store(99);

    const atomic_timing_stats copy(original);
    EXPECT_EQ(copy.total_allocations.load(), 7U);
    EXPECT_EQ(copy.max_alloc_time_us.load(), 99U);
    END_TEST();
}

MEMORYTEST(AtomicTimingStats, CopyAssignment)
{
    atomic_timing_stats source;
    source.total_deallocations.store(3);
    source.cuda_sync_time_us.store(21);

    atomic_timing_stats target;
    target = source;
    EXPECT_EQ(target.total_deallocations.load(), 3U);
    EXPECT_EQ(target.cuda_sync_time_us.load(), 21U);

    // Self-assignment must be a safe no-op.
    target = target;
    EXPECT_EQ(target.total_deallocations.load(), 3U);
    END_TEST();
}

MEMORYTEST(AtomicTimingStats, Reset)
{
    atomic_timing_stats stats;
    stats.total_allocations.store(10);
    stats.total_alloc_time_us.store(100);
    stats.reset();

    EXPECT_EQ(stats.total_allocations.load(), 0U);
    EXPECT_EQ(stats.total_alloc_time_us.load(), 0U);
    EXPECT_EQ(stats.min_alloc_time_us.load(), UINT64_MAX);
    EXPECT_EQ(stats.min_dealloc_time_us.load(), UINT64_MAX);
    END_TEST();
}

// ============================================================================
// unified_resource_stats
// ============================================================================

MEMORYTEST(UnifiedResourceStats, DefaultConstructedIsZero)
{
    const unified_resource_stats stats;
    EXPECT_EQ(stats.num_allocs.load(), 0);
    EXPECT_EQ(stats.average_allocation_size(), 0.0);
    END_TEST();
}

MEMORYTEST(UnifiedResourceStats, AverageAllocationSize)
{
    unified_resource_stats stats;
    stats.num_allocs.store(4);
    stats.total_bytes_allocated.store(400);
    EXPECT_EQ(stats.average_allocation_size(), 100.0);
    END_TEST();
}

MEMORYTEST(UnifiedResourceStats, MemoryEfficiencyWithZeroPeakIsOne)
{
    const unified_resource_stats stats;
    EXPECT_EQ(stats.memory_efficiency(), 1.0);
    END_TEST();
}

MEMORYTEST(UnifiedResourceStats, MemoryEfficiencyWithPeak)
{
    unified_resource_stats stats;
    stats.num_allocs.store(2);
    stats.total_bytes_allocated.store(200);
    stats.peak_bytes_in_use.store(100);
    // average_allocation_size() == 100, peak == 100 -> efficiency == 1.0
    EXPECT_EQ(stats.memory_efficiency(), 1.0);
    END_TEST();
}

MEMORYTEST(UnifiedResourceStats, AllocationSuccessRateNoAttemptsIsHundred)
{
    const unified_resource_stats stats;
    EXPECT_EQ(stats.allocation_success_rate(), 100.0);
    END_TEST();
}

MEMORYTEST(UnifiedResourceStats, AllocationSuccessRateWithFailures)
{
    unified_resource_stats stats;
    stats.num_allocs.store(3);
    stats.failed_allocations.store(1);
    EXPECT_NEAR(stats.allocation_success_rate(), 75.0, 1e-9);
    END_TEST();
}

MEMORYTEST(UnifiedResourceStats, CopyConstructorAndAssignment)
{
    unified_resource_stats source;
    source.num_allocs.store(5);
    source.bytes_reserved.store(1024);
    source.bytes_limit.store(2048);

    const unified_resource_stats copy(source);
    EXPECT_EQ(copy.num_allocs.load(), 5);
    EXPECT_EQ(copy.bytes_reserved.load(), 1024);
    EXPECT_EQ(copy.bytes_limit.load(), 2048);

    unified_resource_stats assigned;
    assigned = source;
    EXPECT_EQ(assigned.num_allocs.load(), 5);

    assigned = assigned;
    EXPECT_EQ(assigned.num_allocs.load(), 5);
    END_TEST();
}

MEMORYTEST(UnifiedResourceStats, Reset)
{
    unified_resource_stats stats;
    stats.num_allocs.store(9);
    stats.bytes_in_use.store(999);
    stats.reset();

    EXPECT_EQ(stats.num_allocs.load(), 0);
    EXPECT_EQ(stats.bytes_in_use.load(), 0);
    END_TEST();
}

MEMORYTEST(UnifiedResourceStats, DebugString)
{
    unified_resource_stats stats;
    stats.num_allocs.store(2);
    stats.num_deallocs.store(1);
    stats.bytes_in_use.store(1024);
    stats.peak_bytes_in_use.store(2048);

    const std::string debug = stats.debug_string();
    EXPECT_NE(debug.find("unified_resource_stats"), std::string::npos);
    EXPECT_NE(debug.find("allocs=2"), std::string::npos);
    END_TEST();
}

// ============================================================================
// unified_cache_stats
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

// ============================================================================
// comprehensive_memory_stats
// ============================================================================

MEMORYTEST(ComprehensiveMemoryStats, DefaultName)
{
    const comprehensive_memory_stats stats;
    EXPECT_EQ(stats.allocator_name, "Unknown");
    END_TEST();
}

MEMORYTEST(ComprehensiveMemoryStats, CustomName)
{
    const comprehensive_memory_stats stats("MyAllocator");
    EXPECT_EQ(stats.allocator_name, "MyAllocator");
    END_TEST();
}

MEMORYTEST(ComprehensiveMemoryStats, OverallEfficiency)
{
    comprehensive_memory_stats stats;
    stats.cache_stats.cache_hits.store(1);
    stats.cache_stats.cache_misses.store(1);
    // resource_stats.memory_efficiency() == 1.0 (default), cache_hit_rate() == 0.5
    EXPECT_NEAR(stats.overall_efficiency(), 0.75, 1e-9);
    END_TEST();
}

MEMORYTEST(ComprehensiveMemoryStats, OperationsPerSecondNoTimingIsZero)
{
    const comprehensive_memory_stats stats;
    EXPECT_EQ(stats.operations_per_second(), 0.0);
    END_TEST();
}

MEMORYTEST(ComprehensiveMemoryStats, OperationsPerSecondWithTiming)
{
    comprehensive_memory_stats stats;
    stats.timing_stats.total_allocations.store(2);
    stats.timing_stats.total_alloc_time_us.store(1000000);  // 1 second total
    EXPECT_NEAR(stats.operations_per_second(), 2.0, 1e-9);
    END_TEST();
}

MEMORYTEST(ComprehensiveMemoryStats, GenerateReport)
{
    const comprehensive_memory_stats stats("ReportAllocator");
    const std::string                report = stats.generate_report();
    EXPECT_NE(report.find("ReportAllocator"), std::string::npos);
    EXPECT_NE(report.find("Resource Stats"), std::string::npos);
    EXPECT_NE(report.find("Cache Performance"), std::string::npos);
    EXPECT_NE(report.find("Overall Efficiency"), std::string::npos);
    END_TEST();
}

// ============================================================================
// memory_fragmentation_metrics
// ============================================================================

MEMORYTEST(MemoryFragmentationMetrics, DefaultConstructedIsZero)
{
    const memory_fragmentation_metrics metrics;
    EXPECT_EQ(metrics.total_free_blocks, 0U);
    EXPECT_EQ(metrics.fragmentation_ratio, 0.0);
    END_TEST();
}

MEMORYTEST(MemoryFragmentationMetrics, CalculateWithEmptyFreeBlocksReturnsDefault)
{
    const auto metrics = memory_fragmentation_metrics::calculate(1024, 512, {});
    EXPECT_EQ(metrics.total_free_blocks, 0U);
    END_TEST();
}

MEMORYTEST(MemoryFragmentationMetrics, CalculateWithZeroAllocatedReturnsDefault)
{
    const auto metrics = memory_fragmentation_metrics::calculate(0, 0, {16, 32});
    EXPECT_EQ(metrics.total_free_blocks, 0U);
    END_TEST();
}

MEMORYTEST(MemoryFragmentationMetrics, CalculateWithFreeBlocks)
{
    const std::vector<size_t> free_blocks = {64, 128, 32};
    const auto metrics = memory_fragmentation_metrics::calculate(1024, 800, free_blocks);

    EXPECT_EQ(metrics.total_free_blocks, 3U);
    EXPECT_EQ(metrics.largest_free_block, 128U);
    EXPECT_EQ(metrics.smallest_free_block, 32U);
    EXPECT_EQ(metrics.average_free_block_size, (64U + 128U + 32U) / 3U);
    EXPECT_GT(metrics.fragmentation_ratio, 0.0);
    EXPECT_GT(metrics.internal_fragmentation, 0.0);
    EXPECT_EQ(metrics.wasted_bytes, 1024U - 800U);
    END_TEST();
}

MEMORYTEST(MemoryFragmentationMetrics, CalculateWithoutRequestedBytes)
{
    // total_requested == 0 must skip the internal-fragmentation branch.
    const auto metrics = memory_fragmentation_metrics::calculate(1024, 0, {64});
    EXPECT_EQ(metrics.internal_fragmentation, 0.0);
    EXPECT_EQ(metrics.wasted_bytes, 0U);
    END_TEST();
}

MEMORYTEST(MemoryFragmentationMetrics, CopyConstructorAndAssignment)
{
    const auto source = memory_fragmentation_metrics::calculate(1024, 800, {64, 128, 32});

    const memory_fragmentation_metrics copy(source);
    EXPECT_EQ(copy.total_free_blocks, source.total_free_blocks);

    memory_fragmentation_metrics assigned;
    assigned = source;
    EXPECT_EQ(assigned.largest_free_block, source.largest_free_block);

    assigned = assigned;
    EXPECT_EQ(assigned.largest_free_block, source.largest_free_block);
    END_TEST();
}

MEMORYTEST(MemoryFragmentationMetrics, Reset)
{
    memory_fragmentation_metrics metrics =
        memory_fragmentation_metrics::calculate(1024, 800, {64, 128, 32});
    metrics.reset();

    EXPECT_EQ(metrics.total_free_blocks, 0U);
    EXPECT_EQ(metrics.largest_free_block, 0U);
    EXPECT_EQ(metrics.fragmentation_ratio, 0.0);
    EXPECT_EQ(metrics.wasted_bytes, 0U);
    END_TEST();
}

MEMORYTEST(MemoryFragmentationMetrics, DebugString)
{
    const auto        metrics = memory_fragmentation_metrics::calculate(1024, 800, {64, 128, 32});
    const std::string debug   = metrics.debug_string();
    EXPECT_NE(debug.find("memory_fragmentation_metrics"), std::string::npos);
    EXPECT_NE(debug.find("free_blocks=3"), std::string::npos);
    END_TEST();
}
