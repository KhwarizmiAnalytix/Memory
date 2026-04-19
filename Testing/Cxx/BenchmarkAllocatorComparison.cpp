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

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include "common/memory_macros.h"
//#include "logger/logger.h"
#include "backend/allocator_bfc.h"
#include "backend/allocator_pool.h"
#include "backend/allocator_tracking.h"
#include "cpu/allocator.h"
#include "cpu/allocator_cpu.h"

using namespace memory;

// =============================================================================
// Benchmark Configuration
// =============================================================================

namespace
{

// Size categories for comprehensive testing
constexpr size_t kSmallSize  = 32;       // < 64 bytes
constexpr size_t kMediumSize = 1024;     // 64 bytes - 4 KB
constexpr size_t kLargeSize  = 65536;    // > 4 KB
constexpr size_t kHugeSize   = 1048576;  // > 1 MB

// Allocation patterns
constexpr int kBatchSize  = 100;
constexpr int kIterations = 1000;

}  // anonymous namespace

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * @brief Perform memory access to prevent optimization
 */
static void touch_memory(void* ptr, size_t size)
{
    /* if (ptr != nullptr && size > 0)
    {
        std::memset(ptr, 0xAA, std::min(size, size_t(64)));
        benchmark::DoNotOptimize(ptr);
    }*/
}

// =============================================================================
// CPU Allocator Benchmarks
// =============================================================================

static void BM_CPUAllocator_SmallAllocation(benchmark::State& state)
{
    Allocator*   allocator = cpu_allocator(0);
    const size_t size      = static_cast<size_t>(state.range(0));

    for (auto _ : state)
    {
        void* ptr = allocator->allocate_raw(64, size);
        touch_memory(ptr, size);
        allocator->deallocate_raw(ptr);
    }

    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * size);
}

static void BM_CPUAllocator_BatchAllocation(benchmark::State& state)
{
    Allocator*   allocator  = cpu_allocator(0);
    const size_t size       = static_cast<size_t>(state.range(0));
    const int    batch_size = static_cast<int>(state.range(1));

    std::vector<void*> ptrs(batch_size);

    for (auto _ : state)
    {
        // Allocate batch
        for (int i = 0; i < batch_size; ++i)
        {
            ptrs[i] = allocator->allocate_raw(64, size);
            touch_memory(ptrs[i], size);
        }

        // Deallocate batch
        for (int i = 0; i < batch_size; ++i)
        {
            allocator->deallocate_raw(ptrs[i]);
        }
    }

    state.SetItemsProcessed(state.iterations() * batch_size);
    state.SetBytesProcessed(state.iterations() * batch_size * size);
}

// =============================================================================
// BFC Allocator Benchmarks
// =============================================================================

static void BM_BFCAllocator_SmallAllocation(benchmark::State& state)
{
    auto sub_allocator = std::make_unique<basic_cpu_allocator>(
        0, std::vector<sub_allocator::Visitor>{}, std::vector<sub_allocator::Visitor>{});

    allocator_bfc::Options opts;
    opts.allow_growth = true;

    allocator_bfc allocator(
        std::move(sub_allocator), 1024ULL * 1024ULL * 1024ULL, "bench_bfc", opts);

    const size_t size = static_cast<size_t>(state.range(0));

    for (auto _ : state)
    {
        void* ptr = allocator.allocate_raw(64, size);
        touch_memory(ptr, size);
        allocator.deallocate_raw(ptr);
    }

    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * size);
}

static void BM_BFCAllocator_BatchAllocation(benchmark::State& state)
{
    auto sub_allocator = std::make_unique<basic_cpu_allocator>(
        0, std::vector<sub_allocator::Visitor>{}, std::vector<sub_allocator::Visitor>{});

    allocator_bfc::Options opts;
    opts.allow_growth       = true;
    const size_t size       = static_cast<size_t>(state.range(0));
    const int    batch_size = static_cast<int>(state.range(1));

    allocator_bfc allocator(
        std::move(sub_allocator),
        1024ULL * 1024ULL * 1024ULL /** batch_size*/,
        "bench_bfc_batch",
        opts);

    std::vector<void*> ptrs(batch_size);

    for (auto _ : state)
    {
        for (int i = 0; i < batch_size; ++i)
        {
            ptrs[i] = allocator.allocate_raw(64, size);
            touch_memory(ptrs[i], size);
        }

        for (int i = 0; i < batch_size; ++i)
        {
            allocator.deallocate_raw(ptrs[i]);
        }
    }

    state.SetItemsProcessed(state.iterations() * batch_size);
    state.SetBytesProcessed(state.iterations() * batch_size * size);
}

// =============================================================================
// Pool Allocator Benchmarks
// =============================================================================

static void BM_PoolAllocator_SmallAllocation(MEMORY_UNUSED benchmark::State& state)
{
    auto base_allocator = std::make_unique<basic_cpu_allocator>(
        0, std::vector<sub_allocator::Visitor>{}, std::vector<sub_allocator::Visitor>{});

    auto pool = std::make_unique<allocator_pool>(
        0, false, std::move(base_allocator), std::make_unique<NoopRounder>(), "bench_pool");

    const size_t size = static_cast<size_t>(state.range(0));

    for (auto _ : state)
    {
        void* ptr = pool->allocate_raw(64, size);
        touch_memory(ptr, size);
        pool->deallocate_raw(ptr);
    }

    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * size);
}

static void BM_PoolAllocator_BatchAllocation(benchmark::State& state)
{
    auto base_allocator = std::make_unique<basic_cpu_allocator>(
        0, std::vector<sub_allocator::Visitor>{}, std::vector<sub_allocator::Visitor>{});

    auto pool = std::make_unique<allocator_pool>(
        0, false, std::move(base_allocator), std::make_unique<NoopRounder>(), "bench_pool_batch");

    const size_t size       = static_cast<size_t>(state.range(0));
    const int    batch_size = static_cast<int>(state.range(1));

    std::vector<void*> ptrs(batch_size);

    for (auto _ : state)
    {
        for (int i = 0; i < batch_size; ++i)
        {
            ptrs[i] = pool->allocate_raw(64, size);
            touch_memory(ptrs[i], size);
        }

        for (int i = 0; i < batch_size; ++i)
        {
            pool->deallocate_raw(ptrs[i]);
        }
    }

    state.SetItemsProcessed(state.iterations() * batch_size);
    state.SetBytesProcessed(state.iterations() * batch_size * size);
}

// =============================================================================
// Fragmentation Benchmarks
// =============================================================================

static void BM_Fragmentation_MixedSizes(benchmark::State& state)
{
    auto base_allocator = std::make_unique<basic_cpu_allocator>(
        0, std::vector<sub_allocator::Visitor>{}, std::vector<sub_allocator::Visitor>{});

    auto pool = std::make_unique<allocator_pool>(
        0,
        false,
        std::move(base_allocator),
        std::make_unique<NoopRounder>(),
        "bench_fragmentation");

    const size_t sizes[]   = {32, 64, 128, 256, 512, 1024, 2048, 4096};
    const int    num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    std::vector<void*> ptrs;
    ptrs.reserve(100);

    for (auto _ : state)
    {
        // Allocate mixed sizes
        for (int i = 0; i < 100; ++i)
        {
            size_t size = sizes[i % num_sizes];
            void*  ptr  = pool->allocate_raw(64, size);
            if (ptr != nullptr)
            {
                touch_memory(ptr, size);
                ptrs.push_back(ptr);
            }
        }

        // Deallocate every other allocation
        for (size_t i = 0; i < ptrs.size(); i += 2)
        {
            pool->deallocate_raw(ptrs[i]);
        }

        // Allocate again
        for (size_t i = 0; i < ptrs.size() / 2; ++i)
        {
            size_t size = sizes[i % num_sizes];
            void*  ptr  = pool->allocate_raw(64, size);
            if (ptr != nullptr)
            {
                touch_memory(ptr, size);
            }
        }

        // Cleanup
        for (size_t i = 1; i < ptrs.size(); i += 2)
        {
            pool->deallocate_raw(ptrs[i]);
        }

        ptrs.clear();
    }

    state.SetItemsProcessed(state.iterations() * 100);
}

// =============================================================================
// Thread Contention Benchmarks
// =============================================================================

static void BM_ThreadContention_CPUAllocator(benchmark::State& state)
{
    EnableCPUAllocatorStats();
    Allocator*   allocator = cpu_allocator(0);
    const size_t size      = 1024;

    for (auto _ : state)
    {
        void* ptr = allocator->allocate_raw(64, size);
        touch_memory(ptr, size);
        allocator->deallocate_raw(ptr);
    }

    state.SetItemsProcessed(state.iterations());
}

static void BM_ThreadContention_PoolAllocator(benchmark::State& state)
{
    static auto base_allocator = std::make_unique<basic_cpu_allocator>(
        0, std::vector<sub_allocator::Visitor>{}, std::vector<sub_allocator::Visitor>{});

    static auto pool = std::make_unique<allocator_pool>(
        0, false, std::move(base_allocator), std::make_unique<NoopRounder>(), "bench_thread_pool");

    const size_t size = 1024;

    for (auto _ : state)
    {
        void* ptr = pool->allocate_raw(64, size);
        touch_memory(ptr, size);
        pool->deallocate_raw(ptr);
    }

    state.SetItemsProcessed(state.iterations());
}

// =============================================================================
// Benchmark Registration
// =============================================================================
// Small allocations (< 64 bytes)
BENCHMARK(BM_CPUAllocator_SmallAllocation)
    ->Name("CPU/Small/Single")
    ->Arg(kSmallSize)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_BFCAllocator_SmallAllocation)
    ->Name("BFC/Small/Single")
    ->Arg(kSmallSize)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_PoolAllocator_SmallAllocation)
    ->Name("Pool/Small/Single")
    ->Arg(kSmallSize)
    ->Unit(benchmark::kMicrosecond);

// Medium allocations (64 bytes - 4 KB)
BENCHMARK(BM_CPUAllocator_SmallAllocation)
    ->Name("CPU/Medium/Single")
    ->Arg(kMediumSize)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_BFCAllocator_SmallAllocation)
    ->Name("BFC/Medium/Single")
    ->Arg(kMediumSize)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_PoolAllocator_SmallAllocation)
    ->Name("Pool/Medium/Single")
    ->Arg(kMediumSize)
    ->Unit(benchmark::kMicrosecond);

// Large allocations (> 4 KB)
BENCHMARK(BM_CPUAllocator_SmallAllocation)
    ->Name("CPU/Large/Single")
    ->Arg(kLargeSize)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_BFCAllocator_SmallAllocation)
    ->Name("BFC/Large/Single")
    ->Arg(kLargeSize)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_PoolAllocator_SmallAllocation)
    ->Name("Pool/Large/Single")
    ->Arg(kLargeSize)
    ->Unit(benchmark::kMicrosecond);

// Batch allocations - Small
BENCHMARK(BM_CPUAllocator_BatchAllocation)
    ->Name("CPU/Small/Batch")
    ->Args({kSmallSize, kBatchSize})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_BFCAllocator_BatchAllocation)
    ->Name("BFC/Small/Batch")
    ->Args({kSmallSize, kBatchSize})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_PoolAllocator_BatchAllocation)
    ->Name("Pool/Small/Batch")
    ->Args({kSmallSize, kBatchSize})
    ->Unit(benchmark::kMicrosecond);

// Batch allocations - Medium
BENCHMARK(BM_CPUAllocator_BatchAllocation)
    ->Name("CPU/Medium/Batch")
    ->Args({kMediumSize, kBatchSize})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_BFCAllocator_BatchAllocation)
    ->Name("BFC/Medium/Batch")
    ->Args({kMediumSize, kBatchSize})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_PoolAllocator_BatchAllocation)
    ->Name("Pool/Medium/Batch")
    ->Args({kMediumSize, kBatchSize})
    ->Unit(benchmark::kMicrosecond);

// Batch allocations - Large
BENCHMARK(BM_CPUAllocator_BatchAllocation)
    ->Name("CPU/Large/Batch")
    ->Args({kLargeSize, kBatchSize})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_BFCAllocator_BatchAllocation)
    ->Name("BFC/Large/Batch")
    ->Args({kLargeSize, kBatchSize})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_PoolAllocator_BatchAllocation)
    ->Name("Pool/Large/Batch")
    ->Args({kLargeSize, kBatchSize})
    ->Unit(benchmark::kMicrosecond);

// Fragmentation test
BENCHMARK(BM_Fragmentation_MixedSizes)
    ->Name("Fragmentation/MixedSizes")
    ->Unit(benchmark::kMicrosecond);

// Thread contention tests
BENCHMARK(BM_ThreadContention_CPUAllocator)
    ->Name("ThreadContention/CPU")
    ->Threads(1)
    ->Threads(2)
    ->Threads(4)
    ->Threads(8)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_ThreadContention_PoolAllocator)
    ->Name("ThreadContention/Pool")
    ->Threads(1)
    ->Threads(2)
    ->Threads(4)
    ->Threads(8)
    ->Unit(benchmark::kMicrosecond);

// Size range sweep
#if 0
BENCHMARK(BM_CPUAllocator_SmallAllocation)
    ->Name("CPU/SizeSweep")
    ->RangeMultiplier(2)
    ->Range(16, 16384)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_BFCAllocator_SmallAllocation)
    ->Name("BFC/SizeSweep")
    ->RangeMultiplier(2)
    ->Range(16, 16384)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_PoolAllocator_SmallAllocation)
    ->Name("Pool/SizeSweep")
    ->RangeMultiplier(2)
    ->Range(16, 16384)
    ->Unit(benchmark::kMicrosecond);

#endif

// BENCHMARK_MAIN() removed - using benchmark_main library instead
