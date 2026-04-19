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
#include "helper/memory_allocator.h"

// Standard aligned allocation
#if defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)
#include <malloc.h>
#else
#include <cstdlib>
#endif

namespace memory
{
namespace benchmarks
{

// =============================================================================
// Allocator Wrapper Classes (same as in tests)
// =============================================================================

class allocator_benchmark_interface
{
public:
    virtual ~allocator_benchmark_interface()                                            = default;
    virtual void*       allocate(std::size_t size, std::size_t alignment = 64) noexcept = 0;
    virtual void        deallocate(void* ptr, std::size_t size = 0) noexcept            = 0;
    virtual const char* name() const noexcept                                           = 0;
};

class mimalloc_benchmark_allocator : public allocator_benchmark_interface
{
public:
    void* allocate(
        MEMORY_UNUSED std::size_t size, MEMORY_UNUSED std::size_t alignment = 64) noexcept override
    {
        return memory::cpu::memory_allocator::allocate_mi(size, alignment);
    }

    void deallocate(MEMORY_UNUSED void* ptr, MEMORY_UNUSED std::size_t size = 0) noexcept override
    {
        memory::cpu::memory_allocator::free_mi(ptr, size);
    }

    const char* name() const noexcept override { return "mimalloc"; }
};

class tbb_scalable_benchmark_allocator : public allocator_benchmark_interface
{
public:
    void* allocate(
        MEMORY_UNUSED std::size_t size, MEMORY_UNUSED std::size_t alignment = 64) noexcept override
    {
        return memory::cpu::memory_allocator::allocate_tbb(size, alignment);
    }

    void deallocate(MEMORY_UNUSED void* ptr, MEMORY_UNUSED std::size_t size = 0) noexcept override
    {
        memory::cpu::memory_allocator::free_tbb(ptr, size);
    }

    const char* name() const noexcept override { return "tbb_scalable"; }
};

class standard_aligned_benchmark_allocator : public allocator_benchmark_interface
{
public:
    void* allocate(std::size_t size, std::size_t alignment = 64) noexcept override
    {
#if defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)
        return _aligned_malloc(size, alignment);
#else
        void* ptr = nullptr;
        if (posix_memalign(&ptr, alignment, size) != 0)
        {
            return nullptr;
        }
        return ptr;
#endif
    }

    void deallocate(void* ptr, MEMORY_UNUSED std::size_t size = 0) noexcept override
    {
        if (ptr != nullptr)
        {
#if defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)
            _aligned_free(ptr);
#else
            ::free(ptr);
#endif
        }
    }

    const char* name() const noexcept override { return "standard_aligned_malloc"; }
};

class malloc_benchmark_allocator : public allocator_benchmark_interface
{
public:
    void* allocate(std::size_t size, MEMORY_UNUSED std::size_t alignment = 64) noexcept override
    {
        return malloc(size);
    }

    void deallocate(void* ptr, MEMORY_UNUSED std::size_t size = 0) noexcept override { free(ptr); }

    const char* name() const noexcept override { return "standard_malloc"; }
};

// =============================================================================
// Benchmark Helper Functions
// =============================================================================

template <typename AllocatorType>
void benchmark_simple_allocation(benchmark::State& state)
{
    AllocatorType     allocator;
    const std::size_t size = static_cast<std::size_t>(state.range(0));

    for (auto _ : state)
    {
        void* ptr = allocator.allocate(size);
        benchmark::DoNotOptimize(ptr);
        if (ptr != nullptr)
        {
            allocator.deallocate(ptr, size);
        }
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * size));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

template <typename AllocatorType>
void benchmark_batch_allocation(benchmark::State& state)
{
    AllocatorType     allocator;
    const std::size_t batch_size      = static_cast<std::size_t>(state.range(0));
    const std::size_t allocation_size = static_cast<std::size_t>(state.range(1));

    for (auto _ : state)
    {
        std::vector<void*> ptrs;
        ptrs.reserve(batch_size);

        // Allocation phase
        for (std::size_t i = 0; i < batch_size; ++i)
        {
            void* ptr = allocator.allocate(allocation_size);
            if (ptr != nullptr)
            {
                ptrs.push_back(ptr);
            }
        }

        benchmark::DoNotOptimize(ptrs.data());

        // Deallocation phase
        for (void* ptr : ptrs)
        {
            allocator.deallocate(ptr, allocation_size);
        }

        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(
        static_cast<int64_t>(state.iterations() * batch_size * allocation_size));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * batch_size));
}

template <typename AllocatorType>
void benchmark_mixed_sizes(benchmark::State& state)
{
    AllocatorType     allocator;
    const std::size_t num_allocations = static_cast<std::size_t>(state.range(0));

    // Pre-generate random sizes for consistency
    std::mt19937                               rng(42);
    std::uniform_int_distribution<std::size_t> size_dist(64, 4096);
    std::vector<std::size_t>                   sizes;
    sizes.reserve(num_allocations);
    for (std::size_t i = 0; i < num_allocations; ++i)
    {
        sizes.push_back(size_dist(rng));
    }

    for (auto _ : state)
    {
        std::vector<void*> ptrs;
        ptrs.reserve(num_allocations);

        // Allocation phase
        for (std::size_t size : sizes)
        {
            void* ptr = allocator.allocate(size);
            if (ptr != nullptr)
            {
                ptrs.push_back(ptr);
            }
        }

        benchmark::DoNotOptimize(ptrs.data());

        // Deallocation phase
        for (std::size_t i = 0; i < ptrs.size(); ++i)
        {
            allocator.deallocate(ptrs[i], sizes[i]);
        }

        benchmark::ClobberMemory();
    }

    std::size_t total_bytes = 0;
    for (std::size_t size : sizes)
        total_bytes += size;

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * total_bytes));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * num_allocations));
}

template <typename AllocatorType>
void benchmark_memory_access_pattern(benchmark::State& state)
{
    AllocatorType     allocator;
    const std::size_t size = static_cast<std::size_t>(state.range(0));

    for (auto _ : state)
    {
        void* ptr = allocator.allocate(size);
        if (ptr != nullptr)
        {
            // sequential write pattern
            std::memset(ptr, 0xAA, size);
            benchmark::DoNotOptimize(ptr);

            // sequential read pattern
            volatile std::size_t checksum = 0;
            auto*                byte_ptr = static_cast<unsigned char*>(ptr);
            for (std::size_t i = 0; i < size; ++i)
            {
                checksum += byte_ptr[i];
            }
            benchmark::DoNotOptimize(checksum);

            allocator.deallocate(ptr, size);
        }
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * size * 2));  // Read + Write
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

// =============================================================================
// Simple Allocation Benchmarks
// =============================================================================

// Quarisma CPU allocator benchmarks
BENCHMARK_TEMPLATE(benchmark_simple_allocation, malloc_benchmark_allocator)
    ->Name("BM_Malloc_SimpleAllocation")
    ->Range(64, 64 << 10)
    ->Unit(benchmark::kMicrosecond);

// Standard aligned allocator benchmarks
BENCHMARK_TEMPLATE(benchmark_simple_allocation, standard_aligned_benchmark_allocator)
    ->Name("BM_StandardAligned_SimpleAllocation")
    ->Range(64, 64 << 10)
    ->Unit(benchmark::kMicrosecond);

#if PROJECT_HAS_MIMALLOC
BENCHMARK_TEMPLATE(benchmark_simple_allocation, mimalloc_benchmark_allocator)
    ->Name("BM_Mimalloc_SimpleAllocation")
    ->Range(64, 64 << 10)
    ->Unit(benchmark::kMicrosecond);
#endif

#if PROJECT_HAS_TBB
BENCHMARK_TEMPLATE(benchmark_simple_allocation, tbb_scalable_benchmark_allocator)
    ->Name("BM_TBBScalable_SimpleAllocation")
    ->Range(64, 64 << 10)
    ->Unit(benchmark::kMicrosecond);
#endif

// =============================================================================
// Batch Allocation Benchmarks
// =============================================================================

// Quarisma CPU allocator batch benchmarks
BENCHMARK_TEMPLATE(benchmark_batch_allocation, malloc_benchmark_allocator)
    ->Name("BM_Malloc_BatchAllocation")
    ->Args({100, 1024})
    ->Args({1000, 1024})
    ->Args({100, 4096})
    ->Args({1000, 4096})
    ->Unit(benchmark::kMicrosecond);

// Standard aligned allocator batch benchmarks
BENCHMARK_TEMPLATE(benchmark_batch_allocation, standard_aligned_benchmark_allocator)
    ->Name("BM_StandardAligned_BatchAllocation")
    ->Args({100, 1024})
    ->Args({1000, 1024})
    ->Args({100, 4096})
    ->Args({1000, 4096})
    ->Unit(benchmark::kMicrosecond);

#if PROJECT_HAS_MIMALLOC
BENCHMARK_TEMPLATE(benchmark_batch_allocation, mimalloc_benchmark_allocator)
    ->Name("BM_Mimalloc_BatchAllocation")
    ->Args({100, 1024})
    ->Args({1000, 1024})
    ->Args({100, 4096})
    ->Args({1000, 4096})
    ->Unit(benchmark::kMicrosecond);
#endif

#if PROJECT_HAS_TBB
BENCHMARK_TEMPLATE(benchmark_batch_allocation, tbb_scalable_benchmark_allocator)
    ->Name("BM_TBBScalable_BatchAllocation")
    ->Args({100, 1024})
    ->Args({1000, 1024})
    ->Args({100, 4096})
    ->Args({1000, 4096})
    ->Unit(benchmark::kMicrosecond);
#endif

// =============================================================================
// Mixed Size Allocation Benchmarks
// =============================================================================

// Quarisma CPU allocator mixed size benchmarks
BENCHMARK_TEMPLATE(benchmark_mixed_sizes, malloc_benchmark_allocator)
    ->Name("BM_Malloc_MixedSizes")
    ->Arg(100)
    ->Arg(500)
    ->Arg(1000)
    ->Unit(benchmark::kMicrosecond);

// Standard aligned allocator mixed size benchmarks
BENCHMARK_TEMPLATE(benchmark_mixed_sizes, standard_aligned_benchmark_allocator)
    ->Name("BM_StandardAligned_MixedSizes")
    ->Arg(100)
    ->Arg(500)
    ->Arg(1000)
    ->Unit(benchmark::kMicrosecond);

#if PROJECT_HAS_MIMALLOC
BENCHMARK_TEMPLATE(benchmark_mixed_sizes, mimalloc_benchmark_allocator)
    ->Name("BM_Mimalloc_MixedSizes")
    ->Arg(100)
    ->Arg(500)
    ->Arg(1000)
    ->Unit(benchmark::kMicrosecond);
#endif

#if PROJECT_HAS_TBB
BENCHMARK_TEMPLATE(benchmark_mixed_sizes, tbb_scalable_benchmark_allocator)
    ->Name("BM_TBBScalable_MixedSizes")
    ->Arg(100)
    ->Arg(500)
    ->Arg(1000)
    ->Unit(benchmark::kMicrosecond);
#endif

// =============================================================================
// Memory Access Pattern Benchmarks
// =============================================================================

// Quarisma CPU allocator memory access benchmarks
BENCHMARK_TEMPLATE(benchmark_memory_access_pattern, malloc_benchmark_allocator)
    ->Name("BM_Malloc_MemoryAccess")
    ->Range(1024, 1024 << 10)
    ->Unit(benchmark::kMicrosecond);

// Standard aligned allocator memory access benchmarks
BENCHMARK_TEMPLATE(benchmark_memory_access_pattern, standard_aligned_benchmark_allocator)
    ->Name("BM_StandardAligned_MemoryAccess")
    ->Range(1024, 1024 << 10)
    ->Unit(benchmark::kMicrosecond);

#if PROJECT_HAS_MIMALLOC
BENCHMARK_TEMPLATE(benchmark_memory_access_pattern, mimalloc_benchmark_allocator)
    ->Name("BM_Mimalloc_MemoryAccess")
    ->Range(1024, 1024 << 10)
    ->Unit(benchmark::kMicrosecond);
#endif

#if PROJECT_HAS_TBB
BENCHMARK_TEMPLATE(benchmark_memory_access_pattern, tbb_scalable_benchmark_allocator)
    ->Name("BM_TBBScalable_MemoryAccess")
    ->Range(1024, 1024 << 10)
    ->Unit(benchmark::kMicrosecond);
#endif

// =============================================================================
// Alignment-Specific Benchmarks
// =============================================================================

template <typename AllocatorType>
void benchmark_aligned_allocation(benchmark::State& state)
{
    AllocatorType     allocator;
    const std::size_t size      = 1024;
    const std::size_t alignment = static_cast<std::size_t>(state.range(0));

    for (auto _ : state)
    {
        void* ptr = allocator.allocate(size, alignment);
        benchmark::DoNotOptimize(ptr);
        if (ptr != nullptr)
        {
            allocator.deallocate(ptr, size);
        }
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * size));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

// Alignment benchmarks for all allocators
BENCHMARK_TEMPLATE(benchmark_aligned_allocation, malloc_benchmark_allocator)
    ->Name("BM_Malloc_AlignedAllocation")
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(benchmark_aligned_allocation, standard_aligned_benchmark_allocator)
    ->Name("BM_StandardAligned_AlignedAllocation")
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Unit(benchmark::kMicrosecond);

#if PROJECT_HAS_MIMALLOC
BENCHMARK_TEMPLATE(benchmark_aligned_allocation, mimalloc_benchmark_allocator)
    ->Name("BM_Mimalloc_AlignedAllocation")
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Unit(benchmark::kMicrosecond);
#endif

#if PROJECT_HAS_TBB
BENCHMARK_TEMPLATE(benchmark_aligned_allocation, tbb_scalable_benchmark_allocator)
    ->Name("BM_TBBScalable_AlignedAllocation")
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Unit(benchmark::kMicrosecond);
#endif

// =============================================================================
// Fragmentation Test Benchmark
// =============================================================================

template <typename AllocatorType>
void benchmark_fragmentation_pattern(benchmark::State& state)
{
    AllocatorType     allocator;
    const std::size_t num_allocations = static_cast<std::size_t>(state.range(0));

    for (auto _ : state)
    {
        std::vector<void*> ptrs;
        ptrs.reserve(num_allocations);

        // Allocate many small blocks
        for (std::size_t i = 0; i < num_allocations; ++i)
        {
            void* ptr = allocator.allocate(64);
            if (ptr != nullptr)
            {
                ptrs.push_back(ptr);
            }
        }

        // Free every other block to create fragmentation
        for (std::size_t i = 1; i < ptrs.size(); i += 2)
        {
            allocator.deallocate(ptrs[i], 64);
            ptrs[i] = nullptr;
        }

        // Try to allocate larger blocks in fragmented space
        std::vector<void*> large_ptrs;
        for (std::size_t i = 0; i < num_allocations / 4; ++i)
        {
            void* ptr = allocator.allocate(256);
            if (ptr != nullptr)
            {
                large_ptrs.push_back(ptr);
            }
        }

        benchmark::DoNotOptimize(large_ptrs.data());

        // Clean up
        for (void* ptr : large_ptrs)
        {
            allocator.deallocate(ptr, 256);
        }

        for (std::size_t i = 0; i < ptrs.size(); i += 2)
        {
            if (ptrs[i] != nullptr)
            {
                allocator.deallocate(ptrs[i], 64);
            }
        }

        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * num_allocations));
}

// Fragmentation benchmarks
BENCHMARK_TEMPLATE(benchmark_fragmentation_pattern, malloc_benchmark_allocator)
    ->Name("BM_Malloc_Fragmentation")
    ->Arg(1000)
    ->Arg(5000)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(benchmark_fragmentation_pattern, standard_aligned_benchmark_allocator)
    ->Name("BM_StandardAligned_Fragmentation")
    ->Arg(1000)
    ->Arg(5000)
    ->Unit(benchmark::kMicrosecond);

#if PROJECT_HAS_MIMALLOC
BENCHMARK_TEMPLATE(benchmark_fragmentation_pattern, mimalloc_benchmark_allocator)
    ->Name("BM_Mimalloc_Fragmentation")
    ->Arg(1000)
    ->Arg(5000)
    ->Unit(benchmark::kMicrosecond);
#endif

#if PROJECT_HAS_TBB
BENCHMARK_TEMPLATE(benchmark_fragmentation_pattern, tbb_scalable_benchmark_allocator)
    ->Name("BM_TBBScalable_Fragmentation")
    ->Arg(1000)
    ->Arg(5000)
    ->Unit(benchmark::kMicrosecond);
#endif

}  // namespace benchmarks
}  // namespace memory