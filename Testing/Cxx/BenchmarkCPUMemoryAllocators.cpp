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

// CPU allocation-backend benchmarks.
//
// Subjects:
//   - standard malloc / aligned malloc          (baseline)
//   - mimalloc / TBB scalable                   (optional third-party backends)
//   - cpu::memory_allocator                     (the production CPU path)
//   - allocator<T> STL facade + data_ptr RAII   (the client-facing layers)
//
// PyTorch (LibTorch) comparisons live in BenchmarkPyTorchComparison.cpp, which
// is only built when MEMORY_ENABLE_LIBTORCH is ON and LibTorch is found.

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdlib>
#include <random>
#include <vector>

#if defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)
#include <malloc.h>
#endif

#include "allocator.h"
#include "common/data_ptr.h"
#include "common/memory_macros.h"
#include "helper/memory_allocator.h"

namespace memory
{
namespace benchmarks
{
namespace
{

constexpr std::size_t kBenchAlignment = 64;

// =============================================================================
// Allocation backends (policy structs, static dispatch)
// =============================================================================

struct malloc_backend
{
    static void* allocate(std::size_t size, std::size_t /*alignment*/) { return std::malloc(size); }
    static void  deallocate(void* ptr, std::size_t /*size*/) { std::free(ptr); }
};

struct aligned_malloc_backend
{
    static void* allocate(std::size_t size, std::size_t alignment)
    {
#if defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)
        return _aligned_malloc(size, alignment);
#else
        void* ptr = nullptr;
        return posix_memalign(&ptr, alignment, size) == 0 ? ptr : nullptr;
#endif
    }
    static void deallocate(void* ptr, std::size_t /*size*/)
    {
#if defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)
        _aligned_free(ptr);
#else
        std::free(ptr);
#endif
    }
};

#if MEMORY_HAS_MIMALLOC
struct mimalloc_backend
{
    static void* allocate(std::size_t size, std::size_t alignment)
    {
        return cpu::memory_allocator::allocate_mi(size, alignment);
    }
    static void deallocate(void* ptr, std::size_t size)
    {
        cpu::memory_allocator::free_mi(ptr, size);
    }
};
#endif

#if MEMORY_HAS_TBB
struct tbb_backend
{
    static void* allocate(std::size_t size, std::size_t alignment)
    {
        return cpu::memory_allocator::allocate_tbb(size, alignment);
    }
    static void deallocate(void* ptr, std::size_t size)
    {
        cpu::memory_allocator::free_tbb(ptr, size);
    }
};
#endif

// The production CPU path: exactly what allocator<T> and data_ptr call.
struct xsigma_cpu_backend
{
    static void* allocate(std::size_t size, std::size_t alignment)
    {
        return cpu::memory_allocator::allocate(size, alignment);
    }
    static void deallocate(void* ptr, std::size_t /*size*/) { cpu::memory_allocator::free(ptr); }
};

// The STL-style facade, to measure the (inlined) wrapper overhead over the raw
// production path.
struct xsigma_stl_backend
{
    static void* allocate(std::size_t size, std::size_t /*alignment*/)
    {
        return allocator<std::byte>::allocate(size, device_enum::CPU);
    }
    static void deallocate(void* ptr, std::size_t /*size*/)
    {
        auto* byte_ptr = static_cast<std::byte*>(ptr);
        allocator<std::byte>::free(byte_ptr, device_enum::CPU);
    }
};

// =============================================================================
// Workloads
// =============================================================================

template <typename Backend>
void benchmark_simple_allocation(benchmark::State& state)
{
    const std::size_t size = static_cast<std::size_t>(state.range(0));

    for (auto _ : state)
    {
        void* ptr = Backend::allocate(size, kBenchAlignment);
        benchmark::DoNotOptimize(ptr);
        if (ptr != nullptr)
        {
            Backend::deallocate(ptr, size);
        }
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * size));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

template <typename Backend>
void benchmark_batch_allocation(benchmark::State& state)
{
    const std::size_t batch_size      = static_cast<std::size_t>(state.range(0));
    const std::size_t allocation_size = static_cast<std::size_t>(state.range(1));

    for (auto _ : state)
    {
        std::vector<void*> ptrs;
        ptrs.reserve(batch_size);

        for (std::size_t i = 0; i < batch_size; ++i)
        {
            void* ptr = Backend::allocate(allocation_size, kBenchAlignment);
            if (ptr != nullptr)
            {
                ptrs.push_back(ptr);
            }
        }

        benchmark::DoNotOptimize(ptrs.data());

        for (void* ptr : ptrs)
        {
            Backend::deallocate(ptr, allocation_size);
        }

        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(
        static_cast<int64_t>(state.iterations() * batch_size * allocation_size));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * batch_size));
}

template <typename Backend>
void benchmark_mixed_sizes(benchmark::State& state)
{
    const std::size_t num_allocations = static_cast<std::size_t>(state.range(0));

    // Pre-generate random sizes so every backend sees the identical sequence.
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

        for (std::size_t size : sizes)
        {
            void* ptr = Backend::allocate(size, kBenchAlignment);
            if (ptr != nullptr)
            {
                ptrs.push_back(ptr);
            }
        }

        benchmark::DoNotOptimize(ptrs.data());

        for (std::size_t i = 0; i < ptrs.size(); ++i)
        {
            Backend::deallocate(ptrs[i], sizes[i]);
        }

        benchmark::ClobberMemory();
    }

    std::size_t total_bytes = 0;
    for (std::size_t size : sizes)
    {
        total_bytes += size;
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * total_bytes));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * num_allocations));
}

template <typename Backend>
void benchmark_aligned_allocation(benchmark::State& state)
{
    const std::size_t size      = 1024;
    const std::size_t alignment = static_cast<std::size_t>(state.range(0));

    for (auto _ : state)
    {
        void* ptr = Backend::allocate(size, alignment);
        benchmark::DoNotOptimize(ptr);
        if (ptr != nullptr)
        {
            Backend::deallocate(ptr, size);
        }
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * size));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

template <typename Backend>
void benchmark_fragmentation_pattern(benchmark::State& state)
{
    const std::size_t num_allocations = static_cast<std::size_t>(state.range(0));

    for (auto _ : state)
    {
        std::vector<void*> ptrs;
        ptrs.reserve(num_allocations);

        // Allocate many small blocks.
        for (std::size_t i = 0; i < num_allocations; ++i)
        {
            void* ptr = Backend::allocate(64, kBenchAlignment);
            if (ptr != nullptr)
            {
                ptrs.push_back(ptr);
            }
        }

        // Free every other block to fragment the arena.
        for (std::size_t i = 1; i < ptrs.size(); i += 2)
        {
            Backend::deallocate(ptrs[i], 64);
            ptrs[i] = nullptr;
        }

        // Allocate larger blocks into the fragmented space.
        std::vector<void*> large_ptrs;
        for (std::size_t i = 0; i < num_allocations / 4; ++i)
        {
            void* ptr = Backend::allocate(256, kBenchAlignment);
            if (ptr != nullptr)
            {
                large_ptrs.push_back(ptr);
            }
        }

        benchmark::DoNotOptimize(large_ptrs.data());

        for (void* ptr : large_ptrs)
        {
            Backend::deallocate(ptr, 256);
        }
        for (std::size_t i = 0; i < ptrs.size(); i += 2)
        {
            if (ptrs[i] != nullptr)
            {
                Backend::deallocate(ptrs[i], 64);
            }
        }

        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * num_allocations));
}

// =============================================================================
// data_ptr (RAII client layer) benchmarks — construct + destruct per iteration
// =============================================================================

void benchmark_data_ptr_simple(benchmark::State& state)
{
    const std::size_t size = static_cast<std::size_t>(state.range(0));

    for (auto _ : state)
    {
        data_ptr<std::byte, false> ptr(size, device_enum::CPU);
        benchmark::DoNotOptimize(ptr.get());
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * size));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

void benchmark_data_ptr_batch(benchmark::State& state)
{
    const std::size_t batch_size      = static_cast<std::size_t>(state.range(0));
    const std::size_t allocation_size = static_cast<std::size_t>(state.range(1));

    for (auto _ : state)
    {
        std::vector<data_ptr<std::byte, false>> ptrs;
        ptrs.reserve(batch_size);
        for (std::size_t i = 0; i < batch_size; ++i)
        {
            ptrs.emplace_back(allocation_size, device_enum::CPU);
        }

        benchmark::DoNotOptimize(ptrs.data());
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(
        static_cast<int64_t>(state.iterations() * batch_size * allocation_size));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * batch_size));
}

}  // namespace

// =============================================================================
// Registration
// =============================================================================

// ---- single allocation / free, size sweep 64 B .. 1 MiB ----

BENCHMARK_TEMPLATE(benchmark_simple_allocation, malloc_backend)
    ->Name("BM_Malloc_SimpleAllocation")
    ->Range(64, 1 << 20)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(benchmark_simple_allocation, aligned_malloc_backend)
    ->Name("BM_StandardAligned_SimpleAllocation")
    ->Range(64, 1 << 20)
    ->Unit(benchmark::kMicrosecond);

#if MEMORY_HAS_MIMALLOC
BENCHMARK_TEMPLATE(benchmark_simple_allocation, mimalloc_backend)
    ->Name("BM_Mimalloc_SimpleAllocation")
    ->Range(64, 1 << 20)
    ->Unit(benchmark::kMicrosecond);
#endif

#if MEMORY_HAS_TBB
BENCHMARK_TEMPLATE(benchmark_simple_allocation, tbb_backend)
    ->Name("BM_TBB_SimpleAllocation")
    ->Range(64, 1 << 20)
    ->Unit(benchmark::kMicrosecond);
#endif

BENCHMARK_TEMPLATE(benchmark_simple_allocation, xsigma_cpu_backend)
    ->Name("BM_XSigmaCpu_SimpleAllocation")
    ->Range(64, 1 << 20)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(benchmark_simple_allocation, xsigma_stl_backend)
    ->Name("BM_XSigmaStl_SimpleAllocation")
    ->Range(64, 1 << 20)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(benchmark_data_ptr_simple)
    ->Name("BM_DataPtr_SimpleAllocation")
    ->Range(64, 1 << 20)
    ->Unit(benchmark::kMicrosecond);

// ---- batch churn: N allocations of S bytes, then free all ----

#define MEMORY_BENCH_BATCH_ARGS \
    ->Args({100, 1024})->Args({1000, 1024})->Args({100, 4096})->Args({1000, 4096})

BENCHMARK_TEMPLATE(benchmark_batch_allocation, malloc_backend)
    ->Name("BM_Malloc_BatchAllocation") MEMORY_BENCH_BATCH_ARGS->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(benchmark_batch_allocation, aligned_malloc_backend)
    ->Name("BM_StandardAligned_BatchAllocation")
        MEMORY_BENCH_BATCH_ARGS->Unit(benchmark::kMicrosecond);

#if MEMORY_HAS_MIMALLOC
BENCHMARK_TEMPLATE(benchmark_batch_allocation, mimalloc_backend)
    ->Name("BM_Mimalloc_BatchAllocation") MEMORY_BENCH_BATCH_ARGS->Unit(benchmark::kMicrosecond);
#endif

#if MEMORY_HAS_TBB
BENCHMARK_TEMPLATE(benchmark_batch_allocation, tbb_backend)
    ->Name("BM_TBB_BatchAllocation") MEMORY_BENCH_BATCH_ARGS->Unit(benchmark::kMicrosecond);
#endif

BENCHMARK_TEMPLATE(benchmark_batch_allocation, xsigma_cpu_backend)
    ->Name("BM_XSigmaCpu_BatchAllocation") MEMORY_BENCH_BATCH_ARGS->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(benchmark_batch_allocation, xsigma_stl_backend)
    ->Name("BM_XSigmaStl_BatchAllocation") MEMORY_BENCH_BATCH_ARGS->Unit(benchmark::kMicrosecond);

BENCHMARK(benchmark_data_ptr_batch)
    ->Name("BM_DataPtr_BatchAllocation") MEMORY_BENCH_BATCH_ARGS->Unit(benchmark::kMicrosecond);

#undef MEMORY_BENCH_BATCH_ARGS

// ---- mixed sizes: N allocations of random 64..4096 B sizes ----

#define MEMORY_BENCH_MIXED_ARGS ->Arg(100)->Arg(500)->Arg(1000)

BENCHMARK_TEMPLATE(benchmark_mixed_sizes, malloc_backend)
    ->Name("BM_Malloc_MixedSizes") MEMORY_BENCH_MIXED_ARGS->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(benchmark_mixed_sizes, aligned_malloc_backend)
    ->Name("BM_StandardAligned_MixedSizes") MEMORY_BENCH_MIXED_ARGS->Unit(benchmark::kMicrosecond);

#if MEMORY_HAS_MIMALLOC
BENCHMARK_TEMPLATE(benchmark_mixed_sizes, mimalloc_backend)
    ->Name("BM_Mimalloc_MixedSizes") MEMORY_BENCH_MIXED_ARGS->Unit(benchmark::kMicrosecond);
#endif

#if MEMORY_HAS_TBB
BENCHMARK_TEMPLATE(benchmark_mixed_sizes, tbb_backend)
    ->Name("BM_TBB_MixedSizes") MEMORY_BENCH_MIXED_ARGS->Unit(benchmark::kMicrosecond);
#endif

BENCHMARK_TEMPLATE(benchmark_mixed_sizes, xsigma_cpu_backend)
    ->Name("BM_XSigmaCpu_MixedSizes") MEMORY_BENCH_MIXED_ARGS->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(benchmark_mixed_sizes, xsigma_stl_backend)
    ->Name("BM_XSigmaStl_MixedSizes") MEMORY_BENCH_MIXED_ARGS->Unit(benchmark::kMicrosecond);

#undef MEMORY_BENCH_MIXED_ARGS

// ---- alignment sweep at fixed 1 KiB size ----
// (plain malloc ignores alignment, so it is intentionally not registered here)

#define MEMORY_BENCH_ALIGN_ARGS ->Arg(16)->Arg(32)->Arg(64)->Arg(128)->Arg(256)->Arg(512)

BENCHMARK_TEMPLATE(benchmark_aligned_allocation, aligned_malloc_backend)
    ->Name("BM_StandardAligned_AlignedAllocation")
        MEMORY_BENCH_ALIGN_ARGS->Unit(benchmark::kMicrosecond);

#if MEMORY_HAS_MIMALLOC
BENCHMARK_TEMPLATE(benchmark_aligned_allocation, mimalloc_backend)
    ->Name("BM_Mimalloc_AlignedAllocation") MEMORY_BENCH_ALIGN_ARGS->Unit(benchmark::kMicrosecond);
#endif

#if MEMORY_HAS_TBB
BENCHMARK_TEMPLATE(benchmark_aligned_allocation, tbb_backend)
    ->Name("BM_TBB_AlignedAllocation") MEMORY_BENCH_ALIGN_ARGS->Unit(benchmark::kMicrosecond);
#endif

BENCHMARK_TEMPLATE(benchmark_aligned_allocation, xsigma_cpu_backend)
    ->Name("BM_XSigmaCpu_AlignedAllocation") MEMORY_BENCH_ALIGN_ARGS->Unit(benchmark::kMicrosecond);

#undef MEMORY_BENCH_ALIGN_ARGS

// ---- fragmentation: many small blocks, free every other, allocate larger ----

#define MEMORY_BENCH_FRAG_ARGS ->Arg(1000)->Arg(5000)

BENCHMARK_TEMPLATE(benchmark_fragmentation_pattern, malloc_backend)
    ->Name("BM_Malloc_Fragmentation") MEMORY_BENCH_FRAG_ARGS->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(benchmark_fragmentation_pattern, aligned_malloc_backend)
    ->Name("BM_StandardAligned_Fragmentation")
        MEMORY_BENCH_FRAG_ARGS->Unit(benchmark::kMicrosecond);

#if MEMORY_HAS_MIMALLOC
BENCHMARK_TEMPLATE(benchmark_fragmentation_pattern, mimalloc_backend)
    ->Name("BM_Mimalloc_Fragmentation") MEMORY_BENCH_FRAG_ARGS->Unit(benchmark::kMicrosecond);
#endif

#if MEMORY_HAS_TBB
BENCHMARK_TEMPLATE(benchmark_fragmentation_pattern, tbb_backend)
    ->Name("BM_TBB_Fragmentation") MEMORY_BENCH_FRAG_ARGS->Unit(benchmark::kMicrosecond);
#endif

BENCHMARK_TEMPLATE(benchmark_fragmentation_pattern, xsigma_cpu_backend)
    ->Name("BM_XSigmaCpu_Fragmentation") MEMORY_BENCH_FRAG_ARGS->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(benchmark_fragmentation_pattern, xsigma_stl_backend)
    ->Name("BM_XSigmaStl_Fragmentation") MEMORY_BENCH_FRAG_ARGS->Unit(benchmark::kMicrosecond);

#undef MEMORY_BENCH_FRAG_ARGS

}  // namespace benchmarks
}  // namespace memory
