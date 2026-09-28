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

// Order 0 baseline: real-hardware CUDA/HIP caching-allocator benchmarks.
// Run the built binary directly (not through ctest's --benchmark_min_time=0.01s
// smoke mode) for real numbers, e.g.:
//   bin/benchmark_memory_cudacachingallocator --benchmark_repetitions=20 \
//     --benchmark_report_aggregates_only=true --benchmark_out=cuda_baseline.json \
//     --benchmark_out_format=json
// See Docs/cpu_gpu_memory_review.md's Order 0 row for where recorded results live.

#include <benchmark/benchmark.h>

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP

#include <cstddef>
#include <vector>

#include "gpu/cuda_caching_allocator.h"
#include "gpu/device_guard.h"
#include "gpu/gpu_runtime.h"

namespace memory
{
namespace benchmarks
{
namespace
{

bool cuda_device_available()
{
    int               device_count = 0;
    const cudaError_t err          = cudaGetDeviceCount(&device_count);
    return err == cudaSuccess && device_count > 0;
}

class bench_stream
{
public:
    bench_stream()
    {
        gpu::throw_on_cuda_error(
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    }
    ~bench_stream() { (void)cudaStreamDestroy(stream); }
    bench_stream(const bench_stream&)            = delete;
    bench_stream& operator=(const bench_stream&) = delete;
    cudaStream_t stream{nullptr};
};

// =============================================================================
// Cold path: empty_cache() before each timed allocate+deallocate pair forces
// every iteration to go through the driver (cudaMalloc/cudaFree), not the
// segment cache.
// =============================================================================
void benchmark_cold_alloc_free(benchmark::State& state)
{
    if (!cuda_device_available())
    {
        state.SkipWithError("No GPU device available");
        return;
    }
    const std::size_t      size = static_cast<std::size_t>(state.range(0));
    gpu::cuda_caching_allocator allocator(0);

    for (auto _ : state)
    {
        allocator.empty_cache();
        void* ptr = allocator.allocate(size);
        benchmark::DoNotOptimize(ptr);
        allocator.deallocate(ptr, size);
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * size));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}
// Iterations bounded explicitly (see Docs/cpu_gpu_memory_review.md's new
// finding on repeated-churn instability): letting Benchmark's own
// convergence pick iteration counts here has been observed to run into the
// tens/hundreds of thousands of real cudaMalloc/cudaFree round trips across
// many short-lived allocator instances, which is the exact regime that
// reproduces a rare (~30-40% per full-suite run), pre-existing, timing-
// dependent segfault -- present before this benchmark existed, not
// introduced by it. Bounding iterations here is a mitigation, not a fix.
BENCHMARK(benchmark_cold_alloc_free)
    ->Name("BM_Cuda_ColdAllocFree")
    ->Range(4096, 1 << 22)
    ->Iterations(200)
    ->Unit(benchmark::kMicrosecond);

// =============================================================================
// Warm path: no empty_cache() between iterations, so after the first
// iteration every allocate() is a same-size cache hit.
// =============================================================================
void benchmark_warm_alloc_free(benchmark::State& state)
{
    if (!cuda_device_available())
    {
        state.SkipWithError("No GPU device available");
        return;
    }
    const std::size_t      size = static_cast<std::size_t>(state.range(0));
    gpu::cuda_caching_allocator allocator(0);

    for (auto _ : state)
    {
        void* ptr = allocator.allocate(size);
        benchmark::DoNotOptimize(ptr);
        allocator.deallocate(ptr, size);
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * size));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}
BENCHMARK(benchmark_warm_alloc_free)
    ->Name("BM_Cuda_WarmAllocFree")
    ->Range(4096, 1 << 22)
    ->Iterations(5000)
    ->Unit(benchmark::kMicrosecond);

// =============================================================================
// Changing sizes: cycles through a small set of distinct sizes so the warm
// cache must serve a mix of size classes rather than always reusing the same
// exact block.
// =============================================================================
void benchmark_changing_size_alloc_free(benchmark::State& state)
{
    if (!cuda_device_available())
    {
        state.SkipWithError("No GPU device available");
        return;
    }
    static const std::vector<std::size_t> kSizes = {
        4096, 65536, 262144, 1 << 20, 2 << 20, 8 << 20};
    gpu::cuda_caching_allocator allocator(0);

    std::size_t i = 0;
    for (auto _ : state)
    {
        const std::size_t size = kSizes[i % kSizes.size()];
        ++i;
        void* ptr = allocator.allocate(size);
        benchmark::DoNotOptimize(ptr);
        allocator.deallocate(ptr, size);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}
BENCHMARK(benchmark_changing_size_alloc_free)
    ->Name("BM_Cuda_ChangingSizeAllocFree")
    ->Iterations(5000)
    ->Unit(benchmark::kMicrosecond);

// =============================================================================
// Multi-stream: round-robins allocate/deallocate across N independently
// created streams, exercising the per-stream free-pool split the segment
// cache maintains (blocks are never reused across streams without an event).
// =============================================================================
void benchmark_multi_stream_alloc_free(benchmark::State& state)
{
    if (!cuda_device_available())
    {
        state.SkipWithError("No GPU device available");
        return;
    }
    const int               stream_count = static_cast<int>(state.range(0));
    constexpr std::size_t    kSize        = 65536;
    gpu::cuda_caching_allocator allocator(0);

    std::vector<bench_stream> streams(static_cast<std::size_t>(stream_count));

    std::size_t i = 0;
    for (auto _ : state)
    {
        cudaStream_t const stream = streams[i % streams.size()].stream;
        ++i;
        void* ptr = allocator.allocate(kSize, stream);
        benchmark::DoNotOptimize(ptr);
        allocator.deallocate(ptr, kSize, stream);
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * kSize));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}
BENCHMARK(benchmark_multi_stream_alloc_free)
    ->Name("BM_Cuda_MultiStreamAllocFree")
    ->Arg(1)
    ->Arg(4)
    ->Arg(16)
    ->Iterations(5000)
    ->Unit(benchmark::kMicrosecond);

}  // namespace
}  // namespace benchmarks
}  // namespace memory

#endif  // MEMORY_HAS_CUDA || MEMORY_HAS_HIP
