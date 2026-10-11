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

// GPU allocation-speed comparison: XSigma vs PyTorch (LibTorch).
//
// GPU half of the XSigma/PyTorch allocation comparison; see
// BenchmarkPyTorchComparisonCpu.cpp for the CPU counterpart (Level 1 raw
// allocator core + Level 2 owning buffer object).
//
// Two backends, each its own guarded section below:
//
//   Metal/MPS — Level 2 (owning buffer object) only. PyTorch does not expose
//   a public raw-allocator API for its MPS backend the way
//   c10::GetCPUAllocator() does for CPU, so there is no Level 1 counterpart:
//
//     memory::data_ptr<float>(numel, device_enum::METAL)  (XSigma's
//       GPU-resident RAII owning buffer, routed through
//       gpu::caching_allocator_for_device / metal_caching_allocator)
//         vs torch::empty({numel}, ...device(kMPS))               (PyTorch's
//       MPS caching allocator)
//
//   CUDA — both levels, since PyTorch does expose a raw CUDA allocator API:
//
//     Level 1: gpu::cuda_caching_allocator::allocate/deallocate (plan task
//       8.6) vs c10::cuda::CUDACachingAllocator::raw_alloc/raw_delete.
//     Level 2: memory::data_ptr<float>(numel, device_enum::CUDA) vs
//       torch::empty({numel}, ...device(kCUDA)).
//
// This file is only added to the build when MEMORY_ENABLE_LIBTORCH is ON and
// find_package(Torch) succeeds (see Testing/Cxx/CMakeLists.txt); each section
// is additionally gated on MEMORY_HAS_METAL / MEMORY_HAS_CUDA so the file
// compiles to an empty translation unit when neither backend nor LibTorch is
// available. HIP has no LibTorch (ROCm) build wired here yet.

#include <benchmark/benchmark.h>

#if MEMORY_HAS_LIBTORCH && MEMORY_HAS_METAL

#include <cstddef>
#include <cstdint>

// LibTorch — must come before any header that pulls <cassert> on MSVC
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996 4100)
#endif
#include <torch/mps.h>
#include <torch/torch.h>
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif

#include "common/data_ptr.h"
#include "common/device.h"

namespace memory
{
namespace benchmarks
{
namespace
{

using data_ptr_t = data_ptr<float>;

// ---------------------------------------------------------------------------
// Level 2 — owning buffer object (float elements), GPU-resident.
// ---------------------------------------------------------------------------

void benchmark_pytorch_mps_tensor_single(benchmark::State& state)
{
    if (!torch::mps::is_available())
    {
        state.SkipWithError("LibTorch MPS device not available");
        return;
    }

    const int64_t numel = state.range(0);
    const auto    options =
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kMPS).requires_grad(false);

    // Warm the MPS caching allocator once so the timed loop measures reuse,
    // matching XSigma's warm caching-allocator path after the first iteration.
    {
        auto warm = torch::empty({numel}, options);
        benchmark::DoNotOptimize(warm.data_ptr());
    }
    torch::mps::synchronize();

    for (auto _ : state)
    {
        torch::Tensor tensor = torch::empty({numel}, options);
        benchmark::DoNotOptimize(tensor.data_ptr<float>());
    }
    torch::mps::synchronize();

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * numel * sizeof(float)));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

void benchmark_xsigma_data_ptr_metal_single(benchmark::State& state)
{
    const std::size_t numel = static_cast<std::size_t>(state.range(0));

    // Warm the caching allocator once, matching the PyTorch benchmark above.
    {
        data_ptr_t warm(numel, device_enum::METAL);
        benchmark::DoNotOptimize(warm.get());
    }

    for (auto _ : state)
    {
        data_ptr_t ptr(numel, device_enum::METAL);
        benchmark::DoNotOptimize(ptr.get());
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * numel * sizeof(float)));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

}  // namespace

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

// Owning buffer object — element sweep 16 .. 256 Ki floats (64 B .. 1 MiB),
// same range as the CPU Level 2 benchmarks for a like-for-like comparison.
BENCHMARK(benchmark_pytorch_mps_tensor_single)
    ->Name("BM_PyTorch_MPS_Tensor_Single")
    ->Range(16, 256 << 10)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(benchmark_xsigma_data_ptr_metal_single)
    ->Name("BM_XSigma_DataPtr_Metal_Single")
    ->Range(16, 256 << 10)
    ->Unit(benchmark::kMicrosecond);

}  // namespace benchmarks
}  // namespace memory

#endif  // MEMORY_HAS_LIBTORCH && MEMORY_HAS_METAL

#if MEMORY_HAS_LIBTORCH && MEMORY_HAS_CUDA

#include <cstddef>
#include <cstdint>

// LibTorch — must come before any header that pulls <cassert> on MSVC
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996 4100)
#endif
#include <torch/torch.h>
// c10::cuda::CUDACachingAllocator::raw_alloc is declared here; not pulled in
// by <torch/torch.h>.
#include <c10/cuda/CUDACachingAllocator.h>
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif

#include "common/data_ptr.h"
#include "common/device.h"
#include "gpu/cuda_caching_allocator.h"
#include "gpu/gpu_runtime.h"

namespace memory
{
namespace benchmarks
{
namespace
{

bool cuda_device_available_pytorch_cmp()
{
    int               device_count = 0;
    const cudaError_t err          = cudaGetDeviceCount(&device_count);
    return err == cudaSuccess && device_count > 0;
}

// ---------------------------------------------------------------------------
// Level 1 — raw allocator core (bytes), warm path (no cache flush between
// iterations). Iteration counts are bounded explicitly, matching
// BenchmarkCudaCachingAllocator.cpp: unbounded convergence here has been
// observed to reproduce a rare, pre-existing, timing-dependent segfault on
// long cudaMalloc/cudaFree churn (Docs/memory_runtime_implementation_plan.md
// Appendix B), independent of which allocator is under test.
// ---------------------------------------------------------------------------

void benchmark_pytorch_cuda_raw_single(benchmark::State& state)
{
    if (!cuda_device_available_pytorch_cmp() || !torch::cuda::is_available())
    {
        state.SkipWithError("No GPU device available");
        return;
    }
    // c10::cuda::CUDACachingAllocator::raw_alloc is a direct client entry point
    // (see the header comment: "Called directly by clients"); unlike
    // torch::empty(), it does not lazily initialize the allocator itself. The
    // normal caller of init() is PyTorch's own CUDA hooks during the first
    // tensor operation; calling raw_alloc before that crashes.
    static bool initialized = false;
    if (!initialized)
    {
        c10::cuda::CUDACachingAllocator::init(torch::cuda::device_count());
        initialized = true;
    }
    const std::size_t nbytes = static_cast<std::size_t>(state.range(0));

    for (auto _ : state)
    {
        void* ptr = c10::cuda::CUDACachingAllocator::raw_alloc(nbytes);
        benchmark::DoNotOptimize(ptr);
        c10::cuda::CUDACachingAllocator::raw_delete(ptr);
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * nbytes));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

void benchmark_xsigma_cuda_raw_single(benchmark::State& state)
{
    if (!cuda_device_available_pytorch_cmp())
    {
        state.SkipWithError("No GPU device available");
        return;
    }
    const std::size_t          nbytes = static_cast<std::size_t>(state.range(0));
    gpu::cuda_caching_allocator allocator(0);

    for (auto _ : state)
    {
        void* ptr = allocator.allocate(nbytes);
        benchmark::DoNotOptimize(ptr);
        allocator.deallocate(ptr, nbytes);
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * nbytes));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

// ---------------------------------------------------------------------------
// Level 2 — owning buffer object (float elements), GPU-resident.
// ---------------------------------------------------------------------------

void benchmark_pytorch_cuda_tensor_single(benchmark::State& state)
{
    if (!cuda_device_available_pytorch_cmp() || !torch::cuda::is_available())
    {
        state.SkipWithError("No GPU device available");
        return;
    }

    const int64_t numel = state.range(0);
    const auto    options =
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false);

    // Warm the CUDA caching allocator once so the timed loop measures reuse,
    // matching XSigma's warm caching-allocator path after the first iteration.
    {
        auto warm = torch::empty({numel}, options);
        benchmark::DoNotOptimize(warm.data_ptr());
    }
    torch::cuda::synchronize();

    for (auto _ : state)
    {
        torch::Tensor tensor = torch::empty({numel}, options);
        benchmark::DoNotOptimize(tensor.data_ptr<float>());
    }
    torch::cuda::synchronize();

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * numel * sizeof(float)));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

void benchmark_xsigma_data_ptr_cuda_single(benchmark::State& state)
{
    if (!cuda_device_available_pytorch_cmp())
    {
        state.SkipWithError("No GPU device available");
        return;
    }
    const std::size_t numel = static_cast<std::size_t>(state.range(0));

    // Warm the caching allocator once, matching the PyTorch benchmark above.
    {
        data_ptr<float> warm(numel, device_enum::CUDA);
        benchmark::DoNotOptimize(warm.get());
    }

    for (auto _ : state)
    {
        data_ptr<float> ptr(numel, device_enum::CUDA);
        benchmark::DoNotOptimize(ptr.get());
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * numel * sizeof(float)));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

}  // namespace

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

// Raw allocator core — byte sweep 4 KiB .. 4 MiB (matches
// BenchmarkCudaCachingAllocator.cpp's warm-path range).
BENCHMARK(benchmark_pytorch_cuda_raw_single)
    ->Name("BM_PyTorch_Cuda_Raw_Single")
    ->Range(4096, 1 << 22)
    ->Iterations(5000)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(benchmark_xsigma_cuda_raw_single)
    ->Name("BM_XSigma_Cuda_Raw_Single")
    ->Range(4096, 1 << 22)
    ->Iterations(5000)
    ->Unit(benchmark::kMicrosecond);

// Owning buffer object — element sweep 16 .. 256 Ki floats (64 B .. 1 MiB),
// same range as the CPU/Metal Level 2 benchmarks for a like-for-like
// comparison.
BENCHMARK(benchmark_pytorch_cuda_tensor_single)
    ->Name("BM_PyTorch_Cuda_Tensor_Single")
    ->Range(16, 256 << 10)
    ->Iterations(5000)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(benchmark_xsigma_data_ptr_cuda_single)
    ->Name("BM_XSigma_DataPtr_Cuda_Single")
    ->Range(16, 256 << 10)
    ->Iterations(5000)
    ->Unit(benchmark::kMicrosecond);

}  // namespace benchmarks
}  // namespace memory

#endif  // MEMORY_HAS_LIBTORCH && MEMORY_HAS_CUDA
