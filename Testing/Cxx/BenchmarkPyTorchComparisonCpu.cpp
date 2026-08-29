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

// CPU allocation-speed comparison: XSigma vs PyTorch (LibTorch).
//
// CPU half of the XSigma/PyTorch allocation comparison; see
// BenchmarkPyTorchComparisonGpu.cpp for the GPU (Metal/MPS) counterpart.
//
// Two levels are compared, each against its closest XSigma counterpart:
//
//   Level 1 — raw allocator core:
//     c10::GetCPUAllocator()->allocate(nbytes)   (PyTorch's CPU DataPtr path,
//     posix_memalign-style, 64-byte aligned)
//       vs cpu::memory_allocator::allocate     (XSigma's production CPU path)
//
//   Level 2 — owning buffer object:
//     torch::empty({numel})                    (full CPU tensor: TensorImpl +
//     Storage + allocator)
//       vs memory::data_ptr<float>             (XSigma's RAII owning buffer)
//
// This file is only added to the build when MEMORY_ENABLE_LIBTORCH is ON and
// find_package(Torch) succeeds (see Testing/Cxx/CMakeLists.txt). Without the
// compile definition it compiles to an empty translation unit.

#include <benchmark/benchmark.h>

#if MEMORY_HAS_LIBTORCH

#include <cstddef>
#include <cstdint>
#include <vector>

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
// c10::GetCPUAllocator is declared here; not pulled in by <torch/torch.h>.
#include <c10/core/CPUAllocator.h>
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif

#include "common/data_ptr.h"
#include "helper/memory_allocator.h"

namespace memory
{
namespace benchmarks
{
namespace
{

// Both libraries 64-byte-align CPU allocations by default
// (PyTorch: c10::gAlignment; XSigma: MEMORY_ALIGNMENT).
constexpr std::size_t kAlignment = 64;

// ---------------------------------------------------------------------------
// Level 1 — raw allocator core (bytes)
// ---------------------------------------------------------------------------

void benchmark_pytorch_c10_single(benchmark::State& state)
{
    const std::size_t nbytes = static_cast<std::size_t>(state.range(0));
    c10::Allocator*   alloc  = c10::GetCPUAllocator();

    for (auto _ : state)
    {
        c10::DataPtr ptr = alloc->allocate(nbytes);
        benchmark::DoNotOptimize(ptr.get());
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * nbytes));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

void benchmark_xsigma_raw_single(benchmark::State& state)
{
    const std::size_t nbytes = static_cast<std::size_t>(state.range(0));

    for (auto _ : state)
    {
        void* ptr = cpu::memory_allocator::allocate(nbytes, kAlignment);
        benchmark::DoNotOptimize(ptr);
        if (ptr != nullptr)
        {
            cpu::memory_allocator::free(ptr);
        }
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * nbytes));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

void benchmark_pytorch_c10_batch(benchmark::State& state)
{
    const std::size_t batch_size = static_cast<std::size_t>(state.range(0));
    const std::size_t nbytes     = static_cast<std::size_t>(state.range(1));
    c10::Allocator*   alloc      = c10::GetCPUAllocator();

    for (auto _ : state)
    {
        std::vector<c10::DataPtr> ptrs;
        ptrs.reserve(batch_size);
        for (std::size_t i = 0; i < batch_size; ++i)
        {
            ptrs.push_back(alloc->allocate(nbytes));
        }

        benchmark::DoNotOptimize(ptrs.data());
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * batch_size * nbytes));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * batch_size));
}

void benchmark_xsigma_raw_batch(benchmark::State& state)
{
    const std::size_t batch_size = static_cast<std::size_t>(state.range(0));
    const std::size_t nbytes     = static_cast<std::size_t>(state.range(1));

    for (auto _ : state)
    {
        std::vector<void*> ptrs;
        ptrs.reserve(batch_size);
        for (std::size_t i = 0; i < batch_size; ++i)
        {
            void* ptr = cpu::memory_allocator::allocate(nbytes, kAlignment);
            if (ptr != nullptr)
            {
                ptrs.push_back(ptr);
            }
        }

        benchmark::DoNotOptimize(ptrs.data());

        for (void* ptr : ptrs)
        {
            cpu::memory_allocator::free(ptr);
        }

        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * batch_size * nbytes));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * batch_size));
}

// ---------------------------------------------------------------------------
// Level 2 — owning buffer object (float elements)
// ---------------------------------------------------------------------------

void benchmark_pytorch_tensor_single(benchmark::State& state)
{
    const int64_t numel   = state.range(0);
    const auto    options = torch::TensorOptions().dtype(torch::kFloat32);

    for (auto _ : state)
    {
        torch::Tensor tensor = torch::empty({numel}, options);
        benchmark::DoNotOptimize(tensor.data_ptr<float>());
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * numel * sizeof(float)));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

void benchmark_xsigma_data_ptr_single(benchmark::State& state)
{
    const std::size_t numel = static_cast<std::size_t>(state.range(0));

    for (auto _ : state)
    {
        data_ptr<float> ptr(numel, device_enum::CPU);
        benchmark::DoNotOptimize(ptr.get());
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * numel * sizeof(float)));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

void benchmark_pytorch_tensor_batch(benchmark::State& state)
{
    const std::size_t batch_size = static_cast<std::size_t>(state.range(0));
    const int64_t     numel      = state.range(1);
    const auto        options    = torch::TensorOptions().dtype(torch::kFloat32);

    for (auto _ : state)
    {
        std::vector<torch::Tensor> tensors;
        tensors.reserve(batch_size);
        for (std::size_t i = 0; i < batch_size; ++i)
        {
            tensors.push_back(torch::empty({numel}, options));
        }

        benchmark::DoNotOptimize(tensors.data());
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(
        static_cast<int64_t>(state.iterations() * batch_size * numel * sizeof(float)));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * batch_size));
}

void benchmark_xsigma_data_ptr_batch(benchmark::State& state)
{
    const std::size_t batch_size = static_cast<std::size_t>(state.range(0));
    const std::size_t numel      = static_cast<std::size_t>(state.range(1));

    for (auto _ : state)
    {
        std::vector<data_ptr<float>> ptrs;
        ptrs.reserve(batch_size);
        for (std::size_t i = 0; i < batch_size; ++i)
        {
            ptrs.emplace_back(numel, device_enum::CPU);
        }

        benchmark::DoNotOptimize(ptrs.data());
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(
        static_cast<int64_t>(state.iterations() * batch_size * numel * sizeof(float)));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * batch_size));
}

}  // namespace

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

// Raw allocator core — byte sweep 64 B .. 1 MiB.
BENCHMARK(benchmark_pytorch_c10_single)
    ->Name("BM_PyTorch_C10_Single")
    ->Range(64, 1 << 20)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(benchmark_xsigma_raw_single)
    ->Name("BM_XSigma_Raw_Single")
    ->Range(64, 1 << 20)
    ->Unit(benchmark::kMicrosecond);

// Owning buffer object — element sweep 16 .. 256 Ki floats (64 B .. 1 MiB).
BENCHMARK(benchmark_pytorch_tensor_single)
    ->Name("BM_PyTorch_Tensor_Single")
    ->Range(16, 256 << 10)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(benchmark_xsigma_data_ptr_single)
    ->Name("BM_XSigma_DataPtr_Single")
    ->Range(16, 256 << 10)
    ->Unit(benchmark::kMicrosecond);

// Batch churn — N buffers of 1 KiB.
BENCHMARK(benchmark_pytorch_c10_batch)
    ->Name("BM_PyTorch_C10_Batch")
    ->Args({100, 1024})
    ->Args({1000, 1024})
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(benchmark_xsigma_raw_batch)
    ->Name("BM_XSigma_Raw_Batch")
    ->Args({100, 1024})
    ->Args({1000, 1024})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(benchmark_pytorch_tensor_batch)
    ->Name("BM_PyTorch_Tensor_Batch")
    ->Args({100, 256})
    ->Args({1000, 256})
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(benchmark_xsigma_data_ptr_batch)
    ->Name("BM_XSigma_DataPtr_Batch")
    ->Args({100, 256})
    ->Args({1000, 256})
    ->Unit(benchmark::kMicrosecond);

}  // namespace benchmarks
}  // namespace memory

#endif  // MEMORY_HAS_LIBTORCH
