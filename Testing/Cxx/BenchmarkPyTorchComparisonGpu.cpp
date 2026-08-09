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

// GPU allocation-speed comparison: XSigma vs PyTorch (LibTorch) — Metal/MPS.
//
// GPU half of the XSigma/PyTorch allocation comparison; see
// BenchmarkPyTorchComparisonCpu.cpp for the CPU counterpart (Level 1 raw
// allocator core + Level 2 owning buffer object).
//
// Only a Level 2 (owning buffer object) comparison exists here — there is no
// Level 1 (raw allocator core) counterpart because PyTorch does not expose a
// public raw-allocator API for its MPS backend the way c10::GetCPUAllocator()
// does for CPU (the Vectorization-side equivalent,
// Library/Vectorization/Testing/Cxx/BenchmarkTensorGpu.cpp's
// LibTorch_MPS_TensorAllocFree, has the same constraint):
//
//   memory::data_ptr<float, false>(numel, device_enum::METAL)  (XSigma's
//     GPU-resident RAII owning buffer, routed through
//     gpu::caching_allocator_for_device / metal_caching_allocator)
//       vs torch::empty({numel}, ...device(kMPS))               (PyTorch's
//     MPS caching allocator)
//
// This file is only added to the build when MEMORY_ENABLE_LIBTORCH is ON and
// find_package(Torch) succeeds (see Testing/Cxx/CMakeLists.txt); the actual
// benchmarks are additionally gated on MEMORY_HAS_METAL so the file compiles
// to an empty translation unit on CUDA/HIP/no-GPU builds. Metal only for
// now — CUDA/HIP LibTorch comparisons already live in
// Library/Vectorization/Testing/Cxx/BenchmarkTensorLibTorch.cpp; add a
// CUDA/HIP branch here if a Memory-scoped comparison is needed later.

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

using data_ptr_t = data_ptr<float, false>;

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

    const int64_t numel   = state.range(0);
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
