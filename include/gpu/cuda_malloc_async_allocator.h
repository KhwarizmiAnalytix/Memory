/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#pragma once

// Optional stream-ordered driver-pool backend.
//
// Enable with -DMEMORY_USE_CUDA_MALLOC_ASYNC=1 at CMake configure time.
// DISABLED by default: cudaMallocAsync has different ordering contracts from
// cudaMalloc and is NOT a drop-in replacement.  Requirements:
//
//   - All allocation, use, and free operations MUST be ordered on the SAME
//     stream unless pool peer-access has been explicitly enabled.
//   - cudaFreeAsync(ptr, stream) is only safe if all work that touches ptr
//     on any OTHER stream has been synchronised before the free call.
//   - Pool capture is required for CUDA graph usage; see gpu_graph_pool.h.
//   - CUDA 11.2+ required.  HIP support requires independent validation.
//
// See NVIDIA stream-ordered allocator documentation before enabling:
// https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/
//        stream-ordered-memory-allocation.html

#if MEMORY_USE_CUDA_MALLOC_ASYNC && (MEMORY_HAS_CUDA || MEMORY_HAS_HIP)

#include <cstddef>
#include <stdexcept>
#include <string>

#include "common/memory_export.h"
#include "common/memory_macros.h"
#include "gpu/device_guard.h"
#include "gpu/gpu_runtime.h"

namespace memory::gpu
{

// Thin wrapper around cudaMallocAsync / cudaFreeAsync.
//
// Suitable only for single-stream workloads or workloads where the caller
// manages cross-stream ordering explicitly.  For all other cases use the
// default caching_allocator (cuda_caching_allocator) which handles
// multi-stream reuse through CUDA events.
class MEMORY_VISIBILITY cuda_malloc_async_allocator
{
public:
    explicit cuda_malloc_async_allocator(int device = 0) : device_(device) {}

    // Allocate @p bytes on @p stream via cudaMallocAsync.
    // Returned memory is not valid until all prior work on @p stream completes.
    void* allocate(size_t bytes, cudaStream_t stream)
    {
        if (bytes == 0) return nullptr;
        void*             ptr = nullptr;
        device_guard const g(device_);
        cudaError_t const  r = cudaMallocAsync(&ptr, bytes, stream);
        if (r != cudaSuccess)
        {
            throw std::runtime_error(
                "cudaMallocAsync failed: " + std::string(cudaGetErrorString(r)));
        }
        return ptr;
    }

    // Free @p ptr on @p stream via cudaFreeAsync.
    // @p ptr must not be accessed after this call on @p stream or any other
    // stream that has not synchronised against @p stream at this point.
    void deallocate(void* ptr, cudaStream_t stream) noexcept
    {
        if (!ptr) return;
        device_guard const g(device_, std::nothrow);
        (void)cudaFreeAsync(ptr, stream);
    }

    int device() const noexcept { return device_; }

private:
    int device_{0};
};

}  // namespace memory::gpu

#endif  // MEMORY_USE_CUDA_MALLOC_ASYNC && (MEMORY_HAS_CUDA || MEMORY_HAS_HIP)
