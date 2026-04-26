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

#include "memory_allocator.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>  // for std::memset

#include "common/memory_macros.h"
#include "util/memory_exception.h"

#if MEMORY_HAS_TBB

#ifdef _MSC_VER
#pragma push_macro("__TBB_NO_IMPLICIT_LINKAGE")
#define __TBB_NO_IMPLICIT_LINKAGE 1  // NOLINT(bugprone-reserved-identifier)
#endif

#include <tbb/scalable_allocator.h>

#ifdef _MSC_VER
#pragma pop_macro("__TBB_NO_IMPLICIT_LINKAGE")
#endif
#endif

#if MEMORY_HAS_NUMA
#include "common/numa.h"
#endif  // MEMORY_HAS_NUMA

#if MEMORY_HAS_MIMALLOC
#include <mimalloc.h>
#endif

#if MEMORY_HAS_CUDA
#include <cuda.h>  // For CUDA Driver API
#include <cuda_runtime.h>
#endif

#if MEMORY_HAS_HIP
#include <hip/hip_runtime.h>
#endif

//#include "logger/logger.h"

namespace memory::cpu::memory_allocator
{

void* allocate(std::size_t nbytes, std::size_t alignment, init_policy_enum init)
{
    MEMORY_CHECK(
        static_cast<std::ptrdiff_t>(nbytes) > 0,
        "cpu allocate() called with negative or zero size: {}",
        nbytes);

    MEMORY_CHECK_DEBUG(
        is_valid_alignment(alignment),
        "cpu allocate() called with invalid alignment: {} (must be power of 2 >= {})",
        alignment,
        sizeof(void*));

    void* ptr = nullptr;

    // Platform-specific allocation
#if MEMORY_HAS_MIMALLOC
    ptr = mi_aligned_alloc(alignment, nbytes);
#elif MEMORY_HAS_TBB
    ptr = scalable_aligned_malloc(nbytes, alignment);
#elif defined(__ANDROID__)
    ptr = memalign(alignment, nbytes);
#elif defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)
    ptr = _aligned_malloc(nbytes, alignment);  // Fixed syntax error
#else
    // POSIX systems
    if (alignment < sizeof(void*))
    {
        ptr = malloc(nbytes);
    }
    else
    {
        // cppcheck-suppress syntaxError
        if MEMORY_UNLIKELY (posix_memalign(&ptr, alignment, nbytes) != 0)
        {
            return nullptr;
        }
    }
#endif
    // cppcheck-suppress syntaxError
    if MEMORY_UNLIKELY (ptr == nullptr)
    {
        return nullptr;
    }

    // NUMA optimization
#if MEMORY_HAS_NUMA
    NUMAMove(ptr, nbytes, GetCurrentNUMANode());
#endif

    // Memory initialization
    switch (init)
    {
    case init_policy_enum::ZERO:
        std::memset(ptr, 0, nbytes);
        break;
#ifndef NDEBUG
    case init_policy_enum::PATTERN:
        std::memset(ptr, 0xCC, nbytes);
        break;
#endif
    case init_policy_enum::UNINITIALIZED:
    default:
        // Do nothing - fastest option
        break;
    }

    return ptr;
}

void free(void* ptr, MEMORY_UNUSED std::size_t nbytes) noexcept
{
    if MEMORY_LIKELY (ptr != nullptr)
    {
#if MEMORY_HAS_MIMALLOC
        mi_free(ptr);
#elif MEMORY_HAS_TBB
        scalable_aligned_free(ptr);
#elif defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)
        _aligned_free(ptr);  // Fixed syntax error
#else
        ::free(ptr);
#endif

#if MEMORY_HAS_ALLOCATION_STATS
        if (nbytes > 0)
        {
            update_free_stats(nbytes);
        }
#endif
    }
}

void* allocate_tbb(MEMORY_UNUSED std::size_t nbytes, MEMORY_UNUSED std::size_t alignment)
{
#if MEMORY_HAS_TBB
    return scalable_aligned_malloc(nbytes, alignment);
#else
    return nullptr;
#endif
}

void free_tbb(MEMORY_UNUSED void* ptr, MEMORY_UNUSED std::size_t nbytes) noexcept
{
#if MEMORY_HAS_TBB
    scalable_aligned_free(ptr);
#endif
}

void* allocate_mi(MEMORY_UNUSED std::size_t nbytes, MEMORY_UNUSED std::size_t alignment)
{
#if MEMORY_HAS_MIMALLOC
    return mi_malloc_aligned(nbytes, alignment);
#else
    return nullptr;
#endif
}

void free_mi(MEMORY_UNUSED void* ptr, MEMORY_UNUSED std::size_t nbytes) noexcept
{
#if MEMORY_HAS_MIMALLOC
    mi_free(ptr);
#endif
}
}  // namespace memory::cpu::memory_allocator

namespace memory::gpu::memory_allocator
{

// CUDA error checking helper macros
#if MEMORY_HAS_CUDA
#define CUDA_CHECK_RETURN_NULL(call)                                                    \
    do                                                                                  \
    {                                                                                   \
        cudaError_t error = call;                                                       \
        if (error != cudaSuccess)                                                       \
        {                                                                               \
            MEMORY_LOG_ERROR("CUDA error in {}: {}", #call, cudaGetErrorString(error)); \
            return nullptr;                                                             \
        }                                                                               \
    } while (0)

#define CUDA_CHECK_RETURN_FALSE(call)                                                   \
    do                                                                                  \
    {                                                                                   \
        const cudaError_t error = call;                                                 \
        if (error != cudaSuccess)                                                       \
        {                                                                               \
            MEMORY_LOG_ERROR("CUDA error in {}: {}", #call, cudaGetErrorString(error)); \
            return false;                                                               \
        }                                                                               \
    } while (0)
#else
#define CUDA_CHECK_RETURN_NULL(call) return nullptr
#define CUDA_CHECK_RETURN_FALSE(call) return false
#endif

// HIP error checking helper macros
#if MEMORY_HAS_HIP
#define HIP_CHECK_RETURN_NULL(call)                                                   \
    do                                                                                \
    {                                                                                 \
        hipError_t error = call;                                                      \
        if (error != hipSuccess)                                                      \
        {                                                                             \
            MEMORY_LOG_ERROR("HIP error in {}: {}", #call, hipGetErrorString(error)); \
            return nullptr;                                                           \
        }                                                                             \
    } while (0)

#define HIP_CHECK_RETURN_FALSE(call)                                                  \
    do                                                                                \
    {                                                                                 \
        hipError_t error = call;                                                      \
        if (error != hipSuccess)                                                      \
        {                                                                             \
            MEMORY_LOG_ERROR("HIP error in {}: {}", #call, hipGetErrorString(error)); \
            return false;                                                             \
        }                                                                             \
    } while (0)
#else
#define HIP_CHECK_RETURN_NULL(call) return nullptr
#define HIP_CHECK_RETURN_FALSE(call) return false
#endif

void* allocate(
    std::size_t nbytes, int device_id, MEMORY_UNUSED void* stream, MEMORY_UNUSED void* memory_pool)
{
    MEMORY_CHECK(
        static_cast<std::ptrdiff_t>(nbytes) > 0,
        "gpu allocate() called with negative or zero size: {}",
        nbytes);

    if (nbytes == 0)
    {
        return nullptr;
    }

    // Set device context
    if (!set_device(device_id))
    {
        return nullptr;
    }

    void* ptr = nullptr;  //NOLINT

#if MEMORY_HAS_CUDA

#ifdef MEMORY_CUDA_ALLOC_SYNC
    // Use synchronous CUDA Runtime API allocation (more commonly available)
    cudaError_t result = cudaMalloc(&ptr, nbytes);
    if (result != cudaSuccess)
    {
        MEMORY_LOG_WARNING(
            "GPU synchronous allocation failed for {} bytes on device {}: {}",
            nbytes,
            device_id,
            cudaGetErrorString(result));
        return nullptr;
    }

#elif defined(MEMORY_CUDA_ALLOC_ASYNC)
    // Use asynchronous CUDA Runtime API allocation
    cudaStream_t gpu_stream_cuda = static_cast<cudaStream_t>(stream);
    cudaError_t  result          = cudaMallocAsync(&ptr, nbytes, gpu_stream_cuda);
    if (result != cudaSuccess)
    {
        MEMORY_LOG_WARNING(
            "GPU asynchronous allocation failed for {} bytes on device {}: {}",
            nbytes,
            device_id,
            cudaGetErrorString(result));
        return nullptr;
    }

#elif defined(MEMORY_CUDA_ALLOC_POOL_ASYNC)
    // Use pool-based asynchronous CUDA Runtime API allocation
    auto* gpu_stream_cuda = static_cast<cudaStream_t>(stream);
    auto* cuda_pool       = static_cast<cudaMemPool_t>(memory_pool);

    cudaError_t result;
    if (cuda_pool != nullptr)
    {
        result = cudaMallocFromPoolAsync(&ptr, nbytes, cuda_pool, gpu_stream_cuda);
    }
    else
    {
        // Fall back to regular async allocation if no pool is specified
        result = cudaMallocAsync(&ptr, nbytes, gpu_stream_cuda);
    }

    if (result != cudaSuccess)
    {
        MEMORY_LOG_WARNING(
            "GPU pool-based allocation failed for {} bytes on device {}: {}",
            nbytes,
            device_id,
            cudaGetErrorString(result));
        return nullptr;
    }

#else
    // Default to synchronous allocation using CUDA Runtime API
    cudaError_t const result = cudaMalloc(&ptr, nbytes);
    if (result != cudaSuccess)
    {
        MEMORY_LOG_WARNING(
            "GPU default allocation failed for {} bytes on device {}: {}",
            nbytes,
            device_id,
            cudaGetErrorString(result));
        return nullptr;
    }
#endif

#elif MEMORY_HAS_HIP

#if defined(MEMORY_HIP_ALLOC_SYNC)
    // Use synchronous HIP allocation
    hipError_t result = hipMalloc(&ptr, nbytes);
    if (result != hipSuccess)
    {
        MEMORY_LOG_WARNING(
            "HIP synchronous allocation failed for {} bytes on device {}: {}",
            nbytes,
            device_id,
            hipGetErrorString(result));
        return nullptr;
    }

#elif defined(MEMORY_HIP_ALLOC_ASYNC)
    // Use asynchronous HIP allocation
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    hipError_t  result     = hipMallocAsync(&ptr, nbytes, hip_stream);
    if (result != hipSuccess)
    {
        MEMORY_LOG_WARNING(
            "HIP asynchronous allocation failed for {} bytes on device {}: {}",
            nbytes,
            device_id,
            hipGetErrorString(result));
        return nullptr;
    }

#elif defined(MEMORY_HIP_ALLOC_POOL_ASYNC)
    // Use pool-based asynchronous HIP allocation
    hipStream_t  hip_stream = static_cast<hipStream_t>(stream);
    hipMemPool_t hip_pool   = static_cast<hipMemPool_t>(memory_pool);

    hipError_t result;
    if (hip_pool != nullptr)
    {
        result = hipMallocFromPoolAsync(&ptr, nbytes, hip_pool, hip_stream);
    }
    else
    {
        // Fallback to regular async allocation if no pool is specified
        result = hipMallocAsync(&ptr, nbytes, hip_stream);
    }

    if (result != hipSuccess)
    {
        MEMORY_LOG_WARNING(
            "HIP pool allocation failed for {} bytes on device {}: {}",
            nbytes,
            device_id,
            hipGetErrorString(result));
        return nullptr;
    }

#else
    // Default to synchronous HIP allocation
    hipError_t result = hipMalloc(&ptr, nbytes);
    if (result != hipSuccess)
    {
        MEMORY_LOG_WARNING(
            "HIP default allocation failed for {} bytes on device {}: {}",
            nbytes,
            device_id,
            hipGetErrorString(result));
        return nullptr;
    }
#endif

#else
    (void)stream;
    (void)memory_pool;
    MEMORY_LOG_ERROR("GPU support not enabled in this build");
#endif

    return ptr;
}

void free(
    void* ptr, MEMORY_UNUSED std::size_t nbytes, int device_id, MEMORY_UNUSED void* stream) noexcept
{
    if (ptr == nullptr)
    {
        return;
    }

    // Set device context (ignore errors in free)
    set_device(device_id);

#if MEMORY_HAS_CUDA

#ifdef MEMORY_CUDA_ALLOC_SYNC
    // Use synchronous CUDA Runtime API deallocation
    cudaError_t result = cudaFree(ptr);
    if (result != cudaSuccess)
    {
        MEMORY_LOG_ERROR(
            "GPU synchronous deallocation failed for {} bytes at {} on device {}: {}",
            nbytes,
            ptr,
            device_id,
            cudaGetErrorString(result));
    }

#elif defined(MEMORY_CUDA_ALLOC_ASYNC)
    // Use asynchronous CUDA Runtime API deallocation
    cudaStream_t gpu_stream_cuda = static_cast<cudaStream_t>(stream);
    cudaError_t  result          = cudaFreeAsync(ptr, gpu_stream_cuda);
    if (result != cudaSuccess)
    {
        MEMORY_LOG_ERROR(
            "GPU asynchronous deallocation failed for {} bytes at {} on device {}: {}",
            nbytes,
            ptr,
            device_id,
            cudaGetErrorString(result));
    }

#elif defined(MEMORY_CUDA_ALLOC_POOL_ASYNC)
    // Use asynchronous CUDA Runtime API deallocation (same for pool and non-pool)
    auto*             gpu_stream_cuda = static_cast<cudaStream_t>(stream);
    const cudaError_t result          = cudaFreeAsync(ptr, gpu_stream_cuda);
    if (result != cudaSuccess)
    {
        MEMORY_LOG_ERROR(
            "GPU pool-based deallocation failed for {} bytes at {} on device {}: {}",
            nbytes,
            ptr,
            device_id,
            cudaGetErrorString(result));
    }

#else
    // Default to synchronous deallocation using CUDA Runtime API
    cudaError_t const result = cudaFree(ptr);
    if (result != cudaSuccess)
    {
        MEMORY_LOG_ERROR(
            "GPU default deallocation failed for {} bytes at {} on device {}: {}",
            nbytes,
            ptr,
            device_id,
            cudaGetErrorString(result));
    }
#endif

#elif MEMORY_HAS_HIP

#if defined(MEMORY_HIP_ALLOC_SYNC)
    // Use synchronous HIP deallocation
    hipError_t result = hipFree(ptr);
    if (result != hipSuccess)
    {
        MEMORY_LOG_ERROR(
            "HIP synchronous deallocation failed for {} bytes at {} on device {}: {}",
            nbytes,
            ptr,
            device_id,
            hipGetErrorString(result));
    }

#elif defined(MEMORY_HIP_ALLOC_ASYNC)
    // Use asynchronous HIP deallocation
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    hipError_t  result     = hipFreeAsync(ptr, hip_stream);
    if (result != hipSuccess)
    {
        MEMORY_LOG_ERROR(
            "HIP asynchronous deallocation failed for {} bytes at {} on device {}: {}",
            nbytes,
            ptr,
            device_id,
            hipGetErrorString(result));
    }

#elif defined(MEMORY_HIP_ALLOC_POOL_ASYNC)
    // Use asynchronous HIP deallocation (same for pool and non-pool)
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    hipError_t  result     = hipFreeAsync(ptr, hip_stream);
    if (result != hipSuccess)
    {
        MEMORY_LOG_ERROR(
            "HIP pool deallocation failed for {} bytes at {} on device {}: {}",
            nbytes,
            ptr,
            device_id,
            hipGetErrorString(result));
    }

#else
    // Default to synchronous HIP deallocation
    hipError_t result = hipFree(ptr);
    if (result != hipSuccess)
    {
        MEMORY_LOG_ERROR(
            "HIP default deallocation failed for {} bytes at {} on device {}: {}",
            nbytes,
            ptr,
            device_id,
            hipGetErrorString(result));
    }
#endif

#else
    (void)nbytes;
    (void)device_id;
    (void)stream;
#endif
}

bool set_device(int device_id) noexcept
{
#if MEMORY_HAS_CUDA
    CUDA_CHECK_RETURN_FALSE(cudaSetDevice(device_id));
    return true;
#elif MEMORY_HAS_HIP
    HIP_CHECK_RETURN_FALSE(hipSetDevice(device_id));
    return true;
#else
    (void)device_id;
    return false;
#endif
}

int get_current_device() noexcept
{
#if MEMORY_HAS_CUDA
    int               device_id = -1;
    cudaError_t const error     = cudaGetDevice(&device_id);
    if (error != cudaSuccess)
    {
        MEMORY_LOG_ERROR("Failed to get current CUDA device: {}", cudaGetErrorString(error));
        return -1;
    }
    return device_id;
#elif MEMORY_HAS_HIP
    int        device_id = -1;
    hipError_t error     = hipGetDevice(&device_id);
    if (error != hipSuccess)
    {
        MEMORY_LOG_ERROR("Failed to get current HIP device: {}", hipGetErrorString(error));
        return -1;
    }
    return device_id;
#else
    return -1;
#endif
}

}  // namespace memory::gpu::memory_allocator