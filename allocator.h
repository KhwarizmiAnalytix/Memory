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

#pragma once

#include <cstddef>    // for size_t, ptrdiff_t
#include <cstdint>    // for uintptr_t
#include <cstring>    // for memcpy
#include <exception>  // for bad_alloc
#include <stdexcept>  // for invalid_argument
#include <type_traits>  // for is_same_v

#include "common/device.h"           // for device_enum
#include "common/memory_macros.h"    // MEMORY_ALIGNMENT, MEMORY_DELETE_CLASS, MEMORY_FORCE_INLINE
#include "helper/memory_allocator.h"  // for cpu::memory_allocator

// GPU support includes
#if MEMORY_HAS_CUDA
#include <cuda_runtime.h>

#include "gpu/cuda_caching_allocator.h"  // for caching_allocator_for_device
#elif MEMORY_HAS_METAL
#include "gpu/metal/metal_buffer_allocator.h"
#include "gpu/metal/metal_caching_allocator.h"
#endif

namespace memory
{
/**
 * @brief Unified memory allocator supporting both CPU and GPU memory management
 *
 * This allocator provides a unified interface for memory allocation across different
 * device types including CPU, CUDA, and HIP devices.
 *
 * Allocation strategy per device:
 * - CPU: direct calls into the raw allocation backend
 *   (cpu::memory_allocator — mimalloc, TBB, or platform aligned malloc,
 *   selected at compile time).
 * - CUDA/HIP: the per-device cuda_caching_allocator (PyTorch-style segment
 *   caching with stream-aware reuse), shared process-wide.
 * - Metal: the per-device metal_caching_allocator (same segment-cache model
 *   on shared-storage MTLBuffers), shared process-wide.
 *
 * Key Features:
 * - Unified interface for CPU and GPU memory allocation
 * - Cached GPU allocation with stream-aware reuse
 * - Asynchronous memory transfers with CUDA streams
 * - Exception-safe RAII memory management
 * - Support for multiple GPU devices
 * - Memory alignment for SIMD and GPU coalescing
 *
 * @tparam T The type of elements to allocate
 * @tparam d_type Default device type (CPU, CUDA, HIP)
 * @tparam alignment Memory alignment requirement in bytes
 */
template <class T, std::size_t alignment = MEMORY_ALIGNMENT>
struct allocator
{
    MEMORY_DELETE_CLASS(allocator)

public:
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using value_type      = T;
    using pointer         = T*;
    using const_pointer   = const T*;

    // Stream type for asynchronous operations
#if MEMORY_HAS_CUDA
    using stream_t = cudaStream_t;
#else
    using stream_t = void*;
#endif

    // Type traits and constants
    static constexpr size_type scalar_size    = sizeof(value_type);
    static constexpr size_type alignment_size = alignment / scalar_size;
    static constexpr size_type alignment_mask = alignment_size - 1;

    /**
     * @brief Allocate memory on the specified device
     * @param n Number of elements to allocate
     * @param type Target device type (CPU, CUDA, HIP)
     * @param device_index device_option index for multi-GPU systems (default: 0)
     * @return Pointer to allocated memory
     * @throws std::bad_alloc if allocation fails
     */
    MEMORY_FORCE_INLINE static pointer allocate(
        size_type n, device_enum type = device_enum::CPU, int device_index = 0)
    {
        if (n == 0)
        {
            return nullptr;
        }

        pointer ptr = nullptr;

        // CPU allocation — direct call into the raw allocation backend
        // (mimalloc / TBB / platform aligned malloc, selected at compile time)
        if (type == device_enum::CPU)
        {
            ptr = static_cast<pointer>(
                memory::cpu::memory_allocator::allocate(n * scalar_size, alignment));
        }
        // GPU allocation via the per-device caching allocator
#if MEMORY_HAS_CUDA
        else if (type == device_enum::CUDA || type == device_enum::HIP)
        {
            ptr = static_cast<pointer>(
                gpu::caching_allocator_for_device(device_index).allocate(n * scalar_size));
        }
#elif MEMORY_HAS_METAL
        // Metal allocation via the per-device caching allocator (Shared MTLBuffers;
        // float only — see below)
        else if (type == device_enum::METAL)
        {
            if constexpr (std::is_same_v<T, double>)
            {
                throw std::invalid_argument(
                    "Metal backend does not support double precision (no fp64 on Apple "
                    "GPU hardware); use device_enum::CPU for double tensors.");
            }
            else
            {
                ptr = static_cast<pointer>(
                    gpu::metal_caching_allocator_for_device(device_index).allocate(n * scalar_size));
            }
        }
#endif
        else
        {
            throw std::invalid_argument("Unsupported device type for allocation");
        }

        if (ptr == nullptr)
        {
            throw std::bad_alloc();
        }
        return ptr;
    }

    /**
     * @brief Free memory allocated on the specified device
     * @param ptr Reference to pointer to memory to free (will be set to nullptr)
     * @param type device_option type where memory was allocated
     * @param device_index device_option index for multi-GPU systems (default: 0)
     * @param count Number of elements (for tracking purposes, default: 0)
     */
    MEMORY_FORCE_INLINE static void free(
        pointer&    ptr,
        device_enum type         = device_enum::CPU,
        int         device_index = 0,
        size_type   count        = 0)
    {
        (void)count;
        if (ptr == nullptr)
        {
            return;
        }

        // CPU deallocation
        if (type == device_enum::CPU)
        {
            memory::cpu::memory_allocator::free(ptr);
        }
        // GPU deallocation via the per-device caching allocator (cached for reuse)
#if MEMORY_HAS_CUDA
        else if (type == device_enum::CUDA || type == device_enum::HIP)
        {
            gpu::caching_allocator_for_device(device_index).deallocate(ptr, 0);
        }
#elif MEMORY_HAS_METAL
        else if (type == device_enum::METAL)
        {
            gpu::metal_caching_allocator_for_device(device_index).deallocate(ptr, 0);
        }
#endif
        else
        {
            throw std::invalid_argument("Unsupported device type for allocation");
        }

        ptr = nullptr;
    }

    /**
     * @brief Copy memory between different memory spaces
     * @param from Source pointer
     * @param n Number of elements to copy
     * @param to Destination pointer
     * @param from_type Source device type
     * @param to_type Destination device type
     * @param from_index Source device index (default: 0)
     * @param to_index Destination device index (default: 0)
     * @param stream Stream for asynchronous operations (default: nullptr)
     */
    MEMORY_FORCE_INLINE static void copy(
        const_pointer from,
        size_type     n,
        pointer       to,
        device_enum   from_type  = device_enum::CPU,
        device_enum   to_type    = device_enum::CPU,
        int           from_index = 0,
        int           to_index   = 0,
        stream_t      stream     = nullptr)
    {
        (void)from_index;
        (void)to_index;
        if (from == nullptr || to == nullptr || n == 0)
        {
            return;
        }

        const auto nbytes = n * scalar_size;

        // CPU-to-CPU copy
        if (from_type == device_enum::CPU && to_type == device_enum::CPU)
        {
            std::memcpy(to, from, nbytes);
            return;
        }

        // GPU-involved copies using direct CUDA operations
#if MEMORY_HAS_CUDA
        if (from_type == device_enum::CUDA || to_type == device_enum::CUDA ||
            from_type == device_enum::HIP || to_type == device_enum::HIP)
        {
            // Determine CUDA memory copy kind
            cudaMemcpyKind copy_kind;

            if (from_type == device_enum::CPU &&
                (to_type == device_enum::CUDA || to_type == device_enum::HIP))
            {
                copy_kind = cudaMemcpyHostToDevice;
            }
            else if (
                (from_type == device_enum::CUDA || from_type == device_enum::HIP) &&
                to_type == device_enum::CPU)
            {
                copy_kind = cudaMemcpyDeviceToHost;
            }
            else if (
                (from_type == device_enum::CUDA || from_type == device_enum::HIP) &&
                (to_type == device_enum::CUDA || to_type == device_enum::HIP))
            {
                copy_kind = cudaMemcpyDeviceToDevice;
            }
            else
            {
                throw std::invalid_argument("Unsupported GPU device combination for memory copy");
            }

            // Perform the memory copy
            cudaError_t result;
            if (stream != nullptr)
            {
                // Asynchronous copy
                result =
                    cudaMemcpyAsync(to, from, nbytes, copy_kind, static_cast<cudaStream_t>(stream));
            }
            else
            {
                // Synchronous copy
                result = cudaMemcpy(to, from, nbytes, copy_kind);
            }

            // Check for CUDA errors
            if (result != cudaSuccess)
            {
                throw std::runtime_error(
                    "CUDA memory copy failed: " + std::string(cudaGetErrorString(result)));
            }

            return;
        }
#elif MEMORY_HAS_METAL
        // Metal buffers are shared-storage (unified memory) on Apple Silicon: the pointer
        // is host-addressable regardless of which side is CPU vs METAL, so every
        // combination (CPU<->METAL, METAL<->METAL) is a plain memcpy — no explicit
        // host/device transfer exists the way it does for CUDA/HIP.
        if (from_type == device_enum::METAL || to_type == device_enum::METAL)
        {
            std::memcpy(to, from, nbytes);
            return;
        }
#endif

        // Fallback for unsupported combinations
        throw std::invalid_argument("Unsupported device combination for memory copy");
    }

    MEMORY_FORCE_INLINE static size_type first_aligned(const_pointer array, size_type size)
    {
        if constexpr ((alignment % scalar_size) != 0)
        {
            return size;
        }

        if ((reinterpret_cast<std::uintptr_t>(array) & (scalar_size - 1)) != 0U)
        {
            return size;
        }

        size_type const first = (alignment_size -
                                 ((reinterpret_cast<std::uintptr_t>(array) / scalar_size) &
                                  alignment_mask)) &
                                alignment_mask;
        return (first < size) ? first : size;
    }

    MEMORY_FORCE_INLINE static size_type last_aligned(
        size_type aligned_start, size_type size, size_type simd_stride)
    {
        return aligned_start +
               (((size - aligned_start) / simd_stride) * simd_stride);
    }

};

/**
 * @brief Helper function to determine optimal alignment for device type
 * @param device_type Target device type
 * @return Recommended alignment in bytes
 */
constexpr std::size_t optimal_alignment(device_enum device_type)
{
    switch (device_type)
    {
    case device_enum::CPU:
        return MEMORY_ALIGNMENT;
    case device_enum::CUDA:
    case device_enum::HIP:
    case device_enum::METAL:
        return 256;  // GPU coalescing alignment
    default:
        return 32;
    }
}

/**
 * @brief Helper function to check if device type supports GPU operations
 * @param device_type device_option type to check
 * @return True if device supports GPU operations
 */
constexpr bool is_gpu_device(device_enum device_type)
{
    return device_type == device_enum::CUDA || device_type == device_enum::HIP ||
           device_type == device_enum::METAL;
}

/**
 * @brief Helper function to check if GPU support is compiled in
 * @return True if GPU support is available
 */
constexpr bool has_gpu_support()
{
#if MEMORY_HAS_CUDA || MEMORY_HAS_METAL
    return true;
#else
    return false;
#endif
}

}  // namespace memory
