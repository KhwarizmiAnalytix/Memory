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

#include <cstddef>      // for size_t, ptrdiff_t
#include <cstdint>      // for uintptr_t
#include <cstring>      // for memcpy
#include <exception>    // for bad_alloc
#include <stdexcept>    // for invalid_argument
#include <string>       // for runtime_error messages
#include <type_traits>  // for is_same_v

#include "common/device.h"            // for device_enum
#include "common/memory_macros.h"     // MEMORY_ALIGNMENT, MEMORY_DELETE_CLASS, MEMORY_FORCE_INLINE
#include "helper/memory_allocator.h"  // for cpu::memory_allocator

// GPU caching allocator (CUDA, HIP, or Metal — compile-time exclusive).
// Unified registry: gpu::caching_allocator_for_device(i).
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP || MEMORY_HAS_METAL
#include "gpu/caching_allocator.h"
#endif

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
#include "gpu/gpu_runtime.h"
#endif

namespace memory
{
constexpr bool is_gpu_device(device_enum device_type)
{
    return device_type == device_enum::CUDA || device_type == device_enum::HIP ||
           device_type == device_enum::METAL;
}

/** True when @p device_type is served by the compiled GPU caching allocator. */
constexpr bool is_active_gpu_device(device_enum device_type)
{
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
    return device_type == device_enum::CUDA || device_type == device_enum::HIP;
#elif MEMORY_HAS_METAL
    return device_type == device_enum::METAL;
#else
    (void)device_type;
    return false;
#endif
}

constexpr bool has_gpu_support()
{
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP || MEMORY_HAS_METAL
    return true;
#else
    return false;
#endif
}

constexpr std::size_t optimal_alignment(device_enum device_type)
{
    switch (device_type)
    {
    case device_enum::CPU:
        return MEMORY_ALIGNMENT;
    case device_enum::CUDA:
    case device_enum::HIP:
    case device_enum::METAL:
        return 256;
    default:
        return 32;
    }
}

/**
 * @brief Unified memory allocator supporting both CPU and GPU memory management
 *
 * Allocation strategy per device:
 * - CPU: direct calls into cpu::memory_allocator (mimalloc / TBB / platform).
 * - CUDA/HIP: gpu::caching_allocator_for_device (PyTorch-style segment cache;
 *   HIP uses the same Impl via gpu/gpu_runtime.h).
 * - Metal: same registry name → metal_caching_allocator (Shared MTLBuffers).
 *
 * @tparam T The type of elements to allocate
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

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
    using stream_t = cudaStream_t;
#else
    using stream_t = void*;
#endif

    static constexpr size_type scalar_size    = sizeof(value_type);
    static constexpr size_type alignment_size = alignment / scalar_size;
    static constexpr size_type alignment_mask = alignment_size - 1;

    /**
     * @brief Allocate memory on the specified device
     */
    MEMORY_FORCE_INLINE static pointer allocate(
        size_type n, device_enum type = device_enum::CPU, int device_index = 0)
    {
        if (n == 0)
        {
            return nullptr;
        }

        pointer ptr = nullptr;

        if (type == device_enum::CPU)
        {
            ptr = static_cast<pointer>(
                memory::cpu::memory_allocator::allocate(n * scalar_size, alignment));
        }
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP || MEMORY_HAS_METAL
        else if (is_active_gpu_device(type))
        {
#if MEMORY_HAS_METAL
            if constexpr (std::is_same_v<T, double>)
            {
                throw std::invalid_argument(
                    "Metal backend does not support double precision (no fp64 on Apple "
                    "GPU hardware); use device_enum::CPU for double tensors.");
            }
#endif
            ptr = static_cast<pointer>(
                gpu::caching_allocator_for_device(device_index).allocate(n * scalar_size));
        }
#endif
        else
        {
            // With no GPU backend compiled in, is_active_gpu_device() is constexpr-false, so
            // this is the only reachable branch for a non-CPU device type. The #if above (not
            // just guarding this branch's body) keeps the two throwing branches from becoming
            // identical clones when GPU is off, which is otherwise a bugprone-branch-clone hit.
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

        if (type == device_enum::CPU)
        {
            memory::cpu::memory_allocator::free(ptr);
        }
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP || MEMORY_HAS_METAL
        else if (is_active_gpu_device(type))
        {
            gpu::caching_allocator_for_device(device_index).deallocate(ptr, 0);
        }
#endif
        else
        {
            throw std::invalid_argument("Unsupported device type for deallocation");
        }

        ptr = nullptr;
    }

    /**
     * @brief Copy memory between device spaces
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

        if (from_type == device_enum::CPU && to_type == device_enum::CPU)
        {
            std::memcpy(to, from, nbytes);
            return;
        }

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
        if (from_type == device_enum::CUDA || to_type == device_enum::CUDA ||
            from_type == device_enum::HIP || to_type == device_enum::HIP)
        {
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

            cudaError_t result =
                (stream != nullptr)
                    ? cudaMemcpyAsync(
                          to, from, nbytes, copy_kind, static_cast<cudaStream_t>(stream))
                    : cudaMemcpy(to, from, nbytes, copy_kind);
            if (result != cudaSuccess)
            {
                throw std::runtime_error(
                    "GPU memory copy failed: " + std::string(cudaGetErrorString(result)));
            }
            return;
        }
#elif MEMORY_HAS_METAL
        // Shared-storage MTLBuffers are host-addressable — all METAL sides are memcpy.
        if (from_type == device_enum::METAL || to_type == device_enum::METAL)
        {
            std::memcpy(to, from, nbytes);
            return;
        }
#endif

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

        size_type const first =
            (alignment_size -
             ((reinterpret_cast<std::uintptr_t>(array) / scalar_size) & alignment_mask)) &
            alignment_mask;
        return (first < size) ? first : size;
    }

    MEMORY_FORCE_INLINE static size_type last_aligned(
        size_type aligned_start, size_type size, size_type simd_stride)
    {
        return aligned_start + (((size - aligned_start) / simd_stride) * simd_stride);
    }
};

}  // namespace memory
