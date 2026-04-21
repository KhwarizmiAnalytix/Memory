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

#include "common/device.h"         // for device_enum
#include "common/memory_macros.h"  // for MEMORY_FORCE_INLINE, MEMORY_ALIGNMENT, XSIG...
#include "cpu/allocator.h"         // for Allocator
#include "cpu/allocator_device.h"
#include "helper/process_state.h"  // for process_state

// GPU support includes
#if MEMORY_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace memory
{
/**
 * @brief Unified memory allocator supporting both CPU and GPU memory management
 *
 * This allocator provides a unified interface for memory allocation across different
 * device types including CPU, CUDA, and HIP devices. It automatically selects
 * the appropriate allocation strategy based on the device type and integrates with
 * Memory's memory pool system for optimal performance in quantitative finance applications.
 *
 * Key Features:
 * - Unified interface for CPU and GPU memory allocation
 * - Automatic device-specific optimization
 * - Memory pooling for reduced allocation overhead
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
    static inline constexpr size_type scalar_size    = sizeof(value_type);
    static inline constexpr size_type alignment_size = alignment / scalar_size;
    static inline constexpr size_type alignment_mask = alignment_size - 1;

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
            return nullptr;

        pointer ptr = nullptr;

        // CPU allocation
        if (type == device_enum::CPU)
        {
            ptr = static_cast<pointer>(
                memory::process_state::singleton()->GetCPUAllocator(0)->allocate_raw(
                    alignment, n * scalar_size));
        }
        // GPU allocation using direct CUDA calls
#if MEMORY_HAS_CUDA
        else if (type == device_enum::CUDA || type == device_enum::HIP)
        {
            // Set the device
            cudaError_t result = cudaSetDevice(device_index);
            if (result != cudaSuccess)
            {
                throw std::runtime_error(
                    "Failed to set CUDA device: " + std::string(cudaGetErrorString(result)));
            }

            // Allocate GPU memory
            const size_type bytes = n * scalar_size;
            result                = cudaMalloc(&ptr, bytes);
            if (result != cudaSuccess)
            {
                throw std::bad_alloc();
            }
        }
#endif
        else
        {
            throw std::invalid_argument("Unsupported device type for allocation");
        }

        if (!ptr)
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
        if (!ptr)
            return;

        // CPU deallocation
        if (type == device_enum::CPU)
        {
#if MEMORY_HAS_NUMA
            int numa_node = GetCurrentNUMANode();
#else
            int numa_node = 0;
#endif
            memory::process_state::singleton()->GetCPUAllocator(numa_node)->deallocate_raw(ptr);
        }
        // GPU deallocation using direct CUDA calls
#if MEMORY_HAS_CUDA
        else if (type == device_enum::CUDA || type == device_enum::HIP)
        {
            // Set the device
            cudaError_t result = cudaSetDevice(device_index);
            if (result != cudaSuccess)
            {
                // Log error but don't throw from free
                // MEMORY_LOG_ERROR("Failed to set CUDA device during deallocation");
            }

            // Free GPU memory
            result = cudaFree(ptr);
            if (result != cudaSuccess)
            {
                // Log error but don't throw from free
                // MEMORY_LOG_ERROR("CUDA free failed");
            }
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
        if (!from || !to || n == 0)
            return;

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
            if (stream)
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

        if (reinterpret_cast<std::uintptr_t>(array) & (scalar_size - 1))
        {
            return size;
        }

        size_type first = (alignment_size - (reinterpret_cast<std::uintptr_t>(array) / scalar_size &
                                             alignment_mask)) &
                          alignment_mask;
        return (first < size) ? first : size;
    }

    MEMORY_FORCE_INLINE static size_type last_aligned(
        size_type aligned_start, size_type size, size_type simd_stride)
    {
        return aligned_start + ((size - aligned_start) / simd_stride) * simd_stride;
    }

    // GPU-specific convenience methods
#if MEMORY_HAS_CUDA

    /**
     * @brief Allocate pinned CPU memory for efficient GPU transfers
     * @param n Number of elements to allocate
     * @return Pointer to pinned CPU memory
     */
    MEMORY_FORCE_INLINE static pointer allocate_pinned(size_type n)
    {
        if (n == 0)
            return nullptr;

        void* ptr = nullptr;  // allocator_device::allocate(n * scalar_size);
        return static_cast<pointer>(ptr);
    }

    /**
     * @brief Free pinned CPU memory
     * @param ptr Reference to pointer to pinned memory (will be set to nullptr)
     */
    MEMORY_FORCE_INLINE static void free_pinned(pointer& ptr)
    {
        if (!ptr)
            return;

        //allocator_device::free(ptr);
        //ptr = nullptr;
    }

    /**
     * @brief Configure GPU memory pool for optimal performance
     * @param device_type device_option type to configure
     * @param device_index device_option index to configure
     * @param min_block_size Minimum block size in bytes
     * @param max_pool_size Maximum pool size in bytes
     *
     * @note This method is deprecated after gpu_allocator removal.
     *       Use gpu_memory_pool directly for advanced memory management.
     */
    static void configure_gpu_pool(
        device_enum device_type    = device_enum::CUDA,
        int         device_index   = 0,
        size_type   min_block_size = 1024,
        size_type   max_pool_size  = 1024ULL * 1024ULL)
    {
        // No-op: Direct CUDA allocation doesn't use pools
        (void)device_type;
        (void)device_index;
        (void)min_block_size;
        (void)max_pool_size;
    }

    /**
     * @brief Get allocated memory statistics for GPU device
     * @param device_type device_option type to query
     * @param device_index device_option index to query
     * @return Number of bytes currently allocated
     *
     * @note This method is deprecated after gpu_allocator removal.
     *       Returns 0 as direct CUDA allocation doesn't track statistics.
     */
    static size_type get_gpu_allocated_bytes(
        device_enum device_type = device_enum::CUDA, int device_index = 0)
    {
        // No tracking available with direct CUDA allocation
        (void)device_type;
        (void)device_index;
        return 0;
    }

#endif  // GPU support
};

/**
 * @brief Helper function to determine optimal alignment for device type
 * @param device_type Target device type
 * @return Recommended alignment in bytes
 */
inline constexpr std::size_t optimal_alignment(device_enum device_type)
{
    switch (device_type)
    {
    case device_enum::CPU:
        return MEMORY_ALIGNMENT;
    case device_enum::CUDA:
    case device_enum::HIP:
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
inline constexpr bool is_gpu_device(device_enum device_type)
{
    return device_type == device_enum::CUDA || device_type == device_enum::HIP;
}

/**
 * @brief Helper function to check if GPU support is compiled in
 * @return True if GPU support is available
 */
inline constexpr bool has_gpu_support()
{
#if MEMORY_HAS_CUDA
    return true;
#else
    return false;
#endif
}

}  // namespace memory
