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
#include <atomic>  // for allocation statistics

#if __cplusplus >= 202002L
#include <bit>  // for std::has_single_bit (C++20)
#endif

#include <cstdlib>

#include "common/memory_export.h"
#include "common/memory_macros.h"

#if MEMORY_HAS_CUDA
#include <cuda.h>  // For CUDA Driver API
#include <cuda_runtime.h>
#endif

#if MEMORY_HAS_HIP
#include <hip/hip_runtime.h>
#endif

namespace memory
{
namespace cpu
{
namespace memory_allocator
{
// Memory initialization options
enum class init_policy_enum : uint8_t
{
    UNINITIALIZED = 0,  // Don't initialize memory (fastest)
    ZERO          = 1,  // Zero-fill memory
    PATTERN       = 2   // Fill with debug pattern (debug builds only)
};

// Validate alignment is power of 2 and >= sizeof(void*)
MEMORY_FORCE_INLINE bool is_valid_alignment(std::size_t alignment) noexcept
{
    return alignment >= sizeof(void*) &&
#if __cplusplus >= 202002L
           std::has_single_bit(alignment);  // C++20
#else
           (alignment & (alignment - 1)) == 0;  // Power of 2 check
#endif
}

// Get default alignment for the platform
MEMORY_FORCE_INLINE MEMORY_FUNCTION_CONSTEXPR std::size_t default_alignment() noexcept
{
    return MEMORY_ALIGNMENT;
}

MEMORY_API void* allocate(
    std::size_t      nbytes,
    std::size_t      alignment = default_alignment(),
    init_policy_enum init      = init_policy_enum::UNINITIALIZED);

MEMORY_API void free(void* ptr, std::size_t nbytes = 0) noexcept;

// TBB-specific allocation and deallocation
MEMORY_API void* allocate_tbb(std::size_t nbytes, std::size_t alignment = default_alignment());

MEMORY_API void free_tbb(void* ptr, std::size_t nbytes = 0) noexcept;

// mimalloc-specific allocation and deallocation
MEMORY_API void* allocate_mi(std::size_t nbytes, std::size_t alignment = default_alignment());

MEMORY_API void free_mi(void* ptr, std::size_t nbytes = 0) noexcept;

// Zero-initialized allocation
MEMORY_FORCE_INLINE void* allocate_zero(
    std::size_t nbytes, std::size_t alignment = default_alignment())
{
    return allocate(nbytes, alignment, init_policy_enum::ZERO);
}

}  // namespace memory_allocator
}  // namespace cpu

namespace gpu
{
namespace memory_allocator
{

/**
 * @brief GPU memory allocation strategy enumeration.
 *
 * Determines the allocation method used at compile time based on CMake flags.
 */
enum class allocation_strategy : uint8_t
{
    SYNC,  ///< Synchronous allocation using cuMemAlloc/cuMemFree or hipMalloc/hipFree
    ASYNC,  ///< Asynchronous allocation using cuMemAllocAsync/cuMemFreeAsync or hipMallocAsync/hipFreeAsync
    POOL_ASYNC  ///< Pool-based async allocation using cuMemAllocFromPoolAsync or hipMallocFromPoolAsync
};

/**
 * @brief Get the current GPU allocation strategy (determined at compile time).
 *
 * @return allocation_strategy The strategy configured via CMake flags
 */
constexpr allocation_strategy get_allocation_strategy() noexcept
{
#if defined(MEMORY_CUDA_ALLOC_SYNC) || defined(MEMORY_HIP_ALLOC_SYNC)
    return allocation_strategy::SYNC;
#elif defined(MEMORY_CUDA_ALLOC_ASYNC) || defined(MEMORY_HIP_ALLOC_ASYNC)
    return allocation_strategy::ASYNC;
#elif defined(MEMORY_CUDA_ALLOC_POOL_ASYNC) || defined(MEMORY_HIP_ALLOC_POOL_ASYNC)
    return allocation_strategy::POOL_ASYNC;
#else
    return allocation_strategy::SYNC;  // Default to synchronous
#endif
}

/**
 * @brief Allocates GPU device memory using the configured allocation strategy.
 *
 * @param nbytes Size of memory block to allocate in bytes
 * @param device_id GPU device ID to allocate memory on
 * @param stream GPU stream for async operations (nullptr = default stream)
 * @param memory_pool Memory pool handle for pool-based allocation (nullptr = default pool)
 * @return Pointer to allocated GPU memory, or nullptr on failure
 *
 * **Thread Safety**: Thread-safe with device-level synchronization
 * **Performance**: O(1) direct GPU API call
 * **Strategy**: Determined at compile time by MEMORY_GPU_ALLOC/MEMORY_HIP_ALLOC flags
 */
MEMORY_API void* allocate(
    std::size_t nbytes, int device_id, void* stream = nullptr, void* memory_pool = nullptr);

/**
 * @brief Deallocates GPU device memory using the configured allocation strategy.
 *
 * @param ptr Pointer to GPU memory to deallocate
 * @param nbytes Size of memory block (for validation and statistics)
 * @param device_id GPU device ID where memory was allocated
 * @param stream GPU stream for async operations (nullptr = default stream)
 *
 * **Thread Safety**: Thread-safe with device-level synchronization
 * **Performance**: O(1) direct GPU API call
 * **Strategy**: Matches allocation strategy used for allocation
 */
MEMORY_API void free(
    void* ptr, std::size_t nbytes = 0, int device_id = 0, void* stream = nullptr) noexcept;

/**
 * @brief Sets the GPU device context for subsequent operations.
 *
 * @param device_id GPU device ID to set as current
 * @return true if device context was set successfully, false otherwise
 */
MEMORY_API bool set_device(int device_id) noexcept;

/**
 * @brief Gets the current GPU device ID.
 *
 * @return Current GPU device ID, or -1 on error
 */
MEMORY_API int get_current_device() noexcept;

}  // namespace memory_allocator
}  // namespace gpu
}  // namespace memory
