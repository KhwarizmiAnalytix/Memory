/*
 * Quarisma: High-Performance Computational Library
 * Copyright 2025 Quarisma Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>

#include "common/memory_export.h"
#include "common/memory_macros.h"
#include "cpu/allocator.h"
#include "backend/sub_allocator.h"

#if MEMORY_HAS_CUDA
#include <cuda.h>  // For CUDA Driver API
#include <cuda_runtime.h>
#endif

#if MEMORY_HAS_HIP
#include <hip/hip_runtime.h>
#endif

namespace memory
{
namespace gpu
{

/**
 * @brief GPU sub-allocator that interfaces with GPU memory management APIs.
 *
 * Provides a bridge between Memory's allocator interface and GPU memory
 * management functions (CUDA/HIP). Handles device context switching and error management.
 */
class MEMORY_VISIBILITY basic_gpu_allocator : public sub_allocator
{
public:
    /**
     * @brief Constructs GPU sub-allocator for specified device.
     *
     * @param device_id GPU device ID to allocate memory on
     * @param alloc_visitors Functions called on each allocation for monitoring
     * @param free_visitors Functions called on each deallocation for monitoring
     * @param numa_node NUMA node affinity (ignored for GPU memory)
     */
    MEMORY_API basic_gpu_allocator(
        int                         device_id,
        const std::vector<Visitor>& alloc_visitors = {},
        const std::vector<Visitor>& free_visitors  = {},
        int                         numa_node      = NUMANOAFFINITY);

    /**
     * @brief Destructor ensures proper cleanup of device context.
     */
    MEMORY_API ~basic_gpu_allocator() override;

    /**
     * @brief Allocates GPU memory using appropriate GPU API.
     *
     * @param alignment Required alignment (handled by GPU API)
     * @param num_bytes Size to allocate in bytes
     * @param bytes_received Actual bytes allocated (output)
     * @return Pointer to GPU memory or nullptr on failure
     */
    MEMORY_API void* Alloc(size_t alignment, size_t num_bytes, size_t* bytes_received) override;

    /**
     * @brief Frees GPU memory using appropriate GPU API.
     *
     * @param ptr Pointer to GPU memory to free
     * @param num_bytes Size of memory block (for validation)
     */
    MEMORY_API void Free(void* ptr, size_t num_bytes) override;

    /**
     * @brief Indicates this allocator supports coalescing.
     *
     * @return true (GPU memory supports coalescing)
     */
    MEMORY_API bool SupportsCoalescing() const override { return true; }

    /**
     * @brief Returns the memory type managed by this allocator.
     *
     * @return DEVICE memory type for GPU allocations
     */
    MEMORY_API allocator_memory_enum GetMemoryType() const noexcept override
    {
        return allocator_memory_enum::DEVICE;
    }

private:
    int                 device_id_;
    std::atomic<size_t> total_allocated_{0};
    std::atomic<size_t> peak_allocated_{0};
    mutable std::mutex  stats_mutex_;
};

/**
 * @brief GPU allocation method enumeration (determined at compile time).
 *
 * The actual allocation method is controlled by the MEMORY_GPU_ALLOC CMake flag:
 * - SYNC: Uses cuMemAlloc/cuMemFree or hipMalloc/hipFree (synchronous allocation)
 * - ASYNC: Uses cuMemAllocAsync/cuMemFreeAsync or hipMallocAsync/hipFreeAsync (asynchronous allocation)
 * - POOL_ASYNC: Uses cuMemAllocFromPoolAsync or hipMallocFromPoolAsync (pool-based async allocation)
 */
enum class gpu_allocation_method
{
#if defined(MEMORY_CUDA_ALLOC_SYNC) || defined(MEMORY_HIP_ALLOC_SYNC)
    SYNC,  ///< Synchronous allocation using cuMemAlloc/cuMemFree or hipMalloc/hipFree
#elif defined(MEMORY_CUDA_ALLOC_ASYNC) || defined(MEMORY_HIP_ALLOC_ASYNC)
    ASYNC,  ///< Asynchronous allocation using cuMemAllocAsync/cuMemFreeAsync or hipMallocAsync/hipFreeAsync
#elif defined(MEMORY_CUDA_ALLOC_POOL_ASYNC) || defined(MEMORY_HIP_ALLOC_POOL_ASYNC)
    POOL_ASYNC,  ///< Pool-based async allocation using cuMemAllocFromPoolAsync or hipMallocFromPoolAsync
#else
    SYNC  ///< Default to synchronous allocation
#endif
};

/**
 * @brief High-performance GPU memory allocator with direct GPU API integration.
 *
 * allocator_gpu provides a direct wrapper around GPU memory allocation functions
 * (CUDA/HIP), eliminating intermediate backend layers for optimal performance.
 * The allocation method is determined at compile time via the MEMORY_GPU_ALLOC CMake flag.
 *
 * **Key Features**:
 * - Direct GPU API integration (no backend layers)
 * - Cross-platform support (CUDA/HIP)
 * - Compile-time allocation strategy selection
 * - GPU device context management
 * - Comprehensive statistics and monitoring
 * - Thread-safe operation
 * - Support for synchronous and asynchronous allocation
 *
 * **Allocation Methods** (controlled by MEMORY_GPU_ALLOC):
 * - SYNC: cuMemAlloc/cuMemFree or hipMalloc/hipFree - O(1) allocation, best for large allocations
 * - ASYNC: cuMemAllocAsync/cuMemFreeAsync or hipMallocAsync/hipFreeAsync - O(1) async, best for stream-based workloads
 * - POOL_ASYNC: cuMemAllocFromPoolAsync or hipMallocFromPoolAsync - O(1) pooled async, best for frequent small allocations
 *
 * **Performance Characteristics**:
 * - Direct GPU API calls eliminate allocation overhead
 * - Device memory bandwidth: Limited by PCIe and GPU memory bandwidth
 * - Memory fragmentation depends on allocation pattern and method
 *
 * **Thread Safety**: Fully thread-safe with device-level synchronization
 * **Memory Type**: GPU device memory (CUDA global memory or HIP device memory)
 */
class MEMORY_VISIBILITY allocator_gpu : public Allocator
{
public:
    /**
     * @brief Configuration options for allocator_gpu behavior.
     */
    struct Options
    {
        bool enable_statistics = true;  ///< Enable allocation statistics tracking

        // Stream-specific options (for async allocation methods)
        void* gpu_stream = nullptr;  ///< GPU stream for async operations (nullptr = default stream)

        // Pool-specific options (for POOL_ASYNC method)
        void*  memory_pool = nullptr;  ///< GPU memory pool handle (nullptr = default pool)
        size_t pool_threshold =
            0;  ///< Minimum allocation size for pool usage (0 = use pool for all sizes)
    };

    /**
     * @brief Constructs GPU allocator with direct GPU API integration.
     *
     * @param device_id GPU device ID to allocate memory on
     * @param options Configuration options for allocator behavior
     * @param name Human-readable name for debugging and profiling
     *
     * **Device Context**: Automatically manages GPU device context
     * **Allocation Method**: Determined at compile time by MEMORY_GPU_ALLOC flag
     * **Thread Safety**: Constructor is not thread-safe
     * **Exception Safety**: Strong guarantee - no partial construction
     */
    MEMORY_API allocator_gpu(int device_id, const Options& options, std::string name);

    /**
     * @brief Destructor that cleans up device context and resources.
     *
     * **Thread Safety**: Destructor is not thread-safe
     * **Resource Cleanup**: Ensures proper cleanup of device resources
     * **Exception Safety**: noexcept - logs errors but doesn't throw
     */
    MEMORY_API ~allocator_gpu() override;

    /**
     * @brief Returns the human-readable name of this allocator.
     *
     * @return Allocator name for debugging and profiling
     */
    std::string Name() const override { return name_; }

    /**
     * @brief Allocates GPU memory with specified alignment using direct GPU API.
     *
     * @param alignment Required alignment in bytes
     * @param num_bytes Size of memory block to allocate
     * @return Pointer to GPU memory or nullptr on failure
     *
     * **Device Context**: Automatically sets correct GPU device
     * **Direct API**: Calls GPU allocation functions directly (no backend layers)
     * **Thread Safety**: Thread-safe through device-level synchronization
     * **Performance**: O(1) direct GPU API call
     */
    MEMORY_API void* allocate_raw(size_t alignment, size_t num_bytes) override;

    /**
     * @brief Allocates GPU memory with allocation attributes.
     *
     * @param alignment Required alignment in bytes
     * @param num_bytes Size of memory block to allocate
     * @param allocation_attr Attributes controlling allocation behavior
     * @return Pointer to GPU memory or nullptr on failure
     */
    MEMORY_API void* allocate_raw(
        size_t alignment, size_t num_bytes, const allocation_attributes& allocation_attr) override;

    /**
     * @brief Deallocates GPU memory using direct GPU API.
     *
     * @param ptr Pointer to GPU memory to deallocate
     *
     * **Device Context**: Automatically sets correct GPU device
     * **Direct API**: Calls GPU deallocation functions directly (no backend layers)
     * **Thread Safety**: Thread-safe through device-level synchronization
     */
    MEMORY_API void deallocate_raw(void* ptr) override;

    /**
     * @brief Indicates whether this allocator tracks allocation sizes.
     *
     * @return true if backend supports size tracking
     */
    bool tracks_allocation_sizes() const noexcept override;

    /**
     * @brief Returns requested size for a given allocation.
     *
     * @param ptr Pointer to allocated memory
     * @return Originally requested size in bytes
     */
    size_t RequestedSize(const void* ptr) const noexcept override;

    /**
     * @brief Returns actual allocated size for a given allocation.
     *
     * @param ptr Pointer to allocated memory
     * @return Actual allocated size in bytes
     */
    size_t AllocatedSize(const void* ptr) const noexcept override;

    /**
     * @brief Returns unique allocation ID for debugging.
     *
     * @param ptr Pointer to allocated memory
     * @return Unique allocation identifier
     */
    int64_t AllocationId(const void* ptr) const override;

    /**
     * @brief Returns comprehensive allocator statistics.
     *
     * @return Statistics structure with memory usage metrics
     */
    std::optional<allocator_stats> GetStats() const override;

    /**
     * @brief Clears statistics counters.
     *
     * @return true if statistics were successfully cleared
     */
    bool ClearStats() override;

    /**
     * @brief Returns memory type managed by this allocator.
     *
     * @return GPU device memory type
     */
    allocator_memory_enum GetMemoryType() const noexcept override
    {
        return allocator_memory_enum::DEVICE;
    }

    /**
     * @brief Returns GPU device ID for this allocator.
     *
     * @return GPU device ID
     */
    int device_id() const noexcept { return device_id_; }

    /**
     * @brief Returns current allocation method (determined at compile time).
     *
     * @return GPU allocation method
     */
    constexpr gpu_allocation_method allocation_method() const noexcept
    {
#if defined(MEMORY_CUDA_ALLOC_SYNC) || defined(MEMORY_HIP_ALLOC_SYNC)
        return gpu_allocation_method::SYNC;
#elif defined(MEMORY_CUDA_ALLOC_ASYNC) || defined(MEMORY_HIP_ALLOC_ASYNC)
        return gpu_allocation_method::ASYNC;
#elif defined(MEMORY_CUDA_ALLOC_POOL_ASYNC) || defined(MEMORY_HIP_ALLOC_POOL_ASYNC)
        return gpu_allocation_method::POOL_ASYNC;
#else
        return gpu_allocation_method::SYNC;
#endif
    }

private:
    /**
     * @brief Sets GPU device context for operations.
     *
     * @return true if device context was set successfully
     */
    bool set_device_context() const;

    /**
     * @brief Allocates GPU memory using the configured GPU allocation method.
     *
     * @param num_bytes Size of memory block to allocate
     * @param stream GPU stream for async operations (can be nullptr)
     * @return Pointer to GPU memory or nullptr on failure
     */
    void* allocate_gpu_memory(size_t num_bytes, void* stream = nullptr) const;

    /**
     * @brief Deallocates GPU memory using the configured GPU allocation method.
     *
     * @param ptr Pointer to GPU memory to deallocate
     * @param num_bytes Size of memory block (for statistics)
     * @param stream GPU stream for async operations (can be nullptr)
     */
    void deallocate_gpu_memory(void* ptr, size_t num_bytes, void* stream = nullptr) const;

    int         device_id_;
    Options     options_;
    std::string name_;

    // Statistics tracking
    mutable std::atomic<size_t> total_allocated_{0};
    mutable std::atomic<size_t> peak_allocated_{0};
    mutable std::atomic<size_t> allocation_count_{0};
    mutable std::atomic<size_t> deallocation_count_{0};

    mutable std::mutex device_mutex_;
};

/**
 * @brief Factory function to create GPU allocator with default options.
 *
 * @param device_id GPU device ID
 * @param name Allocator name for debugging
 * @return Unique pointer to allocator_gpu instance
 *
 * **Allocation Method**: Determined at compile time by MEMORY_GPU_ALLOC flag
 */
MEMORY_API std::unique_ptr<allocator_gpu> create_gpu_allocator(
    int device_id, const std::string& name = "GPU-Direct");

/**
 * @brief Factory function to create GPU allocator with custom options.
 *
 * @param device_id GPU device ID
 * @param options Custom configuration options
 * @param name Allocator name for debugging
 * @return Unique pointer to allocator_gpu instance
 */
MEMORY_API std::unique_ptr<allocator_gpu> create_gpu_allocator(
    int device_id, const allocator_gpu::Options& options, const std::string& name = "GPU-Direct");

}  // namespace gpu
}  // namespace memory
