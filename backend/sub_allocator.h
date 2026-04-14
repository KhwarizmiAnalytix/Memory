/*
 * Quarisma: High-Performance Computational Library
 * Copyright 2025 Quarisma Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "common/memory_export.h"
#include "common/memory_macros.h"

namespace memory
{

/**
 * @brief Enumeration of memory types supported by allocators.
 *
 * Specifies the physical location and characteristics of allocated memory.
 * This information is crucial for performance optimization, data transfer
 * planning, and ensuring memory access patterns are appropriate for the
 * target hardware.
 *
 * **Performance Implications**:
 * - HOST_PAGEABLE: Standard system memory, may be swapped to disk
 * - HOST_PINNED: Locked in physical RAM, faster for device transfers
 * - DEVICE: On-device memory (GPU/accelerator), highest bandwidth for compute
 *
 * **Thread Safety**: Enum values are immutable and thread-safe
 */
enum class allocator_memory_enum : uint8_t
{
    /**
     * @brief Memory type is unknown or not specified.
     *
     * Default value when memory type cannot be determined or is not relevant.
     * Should be avoided in performance-critical code paths.
     */
    UNKNOWN = 0,

    /**
     * @brief Memory located on a compute device (GPU, TPU, etc.).
     *
     * Provides highest bandwidth for device computations but may have
     * limited capacity and require special access patterns.
     *
     * **Characteristics**:
     * - High bandwidth for device operations
     * - Limited capacity compared to host memory
     * - May require asynchronous transfers to/from host
     */
    DEVICE = 1,

    /**
     * @brief Pageable host memory that can be swapped to disk.
     *
     * Standard system memory allocated through malloc/new. May be paged
     * out to disk by the operating system under memory pressure.
     *
     * **Characteristics**:
     * - Large capacity (limited by virtual memory)
     * - May be swapped to disk, causing access latency
     * - Standard CPU cache hierarchy applies
     */
    HOST_PAGEABLE = 2,

    /**
     * @brief Pinned (page-locked) host memory that cannot be swapped.
     *
     * Memory locked in physical RAM, providing faster and more predictable
     * access times. Often used for DMA transfers to/from devices.
     *
     * **Characteristics**:
     * - Guaranteed to remain in physical RAM
     * - Faster device transfer rates
     * - Limited by physical RAM capacity
     * - Higher system resource usage
     */
    HOST_PINNED = 3,

    /**
     * @brief Unified memory accessible from both CPU and GPU.
     *
     * Memory that can be accessed directly by both CPU and GPU without
     * explicit transfers. Managed by the runtime system.
     *
     * **Characteristics**:
     * - Accessible from both CPU and GPU
     * - Automatic migration between memory spaces
     * - May have performance implications due to migration
     */
    UNIFIED = 4,

    /**
     * @brief Remote memory (network, storage, etc.).
     *
     * Memory located on remote systems or storage devices, accessed
     * over network or other interconnects.
     *
     * **Characteristics**:
     * - High latency access
     * - Large capacity potential
     * - Network bandwidth limitations
     */
    REMOTE = 5
};

/**
 * @brief Low-level memory allocator interface for high-level allocator backends.
 *
 * sub_allocator provides the underlying memory allocation/deallocation services
 * for higher-level allocators like allocator_bfc. It's designed for infrequent
 * large allocations that are then managed by sophisticated pooling and caching
 * strategies in the higher-level allocator.
 *
 * **Design Principle**: Coarse-grained allocation with visitor pattern for monitoring
 * **Performance**: Optimized for large, infrequent allocations (not per-object allocation)
 * **Thread Safety**: Implementation-dependent, but typically thread-safe
 *
 * **Typical Usage Pattern**:
 * 1. Higher-level allocator requests large memory regions from sub_allocator
 * 2. Higher-level allocator subdivides regions for individual allocations
 * 3. sub_allocator is called infrequently compared to user allocation requests
 *
 * **Example**:
 * ```cpp
 * // allocator_bfc might request 64MB from sub_allocator
 * size_t received;
 * void* region = sub_alloc->Alloc(64, 64*1024*1024, &received);
 * // Then serve thousands of small allocations from this region
 * ```
 */
class MEMORY_VISIBILITY sub_allocator
{
public:
    /**
     * @brief Visitor function type for allocation/deallocation monitoring.
     *
     * Called whenever memory is allocated or freed to enable tracking,
     * profiling, and debugging. The index parameter identifies the memory
     * domain (NUMA node for CPU, device ID for GPU).
     *
     * @param ptr Pointer to allocated/freed memory
     * @param index Memory domain identifier (NUMA node, GPU ID, etc.)
     * @param size Size of memory region in bytes
     *
     * **Performance**: Should be fast (O(1)) to avoid allocation overhead
     * **Thread Safety**: Must be thread-safe if used with concurrent allocators
     */
    using Visitor = std::function<void(void*, int index, size_t)>;

    /**
     * @brief Constructs sub_allocator with visitor callbacks for monitoring.
     *
     * @param alloc_visitors Functions called on each allocation
     * @param free_visitors Functions called on each deallocation
     *
     * **Visitor Execution**: Visitors are called in registration order
     * **Exception Safety**: Visitor exceptions may propagate
     * **Performance**: Visitor overhead is per large allocation, not per user allocation
     */
    MEMORY_API sub_allocator(
        const std::vector<Visitor>& alloc_visitors = {},
        const std::vector<Visitor>& free_visitors  = {});

    /**
     * @brief Virtual destructor for proper cleanup of derived classes.
     */
    virtual ~sub_allocator() = default;

    /**
     * @brief Allocates a large memory region with specified alignment.
     *
     * Allocates at least num_bytes of memory aligned to the specified boundary.
     * Returns the actual allocated size which may be larger than requested.
     * The entire allocated region is safe to use.
     *
     * @param alignment Required alignment in bytes (must be power of 2)
     * @param num_bytes Minimum size to allocate
     * @param bytes_received Output parameter for actual allocated size
     * @return Pointer to allocated memory, or nullptr on failure
     *
     * **Requirements**:
     * - alignment must be power of 2
     * - bytes_received must not be nullptr
     * - *bytes_received >= num_bytes on success
     *
     * **Performance**: Typically O(1) but may involve system calls
     * **Thread Safety**: Must be thread-safe
     * **Exception Safety**: Should not throw, return nullptr on failure
     *
     * **Example**:
     * ```cpp
     * size_t received;
     * void* ptr = sub_alloc->Alloc(4096, 1024*1024, &received);
     * if (ptr) {
     *     // Can safely use 'received' bytes starting at 'ptr'
     *     assert(received >= 1024*1024);
     * }
     * ```
     */
    virtual void* Alloc(size_t alignment, size_t num_bytes, size_t* bytes_received) = 0;

    /**
     * @brief Frees a previously allocated memory region.
     *
     * Releases memory back to the system. The pointer and size must exactly
     * match a previous successful Alloc() call.
     *
     * @param ptr Pointer returned by previous Alloc() call
     * @param num_bytes Size returned in bytes_received from Alloc()
     *
     * **Requirements**:
     * - ptr must have been returned by Alloc() on this instance
     * - num_bytes must match bytes_received from the Alloc() call
     * - ptr must not be used after this call
     *
     * **Performance**: Typically O(1) but may involve system calls
     * **Thread Safety**: Must be thread-safe
     * **Exception Safety**: Should not throw
     */
    virtual void Free(void* ptr, size_t num_bytes) = 0;

    /**
     * @brief Indicates whether adjacent allocations can be safely coalesced.
     *
     * Returns true if the BFC allocator can treat adjacent memory regions
     * from this sub_allocator as contiguous for coalescing purposes. This
     * enables more efficient memory management and reduced fragmentation.
     *
     * @return true if coalescing is safe, false otherwise
     *
     * **Safety Consideration**: Only return true if adjacent allocations
     * are guaranteed to be physically contiguous in the address space.
     *
     * **Performance Impact**: Coalescing enables better memory utilization
     * **Thread Safety**: Must return consistent values across threads
     *
     * **Example Use Cases**:
     * - System malloc: typically true (virtual memory is contiguous)
     * - GPU memory: may be false if allocations use different memory pools
     * - NUMA memory: may be false across NUMA boundaries
     */
    virtual bool SupportsCoalescing() const = 0;

    /**
     * @brief Returns the type of memory managed by this sub_allocator.
     *
     * @return Memory type enumeration indicating physical memory characteristics
     *
     * **Thread Safety**: Must be thread-safe and return consistent values
     * **Performance**: O(1) - should return cached value
     * **Default**: UNKNOWN (should be overridden by implementations)
     */
    virtual allocator_memory_enum GetMemoryType() const noexcept
    {
        return allocator_memory_enum::UNKNOWN;
    }

protected:
    /**
     * @brief Notifies allocation visitors of new memory allocation.
     *
     * Must be called by Alloc() implementations immediately after successful
     * allocation. Executes all registered allocation visitors in order.
     *
     * @param ptr Pointer to newly allocated memory
     * @param index Memory domain identifier (NUMA node, device ID, etc.)
     * @param num_bytes Size of allocated region
     *
     * **Call Requirement**: Must be called by every Alloc() implementation
     * **Exception Handling**: Visitor exceptions may propagate
     * **Performance**: O(n) where n is number of visitors
     */
    MEMORY_API void VisitAlloc(void* ptr, int index, size_t num_bytes);

    /**
     * @brief Notifies deallocation visitors of memory release.
     *
     * Must be called by Free() implementations immediately before actual
     * deallocation. Executes all registered deallocation visitors in order.
     *
     * @param ptr Pointer to memory being freed
     * @param index Memory domain identifier (NUMA node, device ID, etc.)
     * @param num_bytes Size of region being freed
     *
     * **Call Requirement**: Must be called by every Free() implementation
     * **Exception Handling**: Visitor exceptions may propagate
     * **Performance**: O(n) where n is number of visitors
     */
    MEMORY_API void VisitFree(void* ptr, int index, size_t num_bytes);

    const std::vector<Visitor> alloc_visitors_;  ///< Allocation monitoring callbacks
    const std::vector<Visitor> free_visitors_;   ///< Deallocation monitoring callbacks
};

}  // namespace memory
