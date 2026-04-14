/*
 * Quarisma: High-Performance Computational Library
 *
 * Original work Copyright 2015 The TensorFlow Authors
 * Modified work Copyright 2025 Quarisma Contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * This file contains code modified from TensorFlow (Apache 2.0 licensed)
 * and is part of Quarisma, licensed under a dual-license model:
 *
 *   - Open-source License (GPLv3):
 *       Free for personal, academic, and research use under the terms of
 *       the GNU General Public License v3.0 or later.
 *
 *   - Commercial License:
 *       A commercial license is required for proprietary, closed-source,
 *       or SaaS usage. Contact us to obtain a commercial agreement.
 *
 * MODIFICATIONS FROM ORIGINAL:
 * - Adapted for Quarisma quantitative computing requirements
 * - Added high-performance memory allocation optimizations
 * - Integrated NUMA-aware allocation strategies
 *
 * Contact: licensing@quarisma.co.uk
 * Website: https://www.quarisma.co.uk
 */

#pragma once

#include <stdlib.h>  // for size_t

#include <cassert>     // for assert
#include <cstdint>     // for uint32_t, uint64_t, int64_t, int32_t, uint8_t
#include <functional>  // for function
#include <optional>    // for optional, nullopt
#include <string>      // for string
#include <vector>      // for vector

#include "common/memory_export.h"  // for MEMORY_API
#include "common/memory_macros.h"                // for MEMORY_UNUSED
#include "backend/sub_allocator.h"         // for sub_allocator, allocator_memory_enum
#include "profiler/unified_memory_stats.h"  // for allocator_stats
#include "util/exception.h"               // for check_msg_impl, LOGGING_CHECK

namespace memory
{
constexpr int NUMANOAFFINITY = -1;
/**
 * @brief Attributes for a single allocation call specifying allocation behavior and constraints.
 *
 * Different calls to the same allocator can have different allocation attributes to control
 * retry behavior, logging, and memory lifecycle management. This structure provides fine-grained
 * control over allocation policies.
 *
 * @note This class is move-only to prevent accidental copying of function pointers.
 *
 * **Thread Safety**: Individual instances are not thread-safe, but multiple instances
 * can be used concurrently across different threads.
 *
 * **Performance**: O(1) construction and access to all members.
 */
struct allocation_attributes
{
    /**
     * @brief Default constructor with standard allocation behavior.
     *
     * Creates attributes with retry enabled, logging disabled, and no timing constraints.
     */
    allocation_attributes() = default;

    /**
     * @brief Constructs allocation attributes with specific behavior settings.
     *
     * @param allocation_will_be_logged Whether this allocation should be logged for tracking
     * @param freed_by_func Optional function providing timing constraints for memory reuse
     *
     * **Example Usage**:
     * ```cpp
     * // Critical allocation that should retry on failure
     * allocation_attributes critical_attrs(true, true, nullptr);
     *
     * // Optional scratch space that shouldn't retry
     * allocation_attributes scratch_attrs(false, false, nullptr);
     * ```
     */
    allocation_attributes(
        bool                       retry_on_failure,
        bool                       allocation_will_be_logged,
        std::function<uint64_t()>* freed_by_func) noexcept
        : allocation_will_be_logged(allocation_will_be_logged),
          retry_on_failure(retry_on_failure),
          freed_by_func(freed_by_func)
    {
    }

    /**
     * @brief Move constructor for efficient transfer of attributes.
     */
    allocation_attributes(allocation_attributes&&) noexcept = default;

    /**
     * @brief Move assignment operator for efficient transfer of attributes.
     */
    allocation_attributes& operator=(allocation_attributes&&) noexcept = default;

    /**
     * @brief Controls whether allocation is tracked in execution logs.
     *
     * When true, this allocation will be properly attributed to the executing operation
     * for debugging and profiling purposes. Should be set to true for all tensor
     * allocations during normal execution.
     *
     * **Default**: false (logging disabled)
     * **Performance Impact**: Minimal overhead for tracking metadata
     */
    bool allocation_will_be_logged = false;

    /**
     * @brief Controls whether allocation should retry on failure.
     *
     * When true, the allocator will attempt to retry failed allocations,
     * potentially waiting for memory to become available. When false,
     * the allocator will fail immediately on memory exhaustion.
     *
     * **Default**: true (retry enabled)
     * **Performance Impact**: Retries add latency but improve success rate
     */
    bool retry_on_failure = true;

    /**
     * @brief Optional timing constraint function for memory reuse policies.
     *
     * **EXPERIMENTAL FEATURE**: When provided, this function returns a timing count
     * that constrains which freed memory chunks can be reused. Only chunks with
     * freed_at_count <= returned value are eligible for reuse.
     *
     * **Ownership**: Pointer is not owned by this structure - caller must ensure lifetime
     * **Thread Safety**: Function must be thread-safe if used across multiple threads
     * **Performance**: Called during allocation - should be fast (O(1) preferred)
     *
     * @warning This is an experimental feature and may change in future versions
     */
    std::function<uint64_t()>* freed_by_func = nullptr;  // Not owned.

    // Prevent accidental copying which could lead to dangling function pointers
    allocation_attributes(const allocation_attributes&)            = delete;
    allocation_attributes& operator=(const allocation_attributes&) = delete;
};

/**
 * @brief Abstract interface for high-performance memory allocation and deallocation.
 *
 * The Allocator class provides a unified interface for memory management across different
 * memory types (host, device, pinned) and allocation strategies (pooled, tracking, etc.).
 * All allocators guarantee proper alignment and provide optional statistics collection.
 *
 * **Design Principles**:
 * - Zero-overhead abstraction when statistics are disabled
 * - Consistent alignment guarantees across all implementations
 * - Support for both simple and attribute-based allocation
 * - Thread-safe implementations (implementation-dependent)
 *
 * **Performance Characteristics**:
 * - allocate_raw(): O(1) to O(log n) depending on implementation
 * - deallocate_raw(): O(1) to O(log n) depending on implementation
 * - Statistics collection adds minimal overhead when enabled
 *
 * **Memory Alignment**: All allocations are aligned to Allocator_Alignment (64 bytes)
 * for optimal cache performance and SIMD instruction compatibility.
 *
 * @note Implementations must be thread-safe unless explicitly documented otherwise.
 */
class MEMORY_VISIBILITY Allocator
{
public:
    /**
     * @brief Default memory alignment boundary for all allocations.
     *
     * 64-byte alignment ensures optimal performance for:
     * - CPU cache line alignment (typically 64 bytes)
     * - SIMD instructions (AVX-512 requires 64-byte alignment)
     * - GPU memory coalescing requirements
     *
     * **Compile-time constant**: Can be used in constexpr contexts
     */
    static constexpr size_t Allocator_Alignment = 64;

    /**
     * @brief Virtual destructor ensuring proper cleanup of derived classes.
     *
     * **Thread Safety**: Destructor calls must be externally synchronized
     * **Exception Safety**: noexcept - implementations should not throw
     */
    virtual ~Allocator() = default;

    /**
     * @brief Returns a human-readable identifier for this allocator instance.
     *
     * Used for debugging, logging, and performance profiling. Should be
     * descriptive enough to distinguish between different allocator types
     * and configurations.
     *
     * @return String identifier (e.g., "allocator_bfc", "cpu_allocator", "gpu_allocator")
     *
     * **Performance**: O(1) - should return a cached string
     * **Thread Safety**: Must be thread-safe
     * **Exception Safety**: Should not throw
     *
     * **Example**:
     * ```cpp
     * auto allocator = std::make_unique<allocator_bfc>(...);
     * std::cout << "Using: " << allocator->Name() << std::endl;
     * // Output: "Using: allocator_bfc"
     * ```
     */
    virtual std::string Name() const = 0;

    /**
     * @brief Allocates an uninitialized memory block with specified alignment.
     *
     * Returns a pointer to a memory block of at least num_bytes size, aligned
     * to the specified boundary. The memory content is uninitialized.
     *
     * @param alignment Required alignment in bytes (must be power of 2)
     * @param num_bytes Size of memory block to allocate
     * @return Pointer to allocated memory, or nullptr on failure
     *
     * **Requirements**:
     * - alignment must be a power of 2
     * - alignment must be >= sizeof(void*)
     * - num_bytes can be 0 (implementation-defined behavior)
     *
     * **Performance**: Implementation-dependent, typically O(1) to O(log n)
     * **Thread Safety**: Must be thread-safe
     * **Exception Safety**: Should not throw, return nullptr on failure
     *
     * **Example**:
     * ```cpp
     * void* ptr = allocator->allocate_raw(32, 1024);  // 1KB aligned to 32 bytes
     * if (ptr) {
     *     // Use memory...
     *     allocator->deallocate_raw(ptr);
     * }
     * ```
     */
    virtual void* allocate_raw(size_t alignment, size_t num_bytes) = 0;

    /**
     * @brief Allocates memory with specific allocation attributes and policies.
     *
     * Extended allocation interface supporting retry policies, logging,
     * and timing constraints. Provides fine-grained control over allocation
     * behavior for performance-critical applications.
     *
     * @param alignment Required alignment in bytes (must be power of 2)
     * @param num_bytes Size of memory block to allocate
     * @param allocation_attr Attributes controlling allocation behavior
     * @return Pointer to allocated memory, or nullptr on failure
     *
     * **Default Implementation**: Delegates to simple allocate_raw() method
     * **Override Recommendation**: Implement for allocators supporting advanced features
     *
     * **Performance**: May be slower than simple allocate_raw() due to attribute processing
     * **Thread Safety**: Must be thread-safe
     *
     * **Example**:
     * ```cpp
     * allocation_attributes attrs(true, true, nullptr);  // retry=true, log=true
     * void* ptr = allocator->allocate_raw(64, 2048, attrs);
     * ```
     */
    virtual void* allocate_raw(
        size_t                                       alignment,
        size_t                                       num_bytes,
        MEMORY_UNUSED const allocation_attributes& allocation_attr)
    {
        // Default implementation ignores attributes and delegates to simple version
        return allocate_raw(alignment, num_bytes);
    }

    /**
     * @brief Deallocates a previously allocated memory block.
     *
     * Releases memory pointed to by ptr back to the allocator. The pointer
     * must have been returned by a previous call to allocate_raw() on the
     * same allocator instance.
     *
     * @param ptr Pointer to memory block to deallocate (must not be nullptr)
     *
     * **Requirements**:
     * - ptr must have been returned by allocate_raw() on this allocator
     * - ptr must not be nullptr (undefined behavior)
     * - ptr must not be used after this call (undefined behavior)
     *
     * **Performance**: Implementation-dependent, typically O(1) to O(log n)
     * **Thread Safety**: Must be thread-safe
     * **Exception Safety**: Should not throw
     *
     * **Example**:
     * ```cpp
     * void* ptr = allocator->allocate_raw(64, 1024);
     * // ... use memory ...
     * allocator->deallocate_raw(ptr);  // ptr is now invalid
     * ```
     */
    virtual void deallocate_raw(void* ptr) = 0;

    /**
     * @brief Extended deallocation with size and alignment hints.
     *
     * Some allocators can optimize deallocation when provided with the original
     * allocation size and alignment. This information can enable faster
     * deallocation and better memory pool management.
     *
     * @param ptr Pointer to memory block to deallocate
     * @param alignment Original alignment used for allocation
     * @param num_bytes Original size requested for allocation
     *
     * **Default Implementation**: Ignores hints and delegates to simple deallocate_raw()
     * **Override Recommendation**: Implement for allocators that can benefit from size hints
     *
     * **Performance**: May be faster than simple deallocate_raw() for some allocators
     * **Thread Safety**: Must be thread-safe
     */
    virtual void deallocate_raw(void* ptr, size_t alignment, size_t num_bytes)
    {
        // Default implementation ignores size hints
        static_cast<void>(alignment);
        static_cast<void>(num_bytes);
        deallocate_raw(ptr);
    }

    /**
     * @brief Indicates whether this allocator tracks allocation sizes and metadata.
     *
     * When true, the allocator maintains detailed information about each allocation
     * including requested size, actual allocated size, and unique identifiers.
     * This enables advanced debugging and profiling capabilities.
     *
     * @return true if size tracking is enabled, false otherwise
     *
     * **Implementation Note**: If overridden to return true, must also override:
     * - RequestedSize()
     * - AllocatedSize()
     * - AllocationId()
     *
     * **Performance Impact**: Size tracking adds memory overhead and slight CPU cost
     * **Thread Safety**: Must be thread-safe and return consistent values
     * **Default**: false (no tracking for optimal performance)
     *
     * **Example**:
     * ```cpp
     * if (allocator->tracks_allocation_sizes()) {
     *     size_t actual_size = allocator->AllocatedSize(ptr);
     *     size_t requested_size = allocator->RequestedSize(ptr);
     *     std::cout << "Overhead: " << (actual_size - requested_size) << " bytes\n";
     * }
     * ```
     */
    virtual bool tracks_allocation_sizes() const noexcept { return false; }

    /**
     * @brief Indicates whether allocator returns opaque handles instead of memory pointers.
     *
     * Special-purpose allocators may return opaque handles for tracking tensor usage
     * rather than actual memory pointers. This is used by advanced memory management
     * systems that need to intercept all memory access.
     *
     * @return true if allocator returns opaque handles, false for normal memory pointers
     *
     * **Important**: When true:
     * - allocate_raw() should be called even for num_bytes=0
     * - Returned pointers are handles, not memory addresses
     * - No constructors/destructors should be run on "allocated" memory
     * - Caller must track handle vs. pointer semantics
     *
     * **Performance**: Handle-based allocators may have different performance characteristics
     * **Thread Safety**: Must be thread-safe and return consistent values
     * **Default**: false (returns actual memory pointers)
     *
     * **Use Cases**:
     * - Memory usage tracking and profiling
     * - Lazy allocation strategies
     * - Memory access pattern analysis
     */
    virtual bool AllocatesOpaqueHandle() const noexcept { return false; }

    /**
     * @brief Returns the user-requested size for a previously allocated pointer.
     *
     * Retrieves the original size requested by the user when the memory was allocated.
     * The actual allocated size may be larger due to alignment or allocator overhead.
     *
     * @param ptr Pointer to previously allocated memory (must not be nullptr)
     * @return Original requested size in bytes
     *
     * **Requirements**:
     * - tracks_allocation_sizes() must return true
     * - ptr must not be nullptr
     * - ptr must have been allocated by this allocator instance
     *
     * **Performance**: O(1) to O(log n) depending on tracking implementation
     * **Thread Safety**: Must be thread-safe
     * **Exception Safety**: Should not throw, logs error and returns 0 on failure
     *
     * **Example**:
     * ```cpp
     * void* ptr = allocator->allocate_raw(64, 1000);
     * if (allocator->tracks_allocation_sizes()) {
     *     size_t requested = allocator->RequestedSize(ptr);  // Returns 1000
     *     size_t actual = allocator->AllocatedSize(ptr);     // May return 1024
     * }
     * ```
     */
    virtual size_t RequestedSize(MEMORY_UNUSED const void* ptr) const
    {
        //LOGGING_LOG_ERROR("Allocator '" << Name() << "' doesn't track allocation sizes");
        return size_t{0};
    }

    /**
     * @brief Returns the actual allocated size for a previously allocated pointer.
     *
     * Retrieves the actual size of the memory block allocated, which may be larger
     * than the requested size due to alignment requirements or allocator overhead.
     * Always >= RequestedSize(ptr).
     *
     * @param ptr Pointer to previously allocated memory (must not be nullptr)
     * @return Actual allocated size in bytes
     *
     * **Requirements**:
     * - tracks_allocation_sizes() must return true
     * - ptr must not be nullptr
     * - ptr must have been allocated by this allocator instance
     *
     * **Guarantee**: AllocatedSize(ptr) >= RequestedSize(ptr)
     * **Performance**: O(1) to O(log n) depending on tracking implementation
     * **Thread Safety**: Must be thread-safe
     * **Default Implementation**: Returns RequestedSize(ptr)
     *
     * **Example**:
     * ```cpp
     * void* ptr = allocator->allocate_raw(32, 100);  // Request 100 bytes, 32-byte aligned
     * size_t actual = allocator->AllocatedSize(ptr);  // May return 128 due to alignment
     * ```
     */
    virtual size_t AllocatedSize(const void* ptr) const { return RequestedSize(ptr); }

    /**
     * @brief Returns a unique identifier for the allocation, if available.
     *
     * Provides a unique ID assigned to each allocation for tracking and debugging
     * purposes. IDs are unique within this allocator instance and non-zero for
     * valid allocations.
     *
     * @param ptr Pointer to previously allocated memory (must not be nullptr)
     * @return Unique allocation ID (>0), or 0 if not available
     *
     * **Requirements**:
     * - tracks_allocation_sizes() must return true
     * - ptr must not be nullptr
     * - ptr must have been allocated by this allocator instance
     *
     * **Uniqueness**: Each allocation gets a different non-zero ID
     * **Performance**: O(1) to O(log n) depending on tracking implementation
     * **Thread Safety**: Must be thread-safe
     * **Default**: Returns 0 (no ID tracking)
     *
     * **Use Cases**:
     * - Memory leak detection
     * - Allocation lifetime tracking
     * - Performance profiling correlation
     *
     * **Example**:
     * ```cpp
     * void* ptr1 = allocator->allocate_raw(64, 1024);
     * void* ptr2 = allocator->allocate_raw(64, 1024);
     * int64_t id1 = allocator->AllocationId(ptr1);  // e.g., returns 1
     * int64_t id2 = allocator->AllocationId(ptr2);  // e.g., returns 2
     * assert(id1 != id2);  // IDs are unique
     * ```
     */
    virtual int64_t AllocationId(MEMORY_UNUSED const void* ptr) const { return 0; }

    /**
     * @brief Returns allocated size with potentially slow computation.
     *
     * Attempts to determine the allocated size even when tracks_allocation_sizes()
     * returns false. May use expensive system calls or memory introspection.
     * Should only be used for debugging or when performance is not critical.
     *
     * @param ptr Pointer to previously allocated memory (must not be nullptr)
     * @return Allocated size in bytes, or 0 if cannot be determined
     *
     * **Requirements**:
     * - ptr must not be nullptr
     * - ptr must have been allocated by this allocator instance
     *
     * **Performance**: Can be extremely slow (O(n) or system call overhead)
     * **Thread Safety**: Must be thread-safe
     * **Use Cases**: Debugging, memory analysis tools, diagnostics
     *
     * **Implementation Strategy**:
     * - If tracks_allocation_sizes() is true: delegate to fast AllocatedSize()
     * - Otherwise: attempt system-specific size queries (malloc_usable_size, etc.)
     * - Return 0 if size cannot be determined
     *
     * **Example**:
     * ```cpp
     * void* ptr = allocator->allocate_raw(64, 1024);
     * // This may be slow but works even without size tracking
     * size_t size = allocator->AllocatedSizeSlow(ptr);
     * ```
     */
    virtual size_t AllocatedSizeSlow(const void* ptr) const
    {
        if constexpr (true)
        {  // Use C++17 if constexpr for potential optimization
            if (tracks_allocation_sizes())
            {
                return AllocatedSize(ptr);
            }
        }
        return 0;
    }

    /**
     * @brief Retrieves comprehensive allocator statistics.
     *
     * Returns detailed runtime statistics about allocator performance and memory usage.
     * Statistics may include allocation counts, memory usage, fragmentation metrics,
     * and performance counters.
     *
     * @return Optional statistics structure, std::nullopt if not supported
     *
     * **Performance**: O(1) for cached stats, O(n) if computed on-demand
     * **Thread Safety**: Must be thread-safe
     * **Consistency**: Statistics should represent a consistent snapshot
     *
     * **Implementation Notes**:
     * - Return std::nullopt for allocators without statistics support
     * - Consider caching expensive computations
     * - Ensure thread-safe access to internal counters
     *
     * **Example**:
     * ```cpp
     * if (auto stats = allocator->GetStats()) {
     *     std::cout << "Memory in use: " << stats->bytes_in_use << " bytes\n";
     *     std::cout << "Peak usage: " << stats->peak_bytes_in_use << " bytes\n";
     *     std::cout << "Allocations: " << stats->num_allocs << "\n";
     * }
     * ```
     */
    virtual std::optional<allocator_stats> GetStats() const { return std::nullopt; }

    /**
     * @brief Resets allocator statistics to initial state.
     *
     * Clears accumulated statistics while preserving current usage information.
     * Peak values are reset to current values, counters are reset to zero.
     *
     * @return true if statistics were cleared, false if not supported
     *
     * **Requirements**: GetStats() must be overridden to return valid statistics
     * **Behavior**:
     * - Preserves: bytes_in_use, allocated memory state
     * - Resets: peak_bytes_in_use = bytes_in_use
     * - Clears: num_allocs, allocation counters
     *
     * **Performance**: O(1) - simple counter reset
     * **Thread Safety**: Must be thread-safe
     * **Use Cases**: Periodic monitoring, benchmark resets, memory profiling
     *
     * **Example**:
     * ```cpp
     * // Reset stats before benchmark
     * if (allocator->ClearStats()) {
     *     run_benchmark();
     *     auto stats = allocator->GetStats();
     *     // stats now reflect only benchmark allocations
     * }
     * ```
     */
    virtual bool ClearStats() { return false; }

    /**
     * @brief Sets the safe frontier for timestamped memory management.
     *
     * **EXPERIMENTAL FEATURE**: Establishes a timestamp boundary for safe memory reuse.
     * Memory freed before this timestamp can be safely reused without synchronization
     * concerns. Used by advanced allocators with temporal memory safety guarantees.
     *
     * @param count Timestamp representing the safe frontier
     *
     * **Thread Safety**: Must be thread-safe
     * **Performance**: O(1) - simple atomic update
     * **Default**: No-op for allocators without timestamp support
     *
     * **Use Cases**:
     * - GPU memory management with stream synchronization
     * - Asynchronous computation memory safety
     * - Temporal memory reuse optimization
     *
     * @warning This is an experimental feature subject to change
     */
    virtual void SetSafeFrontier(MEMORY_UNUSED uint64_t count) noexcept {}

    /**
     * @brief Returns the type of memory managed by this allocator.
     *
     * Indicates the physical location and characteristics of memory returned
     * by this allocator. Critical for performance optimization and ensuring
     * appropriate usage patterns.
     *
     * @return Memory type enumeration value
     *
     * **Performance**: O(1) - should return cached value
     * **Thread Safety**: Must be thread-safe and return consistent values
     * **Default**: UNKNOWN (should be overridden by implementations)
     *
     * **Usage Guidelines**:
     * - HOST_PAGEABLE: Standard CPU memory, may be swapped
     * - HOST_PINNED: Locked CPU memory, optimal for device transfers
     * - DEVICE: GPU/accelerator memory, highest compute bandwidth
     * - UNKNOWN: Avoid in performance-critical code
     *
     * **Example**:
     * ```cpp
     * switch (allocator->GetMemoryType()) {
     *     case allocator_memory_enum::DEVICE:
     *         // Optimize for device computation
     *         break;
     *     case allocator_memory_enum::HOST_PINNED:
     *         // Optimize for host-device transfers
     *         break;
     *     default:
     *         // Generic handling
     *         break;
     * }
     * ```
     */
    virtual allocator_memory_enum GetMemoryType() const noexcept
    {
        return allocator_memory_enum::UNKNOWN;
    }
};

/**
 * @brief Specifies memory allocation requirements and device compatibility constraints.
 *
 * allocator_attributes provides a flexible way to specify memory allocation requirements
 * when operations need access to different types of memory (host, device, pinned, etc.).
 * This enables fine-grained control over memory placement for optimal performance.
 *
 * **Design**: Bit-packed structure for efficient storage and comparison
 * **Performance**: O(1) operations for all attribute queries and modifications
 * **Thread Safety**: Individual instances are not thread-safe, but immutable after construction
 *
 * **Use Cases**:
 * - Cross-device memory allocation (GPU ops needing CPU memory)
 * - DMA-optimized memory regions
 * - Network-compatible memory for distributed computing
 * - Specialized allocator selection
 *
 * **Example Usage**:
 * ```cpp
 * // Request CPU memory for a GPU operation
 * allocator_attributes cpu_attrs;
 * cpu_attrs.set_on_host(true);
 * Allocator* cpu_alloc = device->GetAllocator(cpu_attrs);
 *
 * // Request GPU-compatible memory for transfers
 * allocator_attributes gpu_attrs;
 * gpu_attrs.set_gpu_compatible(true);
 * Allocator* gpu_alloc = device->GetAllocator(gpu_attrs);
 * ```
 */
struct MEMORY_VISIBILITY allocator_attributes
{
    /**
     * @brief Sets whether memory should be allocated on host (CPU).
     *
     * @param v true to request host memory, false for device memory
     *
     * **Performance Impact**: Host memory typically has larger capacity but lower bandwidth
     * **Use Cases**: Large buffers, intermediate results, CPU-only operations
     */
    void set_on_host(bool v) noexcept { value = (value & ~0x1u) | static_cast<uint32_t>(v); }

    /**
     * @brief Checks if host memory allocation is requested.
     * @return true if host memory is requested
     */
    bool on_host() const noexcept { return (value & 0x1u) != 0; }

    /**
     * @brief Sets whether memory should be compatible with network operations.
     *
     * @param v true to request network-compatible memory
     *
     * **Performance Impact**: May require special alignment or memory regions
     * **Use Cases**: Distributed computing, RDMA operations, network transfers
     */
    void set_nic_compatible(bool v) noexcept
    {
        value = (value & ~0x2u) | (static_cast<uint32_t>(v) << 1);
    }

    /**
     * @brief Checks if network-compatible memory is requested.
     * @return true if network-compatible memory is requested
     */
    bool nic_compatible() const noexcept { return (value & 0x2u) != 0; }

    /**
     * @brief Sets whether memory should be compatible with GPU operations.
     *
     * @param v true to request GPU-compatible memory
     *
     * **Performance Impact**: Optimized for GPU access patterns and transfers
     * **Use Cases**: GPU computations, device-to-device transfers, unified memory
     */
    void set_gpu_compatible(bool v) noexcept
    {
        value = (value & ~0x4u) | (static_cast<uint32_t>(v) << 2);
    }

    /**
     * @brief Checks if GPU-compatible memory is requested.
     * @return true if GPU-compatible memory is requested
     */
    bool gpu_compatible() const noexcept { return (value & 0x4u) != 0; }

    /**
     * @brief Merges attributes from another instance using bitwise OR.
     *
     * Combines attribute flags and handles scope_id conflicts. If both instances
     * have non-zero scope_ids, they must match or one must be zero.
     *
     * @param other Attributes to merge with this instance
     *
     * **Exception Safety**: May throw if scope_ids conflict
     * **Performance**: O(1) bitwise operations
     *
     * **Example**:
     * ```cpp
     * allocator_attributes base;
     * base.set_on_host(true);
     *
     * allocator_attributes extra;
     * extra.set_gpu_compatible(true);
     *
     * base.Merge(extra);  // Now has both on_host and gpu_compatible
     * ```
     */
    void Merge(const allocator_attributes& other)
    {
        value |= other.value;
        if (scope_id != other.scope_id)
        {
            LOGGING_CHECK(
                scope_id == 0 || other.scope_id == 0,
                "Cannot merge allocator_attributes with conflicting scope_ids: ",
                scope_id,
                " vs ",
                other.scope_id,
                ". At least one must be zero.");

            scope_id = (scope_id == 0) ? other.scope_id : scope_id;
        }
    }

    /**
     * @brief Checks if this instance's requirements are subset of another's.
     *
     * Returns true if all attributes set in this instance are also set in other.
     * Useful for determining allocator compatibility and requirement satisfaction.
     *
     * @param other Attributes to compare against
     * @return true if this is subset of or equal to other
     *
     * **Performance**: O(1) bitwise comparison
     * **Use Cases**: Allocator selection, compatibility checking
     *
     * **Example**:
     * ```cpp
     * allocator_attributes required;
     * required.set_on_host(true);
     *
     * allocator_attributes available;
     * available.set_on_host(true);
     * available.set_gpu_compatible(true);
     *
     * assert(required.IsEqualOrLessRestrictiveThan(available));  // true
     * ```
     */
    bool IsEqualOrLessRestrictiveThan(const allocator_attributes& other) const noexcept
    {
        return (value | other.value) == other.value;
    }

    /**
     * @brief Bit-packed attribute flags.
     *
     * **Layout**:
     * - Bits 0-3: Standard attributes (host, nic, gpu, pjrt)
     * - Bits 4-23: Reserved for future use
     * - Bits 24-31: device_option-specific attributes
     *
     * **device_option-Specific Usage**: Upper 8 bits (24-31) are reserved for
     * device-specific interpretations. device_option implementations can use these
     * bits for custom allocation policies.
     */
    uint32_t value{0};

    /**
     * @brief Scope identifier for specialized allocators.
     *
     * **EXPERIMENTAL**: When non-zero, delegates allocation to a named
     * special-purpose allocator on the same device. Enables fine-grained
     * control over memory allocation policies.
     *
     * **Values**:
     * - 0: Use default allocator selection
     * - >0: Use specialized allocator with this ID
     *
     * **Thread Safety**: Should be set once during initialization
     * **Use Cases**: Memory pools, specialized allocation strategies
     */
    int32_t scope_id{0};

    /**
     * @brief Generates human-readable string representation of attributes.
     *
     * @return Formatted string showing all set attributes
     *
     * **Performance**: O(1) - simple string formatting
     * **Thread Safety**: Safe to call concurrently on different instances
     *
     * **Example Output**:
     * ```
     * "allocator_attributes: on_host=true, gpu_compatible=true, scope_id=0"
     * ```
     */
    std::string debug_string() const;
};

/**
 * @brief Returns the base CPU allocator singleton instance.
 *
 * Provides access to a simple, process-wide CPU allocator implementation.
 * This is a fallback allocator intended for infrastructure use only.
 *
 * @return Pointer to singleton CPU allocator (never nullptr)
 *
 * **Thread Safety**: Thread-safe singleton initialization
 * **Performance**: O(1) after initialization
 * **Lifetime**: Lives for entire process duration
 *
 * @warning Intended for restricted infrastructure use only.
 *          Prefer process_state::GetCPUAllocator() when available.
 */
Allocator* allocator_cpu_base();

/**
 * @brief Returns a NUMA-aware CPU allocator for the specified node.
 *
 * Attempts to use process_state::GetCPUAllocator() for NUMA-optimized allocation.
 * Falls back to allocator_cpu_base() if process_state is not available.
 *
 * @param numa_node NUMA node ID, or NUMANOAFFINITY for no preference
 * @return Pointer to CPU allocator optimized for the specified NUMA node
 *
 * **NUMA Optimization**: When available, allocates memory local to specified node
 * **Fallback**: Uses base CPU allocator if NUMA support unavailable
 * **Thread Safety**: Thread-safe
 * **Performance**: O(1) - cached allocator instances
 *
 * **Example**:
 * ```cpp
 * // Get allocator for current NUMA node
 * Allocator* local_alloc = cpu_allocator();
 *
 * // Get allocator for specific NUMA node
 * Allocator* node_alloc = cpu_allocator(1);
 * ```
 */
MEMORY_API Allocator* cpu_allocator(int numa_node = NUMANOAFFINITY);

/**
 * @brief Enables statistics collection in the default CPU allocator.
 *
 * Activates comprehensive statistics tracking including allocation counts,
 * memory usage, and performance metrics. Statistics are disabled by default
 * for optimal performance.
 *
 * **Performance Impact**: Minimal overhead for counter updates
 * **Thread Safety**: Safe to call from any thread
 * **Global Effect**: Affects all CPU allocator instances
 *
 * **Example**:
 * ```cpp
 * EnableCPUAllocatorStats();
 * // Now CPU allocators will collect statistics
 * auto stats = cpu_allocator()->GetStats();
 * ```
 */
MEMORY_API void EnableCPUAllocatorStats() noexcept;

/**
 * @brief Disables statistics collection in the default CPU allocator.
 *
 * Deactivates statistics tracking to minimize performance overhead.
 * This is the default state for optimal performance.
 *
 * **Performance Benefit**: Eliminates statistics collection overhead
 * **Thread Safety**: Safe to call from any thread
 * **Global Effect**: Affects all CPU allocator instances
 */
MEMORY_API void DisableCPUAllocatorStats() noexcept;

/**
 * @brief Checks if CPU allocator statistics collection is enabled.
 *
 * @return true if statistics are being collected, false otherwise
 *
 * **Thread Safety**: Safe to call from any thread
 * **Performance**: O(1) - simple flag check
 */
MEMORY_API bool CPUAllocatorStatsEnabled() noexcept;

/**
 * @brief Enables comprehensive statistics collection in CPU allocators.
 *
 * Activates detailed statistics including fragmentation metrics, allocation
 * patterns, and performance counters. More comprehensive than basic stats
 * but with higher overhead.
 *
 * **Performance Impact**: Higher overhead than basic statistics
 * **Thread Safety**: Safe to call from any thread
 * **Use Cases**: Detailed performance analysis, memory debugging
 */
void EnableCPUAllocatorFullStats();

/**
 * @brief Checks if full CPU allocator statistics collection is enabled.
 *
 * @return true if full statistics are being collected, false otherwise
 *
 * **Thread Safety**: Safe to call from any thread
 * **Performance**: O(1) - simple flag check
 */
bool CPUAllocatorFullStatsEnabled();

}  // namespace memory
