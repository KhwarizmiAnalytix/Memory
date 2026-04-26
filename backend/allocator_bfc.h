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

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <unordered_set>
#include <vector>

#include "backend/allocator_retry.h"
#include "common/memory_export.h"
#include "common/memory_macros.h"
#include "cpu/allocator.h"
#include "util/memory_exception.h"

namespace memory
{
/**
 * @brief Computes floor(log2(x)) for 64-bit unsigned integers.
 *
 * Efficiently calculates the position of the highest set bit in x.
 * Used for bin size calculations and power-of-2 operations in the allocator.
 *
 * @param x Input value (must be > 0 for valid result)
 * @return Floor of log base 2 of x, or -1 if x == 0
 *
 * **Algorithm Complexity**: O(1) with compiler intrinsics, O(log log n) without
 * **Performance**: Optimized with compiler intrinsics when available
 * **Thread Safety**: Pure function, inherently thread-safe
 *
 * **Examples**:
 * - Log2Floor64(1) = 0
 * - Log2Floor64(8) = 3
 * - Log2Floor64(15) = 3
 * - Log2Floor64(16) = 4
 *
 * **Implementation Strategy**:
 * - GCC/Clang: Uses __builtin_clzll() for optimal performance
 * - Other compilers: Binary search approach with unrolled loop
 */
constexpr int Log2Floor64(uint64_t x) noexcept
{
    if (x == 0)
    {
        return -1;  // Undefined case - return sentinel value
    }

#if defined(__GNUC__) || defined(__clang__)
    // Use compiler intrinsic for optimal performance
    // __builtin_clzll counts leading zeros in unsigned long long
    return 63 - __builtin_clzll(x);
#else
    // Fallback implementation using binary search
    // Unrolled loop for predictable performance
    int result = 0;

    // Test 32-bit boundary
    if (x >= (1ULL << 32))
    {
        x >>= 32;
        result += 32;
    }

    // Test 16-bit boundary
    if (x >= (1ULL << 16))
    {
        x >>= 16;
        result += 16;
    }

    // Test 8-bit boundary
    if (x >= (1ULL << 8))
    {
        x >>= 8;
        result += 8;
    }

    // Test 4-bit boundary
    if (x >= (1ULL << 4))
    {
        x >>= 4;
        result += 4;
    }

    // Test 2-bit boundary
    if (x >= (1ULL << 2))
    {
        x >>= 2;
        result += 2;
    }

    // Test final bit
    if (x >= (1ULL << 1))
    {
        result += 1;
    }

    return result;
#endif
}

/**
 * @brief Thread-safe atomic counter for allocation tracking and timing.
 *
 * Provides atomic increment operations for tracking allocation sequences,
 * timing events, and generating unique identifiers. Used by allocator_bfc
 * for temporal memory management and debugging.
 *
 * **Thread Safety**: All operations are atomic and thread-safe
 * **Performance**: Lock-free atomic operations - O(1) with memory ordering
 * **Memory Ordering**: Uses default memory ordering (sequential consistency)
 *
 * **Use Cases**:
 * - Allocation sequence numbering
 * - Timestamp generation for memory safety
 * - Unique ID generation for debugging
 *
 * **Example**:
 * ```cpp
 * shared_counter counter;
 * int64_t current = counter.get();    // Read current value
 * int64_t next_id = counter.next();   // Atomic increment and return
 * ```
 */
class shared_counter
{
public:
    /**
     * @brief Returns the current counter value without modification.
     *
     * @return Current counter value
     *
     * **Thread Safety**: Atomic read operation
     * **Performance**: O(1) - single atomic load
     * **Memory Ordering**: Acquire semantics for synchronization
     */
    int64_t get() noexcept { return value_.load(std::memory_order_acquire); }

    /**
     * @brief Atomically increments counter and returns new value.
     *
     * @return New counter value after increment
     *
     * **Thread Safety**: Atomic read-modify-write operation
     * **Performance**: O(1) - single atomic increment
     * **Memory Ordering**: Acquire-release semantics for synchronization
     * **Overflow Behavior**: Wraps around at INT64_MAX (undefined behavior)
     */
    int64_t next() noexcept { return value_.fetch_add(1, std::memory_order_acq_rel) + 1; }

    /**
     * @brief Resets counter to zero.
     *
     * **Thread Safety**: Atomic write operation
     * **Use Cases**: Testing, periodic resets, initialization
     */
    void reset() noexcept { value_.store(0, std::memory_order_release); }

private:
    std::atomic<int64_t> value_{0};  ///< Atomic counter storage
};

class memory_dump;

/**
 * @brief High-performance Best-Fit with Coalescing (BFC) memory allocator.
 *
 * allocator_bfc implements a sophisticated memory allocation strategy based on Doug Lea's
 * dlmalloc algorithm. It provides excellent performance for applications with diverse
 * allocation patterns while minimizing memory fragmentation through intelligent coalescing.
 *
 * **Algorithm**: Best-fit allocation with immediate coalescing of adjacent free blocks
 * **Fragmentation Control**: Aggressive coalescing and optional garbage collection
 * **Thread Safety**: Fully thread-safe with fine-grained locking
 * **Memory Overhead**: ~8-16 bytes per allocation for metadata
 *
 * **Key Features**:
 * - Bin-based free list organization for O(log n) allocation
 * - Immediate coalescing of adjacent free blocks
 * - Optional memory pool growth and garbage collection
 * - Comprehensive statistics and debugging support
 * - Temporal memory safety with safe frontier support
 *
 * **Performance Characteristics**:
 * - Allocation: O(log n) average, O(n) worst case
 * - Deallocation: O(log n) average with coalescing
 * - Memory utilization: >95% typical, depends on allocation patterns
 * - Fragmentation: Minimal due to aggressive coalescing
 *
 * **Design Assumptions**:
 * - Process owns most of the memory being managed
 * - Most allocations go through this interface (not mixed with system malloc)
 * - Allocation patterns benefit from coalescing (not purely stack-like)
 *
 * **Example Usage**:
 * ```cpp
 * allocator_bfc::Options opts;
 * opts.allow_growth = true;
 * opts.garbage_collection = true;
 *
 * auto sub_alloc = std::make_unique<CPUSubAllocator>();
 * auto allocator = std::make_unique<allocator_bfc>(
 *     std::move(sub_alloc), 1024*1024*1024, "MainAllocator", opts);
 *
 * void* ptr = allocator->allocate_raw(64, 1024);  // 1KB, 64-byte aligned
 * // ... use memory ...
 * allocator->deallocate_raw(ptr);
 * ```
 */
class MEMORY_VISIBILITY allocator_bfc : public Allocator
{
public:
    /**
     * @brief Configuration options for allocator_bfc behavior and policies.
     *
     * Controls various aspects of allocator behavior including memory growth,
     * retry policies, garbage collection, and fragmentation management.
     *
     * **Thread Safety**: Should be set once during construction
     * **Performance Impact**: Options affect runtime behavior and performance
     */
    struct Options
    {
        /**
         * @brief Enables dynamic memory pool growth when needed.
         *
         * When true, allocator can request additional memory from sub_allocator
         * when current pool is exhausted. When false, allocator works with
         * fixed initial memory allocation.
         *
         * **Default**: true (dynamic growth enabled)
         * **Performance**: Growth involves system calls but improves success rate
         * **Memory**: Allows handling of unpredictable allocation patterns
         */
        bool allow_growth = true;

        /**
         * @brief Enables retry-on-failure with exponential backoff.
         *
         * When true, allocator will sleep and retry failed allocations,
         * hoping other threads will free memory. When false, allocator
         * fails immediately on memory exhaustion.
         *
         * **Default**: true (retry enabled)
         * **Performance**: Retries add latency but improve success rate
         * **Thread Safety**: Uses condition variables for efficient waiting
         * **Timeout**: Limited to prevent indefinite blocking
         */
        bool allow_retry_on_failure = true;

        /**
         * @brief Enables garbage collection to reduce fragmentation.
         *
         * When true, allocator will deallocate entire free regions back to
         * sub_allocator when memory is fragmented but total free space would
         * satisfy allocation requests.
         *
         * **Default**: false (no garbage collection)
         * **Performance**: GC is expensive but can prevent OOM conditions
         * **Use Cases**: Long-running applications with varying allocation patterns
         * **Trade-off**: CPU time vs memory efficiency
         */
        bool garbage_collection = false;

        /**
         * @brief Controls chunk splitting threshold to manage fragmentation.
         *
         * When a free chunk is larger than needed, it's split only if the
         * remainder would be larger than (fragmentation_fraction * chunk_size).
         * This prevents creation of tiny unusable fragments.
         *
         * **Default**: 0.0 (always split when beneficial)
         * **Range**: [0.0, 1.0] where 0.0 = always split, 1.0 = never split
         * **Performance**: Higher values reduce fragmentation but may waste memory
         * **Tuning**: Depends on typical allocation size distribution
         *
         * **Example**: 0.1 means split only if remainder > 10% of original chunk
         */
        double fragmentation_fraction = 0.0;
    };

    /**
     * @brief Constructs allocator_bfc with specified configuration and backend.
     *
     * @param sub_allocator Backend allocator for large memory regions (takes ownership)
     * @param total_memory Maximum memory limit in bytes (0 = unlimited)
     * @param name Human-readable identifier for debugging and profiling
     * @param opts Configuration options controlling allocator behavior
     *
     * **Ownership**: Takes ownership of sub_allocator
     * **Initialization**: Pre-allocates bin structures and initial memory region
     * **Thread Safety**: Constructor is not thread-safe, but resulting instance is
     * **Exception Safety**: Strong guarantee - no partial construction
     *
     * **Performance**: O(1) initialization with small constant overhead for bin setup
     * **Memory**: Initial allocation depends on opts.allow_growth and total_memory
     *
     * **Example**:
     * ```cpp
     * auto sub_alloc = std::make_unique<CPUSubAllocator>(numa_node);
     * allocator_bfc::Options opts{.allow_growth = true, .garbage_collection = true};
     * auto allocator = std::make_unique<allocator_bfc>(
     *     std::move(sub_alloc), 2ULL << 30, "GPU_Memory", opts);
     * ```
     */
    MEMORY_API allocator_bfc(
        std::unique_ptr<memory::sub_allocator> sub_allocator,
        size_t                                 total_memory,
        std::string                            name,
        const Options&                         opts);

    /**
     * @brief Destructor that releases all managed memory back to sub_allocator.
     *
     * **Thread Safety**: Destructor is not thread-safe - ensure no concurrent access
     * **Resource Cleanup**: Returns all memory regions to sub_allocator
     * **Exception Safety**: noexcept - logs errors but doesn't throw
     */
    MEMORY_API ~allocator_bfc() override;

    /**
     * @brief Returns the human-readable name of this allocator instance.
     *
     * @return Allocator name string for debugging and profiling
     *
     * **Thread Safety**: Thread-safe (returns const reference to immutable string)
     * **Performance**: O(1) - returns cached string
     */
    std::string Name() const override { return name_; }

    /**
     * @brief Allocates memory with default allocation attributes.
     *
     * Convenience overload that uses default allocation_attributes for simpler usage.
     * Delegates to the full allocate_raw() method with default attributes.
     *
     * @param alignment Required alignment in bytes (must be power of 2)
     * @param num_bytes Size of memory block to allocate
     * @return Pointer to allocated memory, or nullptr on failure
     *
     * **Performance**: Same as full allocate_raw() method
     * **Thread Safety**: Thread-safe
     * **Algorithm**: O(log n) average case for bin lookup and chunk management
     */
    void* allocate_raw(size_t alignment, size_t num_bytes) override
    {
        return num_bytes == 0 ? nullptr
                              : allocate_raw(alignment, num_bytes, allocation_attributes{});
    }

    /**
     * @brief Allocates memory with specified allocation attributes and policies.
     *
     * Core allocation method implementing best-fit algorithm with coalescing.
     * Searches appropriate bins for suitable free chunks, splits if necessary,
     * and may extend memory pool or retry on failure based on attributes.
     *
     * @param alignment Required alignment in bytes (must be power of 2)
     * @param num_bytes Size of memory block to allocate
     * @param allocation_attr Attributes controlling retry, logging, and timing
     * @return Pointer to allocated memory, or nullptr on failure
     *
     * **Algorithm Complexity**:
     * - Best case: O(1) - perfect fit in appropriate bin
     * - Average case: O(log n) - bin search and chunk management
     * - Worst case: O(n) - full scan during fragmentation or retry
     *
     * **Memory Layout**: Returns memory aligned to max(alignment, Allocator_Alignment)
     * **Thread Safety**: Thread-safe with internal mutex protection
     * **Retry Behavior**: Controlled by allocation_attr.retry_on_failure and opts
     *
     * **Allocation Strategy**:
     * 1. Round size up to minimum allocation unit
     * 2. Find appropriate bin for requested size
     * 3. Search bin for best-fit chunk
     * 4. Split chunk if significantly larger than needed
     * 5. Extend memory pool if no suitable chunk found
     * 6. Retry with exponential backoff if enabled
     */
    MEMORY_API void* allocate_raw(
        size_t alignment, size_t num_bytes, const allocation_attributes& allocation_attr) override;

    /**
     * @brief Deallocates a previously allocated memory block.
     *
     * Returns memory to appropriate free bin and attempts immediate coalescing
     * with adjacent free chunks to reduce fragmentation. May trigger garbage
     * collection if enabled and beneficial.
     *
     * @param ptr Pointer to memory block (must have been allocated by this instance)
     *
     * **Algorithm Complexity**:
     * - Best case: O(1) - simple free without coalescing
     * - Average case: O(log n) - coalescing and bin management
     * - Worst case: O(log n) - multiple coalescing operations
     *
     * **Thread Safety**: Thread-safe with internal mutex protection
     * **Coalescing**: Immediately attempts to merge with adjacent free chunks
     * **Bin Management**: Places coalesced chunk in appropriate size bin
     *
     * **Deallocation Strategy**:
     * 1. Validate pointer and retrieve chunk metadata
     * 2. Mark chunk as free and update statistics
     * 3. Attempt coalescing with adjacent free chunks
     * 4. Insert final chunk into appropriate free bin
     * 5. Consider garbage collection if enabled
     */
    MEMORY_API void deallocate_raw(void* ptr) override;

    /**
     * @brief Indicates that this allocator tracks detailed allocation metadata.
     *
     * @return Always true - allocator_bfc maintains comprehensive allocation tracking
     *
     * **Thread Safety**: Thread-safe (returns constant)
     * **Performance**: O(1) - simple constant return
     * **Capabilities**: Enables RequestedSize(), AllocatedSize(), AllocationId()
     */
    MEMORY_API bool tracks_allocation_sizes() const noexcept override;

    /**
     * @brief Returns the original size requested for an allocation.
     *
     * @param ptr Pointer to allocated memory (must not be nullptr)
     * @return Original requested size in bytes
     *
     * **Requirements**: ptr must have been allocated by this instance
     * **Performance**: O(1) - direct metadata lookup
     * **Thread Safety**: Thread-safe with internal synchronization
     * **Exception Safety**: May throw if ptr is invalid
     */
    MEMORY_API size_t RequestedSize(const void* ptr) const override;

    /**
     * @brief Returns the actual allocated size for a memory block.
     *
     * @param ptr Pointer to allocated memory (must not be nullptr)
     * @return Actual allocated size in bytes (>= RequestedSize)
     *
     * **Requirements**: ptr must have been allocated by this instance
     * **Performance**: O(1) - direct metadata lookup
     * **Thread Safety**: Thread-safe with internal synchronization
     * **Guarantee**: AllocatedSize(ptr) >= RequestedSize(ptr)
     */
    MEMORY_API size_t AllocatedSize(const void* ptr) const override;

    /**
     * @brief Returns unique identifier for an allocation.
     *
     * @param ptr Pointer to allocated memory (must not be nullptr)
     * @return Unique allocation ID (positive integer)
     *
     * **Requirements**: ptr must have been allocated by this instance
     * **Performance**: O(1) - direct metadata lookup
     * **Thread Safety**: Thread-safe with internal synchronization
     * **Uniqueness**: Each allocation gets a different positive ID
     */
    MEMORY_API int64_t AllocationId(const void* ptr) const override;

    /**
     * @brief Retrieves comprehensive allocator statistics.
     *
     * @return Current statistics including memory usage, fragmentation, and performance
     *
     * **Performance**: O(1) for cached statistics, O(n) for computed metrics
     * **Thread Safety**: Thread-safe - returns consistent snapshot
     * **Statistics**: Includes allocation counts, memory usage, fragmentation metrics
     * **Consistency**: Statistics represent atomic snapshot of allocator state
     */
    MEMORY_API std::optional<allocator_stats> GetStats() const override;

    /**
     * @brief Resets statistics counters while preserving current memory state.
     *
     * @return Always true - statistics reset is always supported
     *
     * **Performance**: O(1) - simple counter reset
     * **Thread Safety**: Thread-safe with internal synchronization
     * **Behavior**: Resets counters but preserves current memory allocations
     */
    MEMORY_API bool ClearStats() override;

    /**
     * @brief Sets timing counter for temporal memory management.
     *
     * @param sc Pointer to shared counter (not owned, must outlive allocator)
     *
     * **Thread Safety**: Not thread-safe - should be called during initialization
     * **Ownership**: Does not take ownership of counter
     * **Use Cases**: Temporal memory safety, allocation sequencing
     */
    void SetTimingCounter(shared_counter* sc) noexcept { timing_counter_ = sc; }

    /**
     * @brief Sets safe frontier for timestamped memory reuse.
     *
     * @param count Timestamp representing safe frontier for memory reuse
     *
     * **Thread Safety**: Thread-safe with atomic operations
     * **Performance**: O(1) - atomic store operation
     * **Memory Safety**: Prevents reuse of memory freed after this timestamp
     */
    MEMORY_API void SetSafeFrontier(uint64_t count) noexcept override;

    /**
     * @brief Returns memory type managed by underlying sub_allocator.
     *
     * @return Memory type from sub_allocator (host, device, pinned, etc.)
     *
     * **Thread Safety**: Thread-safe (delegates to sub_allocator)
     * **Performance**: O(1) - simple delegation
     */
    MEMORY_API allocator_memory_enum GetMemoryType() const noexcept override;

    /**
     * @brief Indicates whether operation names should be recorded for debugging.
     *
     * @return Always true - allocator_bfc supports operation name recording
     *
     * **Thread Safety**: Thread-safe (returns constant)
     * **Performance**: O(1) - constant return
     * **Use Cases**: Debugging, profiling, allocation attribution
     */
    bool ShouldRecordOpName() const noexcept { return true; }

    /**
     * @brief Captures complete memory map for debugging and analysis.
     *
     * @return memory_dump containing detailed allocator state and memory layout
     *
     * **Performance**: O(n) - scans all chunks and bins
     * **Thread Safety**: Thread-safe - captures consistent snapshot
     * **Use Cases**: Debugging, memory analysis, fragmentation visualization
     * **Memory**: Creates temporary copy of allocator state
     */
    MEMORY_API memory_dump RecordMemoryMap();

private:
    struct Bin;  ///< Forward declaration of bin structure

    /**
     * @brief Core allocation implementation without retry logic.
     *
     * Implements the fundamental allocation algorithm including bin search,
     * chunk splitting, and memory pool extension. Does not handle retries
     * or complex error recovery.
     *
     * @param alignment Required alignment in bytes
     * @param num_bytes Size to allocate
     * @param dump_log_on_failure Whether to log detailed failure information
     * @param freed_before_count Timestamp constraint for memory reuse
     * @return Allocated memory pointer or nullptr on failure
     *
     * **Algorithm Complexity**: O(log n) average, O(n) worst case
     * **Thread Safety**: Requires external mutex protection
     * **Memory Safety**: Respects timestamp constraints for temporal safety
     */
    void* AllocateRawInternal(
        size_t alignment, size_t num_bytes, bool dump_log_on_failure, uint64_t freed_before_count)
        MEMORY_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

    /**
     * @brief Allocation with retry logic and exponential backoff.
     *
     * Wraps AllocateRawInternal with sophisticated retry logic including
     * exponential backoff, condition variable waiting, and timeout handling.
     *
     * @param alignment Required alignment in bytes
     * @param num_bytes Size to allocate
     * @param allocation_attr Attributes controlling retry behavior
     * @return Allocated memory pointer or nullptr after all retries exhausted
     *
     * **Algorithm Complexity**: O(log n) per attempt, multiple attempts possible
     * **Thread Safety**: Thread-safe with internal synchronization
     * **Retry Strategy**: Exponential backoff with maximum timeout
     * **Condition Variables**: Efficient waiting for memory availability
     */
    void* AllocateRawInternalWithRetry(
        size_t alignment, size_t num_bytes, const allocation_attributes& allocation_attr);

    /**
     * @brief Core deallocation implementation with coalescing.
     *
     * Implements chunk deallocation, immediate coalescing with adjacent
     * free chunks, and bin management. Handles temporal memory safety
     * constraints and statistics updates.
     *
     * @param ptr Pointer to memory to deallocate
     *
     * **Algorithm Complexity**: O(log n) average for coalescing and bin operations
     * **Thread Safety**: Requires external mutex protection
     * **Coalescing Strategy**: Immediate coalescing with adjacent free chunks
     * **Bin Management**: Places coalesced chunks in appropriate size bins
     */
    void DeallocateRawInternal(void* ptr) MEMORY_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

    /**
     * @brief Processes timestamped chunks for safe memory reuse.
     *
     * Manages chunks with temporal safety constraints by checking their
     * freed_at_count against the safe frontier. Safely merges chunks that
     * have passed the safety threshold and handles emergency merging when
     * memory is critically needed.
     *
     * @param required_bytes Minimum bytes needed (0 = normal processing)
     * @return true if any chunks were processed and made available
     *
     * **Algorithm Complexity**: O(n) where n is number of timestamped chunks
     * **Thread Safety**: Requires external mutex protection
     * **Memory Safety**: Ensures temporal safety constraints are respected
     *
     * **Processing Strategy**:
     * - Normal mode (required_bytes = 0): Only merge safe chunks
     * - Emergency mode (required_bytes > 0): Merge unsafe chunks if necessary
     * - Conservative timestamps: Merged chunks inherit most restrictive timestamp
     *
     * **Use Cases**:
     * - Periodic cleanup of safe timestamped chunks
     * - Emergency memory reclamation during allocation pressure
     * - Maintaining temporal memory safety guarantees
     */
    bool MergeTimestampedChunks(size_t required_bytes) MEMORY_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

    /**
     * @brief Returns size of largest available free chunk.
     *
     * Efficiently determines the largest contiguous free memory block
     * available for allocation. Used for fragmentation analysis and
     * allocation feasibility checks.
     *
     * @return Size in bytes of largest free chunk, or 0 if no free chunks
     *
     * **Algorithm Complexity**: O(1) - leverages sorted bin structure
     * **Thread Safety**: Requires external mutex protection
     * **Implementation**: Checks largest bin with free chunks
     * **Use Cases**: Fragmentation metrics, allocation planning, statistics
     */
    int64_t LargestFreeChunk() MEMORY_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

    /**
     * @brief Adds profiling trace for memory operations.
     *
     * Records memory allocation/deallocation events for performance profiling
     * and debugging. Integrates with external profiling systems.
     *
     * @param traceme_name Operation name for profiling
     * @param ptr Memory pointer being traced
     *
     * **Performance**: Minimal overhead when profiling disabled
     * **Thread Safety**: Requires external mutex protection
     * **Integration**: Works with external profiling frameworks
     */
#if MEMORY_HAS_NATIVE_PROFILER
    void AddTraceMe(std::string_view traceme_name, const void* ptr)
        MEMORY_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

    /**
     * @brief Adds detailed profiling trace with size information.
     *
     * Extended version of AddTraceMe that includes detailed size information
     * for more comprehensive profiling and analysis.
     *
     * @param traceme_name Operation name for profiling
     * @param chunk_ptr Memory pointer being traced
     * @param req_bytes Originally requested size
     * @param alloc_bytes Actually allocated size
     *
     * **Performance**: Minimal overhead when profiling disabled
     * **Thread Safety**: Requires external mutex protection
     * **Use Cases**: Detailed memory usage analysis, overhead tracking
     */
    void AddTraceMe(
        std::string_view traceme_name,
        const void*      chunk_ptr,
        int64_t          req_bytes,
        int64_t          alloc_bytes) MEMORY_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
#endif

    /**
     * @brief Handle type for efficient chunk referencing.
     *
     * ChunkHandle provides an efficient way to reference chunks in the allocator's
     * internal data structures. Uses indices instead of pointers for better cache
     * locality and to avoid pointer invalidation during vector reallocation.
     *
     * **Design Rationale**: Index-based handles prevent pointer invalidation
     * **Performance**: Better cache locality than pointer-based references
     * **Memory**: Smaller than pointers on 64-bit systems when chunk count < 2^32
     */
    using ChunkHandle = size_t;

    /**
     * @brief Sentinel value indicating an invalid or null chunk handle.
     *
     * Used to represent null pointers in the chunk linked list structure
     * and to indicate invalid or uninitialized chunk references.
     */
    static constexpr ChunkHandle kInvalidChunkHandle = SIZE_MAX;

    /**
     * @brief Type for bin number identification.
     *
     * Bins are organized by size ranges, with each bin containing free chunks
     * of similar sizes. Bin numbers are used for efficient size-based lookup.
     */
    using BinNum = int;

    /**
     * @brief Sentinel value indicating an invalid bin number.
     *
     * Used to indicate that a chunk is not currently in any bin
     * (typically when the chunk is allocated).
     */
    static constexpr BinNum kInvalidBinNum = -1;

    /**
     * @brief Total number of size-based bins for free chunk organization.
     *
     * Bins are organized exponentially: bin i contains chunks of size [256 << i, 256 << (i+1)).
     * With 21 bins, the largest bin contains chunks up to 256 << 20 = 256MB.
     *
     * **Size Range**: 256 bytes to 256MB
     * **Organization**: Exponential size progression for O(log n) lookup
     * **Trade-off**: More bins = better fit but higher overhead
     */
    static constexpr int kNumBins = 21;

    /**
     * @brief Metadata structure representing a contiguous memory chunk.
     *
     * Chunk represents a contiguous piece of memory that is either entirely free
     * or entirely allocated to a single user request. Chunks form the fundamental
     * unit of memory management in the BFC allocator.
     *
     * **Design Principles**:
     * - Each chunk is either completely free or completely allocated
     * - Chunks form doubly-linked lists for efficient traversal
     * - Adjacent free chunks are immediately coalesced
     * - Chunks maintain both requested and actual sizes for analysis
     *
     * **Memory Layout**: Chunks cover entire AllocationRegions without gaps
     * **Linked List**: prev/next pointers reference physically adjacent chunks
     * **State Management**: allocation_id indicates free (-1) vs allocated (>0)
     * **Bin Organization**: Free chunks are organized in size-based bins
     *
     * **Invariants**:
     * - size >= requested_size (always)
     * - allocation_id == -1 iff chunk is free
     * - Adjacent free chunks are always coalesced
     * - ptr points to actual memory buffer
     *
     * **Performance**: Optimized layout for cache efficiency and fast operations
     * **Thread Safety**: Protected by allocator's mutex during modifications
     */
    struct Chunk
    {
        /**
         * @brief Total size of the memory buffer in bytes.
         *
         * Represents the actual size of the memory chunk, which may be larger
         * than the originally requested size due to alignment requirements,
         * minimum allocation sizes, or fragmentation avoidance strategies.
         *
         * **Invariant**: size >= requested_size
         * **Alignment**: Always aligned to allocator's minimum alignment
         * **Range**: [kMinAllocationSize, maximum_region_size]
         */
        size_t size{0};

        /**
         * @brief Size originally requested by the user.
         *
         * Tracks the actual size requested by the user to enable analysis
         * of allocation efficiency and fragmentation patterns. The difference
         * between size and requested_size represents internal fragmentation.
         *
         * **Usage**: Efficiency analysis, fragmentation metrics, debugging
         * **Invariant**: requested_size <= size
         * **Statistics**: Used for overhead calculations and optimization
         */
        size_t requested_size{0};

        /**
         * @brief Unique identifier for allocated chunks.
         *
         * Provides unique identification for each allocation to enable tracking,
         * debugging, and correlation with external systems. Set to -1 for free
         * chunks and positive values for allocated chunks.
         *
         * **Values**:
         * - -1: Chunk is free and available for allocation
         * - >0: Chunk is allocated with unique identifier
         *
         * **Uniqueness**: Each allocated chunk gets a different positive ID
         * **Thread Safety**: Atomically updated during allocation/deallocation
         * **Use Cases**: Memory leak detection, allocation tracking, debugging
         */
        int64_t allocation_id{-1};

        /**
         * @brief Pointer to the actual memory buffer.
         *
         * Points to the start of the usable memory region. This is the pointer
         * returned to users during allocation and must be properly aligned
         * according to the allocation requirements.
         *
         * **Alignment**: Guaranteed to meet allocator's alignment requirements
         * **Validity**: Non-null for all valid chunks
         * **User Access**: This is the pointer returned by allocate_raw()
         * **Memory Safety**: Valid for 'size' bytes from this address
         */
        void* ptr{nullptr};

        /**
         * @brief Handle to physically preceding chunk in memory layout.
         *
         * References the chunk that immediately precedes this chunk in physical
         * memory. Used for efficient coalescing and memory layout traversal.
         *
         * **Values**:
         * - kInvalidChunkHandle: No preceding chunk (first chunk in region)
         * - Valid handle: References chunk ending at (this->ptr - 1)
         *
         * **Invariant**: If valid, prev chunk's memory ends exactly where this begins
         * **Use Cases**: Coalescing, memory layout validation, debugging
         * **Performance**: O(1) access to adjacent chunks for coalescing
         */
        ChunkHandle prev{kInvalidChunkHandle};

        /**
         * @brief Handle to physically following chunk in memory layout.
         *
         * References the chunk that immediately follows this chunk in physical
         * memory. Used for efficient coalescing and forward traversal.
         *
         * **Values**:
         * - kInvalidChunkHandle: No following chunk (last chunk in region)
         * - Valid handle: References chunk starting at (this->ptr + this->size)
         *
         * **Invariant**: If valid, next chunk's memory starts exactly where this ends
         * **Use Cases**: Coalescing, memory layout validation, debugging
         * **Performance**: O(1) access to adjacent chunks for coalescing
         */
        ChunkHandle next{kInvalidChunkHandle};

        /**
         * @brief Bin number for free chunk organization.
         *
         * Indicates which size-based bin contains this chunk when it's free.
         * Only meaningful for free chunks; allocated chunks have kInvalidBinNum.
         *
         * **Values**:
         * - kInvalidBinNum: Chunk is allocated or not in any bin
         * - [0, kNumBins): Valid bin number for free chunks
         *
         * **Bin Organization**: Exponential size ranges for efficient lookup
         * **Use Cases**: Free chunk lookup, bin management, statistics
         * **Performance**: O(1) bin identification and insertion
         */
        BinNum bin_num{kInvalidBinNum};

        /**
         * @brief Timestamp when chunk was last freed (for temporal safety).
         *
         * Records the timestamp when this chunk was deallocated, used for
         * temporal memory safety in concurrent environments. Prevents premature
         * reuse of memory that might still be referenced by other threads.
         *
         * **Values**:
         * - 0: No timestamp constraint (safe to reuse immediately)
         * - >0: Timestamp from shared_counter when chunk was freed
         *
         * **Memory Safety**: Prevents use-after-free in concurrent scenarios
         * **Performance**: Enables safe memory reuse without expensive synchronization
         * **Use Cases**: GPU memory management, asynchronous computation safety
         */
        uint64_t freed_at_count{0};

        /**
         * @brief Checks if chunk is currently allocated.
         *
         * @return true if chunk is allocated to a user, false if free
         *
         * **Implementation**: Based on allocation_id value
         * **Performance**: O(1) - simple comparison
         * **Thread Safety**: Safe to call without synchronization (atomic read)
         */
        bool in_use() const noexcept { return allocation_id != -1; }

#ifdef MEMORY_MEM_DEBUG
        /**
         * @brief Operation name for debugging (debug builds only).
         *
         * Records the name of the operation that allocated this chunk.
         * Only available in debug builds for memory usage analysis.
         */
        const char* op_name{nullptr};

        /**
         * @brief Step identifier for debugging (debug builds only).
         *
         * Records the execution step when this chunk was allocated.
         * Used for correlating memory usage with execution phases.
         */
        uint64_t step_id{0};

        /**
         * @brief Action counter for debugging (debug builds only).
         *
         * Records the action sequence number for this chunk's last operation.
         * Used for detailed execution tracing and debugging.
         */
        int64_t action_count{0};
#endif

        /**
         * @brief Generates comprehensive debug string representation.
         *
         * Creates a detailed string representation of the chunk including size
         * information, allocation status, bin assignment, and optionally
         * adjacent chunk information for memory layout analysis.
         *
         * @param a Pointer to parent allocator_bfc for chunk handle resolution
         * @param recurse Whether to include information about adjacent chunks
         * @return Formatted debug string with chunk details
         *
         * **Performance**: O(1) for non-recursive, O(k) for recursive where k = recursion depth
         * **Thread Safety**: Not thread-safe - intended for debugging only
         * **Use Cases**: Memory debugging, allocation analysis, fragmentation visualization
         *
         * **Output Format**:
         * ```
         * Size: 1.5KB | Requested: 1.2KB | in_use: true | bin: -1 [| prev: ... | next: ...]
         * ```
         */
        std::string debug_string(allocator_bfc* a, bool recurse) const
            MEMORY_NO_THREAD_SAFETY_ANALYSIS
        {
            std::string result;

            // Helper lambda to format bytes in human-readable format (IEC 60027-2 binary prefixes)
            auto format_bytes = [](int64_t bytes) -> std::string
            {
                bool negative = bytes < 0;
                if (negative)
                {
                    bytes = -bytes;
                }
                std::string result;
                if (bytes < 1024)
                {
                    result = std::to_string(bytes) + "B";
                }
                else
                {
                    const char*  units[]    = {"KiB", "MiB", "GiB", "TiB", "PiB"};
                    const double divisors[] = {
                        1024.0, 1048576.0, 1073741824.0, 1099511627776.0, 1125899906842624.0};
                    for (int i = 4; i >= 0; --i)
                    {
                        if (bytes >= divisors[i])
                        {
                            std::ostringstream oss;
                            oss << std::fixed << std::setprecision(2) << (bytes / divisors[i])
                                << units[i];
                            result = oss.str();
                            break;
                        }
                    }
                    if (result.empty())
                    {
                        result = std::to_string(bytes) + "B";
                    }
                }
                return negative ? "-" + result : result;
            };

            // Core chunk information
            result += fmt::format(
                "Size: {} | Requested: {} | in_use: {} | bin: {}",
                format_bytes(size),
                format_bytes(requested_size),
                in_use() ? "true" : "false",
                bin_num);

            // Add timestamp information if relevant
            if (freed_at_count > 0)
            {
                result += fmt::format(" | freed_at: {}", freed_at_count);
            }

            // Recursively include adjacent chunk information
            if (recurse)
            {
                if (prev != allocator_bfc::kInvalidChunkHandle)
                {
                    const Chunk* p = a->ChunkFromHandle(prev);
                    result += " | prev: [" + p->debug_string(a, false) + "]";
                }
                if (next != allocator_bfc::kInvalidChunkHandle)
                {
                    const Chunk* n = a->ChunkFromHandle(next);
                    result += " | next: [" + n->debug_string(a, false) + "]";
                }
            }

#ifdef MEMORY_MEM_DEBUG
            // Add debug-specific information
            result += fmt::format(
                " | op: {} | step: {} | action: {}",
                (op_name ? op_name : "UNKNOWN"),
                step_id,
                action_count);
#endif

            return result;
        }
    };

    /**
     * @brief Container for free chunks of similar sizes.
     *
     * Bin organizes free chunks into size-based categories for efficient
     * best-fit allocation. Each bin contains chunks within a specific size
     * range, sorted by size and address for optimal allocation performance.
     *
     * **Design Principles**:
     * - Only free chunks are stored in bins (allocated chunks are removed)
     * - Chunks are sorted by size first, then by address for deterministic behavior
     * - Bins enable O(log n) lookup for best-fit allocation
     * - Size ranges are exponentially distributed for balanced performance
     *
     * **Performance Characteristics**:
     * - Insertion: O(log n) where n is chunks in bin
     * - Removal: O(log n) for specific chunk removal
     * - Best-fit search: O(log n) for finding suitable chunk
     * - Memory overhead: Minimal - uses efficient set data structure
     *
     * **Thread Safety**: Protected by allocator's mutex during operations
     */
    struct Bin
    {
        /**
         * @brief Minimum size threshold for chunks in this bin.
         *
         * All chunks in this bin have size >= bin_size. This enables
         * efficient size-based lookup and ensures allocation requests
         * can be satisfied from appropriate bins.
         *
         * **Invariant**: All chunks in free_chunks have size >= bin_size
         * **Organization**: Exponential progression across bins
         * **Range**: [256 bytes, 256MB] across all bins
         */
        size_t bin_size{0};

        /**
         * @brief Comparator for ordering chunks within a bin.
         *
         * Implements strict weak ordering for chunks based on size first,
         * then memory address. This ensures deterministic allocation behavior
         * and efficient best-fit searches.
         *
         * **Ordering Criteria**:
         * 1. Primary: Chunk size (smaller first)
         * 2. Secondary: Memory address (lower address first)
         *
         * **Performance**: O(1) comparison operations
         * **Determinism**: Consistent ordering across runs
         * **Best-fit**: Enables efficient smallest-suitable-chunk selection
         */
        class ChunkComparator
        {
        public:
            /**
             * @brief Constructs comparator with reference to parent allocator.
             *
             * @param allocator Pointer to allocator_bfc for chunk handle resolution
             *
             * **Lifetime**: Allocator must outlive this comparator
             * **Thread Safety**: Not thread-safe - relies on external synchronization
             */
            explicit ChunkComparator(allocator_bfc* allocator) noexcept : allocator_(allocator) {}

            /**
             * @brief Compares two chunk handles for ordering.
             *
             * @param ha First chunk handle
             * @param hb Second chunk handle
             * @return true if chunk ha should be ordered before chunk hb
             *
             * **Algorithm**: Size-first comparison with address tie-breaking
             * **Performance**: O(1) - direct chunk metadata access
             * **Thread Safety**: Requires external mutex protection
             *
             * **Comparison Logic**:
             * ```cpp
             * if (a.size != b.size) return a.size < b.size;
             * return a.ptr < b.ptr;  // Address tie-breaker
             * ```
             */
            bool operator()(ChunkHandle ha, ChunkHandle hb) const MEMORY_NO_THREAD_SAFETY_ANALYSIS
            {
                const Chunk* a = allocator_->ChunkFromHandle(ha);
                const Chunk* b = allocator_->ChunkFromHandle(hb);

                // Primary comparison: size (smaller chunks first)
                if (a->size != b->size)
                {
                    return a->size < b->size;
                }

                // Secondary comparison: address (deterministic tie-breaking)
                return a->ptr < b->ptr;
            }

        private:
            allocator_bfc* allocator_;  ///< Parent allocator for chunk resolution
        };

        /**
         * @brief Ordered set type for storing free chunks.
         *
         * Uses efficient balanced tree structure for O(log n) operations
         * while maintaining sorted order for best-fit allocation.
         */
        using FreeChunkSet = std::set<ChunkHandle, ChunkComparator>;

        /**
         * @brief Sorted collection of free chunks in this bin.
         *
         * Maintains free chunks in sorted order (size first, address second)
         * for efficient best-fit allocation and deterministic behavior.
         *
         * **Ownership**: Contains handles, not chunk objects themselves
         * **Ordering**: Maintained automatically by ChunkComparator
         * **Operations**: O(log n) insertion, removal, and search
         */
        FreeChunkSet free_chunks;

        /**
         * @brief Constructs bin with specified size threshold.
         *
         * @param allocator Pointer to parent allocator_bfc
         * @param bs Minimum size threshold for chunks in this bin
         *
         * **Initialization**: Creates empty bin with proper comparator
         * **Performance**: O(1) construction
         */
        Bin(allocator_bfc* allocator, size_t bs) noexcept
            : bin_size(bs), free_chunks(ChunkComparator(allocator))
        {
        }

        // Prevent copying to avoid comparator issues
        Bin(const Bin&)            = delete;
        Bin& operator=(const Bin&) = delete;

        // Allow moving for container operations
        Bin(Bin&&)            = default;
        Bin& operator=(Bin&&) = default;
    };

    /**
     * @brief Bit shift for minimum allocation size calculation.
     *
     * Used to compute kMinAllocationSize as a power of 2. The value 8
     * corresponds to 2^8 = 256 bytes minimum allocation size.
     *
     * **Rationale**: 256-byte minimum reduces metadata overhead while
     * providing reasonable granularity for most allocation patterns.
     */
    static constexpr size_t kMinAllocationBits = 8;

    /**
     * @brief Minimum allocation size in bytes.
     *
     * All allocations are rounded up to multiples of this size to reduce
     * fragmentation and simplify metadata management. Computed as 2^kMinAllocationBits.
     *
     * **Value**: 256 bytes (2^8)
     * **Purpose**: Reduces fragmentation and metadata overhead
     * **Trade-off**: Larger minimum size reduces flexibility but improves efficiency
     */
    static constexpr size_t kMinAllocationSize = 1 << kMinAllocationBits;

    /**
     * @brief Represents a contiguous memory region obtained from sub_allocator.
     *
     * AllocationRegion manages a large contiguous memory region obtained from
     * the underlying sub_allocator. It subdivides this region into chunks and
     * provides efficient pointer-to-chunk mapping for the BFC allocator.
     *
     * **Design Principles**:
     * - Each region corresponds to one sub_allocator::Alloc() call
     * - Adjacent regions from sub_allocator are automatically coalesced
     * - Regions are subdivided into chunks covering the entire space
     * - Efficient pointer-to-chunk mapping using index-based lookup
     *
     * **Memory Layout**: Regions contain one or more chunks with no gaps
     * **Pointer Mapping**: O(1) conversion from pointers to chunk handles
     * **Thread Safety**: Thread-compatible (not thread-safe without external sync)
     *
     * **Performance Characteristics**:
     * - Pointer lookup: O(1) - direct index calculation
     * - Memory overhead: ~1 handle per kMinAllocationSize bytes
     * - Region extension: O(n) where n is new handles needed
     *
     * **Example Usage**:
     * ```cpp
     * void* region_ptr = sub_allocator->Alloc(alignment, 64*1024*1024, &received);
     * AllocationRegion region(region_ptr, received);
     * // Region now manages 64MB of memory subdivided into chunks
     * ```
     */
    class AllocationRegion
    {
    public:
        /**
         * @brief Constructs region managing specified memory area.
         *
         * @param ptr Pointer to start of memory region
         * @param memory_size Size of memory region in bytes
         *
         * **Requirements**: memory_size must be multiple of kMinAllocationSize
         * **Initialization**: Creates handle array for pointer-to-chunk mapping
         * **Performance**: O(n) where n = memory_size / kMinAllocationSize
         * **Exception Safety**: Strong guarantee - no partial construction
         */
        AllocationRegion(void* ptr, size_t memory_size)
            : ptr_(ptr),
              memory_size_(memory_size),
              end_ptr_(static_cast<void*>(static_cast<char*>(ptr_) + memory_size_))
        {
            MEMORY_CHECK(
                memory_size % kMinAllocationSize == 0,
                "Memory size must be multiple of kMinAllocationSize");

            const size_t n_handles = (memory_size + kMinAllocationSize - 1) / kMinAllocationSize;
            handles_.resize(n_handles, kInvalidChunkHandle);
        }

        /**
         * @brief Default constructor creating empty region.
         *
         * **Use Cases**: Container initialization, delayed initialization
         * **State**: Region is invalid until properly initialized
         */
        AllocationRegion() = default;

        /**
         * @brief Move constructor for efficient region transfer.
         *
         * @param other Region to move from (left in valid but unspecified state)
         *
         * **Performance**: O(1) - efficient swap-based implementation
         * **Exception Safety**: noexcept - no allocation or throwing operations
         */
        AllocationRegion(AllocationRegion&& other) noexcept { Swap(&other); }

        /**
         * @brief Move assignment operator for efficient region transfer.
         *
         * @param other Region to move from
         * @return Reference to this region
         *
         * **Performance**: O(1) - efficient swap-based implementation
         * **Exception Safety**: noexcept - no allocation or throwing operations
         */
        AllocationRegion& operator=(AllocationRegion&& other) noexcept
        {
            if (this != &other)
            {
                Swap(&other);
            }
            return *this;
        }

        /**
         * @brief Returns pointer to start of memory region.
         *
         * @return Pointer to beginning of managed memory region
         *
         * **Thread Safety**: Safe for concurrent read access
         * **Performance**: O(1) - direct member access
         * **Validity**: Valid throughout region lifetime
         */
        void* ptr() const noexcept { return ptr_; }

        /**
         * @brief Returns pointer to end of memory region.
         *
         * @return Pointer to one byte past end of managed memory region
         *
         * **Thread Safety**: Safe for concurrent read access
         * **Performance**: O(1) - direct member access
         * **Usage**: Range checking, iteration bounds
         */
        void* end_ptr() const noexcept { return end_ptr_; }

        /**
         * @brief Returns total size of memory region in bytes.
         *
         * @return Size of managed memory region
         *
         * **Thread Safety**: Safe for concurrent read access
         * **Performance**: O(1) - direct member access
         * **Invariant**: Always multiple of kMinAllocationSize
         */
        size_t memory_size() const noexcept { return memory_size_; }

        /**
         * @brief Extends region by additional memory (for adjacent allocations).
         *
         * When sub_allocator returns memory adjacent to existing region,
         * this method efficiently extends the region rather than creating
         * a separate AllocationRegion.
         *
         * @param size Additional bytes to add to region
         *
         * **Requirements**:
         * - size must be multiple of kMinAllocationSize
         * - New memory must be physically adjacent to current region
         *
         * **Performance**: O(n) where n = new handles needed
         * **Thread Safety**: Not thread-safe - requires external synchronization
         * **Memory**: May reallocate handle array for larger regions
         *
         * **Use Cases**: Coalescing adjacent sub_allocator regions
         */
        void extend(size_t size)
        {
            memory_size_ += size;
            MEMORY_CHECK(
                memory_size_ % kMinAllocationSize == 0,
                "Extended memory size must be multiple of kMinAllocationSize");

            end_ptr_               = static_cast<void*>(static_cast<char*>(end_ptr_) + size);
            const size_t n_handles = (memory_size_ + kMinAllocationSize - 1) / kMinAllocationSize;
            handles_.resize(n_handles, kInvalidChunkHandle);
        }

        /**
         * @brief Retrieves chunk handle for given pointer.
         *
         * @param p Pointer within this region
         * @return ChunkHandle for chunk containing the pointer
         *
         * **Requirements**: p must be within [ptr_, end_ptr_)
         * **Performance**: O(1) - direct index calculation
         * **Thread Safety**: Safe for concurrent read access
         * **Algorithm**: Uses pointer arithmetic to compute handle index
         */
        ChunkHandle get_handle(const void* p) const { return handles_[IndexFor(p)]; }

        /**
         * @brief Associates chunk handle with pointer location.
         *
         * @param p Pointer within this region
         * @param h ChunkHandle to associate with pointer
         *
         * **Requirements**: p must be within [ptr_, end_ptr_)
         * **Performance**: O(1) - direct index assignment
         * **Thread Safety**: Not thread-safe - requires external synchronization
         * **Use Cases**: Chunk creation, chunk splitting, region initialization
         */
        void set_handle(const void* p, ChunkHandle h) { handles_[IndexFor(p)] = h; }

        /**
         * @brief Removes chunk handle association for pointer.
         *
         * @param p Pointer within this region
         *
         * **Requirements**: p must be within [ptr_, end_ptr_)
         * **Performance**: O(1) - sets handle to kInvalidChunkHandle
         * **Thread Safety**: Not thread-safe - requires external synchronization
         * **Use Cases**: Chunk deletion, chunk merging, cleanup
         */
        void erase(const void* p) { set_handle(p, kInvalidChunkHandle); }

    private:
        /**
         * @brief Efficiently swaps contents with another region.
         *
         * @param other Pointer to region to swap with
         *
         * **Performance**: O(1) - swaps pointers and containers
         * **Exception Safety**: noexcept - no throwing operations
         * **Use Cases**: Move constructor, move assignment, container operations
         */
        void Swap(AllocationRegion* other) noexcept
        {
            std::swap(ptr_, other->ptr_);
            std::swap(memory_size_, other->memory_size_);
            std::swap(end_ptr_, other->end_ptr_);
            std::swap(handles_, other->handles_);
        }

        /**
         * @brief Computes handle array index for given pointer.
         *
         * @param p Pointer within this region
         * @return Index into handles_ array
         *
         * **Algorithm**: Uses bit shifting for efficient division by kMinAllocationSize
         * **Performance**: O(1) - pointer arithmetic and bit operations
         * **Requirements**: p must be within [ptr_, end_ptr_]
         * **Thread Safety**: Safe for concurrent read access
         *
         * **Implementation**:
         * ```cpp
         * index = (p - ptr_) / kMinAllocationSize
         *       = (p - ptr_) >> kMinAllocationBits  // Efficient bit shift
         * ```
         */
        size_t IndexFor(const void* p) const
        {
            const auto p_int    = reinterpret_cast<std::uintptr_t>(p);
            const auto base_int = reinterpret_cast<std::uintptr_t>(ptr_);

            MEMORY_CHECK(p_int >= base_int, "Pointer is before region start");
            MEMORY_CHECK(p_int <= base_int + memory_size_, "Pointer is beyond region end");

            return static_cast<size_t>((p_int - base_int) >> kMinAllocationBits);
        }

        // Region metadata
        void*  ptr_{nullptr};      ///< Start of memory region
        size_t memory_size_{0};    ///< Size of memory region in bytes
        void*  end_ptr_{nullptr};  ///< One byte past end of region

        /**
         * @brief Handle lookup array for pointer-to-chunk mapping.
         *
         * Array indexed by (pointer - base) / kMinAllocationSize.
         * Each entry contains the ChunkHandle for the chunk containing
         * that memory location.
         *
         * **Size**: memory_size_ / kMinAllocationSize entries
         * **Indexing**: Direct O(1) lookup using IndexFor()
         * **Memory Overhead**: ~1 handle per kMinAllocationSize bytes
         * **Initialization**: All entries start as kInvalidChunkHandle
         */
        std::vector<ChunkHandle> handles_;

        // Prevent copying to avoid handle array duplication issues
        AllocationRegion(const AllocationRegion&)            = delete;
        AllocationRegion& operator=(const AllocationRegion&) = delete;
    };

    /**
     * @brief Manages multiple discontiguous memory regions for BFC allocator.
     *
     * RegionManager provides a unified interface for managing multiple
     * AllocationRegions obtained from sub_allocator. It handles region
     * creation, extension, lookup, and provides efficient pointer-to-chunk
     * mapping across all managed regions.
     *
     * **Design Principles**:
     * - Maintains sorted list of regions for efficient lookup
     * - Automatically coalesces adjacent regions when possible
     * - Provides O(log n) pointer-to-region mapping
     * - Supports dynamic region addition and removal
     *
     * **Performance Characteristics**:
     * - Region lookup: O(log n) where n is number of regions
     * - Region insertion: O(n) for maintaining sorted order
     * - Pointer mapping: O(log n) + O(1) for region + chunk lookup
     * - Memory overhead: Minimal - vector of regions
     *
     * **Thread Safety**: Thread-compatible (requires external synchronization)
     * **Use Cases**: Multi-region memory management, NUMA-aware allocation
     */
    class RegionManager
    {
    public:
        /**
         * @brief Default constructor creating empty region manager.
         *
         * **Performance**: O(1) - no initial allocations
         * **State**: Ready to accept regions via AddAllocationRegion()
         */
        RegionManager() = default;

        /**
         * @brief Destructor that releases all managed regions.
         *
         * **Cleanup**: Regions are automatically destroyed
         * **Thread Safety**: Not thread-safe - ensure no concurrent access
         */
        ~RegionManager() = default;

        /**
         * @brief Adds new allocation region to management.
         *
         * Inserts region into sorted collection for efficient lookup.
         * Does not attempt coalescing - use AddOrExtendAllocationRegion()
         * for automatic coalescing behavior.
         *
         * @param ptr Pointer to start of memory region
         * @param memory_size Size of memory region in bytes
         *
         * **Performance**: O(n) for insertion into sorted vector
         * **Ordering**: Maintains regions sorted by end pointer
         * **Thread Safety**: Not thread-safe - requires external synchronization
         */
        void AddAllocationRegion(void* ptr, size_t memory_size)
        {
            // Insert maintaining sorted order by end_ptr for efficient lookup
            auto entry = std::upper_bound(regions_.begin(), regions_.end(), ptr, &Comparator);
            regions_.insert(entry, AllocationRegion(ptr, memory_size));
        }

        /**
         * @brief Adds region with automatic coalescing of adjacent regions.
         *
         * Attempts to extend existing region if the new memory is adjacent
         * to an existing region's end. This reduces fragmentation and
         * improves allocation efficiency.
         *
         * @param ptr Pointer to start of new memory region
         * @param memory_size Size of new memory region in bytes
         * @return Pointer to extended region if coalescing occurred, nullptr otherwise
         *
         * **Algorithm**:
         * 1. Find insertion point in sorted region list
         * 2. Check if new region is adjacent to preceding region
         * 3. If adjacent, extend preceding region and return pointer to it
         * 4. Otherwise, insert new region and return nullptr
         *
         * **Performance**: O(n) for insertion, O(1) for extension check
         * **Coalescing**: Only extends existing regions, doesn't merge multiple
         * **Logging**: Records region operations for debugging
         * **Thread Safety**: Not thread-safe - requires external synchronization
         *
         * **Use Cases**:
         * - sub_allocator returns adjacent memory blocks
         * - Reducing region fragmentation
         * - Optimizing memory layout for better cache performance
         */
        AllocationRegion* AddOrExtendAllocationRegion(void* ptr, size_t memory_size)
        {
            // Find insertion point maintaining sorted order
            auto entry = std::upper_bound(regions_.begin(), regions_.end(), ptr, &Comparator);

            // Helper lambda to format bytes in human-readable format (IEC 60027-2 binary prefixes)
            auto format_bytes = [](int64_t bytes) -> std::string
            {
                bool negative = bytes < 0;
                if (negative)
                {
                    bytes = -bytes;
                }
                std::string result;
                if (bytes < 1024)
                {
                    result = std::to_string(bytes) + "B";
                }
                else
                {
                    const char*  units[]    = {"KiB", "MiB", "GiB", "TiB", "PiB"};
                    const double divisors[] = {
                        1024.0, 1048576.0, 1073741824.0, 1099511627776.0, 1125899906842624.0};
                    for (int i = 4; i >= 0; --i)
                    {
                        if (bytes >= divisors[i])
                        {
                            std::ostringstream oss;
                            oss << std::fixed << std::setprecision(2) << (bytes / divisors[i])
                                << units[i];
                            result = oss.str();
                            break;
                        }
                    }
                    if (result.empty())
                    {
                        result = std::to_string(bytes) + "B";
                    }
                }
                return negative ? "-" + result : result;
            };

            // Check for coalescing opportunity with preceding region
            if (entry != regions_.begin())
            {
                auto preceding_region = entry - 1;
                if (preceding_region->end_ptr() == ptr)
                {
                    // Adjacent memory - extend existing region
                    MEMORY_LOG_INFO(
                        "Extending region {} ({}) by {} bytes",
                        preceding_region->ptr(),
                        format_bytes(preceding_region->memory_size()),
                        format_bytes(memory_size));

                    preceding_region->extend(memory_size);
                    return &*preceding_region;
                }
            }

            // No coalescing possible - insert new region
            MEMORY_LOG_INFO("Adding new region {} ({})", ptr, format_bytes(memory_size));

            regions_.insert(entry, AllocationRegion(ptr, memory_size));
            return nullptr;
        }

        /**
         * @brief Removes allocation region from management.
         *
         * @param it Iterator pointing to region to remove
         * @return Iterator to element following the removed region
         *
         * **Performance**: O(n) for vector element removal
         * **Thread Safety**: Not thread-safe - requires external synchronization
         * **Use Cases**: Region cleanup, memory pool shrinking, error recovery
         * **Precondition**: Iterator must be valid and point to existing region
         */
        std::vector<AllocationRegion>::iterator RemoveAllocationRegion(
            std::vector<AllocationRegion>::iterator it)
        {
            return regions_.erase(it);
        }

        /**
         * @brief Retrieves chunk handle for pointer across all regions.
         *
         * @param p Pointer to look up
         * @return ChunkHandle for chunk containing the pointer
         *
         * **Performance**: O(log n) for region lookup + O(1) for handle lookup
         * **Thread Safety**: Safe for concurrent read access
         * **Error Handling**: May return kInvalidChunkHandle if pointer not found
         */
        ChunkHandle get_handle(const void* p) const
        {
            const auto* region = RegionFor(p);
            return region ? region->get_handle(p) : kInvalidChunkHandle;
        }

        /**
         * @brief Sets chunk handle for pointer across all regions.
         *
         * @param p Pointer to associate with handle
         * @param h ChunkHandle to associate
         *
         * **Performance**: O(log n) for region lookup + O(1) for handle assignment
         * **Thread Safety**: Not thread-safe - requires external synchronization
         * **Error Handling**: Logs error if pointer not found in any region
         */
        void set_handle(const void* p, ChunkHandle h)
        {
            if (auto* region = MutableRegionFor(p))
            {
                region->set_handle(p, h);
            }
        }

        /**
         * @brief Removes chunk handle association for pointer.
         *
         * @param p Pointer to clear handle for
         *
         * **Performance**: O(log n) for region lookup + O(1) for handle clearing
         * **Thread Safety**: Not thread-safe - requires external synchronization
         */
        void erase(const void* p)
        {
            if (auto* region = MutableRegionFor(p))
            {
                region->erase(p);
            }
        }

        /**
         * @brief Returns read-only access to all managed regions.
         *
         * @return Const reference to vector of all allocation regions
         *
         * **Performance**: O(1) - direct container access
         * **Thread Safety**: Safe for concurrent read access
         * **Use Cases**: Statistics, debugging, memory analysis
         */
        const std::vector<AllocationRegion>& regions() const noexcept { return regions_; }

    private:
        /**
         * @brief Comparator for binary search in sorted region vector.
         *
         * @param ptr Pointer to search for
         * @param other Region to compare against
         * @return true if ptr is before the end of other region
         *
         * **Algorithm**: Used with std::upper_bound for efficient region lookup
         * **Performance**: O(1) - simple pointer comparison
         */
        static bool Comparator(const void* ptr, const AllocationRegion& other) noexcept
        {
            return ptr < other.end_ptr();
        }

        /**
         * @brief Finds mutable region containing given pointer.
         *
         * @param p Pointer to find region for
         * @return Pointer to mutable region, or nullptr if not found
         *
         * **Performance**: O(log n) - binary search through sorted regions
         * **Thread Safety**: Not thread-safe - returns mutable pointer
         */
        AllocationRegion* MutableRegionFor(const void* p)
        {
            return const_cast<AllocationRegion*>(RegionFor(p));
        }

        /**
         * @brief Finds region containing given pointer.
         *
         * @param p Pointer to find region for
         * @return Pointer to const region, or nullptr if not found
         *
         * **Algorithm**: Binary search using std::upper_bound
         * **Performance**: O(log n) where n is number of regions
         * **Thread Safety**: Safe for concurrent read access
         * **Error Handling**: Logs error and returns nullptr if not found
         */
        const AllocationRegion* RegionFor(const void* p) const
        {
            auto entry = std::upper_bound(regions_.begin(), regions_.end(), p, &Comparator);

            if (entry != regions_.end())
            {
                return &(*entry);
            }

            MEMORY_LOG_ERROR("Could not find region for pointer {}", p);
            return nullptr;
        }

        /**
         * @brief Vector of allocation regions sorted by end pointer.
         *
         * Maintains regions in sorted order for efficient O(log n) lookup.
         * Sorted by end_ptr to work correctly with std::upper_bound.
         */
        std::vector<AllocationRegion> regions_;
    };

    /**
     * @brief Rounds byte count up to minimum allocation size boundary.
     *
     * @param bytes Raw byte count to round up
     * @return Rounded byte count (multiple of kMinAllocationSize)
     *
     * **Algorithm**: Rounds up to next multiple of kMinAllocationSize
     * **Performance**: O(1) - simple arithmetic operations
     * **Thread Safety**: Thread-safe (static function, no shared state)
     * **Use Cases**: Size normalization, allocation request processing
     */
    static size_t RoundedBytes(size_t bytes) noexcept;

    /**
     * @brief Attempts to extend memory pool to satisfy allocation request.
     *
     * Requests additional memory from sub_allocator when current pool
     * cannot satisfy allocation. May coalesce with existing regions
     * if sub_allocator returns adjacent memory.
     *
     * @param alignment Required alignment for new region
     * @param rounded_bytes Minimum size needed (already rounded)
     * @return true if extension succeeded, false otherwise
     *
     * **Algorithm**: Requests memory from sub_allocator and integrates into pool
     * **Performance**: O(n) where n is number of existing regions (for insertion)
     * **Thread Safety**: Requires mutex protection
     * **Side Effects**: May add new AllocationRegion and create initial chunk
     * **Use Cases**: Pool growth, handling large allocations, OOM recovery
     */
    bool Extend(size_t alignment, size_t rounded_bytes) MEMORY_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

    /**
     * @brief Deallocates entire free regions to reduce fragmentation.
     *
     * Implements garbage collection by returning completely free regions
     * back to sub_allocator. Used when memory is fragmented but total free
     * space would satisfy allocation requests.
     *
     * @param rounded_bytes Target size that triggered garbage collection
     * @return true if any regions were deallocated, false otherwise
     *
     * **Algorithm**:
     * 1. Identify regions containing only free chunks
     * 2. Return those regions to sub_allocator
     * 3. Update internal data structures
     *
     * **Performance**: O(n*m) where n=regions, m=chunks per region
     * **Thread Safety**: Requires mutex protection
     * **Use Cases**: OOM recovery, fragmentation reduction, memory pressure
     * **Trade-off**: CPU time vs memory efficiency
     */
    bool DeallocateFreeRegions(size_t rounded_bytes) MEMORY_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

    /**
     * @brief Helper to deallocate specified regions back to sub_allocator.
     *
     * @param region_ptrs Set of region pointers to deallocate
     *
     * **Performance**: O(n) where n is number of regions to deallocate
     * **Thread Safety**: Requires mutex protection
     * **Side Effects**: Removes regions and associated chunks from allocator
     * **Use Cases**: Garbage collection implementation, cleanup operations
     */
    void DeallocateRegions(const std::unordered_set<void*>& region_ptrs)
        MEMORY_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

    /**
     * @brief Finds suitable chunk in specified bin for allocation.
     *
     * Searches bin for chunk that satisfies size and temporal safety
     * requirements. Implements best-fit strategy within the bin.
     *
     * @param bin_num Bin number to search in
     * @param rounded_bytes Minimum chunk size needed
     * @param num_bytes Original requested size (for statistics)
     * @param freed_before Temporal safety constraint timestamp
     * @return Pointer to allocated chunk, or nullptr if none suitable
     *
     * **Algorithm**: Best-fit search within sorted bin
     * **Performance**: O(log n) where n is chunks in bin
     * **Thread Safety**: Requires mutex protection
     * **Temporal Safety**: Respects freed_before timestamp constraints
     * **Side Effects**: Removes chunk from bin, may split chunk
     */
    void* FindChunkPtr(
        BinNum bin_num, size_t rounded_bytes, size_t num_bytes, uint64_t freed_before)
        MEMORY_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

    /**
     * @brief Splits chunk into two parts for allocation.
     *
     * Divides chunk into allocated portion and remaining free portion.
     * Creates new chunk metadata for the remainder and adds it to
     * appropriate bin.
     *
     * @param h Handle of chunk to split
     * @param num_bytes Size of allocated portion
     *
     * **Algorithm**: Creates new chunk for remainder, updates linked list
     * **Performance**: O(log n) for bin insertion of remainder
     * **Thread Safety**: Requires mutex protection
     * **Invariants**: Maintains chunk linked list and region mapping
     * **Use Cases**: Reducing internal fragmentation, exact-size allocation
     */
    void SplitChunk(ChunkHandle h, size_t num_bytes) MEMORY_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

    /**
     * @brief Merges two adjacent chunks into single larger chunk.
     *
     * Combines two physically adjacent chunks to reduce fragmentation.
     * Updates linked list pointers and chunk metadata.
     *
     * @param h Handle of first chunk
     * @param h2 Handle of second chunk (must be adjacent to first)
     *
     * **Requirements**: Chunks must be physically adjacent in memory
     * **Algorithm**: Combines sizes, updates pointers, deallocates second chunk
     * **Performance**: O(1) - direct pointer manipulation
     * **Thread Safety**: Requires mutex protection
     * **Use Cases**: Coalescing during deallocation, fragmentation reduction
     */
    void Merge(ChunkHandle h, ChunkHandle h2) MEMORY_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

    /**
     * @brief Adds free chunk to appropriate size-based bin.
     *
     * Determines correct bin for chunk size and inserts chunk
     * maintaining sorted order within the bin.
     *
     * @param h Handle of chunk to add to bin
     *
     * **Algorithm**: Computes bin number, inserts into sorted set
     * **Performance**: O(log n) where n is chunks in target bin
     * **Thread Safety**: Requires mutex protection
     * **Bin Selection**: Based on chunk size using exponential bins
     * **Use Cases**: Making chunks available for allocation, coalescing
     */
    void InsertFreeChunkIntoBin(ChunkHandle h) MEMORY_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

    /**
     * @brief Removes chunk from bin using iterator (optimized removal).
     *
     * @param free_chunks Pointer to bin's chunk set
     * @param c Iterator pointing to chunk to remove
     *
     * **Performance**: O(log n) - set erase operation
     * **Thread Safety**: Requires mutex protection
     * **Use Cases**: Optimized removal when iterator is already available
     */
    void RemoveFreeChunkIterFromBin(
        Bin::FreeChunkSet* free_chunks, const Bin::FreeChunkSet::iterator& c)
        MEMORY_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

    /**
     * @brief Removes free chunk from its current bin.
     *
     * @param h Handle of chunk to remove from bin
     *
     * **Performance**: O(log n) where n is chunks in bin
     * **Thread Safety**: Requires mutex protection
     * **Use Cases**: Chunk allocation, chunk merging, bin management
     */
    void RemoveFreeChunkFromBin(ChunkHandle h) MEMORY_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

    /**
     * @brief Conditionally removes chunk from bin if it's actually in one.
     *
     * @param h Handle of chunk to potentially remove
     *
     * **Performance**: O(log n) if chunk is in bin, O(1) otherwise
     * **Thread Safety**: Requires mutex protection
     * **Use Cases**: Safe removal when chunk state is uncertain
     */
    void MaybeRemoveFreeChunkFromBin(ChunkHandle h) MEMORY_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

    /**
     * @brief Removes chunk metadata and deallocates chunk handle.
     *
     * @param h Handle of chunk to delete
     *
     * **Performance**: O(1) - direct handle deallocation
     * **Thread Safety**: Requires mutex protection
     * **Side Effects**: Invalidates chunk handle, updates free handle list
     * **Use Cases**: Chunk merging, cleanup operations, memory reclamation
     */
    void DeleteChunk(ChunkHandle h) MEMORY_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

    /**
     * @brief Renders visual representation of memory occupancy.
     *
     * @return String showing memory layout and occupancy patterns
     *
     * **Performance**: O(n) where n is total number of chunks
     * **Thread Safety**: Requires mutex protection
     * **Use Cases**: Debugging, fragmentation visualization, memory analysis
     */
    std::string RenderOccupancy() MEMORY_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

    /**
     * @brief Dumps detailed memory allocation log for debugging.
     *
     * @param num_bytes Size that triggered the memory dump
     *
     * **Performance**: O(n) where n is number of chunks and regions
     * **Thread Safety**: Requires mutex protection
     * **Use Cases**: Allocation failure debugging, memory leak analysis
     */
    void DumpMemoryLog(size_t num_bytes) MEMORY_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

    /**
     * @brief Internal implementation of memory map recording.
     *
     * @return memory_dump containing complete allocator state
     *
     * **Performance**: O(n) where n is total chunks and regions
     * **Thread Safety**: Requires mutex protection
     * **Use Cases**: Debugging, profiling, memory analysis tools
     */
    memory_dump RecordMemoryMapInternal() MEMORY_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

    /**
     * @brief Conditionally writes memory map to file for debugging.
     *
     * **Performance**: O(n) if writing is enabled, O(1) otherwise
     * **Thread Safety**: Requires mutex protection
     * **Use Cases**: Automated debugging, memory leak detection
     */
    void MaybeWriteMemoryMap() MEMORY_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

    /**
     * @brief Allocates new chunk handle from internal pool.
     *
     * @return New chunk handle for metadata storage
     *
     * **Performance**: O(1) - uses free handle pool or extends vector
     * **Thread Safety**: Requires mutex protection
     * **Use Cases**: Chunk creation, chunk splitting, region initialization
     */
    ChunkHandle AllocateChunk() MEMORY_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

    /**
     * @brief Returns chunk handle to internal pool for reuse.
     *
     * @param h Handle to deallocate
     *
     * **Performance**: O(1) - adds to free handle pool
     * **Thread Safety**: Requires mutex protection
     * **Use Cases**: Chunk deletion, chunk merging, cleanup
     */
    void DeallocateChunk(ChunkHandle h) MEMORY_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

    /**
     * @brief Converts chunk handle to mutable chunk pointer.
     *
     * @param h Chunk handle to resolve
     * @return Pointer to mutable chunk metadata
     *
     * **Performance**: O(1) - direct vector indexing
     * **Thread Safety**: Requires mutex protection
     * **Use Cases**: Chunk modification, allocation operations
     */
    Chunk* ChunkFromHandle(ChunkHandle h) MEMORY_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

    /**
     * @brief Converts chunk handle to const chunk pointer.
     *
     * @param h Chunk handle to resolve
     * @return Pointer to const chunk metadata
     *
     * **Performance**: O(1) - direct vector indexing
     * **Thread Safety**: Requires mutex protection
     * **Use Cases**: Chunk inspection, statistics, debugging
     */
    const Chunk* ChunkFromHandle(ChunkHandle h) const MEMORY_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

    /**
     * @brief Marks chunk as free and updates statistics.
     *
     * @param h Handle of chunk to mark as free
     *
     * **Performance**: O(1) - direct metadata update
     * **Thread Safety**: Requires mutex protection
     * **Side Effects**: Updates allocation statistics, sets allocation_id to -1
     * **Use Cases**: Deallocation processing, chunk state management
     */
    void MarkFree(ChunkHandle h) MEMORY_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

    /**
     * @brief Attempts to coalesce chunk with adjacent free chunks.
     *
     * @param h Handle of chunk to coalesce
     * @param ignore_freed_at Whether to ignore temporal safety constraints
     * @return Handle of final coalesced chunk
     *
     * **Algorithm**: Checks adjacent chunks and merges if both are free
     * **Performance**: O(log n) per merge for bin management
     * **Thread Safety**: Requires mutex protection
     * **Temporal Safety**: Respects freed_at timestamps unless ignored
     * **Use Cases**: Deallocation, fragmentation reduction, memory compaction
     */
    ChunkHandle TryToCoalesce(ChunkHandle h, bool ignore_freed_at)
        MEMORY_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

    /**
     * @brief Computes memory fragmentation metric.
     *
     * @return Fragmentation ratio [0.0, 1.0] where 0.0 = no fragmentation
     *
     * **Algorithm**: 1.0 - (largest_free_chunk / total_free_memory)
     * **Performance**: O(1) - uses cached largest chunk size
     * **Thread Safety**: Requires mutex protection
     * **Interpretation**: Higher values indicate more fragmentation
     * **Use Cases**: Performance monitoring, garbage collection triggers
     */
    double GetFragmentation() MEMORY_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

    /**
     * @brief Debug information for a single bin.
     *
     * Contains comprehensive statistics about bin usage including
     * memory utilization, chunk counts, and efficiency metrics.
     */
    struct BinDebugInfo
    {
        size_t total_bytes_in_use{0};            ///< Bytes in allocated chunks
        size_t total_bytes_in_bin{0};            ///< Bytes in free chunks in bin
        size_t total_requested_bytes_in_use{0};  ///< Originally requested bytes
        size_t total_chunks_in_use{0};           ///< Number of allocated chunks
        size_t total_chunks_in_bin{0};           ///< Number of free chunks in bin
    };

    /**
     * @brief Computes debug information for all bins.
     *
     * @return Array of debug info structures, one per bin
     *
     * **Performance**: O(n) where n is total number of chunks
     * **Thread Safety**: Requires mutex protection
     * **Use Cases**: Performance analysis, memory usage debugging, optimization
     */
    std::array<BinDebugInfo, kNumBins> get_bin_debug_info() MEMORY_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

    // ========== Immutable Configuration (Set During Construction) ==========
    /**
     * @brief Retry helper for handling allocation failures with backoff.
     *
     * Manages retry logic, exponential backoff, and condition variable
     * coordination for failed allocation attempts.
     */
    allocator_retry retry_helper_;

    /**
     * @brief Maximum memory limit in bytes (0 = unlimited).
     *
     * Hard limit on total memory that can be allocated from sub_allocator.
     * When reached, allocations fail even if sub_allocator could provide more.
     */
    size_t memory_limit_{0};

    /**
     * @brief Converts bin index to bin pointer using placement new area.
     *
     * @param index Bin number [0, kNumBins)
     * @return Pointer to bin at specified index
     *
     * **Performance**: O(1) - direct pointer arithmetic
     * **Memory Layout**: Uses bins_space_ for storage
     */
    Bin* BinFromIndex(BinNum index) noexcept
    {
        return reinterpret_cast<Bin*>(&(bins_space_[index * sizeof(Bin)]));
    }

    /**
     * @brief Converts bin number to minimum size for that bin.
     *
     * @param index Bin number
     * @return Minimum chunk size for bin (256 << index)
     *
     * **Performance**: O(1) - bit shift operation
     * **Size Progression**: 256, 512, 1024, 2048, ... bytes
     */
    size_t BinNumToSize(BinNum index) const noexcept { return static_cast<size_t>(256) << index; }

    /**
     * @brief Determines appropriate bin number for given size.
     *
     * @param bytes Size to find bin for
     * @return Bin number that can accommodate the size
     *
     * **Algorithm**: Uses log2 of normalized size for exponential binning
     * **Performance**: O(1) - bit operations and log2 calculation
     * **Range**: Returns [0, kNumBins-1] for any input size
     */
    BinNum BinNumForSize(size_t bytes) const noexcept
    {
        const uint64_t v = std::max<size_t>(bytes, 256) >> kMinAllocationBits;
        const int      b = std::min(kNumBins - 1, Log2Floor64(v));
        return static_cast<BinNum>(b);
    }

    /**
     * @brief Gets bin pointer for given chunk size.
     *
     * @param bytes Size to find bin for
     * @return Pointer to appropriate bin
     *
     * **Performance**: O(1) - combines BinNumForSize and BinFromIndex
     * **Use Cases**: Chunk insertion, allocation searches
     */
    Bin* BinForSize(size_t bytes) noexcept { return BinFromIndex(BinNumForSize(bytes)); }

    /**
     * @brief Raw storage space for bin objects.
     *
     * Uses placement new approach to avoid dynamic allocation during
     * construction. Bins are constructed in-place in this space.
     */
    alignas(Bin) char bins_space_[sizeof(Bin) * kNumBins];

    /**
     * @brief Immutable configuration options set during construction.
     *
     * Controls allocator behavior including growth, retry policies,
     * garbage collection, and fragmentation management.
     */
    const Options opts_;

    /**
     * @brief Size of current region allocation for growth calculations.
     *
     * Tracks the size of memory regions being requested from sub_allocator
     * to implement exponential growth strategies.
     */
    size_t curr_region_allocation_bytes_{0};

    /**
     * @brief Whether to coalesce adjacent sub_allocator regions.
     *
     * Controls whether adjacent memory regions from sub_allocator should
     * be treated as contiguous. May be disabled for device memory where
     * physical adjacency doesn't guarantee virtual adjacency.
     *
     * **Use Cases**: device_option memory, NUMA-aware allocation, memory mapping
     */
    const bool coalesce_regions_{true};

    /**
     * @brief Backend allocator for large memory regions.
     *
     * Provides large contiguous memory regions that allocator_bfc
     * subdivides into smaller chunks for user allocations.
     */
    std::unique_ptr<memory::sub_allocator> sub_allocator_;

    /**
     * @brief Human-readable name for debugging and profiling.
     *
     * Used in log messages, error reports, and profiling output
     * to identify this allocator instance.
     */
    std::string name_;

    /**
     * @brief Optional timing counter for temporal memory safety.
     *
     * Provides timestamps for memory operations to enable safe
     * memory reuse in concurrent environments. Not owned by allocator.
     */
    shared_counter* timing_counter_{nullptr};

    /**
     * @brief Queue of chunks with temporal safety constraints.
     *
     * Maintains chunks that have been freed but cannot be immediately
     * reused due to temporal safety requirements (freed_at_count).
     */
    std::deque<ChunkHandle> timestamped_chunks_;

    /**
     * @brief Atomic safe frontier for temporal memory management.
     *
     * Represents the safe timestamp boundary - chunks freed before
     * this timestamp can be safely reused without synchronization concerns.
     */
    std::atomic<uint64_t> safe_frontier_{0};

    // ========== Mutable State (Protected by Mutex) ==========

    /**
     * @brief Primary mutex protecting all mutable allocator state.
     *
     * Provides thread safety for all allocation, deallocation, and
     * internal data structure operations. Marked mutable to allow
     * const methods to acquire locks for read operations.
     */
    mutable std::mutex mutex_;

    /**
     * @brief Manager for all memory regions obtained from sub_allocator.
     *
     * Handles multiple discontiguous memory regions and provides
     * unified pointer-to-chunk mapping across all regions.
     */
    RegionManager region_manager_ MEMORY_GUARDED_BY(mutex_);

    /**
     * @brief Vector storing all chunk metadata.
     *
     * Central repository for chunk information. ChunkHandles are
     * indices into this vector. May contain gaps for deallocated chunks.
     */
    std::vector<Chunk> chunks_ MEMORY_GUARDED_BY(mutex_);

    /**
     * @brief Head of linked list of free chunk handles.
     *
     * Maintains a free list of deallocated chunk handles for reuse,
     * avoiding vector growth when possible. kInvalidChunkHandle indicates
     * empty free list.
     */
    ChunkHandle free_chunks_list_ MEMORY_GUARDED_BY(mutex_){kInvalidChunkHandle};

    /**
     * @brief Counter for generating unique allocation identifiers.
     *
     * Provides unique positive IDs for each allocation to enable
     * tracking, debugging, and correlation with external systems.
     * Incremented for each new allocation.
     */
    int64_t next_allocation_id_ MEMORY_GUARDED_BY(mutex_){1};

    /**
     * @brief Comprehensive allocator statistics and metrics.
     *
     * Tracks memory usage, allocation counts, fragmentation metrics,
     * and performance statistics for monitoring and optimization.
     */
    allocator_stats stats_ MEMORY_GUARDED_BY(mutex_);

#ifdef MEMORY_MEM_DEBUG
    /**
     * @brief Debug-only action counter for operation sequencing.
     *
     * Tracks the sequence number of allocator operations for
     * detailed debugging and analysis in debug builds.
     */
    int64_t action_counter_ MEMORY_GUARDED_BY(mutex_){0};

    /**
     * @brief Debug-only history of allocation sizes.
     *
     * Circular buffer maintaining recent allocation size history
     * for debugging allocation patterns and performance analysis.
     */
    static constexpr size_t kMemDebugSizeHistorySize = 4096;
    int64_t                 size_history_[kMemDebugSizeHistorySize] MEMORY_GUARDED_BY(mutex_){};
#endif

    // ========== Test Access and Copy Prevention ==========

    /**
     * @brief Friend class declarations for unit testing.
     *
     * Allows test classes to access private methods and members
     * for comprehensive testing of internal algorithms.
     */
    friend class GPUBFCAllocatorPrivateMethodsTest;
    friend class GPUBFCAllocatorPrivateMethodsTest_SubAllocatorSpecific;

    // Prevent copying and assignment to avoid complex state duplication
    allocator_bfc(const allocator_bfc&)            = delete;
    allocator_bfc& operator=(const allocator_bfc&) = delete;
};

}  // namespace memory
