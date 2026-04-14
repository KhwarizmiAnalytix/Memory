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

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#if defined(__clang__)
#define NO_THREAD_SAFETY_ANALYSIS __attribute__((no_thread_safety_analysis))
#else
#define NO_THREAD_SAFETY_ANALYSIS
#endif

#include "cpu/allocator.h"
#include "common/memory_export.h"

namespace memory
{

/**
 * @brief Computes ceiling of log base 2 for 64-bit integers.
 *
 * Efficiently calculates the smallest integer n such that 2^n >= x.
 * Optimized with compiler intrinsics when available for maximum performance.
 *
 * @param x Input value (must be > 0)
 * @return Ceiling of log2(x), or 0 if x <= 1
 *
 * **Algorithm**:
 * - Uses compiler intrinsics (CLZ) when available for O(1) performance
 * - Falls back to bit manipulation for portable implementation
 * - Handles powers of 2 correctly
 *
 * **Performance**:
 * - O(1) with compiler intrinsics
 * - O(log log n) with fallback implementation
 *
 * **Thread Safety**: Thread-safe (pure function)
 * **Use Cases**: Power-of-2 size calculations, bin sizing, memory alignment
 *
 * **Examples**:
 * - Log2Ceiling64(1) = 0
 * - Log2Ceiling64(8) = 3
 * - Log2Ceiling64(9) = 4
 * - Log2Ceiling64(1024) = 10
 */
MEMORY_NODISCARD constexpr int Log2Ceiling64(uint64_t x) noexcept
{
    if (x <= 1)
    {
        return 0;
    }

#if defined(__GNUC__) || defined(__clang__)
    // Optimized path using compiler intrinsics
    // __builtin_clzll counts leading zeros in 64-bit integer
    const int floor_log = 63 - __builtin_clzll(x);
    return floor_log + (x > (1ULL << floor_log) ? 1 : 0);
#else
    // Portable fallback implementation using bit manipulation
    int      log   = 0;
    uint64_t value = x - 1;  // Handle powers of 2 correctly

    // Binary search approach for efficient bit position finding
    if (value >= (1ULL << 32))
    {
        value >>= 32;
        log += 32;
    }
    if (value >= (1ULL << 16))
    {
        value >>= 16;
        log += 16;
    }
    if (value >= (1ULL << 8))
    {
        value >>= 8;
        log += 8;
    }
    if (value >= (1ULL << 4))
    {
        value >>= 4;
        log += 4;
    }
    if (value >= (1ULL << 2))
    {
        value >>= 2;
        log += 2;
    }
    if (value >= (1ULL << 1))
    {
        log += 1;
    }

    return log + 1;
#endif
}

/**
 * @brief Abstract interface for size rounding strategies.
 *
 * Defines the contract for objects that implement different size rounding
 * policies for memory allocation optimization. Enables pluggable rounding
 * strategies in pool allocators and other memory management systems.
 *
 * **Design Pattern**: Strategy pattern for size rounding algorithms
 * **Use Cases**:
 * - Memory pool size standardization
 * - Allocation size optimization
 * - Fragmentation reduction
 * - Cache-friendly size alignment
 *
 * **Common Implementations**:
 * - NoopRounder: Pass-through (no rounding)
 * - Pow2Rounder: Round to next power of 2
 * - Custom rounders: Application-specific strategies
 */
class round_up_interface
{
public:
    /**
     * @brief Virtual destructor for proper cleanup.
     *
     * Ensures derived classes are properly destroyed when accessed
     * through base class pointers.
     */
    virtual ~round_up_interface() = default;

    /**
     * @brief Rounds up size according to implementation strategy.
     *
     * @param num_bytes Original size in bytes
     * @return Rounded size in bytes (>= num_bytes)
     *
     * **Contract**: Must return value >= num_bytes
     * **Thread Safety**: Implementation-dependent (should be thread-safe)
     * **Performance**: Implementation-dependent
     */
    MEMORY_NODISCARD virtual size_t RoundUp(size_t num_bytes) = 0;
};

/**
 * @brief High-performance memory pool allocator with LRU eviction policy.
 *
 * allocator_pool implements a sophisticated memory pooling strategy that
 * maintains a cache of previously allocated memory buffers for reuse.
 * It uses a Least Recently Used (LRU) eviction policy to manage pool
 * size and provides significant performance improvements for applications
 * with predictable allocation patterns.
 *
 * **Key Features**:
 * - LRU-based pool management for optimal cache utilization
 * - Configurable pool size limits with optional auto-resizing
 * - Pluggable size rounding strategies for allocation optimization
 * - Comprehensive statistics for performance monitoring
 * - Thread-safe operation with fine-grained locking
 * - Integration with sub_allocator backend for large allocations
 *
 * **Performance Characteristics**:
 * - Pool hit: O(log n) where n is pool size
 * - Pool miss: O(log n) + backend allocation time
 * - Eviction: O(1) - removes LRU item
 * - Memory overhead: ~32 bytes per pooled buffer
 *
 * **Design Principles**:
 * - Minimize allocation/deallocation overhead for repeated patterns
 * - Maintain working set in memory for optimal performance
 * - Provide detailed statistics for optimization and monitoring
 * - Support various size rounding strategies for different use cases
 *
 * **Use Cases**:
 * - Temporary buffer allocation in computational loops
 * - Frequent allocation/deallocation patterns
 * - Memory-intensive applications requiring predictable performance
 *
 * **Thread Safety**: Fully thread-safe with internal mutex protection
 */
class MEMORY_VISIBILITY allocator_pool : public Allocator
{
public:
    /**
     * @brief Constructs pool allocator with specified configuration.
     *
     * @param pool_size_limit Maximum number of buffers to keep in pool
     * @param auto_resize Whether to automatically increase pool size
     * @param allocator Backend sub_allocator (takes ownership)
     * @param size_rounder Size rounding strategy (takes ownership)
     * @param name Human-readable name for debugging
     *
     * **Pool Size Limit**:
     * - 0: Effectively a thin wrapper (no pooling)
     * - >0: Maximum number of cached buffers
     *
     * **Auto-Resize Behavior**:
     * - true: Gradually increases pool_size_limit to reduce deallocations
     * - false: Fixed pool size throughout lifetime
     * - Auto-resize only increases, never decreases pool size
     *
     * **Ownership**: Takes ownership of allocator and size_rounder
     * **Thread Safety**: Constructor is not thread-safe
     * **Exception Safety**: Strong guarantee - no partial construction
     */
    MEMORY_API allocator_pool(
        size_t                                   pool_size_limit,
        bool                                     auto_resize,
        std::unique_ptr<memory::sub_allocator> allocator,
        std::unique_ptr<round_up_interface>      size_rounder,
        std::string                              name);

    /**
     * @brief Destructor that cleans up pool and releases all resources.
     *
     * **Cleanup**: Deallocates all pooled buffers and releases backend allocator
     * **Thread Safety**: Destructor is not thread-safe
     * **Exception Safety**: noexcept - logs errors but doesn't throw
     */
    MEMORY_API ~allocator_pool() override;

    /**
     * @brief Returns human-readable name of the allocator.
     *
     * @return Name string provided during construction
     *
     * **Thread Safety**: Thread-safe (returns immutable string)
     * **Use Cases**: Debugging, logging, performance monitoring
     */
    std::string Name() const override { return name_; }

    /**
     * @brief Allocates memory with alignment requirements through pool.
     *
     * Primary allocation interface that attempts to satisfy requests from
     * the pool cache before falling back to the underlying allocator.
     *
     * @param alignment Required alignment in bytes
     * @param num_bytes Size of memory block to allocate
     * @return Pointer to allocated memory, or nullptr on failure
     *
     * **Algorithm**:
     * 1. Round size using configured size rounder
     * 2. Search pool for suitable cached buffer
     * 3. If found, remove from pool and return (pool hit)
     * 4. If not found, allocate from backend (pool miss)
     *
     * **Performance**:
     * - Pool hit: O(log n) - multimap search
     * - Pool miss: O(log n) + backend allocation time
     *
     * **Thread Safety**: Thread-safe with internal synchronization
     * **Statistics**: Updates allocation counters and hit/miss ratios
     */
    MEMORY_API void* allocate_raw(size_t alignment, size_t num_bytes) override;

    /**
     * @brief Deallocates memory by returning it to pool or backend.
     *
     * Returns memory to the pool for potential reuse, or deallocates
     * through backend allocator if pool is full.
     *
     * @param ptr Pointer to memory to deallocate
     *
     * **Algorithm**:
     * 1. Determine original allocation size
     * 2. If pool has space, add to pool cache
     * 3. If pool full, evict LRU item and add new item
     * 4. Update LRU ordering and statistics
     *
     * **Performance**: O(log n) for pool operations
     * **Thread Safety**: Thread-safe with internal synchronization
     * **LRU Management**: Automatically maintains LRU ordering
     */
    MEMORY_API void deallocate_raw(void* ptr) override;

    /**
     * @brief Allocates memory region from pool or backend allocator.
     *
     * Lower-level allocation interface that bypasses alignment handling
     * and works directly with rounded sizes for optimal pool utilization.
     *
     * @param num_bytes Size of memory region to allocate
     * @return Pointer to allocated memory, or nullptr on failure
     *
     * **Size Handling**: Uses configured size rounder for optimization
     * **Pool Strategy**: Searches pool first, then backend allocation
     * **Performance**: Optimized path for pool-aware allocations
     * **Thread Safety**: Thread-safe with internal synchronization
     */
    MEMORY_API void* Get(size_t num_bytes);

    /**
     * @brief Returns memory region to pool for reuse.
     *
     * Adds memory region to pool cache for future reuse, implementing
     * LRU eviction policy when pool reaches capacity limits.
     *
     * @param ptr Pointer to memory region to return
     * @param num_bytes Size of memory region
     *
     * **Important**: Accessing ptr after this call is undefined behavior
     * **LRU Policy**: Evicts least recently used item if pool is full
     * **Performance**: O(log n) for pool insertion and LRU management
     * **Thread Safety**: Thread-safe with internal synchronization
     * **Auto-Resize**: May trigger pool size increase if enabled
     */
    MEMORY_API void Put(void* ptr, size_t num_bytes);

    /**
     * @brief Clears all cached buffers from pool.
     *
     * Deallocates all buffers currently cached in the pool, effectively
     * resetting the pool to empty state. Useful for memory pressure
     * situations or explicit cleanup.
     *
     * **Effect**: Pool becomes empty, all cached buffers deallocated
     * **Performance**: O(n) where n is current pool size
     * **Thread Safety**: Thread-safe with internal synchronization
     * **Use Cases**: Memory pressure relief, explicit cleanup, testing
     */
    MEMORY_API void Clear();

    // The following accessors permit monitoring the effectiveness of
    // the pool at avoiding repeated malloc/frees on the underlying
    // allocator.  Read locks are not taken on the theory that value
    // consistency with other threads is not important.

    // Number of Get() requests satisfied from pool.
    int64_t get_from_pool_count() const NO_THREAD_SAFETY_ANALYSIS { return get_from_pool_count_; }
    // Number of Put() requests.
    int64_t put_count() const NO_THREAD_SAFETY_ANALYSIS { return put_count_; }
    // Number of Get() requests requiring a fresh allocation.
    int64_t allocated_count() const NO_THREAD_SAFETY_ANALYSIS { return allocated_count_; }
    // Number of pool evictions.
    int64_t evicted_count() const NO_THREAD_SAFETY_ANALYSIS { return evicted_count_; }
    // Current size limit.
    size_t size_limit() const NO_THREAD_SAFETY_ANALYSIS { return pool_size_limit_; }

    allocator_memory_enum GetMemoryType() const noexcept override
    {
        return allocator_->GetMemoryType();
    }

private:
    struct PtrRecord
    {
        void*      ptr;
        size_t     num_bytes;
        PtrRecord* prev;
        PtrRecord* next;
    };

    // Remove "pr" from the double-linked LRU list.
    void RemoveFromList(PtrRecord* ptr) MEMORY_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

    // Add "pr" to the head of the double-linked LRU list.
    void AddToList(PtrRecord* pr) MEMORY_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

    // Delete the least recently used record.
    void EvictOne() MEMORY_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

    const std::string                             name_;
    const bool                                    has_size_limit_;
    const bool                                    auto_resize_;
    size_t                                        pool_size_limit_;
    std::unique_ptr<memory::sub_allocator>      allocator_;
    std::unique_ptr<round_up_interface>           size_rounder_;
    std::mutex                                    mutex_;
    std::multimap<const size_t, PtrRecord*> pool_ MEMORY_GUARDED_BY(mutex_);
    PtrRecord* lru_head_                          MEMORY_GUARDED_BY(mutex_) = nullptr;
    PtrRecord* lru_tail_                          MEMORY_GUARDED_BY(mutex_) = nullptr;
    int64_t get_from_pool_count_                  MEMORY_GUARDED_BY(mutex_) = 0;
    int64_t put_count_                            MEMORY_GUARDED_BY(mutex_) = 0;
    int64_t allocated_count_                      MEMORY_GUARDED_BY(mutex_) = 0;
    int64_t evicted_count_                        MEMORY_GUARDED_BY(mutex_) = 0;
};

// Do-nothing rounder. Passes through sizes unchanged.
class NoopRounder : public round_up_interface
{
public:
    size_t RoundUp(size_t num_bytes) override { return num_bytes; }
};

// Power of 2 rounder: rounds up to nearest power of 2 size.
class Pow2Rounder : public round_up_interface
{
public:
    size_t RoundUp(size_t num_bytes) override { return 1uLL << Log2Ceiling64(num_bytes); }
};

class MEMORY_VISIBILITY basic_cpu_allocator : public sub_allocator
{
public:
    basic_cpu_allocator(
        int                         numa_node,
        const std::vector<Visitor>& alloc_visitors,
        const std::vector<Visitor>& free_visitors)
        : sub_allocator(alloc_visitors, free_visitors), numa_node_(numa_node)
    {
    }

    MEMORY_API ~basic_cpu_allocator() override;

    MEMORY_API void* Alloc(size_t alignment, size_t num_bytes, size_t* bytes_received) override;

    MEMORY_API void Free(void* ptr, size_t num_bytes) override;

    bool SupportsCoalescing() const override { return false; }

    allocator_memory_enum GetMemoryType() const noexcept override
    {
        return allocator_memory_enum::HOST_PAGEABLE;
    }

private:
    int numa_node_;

    basic_cpu_allocator(const basic_cpu_allocator&) = delete;
    void operator=(const basic_cpu_allocator&)      = delete;
};
}  // namespace memory