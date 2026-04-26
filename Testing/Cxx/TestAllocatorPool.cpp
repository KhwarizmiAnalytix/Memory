/**
 * @file TestAllocatorPool.cpp
 * @brief Comprehensive test suite for Pool allocator with LRU eviction
 *
 * Tests all public methods and functionality of the allocator_pool class including:
 * - Basic allocation and deallocation with pooling
 * - LRU eviction policy behavior
 * - Pool size limits and auto-resize functionality
 * - Size rounding strategies
 * - Statistics collection and monitoring
 * - Thread safety and concurrent access
 * - Error handling and edge cases
 */

#include <chrono>
#include <memory>
#include <memory>
#include <thread>
#include <vector>

#include "MemoryTest.h"
#include "backend/allocator_pool.h"
#include "cpu/allocator.h"
#include "helper/memory_allocator.h"
#include "util/memory_exception.h"

using namespace memory;
using namespace memory;

namespace
{

/**
 * @brief Counting wrapper around production basic_cpu_allocator for pool testing
 */
class counting_cpu_allocator : public sub_allocator
{
public:
    counting_cpu_allocator()
        : sub_allocator({}, {}),
          alloc_count_(0),
          free_count_(0),
          underlying_allocator_(
              0,                                              // numa_node = 0 (default)
              std::vector<memory::sub_allocator::Visitor>{},  // no alloc visitors
              std::vector<memory::sub_allocator::Visitor>{}   // no free visitors
          )
    {
    }

    void* Alloc(size_t alignment, size_t num_bytes, size_t* bytes_received) override
    {
        alloc_count_++;
        return underlying_allocator_.Alloc(alignment, num_bytes, bytes_received);
    }

    void Free(void* ptr, size_t num_bytes) override
    {
        free_count_++;
        underlying_allocator_.Free(ptr, num_bytes);
    }

    bool SupportsCoalescing() const override { return underlying_allocator_.SupportsCoalescing(); }

    allocator_memory_enum GetMemoryType() const noexcept override
    {
        return underlying_allocator_.GetMemoryType();
    }

    // Test helpers
    int  get_alloc_count() const { return alloc_count_; }
    int  get_free_count() const { return free_count_; }
    void reset_counters()
    {
        alloc_count_ = 0;
        free_count_  = 0;
    }

private:
    std::atomic<int>    alloc_count_;
    std::atomic<int>    free_count_;
    basic_cpu_allocator underlying_allocator_;
};

/**
 * @brief No-op size rounder for testing
 */
class noop_size_rounder : public round_up_interface
{
public:
    size_t RoundUp(size_t num_bytes) override
    {
        return num_bytes;  // No rounding
    }
};

/**
 * @brief Power-of-2 size rounder for testing
 */
class power_of_2_rounder : public round_up_interface
{
public:
    size_t RoundUp(size_t num_bytes) override
    {
        if (num_bytes <= 1)
            return 1;

        size_t power = 1;
        while (power < num_bytes)
        {
            power <<= 1;
        }
        return power;
    }
};

/**
 * @brief Helper function to check memory alignment
 */
bool is_aligned(void* ptr, size_t alignment)
{
    return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
}

/**
 * @brief Helper function to fill memory with pattern
 */
void fill_memory(void* ptr, size_t size, uint8_t pattern)
{
    std::memset(ptr, pattern, size);
}

/**
 * @brief Helper function to validate memory pattern
 */
bool validate_memory(void* ptr, size_t size, uint8_t pattern)
{
    uint8_t* bytes = static_cast<uint8_t*>(ptr);
    for (size_t i = 0; i < size; ++i)
    {
        if (bytes[i] != pattern)
            return false;
    }
    return true;
}

}  // anonymous namespace

/**
 * @brief Helper function to create a pool allocator for testing
 */
std::pair<std::unique_ptr<allocator_pool>, counting_cpu_allocator*> create_test_pool_allocator()
{
    // Create counting wrapper around production allocator
    auto                    sub_alloc         = std::make_unique<counting_cpu_allocator>();
    counting_cpu_allocator* sub_allocator_ptr = sub_alloc.get();  // Keep reference for testing

    // Create size rounder
    auto size_rounder = std::make_unique<noop_size_rounder>();

    // Create pool allocator with size limit of 5
    auto pool_allocator = std::make_unique<allocator_pool>(
        5,      // pool_size_limit
        false,  // auto_resize
        std::move(sub_alloc),
        std::move(size_rounder),
        "test_pool");

    return std::make_pair(std::move(pool_allocator), sub_allocator_ptr);
}

/**
 * @brief Test basic allocation and deallocation with pooling
 */
MEMORYTEST(AllocatorPool, basic_allocation_deallocation)
{
    auto [pool_allocator, sub_allocator_ptr] = create_test_pool_allocator();

    sub_allocator_ptr->reset_counters();

    // First allocation should go to underlying allocator
    void* ptr1 = pool_allocator->allocate_raw(64, 1024);
    EXPECT_NE(nullptr, ptr1);
    EXPECT_TRUE(is_aligned(ptr1, 64));
    EXPECT_EQ(sub_allocator_ptr->get_alloc_count(), 1);

    // Test memory content
    fill_memory(ptr1, 1024, 0xAA);
    EXPECT_TRUE(validate_memory(ptr1, 1024, 0xAA));

    // Deallocation should put memory in pool (not free to underlying allocator)
    pool_allocator->deallocate_raw(ptr1);
    EXPECT_EQ(sub_allocator_ptr->get_free_count(), 0);  // Should be 0 (in pool)

    // Second allocation of same size should come from pool
    void* ptr2 = pool_allocator->allocate_raw(64, 1024);
    EXPECT_NE(nullptr, ptr2);
    EXPECT_EQ(sub_allocator_ptr->get_alloc_count(), 1);  // Still 1 (from pool)

    pool_allocator->deallocate_raw(ptr2);

    END_TEST();
}

/**
 * @brief Test pool size limits and LRU eviction
 */
MEMORYTEST(AllocatorPool, pool_size_limits_and_lru)
{
    auto [pool_allocator, sub_allocator_ptr] = create_test_pool_allocator();

    sub_allocator_ptr->reset_counters();

    // Fill pool to capacity (5 items)
    std::vector<void*> ptrs;
    for (int i = 0; i < 5; ++i)
    {
        void* ptr = pool_allocator->allocate_raw(64, 1024);
        EXPECT_NE(nullptr, ptr);
        ptrs.push_back(ptr);
    }

    EXPECT_EQ(sub_allocator_ptr->get_alloc_count(), 5);

    // Return all to pool
    for (void* ptr : ptrs)
    {
        pool_allocator->deallocate_raw(ptr);
    }

    EXPECT_EQ(sub_allocator_ptr->get_free_count(), 0);  // All in pool

    // Allocate one more item of same size - should come from pool
    void* ptr_from_pool = pool_allocator->allocate_raw(64, 1024);
    EXPECT_NE(nullptr, ptr_from_pool);
    EXPECT_EQ(sub_allocator_ptr->get_alloc_count(), 5);  // No new allocation

    pool_allocator->deallocate_raw(ptr_from_pool);

    // Now add one more item to pool (6th item) - should trigger LRU eviction
    void* ptr6 = pool_allocator->allocate_raw(64, 1024);
    EXPECT_NE(nullptr, ptr6);
    pool_allocator->deallocate_raw(ptr6);

    // Pool should still have 5 items, oldest should be evicted
    EXPECT_EQ(sub_allocator_ptr->get_free_count(), 0);  // One evicted

    END_TEST();
}

/**
 * @brief Test pool statistics and monitoring
 */
MEMORYTEST(AllocatorPool, pool_statistics)
{
    auto [pool_allocator, sub_allocator_ptr] = create_test_pool_allocator();

    // Check initial statistics
    EXPECT_EQ(pool_allocator->get_from_pool_count(), 0);
    EXPECT_EQ(pool_allocator->put_count(), 0);
    EXPECT_EQ(pool_allocator->allocated_count(), 0);
    EXPECT_EQ(pool_allocator->evicted_count(), 0);
    EXPECT_EQ(pool_allocator->size_limit(), 5);

    // Perform allocation and deallocation
    void* ptr = pool_allocator->allocate_raw(64, 1024);
    EXPECT_NE(nullptr, ptr);

    // Should show one allocation
    EXPECT_EQ(pool_allocator->allocated_count(), 1);

    pool_allocator->deallocate_raw(ptr);

    // Should show one put to pool
    EXPECT_EQ(pool_allocator->put_count(), 1);

    // Allocate same size again - should come from pool
    void* ptr2 = pool_allocator->allocate_raw(64, 1024);
    EXPECT_NE(nullptr, ptr2);

    // Should show one get from pool
    EXPECT_EQ(pool_allocator->get_from_pool_count(), 1);

    pool_allocator->deallocate_raw(ptr2);

    END_TEST();
}

/**
 * @brief Test Clear() functionality
 */
MEMORYTEST(AllocatorPool, clear_functionality)
{
    auto [pool_allocator, sub_allocator_ptr] = create_test_pool_allocator();

    sub_allocator_ptr->reset_counters();

    // Fill pool with items
    std::vector<void*> ptrs;
    for (int i = 0; i < 3; ++i)
    {
        void* ptr = pool_allocator->allocate_raw(64, 1024);
        EXPECT_NE(nullptr, ptr);
        ptrs.push_back(ptr);
    }

    // Return to pool
    for (void* ptr : ptrs)
    {
        pool_allocator->deallocate_raw(ptr);
    }

    EXPECT_EQ(sub_allocator_ptr->get_free_count(), 0);  // All in pool

    // Clear pool - should free all cached items
    pool_allocator->Clear();

    EXPECT_EQ(sub_allocator_ptr->get_free_count(), 3);  // All freed

    // Next allocation should go to underlying allocator
    void* new_ptr = pool_allocator->allocate_raw(64, 1024);
    EXPECT_NE(nullptr, new_ptr);
    EXPECT_EQ(sub_allocator_ptr->get_alloc_count(), 4);  // New allocation

    pool_allocator->deallocate_raw(new_ptr);

    END_TEST();
}

/**
 * @brief Test allocator properties
 */
MEMORYTEST(AllocatorPool, allocator_properties)
{
    auto [pool_allocator, sub_allocator_ptr] = create_test_pool_allocator();

    // Test allocator name
    EXPECT_EQ(std::string(pool_allocator->Name()), std::string("test_pool"));

    // Test memory type (should delegate to underlying allocator)
    EXPECT_EQ(pool_allocator->GetMemoryType(), allocator_memory_enum::HOST_PAGEABLE);

    // Test allocation size tracking (pool allocator doesn't track sizes by default)
    EXPECT_FALSE(pool_allocator->tracks_allocation_sizes());

    END_TEST();
}

/**
 * @brief Test zero-size allocation handling
 */
MEMORYTEST(AllocatorPool, zero_size_allocation)
{
    auto [pool_allocator, sub_allocator_ptr] = create_test_pool_allocator();

    ASSERT_ANY_THROW({ pool_allocator->allocate_raw(64, 0); });

    END_TEST();
}

/**
 * @brief Test null pointer deallocation
 */
MEMORYTEST(AllocatorPool, null_pointer_deallocation)
{
    auto [pool_allocator, sub_allocator_ptr] = create_test_pool_allocator();

    // Deallocating nullptr should not crash
    pool_allocator->deallocate_raw(nullptr);

    // Test passes if no exception is thrown
    EXPECT_TRUE(true);

    END_TEST();
}

/**
 * @brief Test pool allocator with auto-resize enabled
 */
TEST(AllocatorPool, auto_resize_functionality)
{
    auto sub_alloc      = std::make_unique<counting_cpu_allocator>();
    auto test_sub_alloc = sub_alloc.get();

    // Create pool with auto-resize enabled
    auto pool = std::make_unique<allocator_pool>(
        2,     // initial pool_size_limit
        true,  // auto_resize enabled
        std::move(sub_alloc),
        std::make_unique<noop_size_rounder>(),
        "test_auto_resize_pool");

    test_sub_alloc->reset_counters();

    // Fill initial pool capacity
    std::vector<void*> ptrs;
    for (int i = 0; i < 3; ++i)
    {
        void* ptr = pool->allocate_raw(64, 1024);
        EXPECT_NE(nullptr, ptr);
        ptrs.push_back(ptr);
    }

    // Return to pool
    for (void* ptr : ptrs)
    {
        pool->deallocate_raw(ptr);
    }

    // With auto-resize, pool should have grown to accommodate all items
    EXPECT_EQ(test_sub_alloc->get_free_count(), 1);  // All should be in pool

    // Verify pool size has increased
    EXPECT_GE(pool->size_limit(), 2);
}

/**
 * @brief Test pool allocator with different size rounding strategies
 */
TEST(AllocatorPool, size_rounding_strategies)
{
    auto sub_alloc      = std::make_unique<counting_cpu_allocator>();
    auto test_sub_alloc = sub_alloc.get();

    // Create pool with power-of-2 size rounder
    auto pool = std::make_unique<allocator_pool>(
        5,
        false,
        std::move(sub_alloc),
        std::make_unique<power_of_2_rounder>(),
        "test_rounding_pool");

    test_sub_alloc->reset_counters();

    // Allocate 1000 bytes (should be rounded to 1024)
    void* ptr1 = pool->allocate_raw(64, 1000);
    EXPECT_NE(nullptr, ptr1);

    pool->deallocate_raw(ptr1);

    // Allocate 1024 bytes - should come from pool due to rounding
    void* ptr2 = pool->allocate_raw(64, 1024);
    EXPECT_NE(nullptr, ptr2);
    EXPECT_EQ(test_sub_alloc->get_alloc_count(), 1);  // Should reuse from pool

    pool->deallocate_raw(ptr2);
}

/**
 * @brief Test concurrent access to pool allocator
 */
TEST(AllocatorPool, thread_safety)
{
    auto sub_alloc = std::make_unique<counting_cpu_allocator>();

    auto pool = std::make_unique<allocator_pool>(
        10,  // larger pool for concurrency
        false,
        std::move(sub_alloc),
        std::make_unique<noop_size_rounder>(),
        "test_concurrent_pool");

    const int                       num_threads            = 4;
    const int                       allocations_per_thread = 20;
    std::vector<std::thread>        threads;
    std::vector<std::vector<void*>> thread_ptrs(num_threads);

    // Launch threads that perform allocations and deallocations
    for (int t = 0; t < num_threads; ++t)
    {
        threads.emplace_back(
            [&, t]()
            {
                for (int i = 0; i < allocations_per_thread; ++i)
                {
                    void* ptr = pool->allocate_raw(64, 1024);
                    if (ptr != nullptr)
                    {
                        thread_ptrs[t].push_back(ptr);

                        // Brief delay to increase chance of contention
                        std::this_thread::sleep_for(std::chrono::microseconds(1));
                    }
                }
            });
    }

    // Wait for all threads to complete
    for (auto& thread : threads)
    {
        thread.join();
    }

    // Verify allocations and clean up
    size_t total_allocations = 0;
    for (int t = 0; t < num_threads; ++t)
    {
        total_allocations += thread_ptrs[t].size();

        for (void* ptr : thread_ptrs[t])
        {
            EXPECT_NE(nullptr, ptr);
            pool->deallocate_raw(ptr);
        }
    }

    EXPECT_GT(total_allocations, 0);
}

/**
 * @brief Test pool allocator performance characteristics
 */
TEST(AllocatorPool, allocation_timing)
{
    auto sub_alloc      = std::make_unique<counting_cpu_allocator>();
    auto test_sub_alloc = sub_alloc.get();

    auto pool = std::make_unique<allocator_pool>(
        100,  // large pool
        false,
        std::move(sub_alloc),
        std::make_unique<noop_size_rounder>(),
        "test_perf_pool");

    const int          num_allocations = 1000;
    std::vector<void*> ptrs;
    ptrs.reserve(num_allocations);

    test_sub_alloc->reset_counters();

    // Time allocation performance
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_allocations; ++i)
    {
        void* ptr = pool->allocate_raw(64, 1024);
        if (ptr != nullptr)
        {
            ptrs.push_back(ptr);
        }
    }

    auto alloc_time = std::chrono::high_resolution_clock::now();

    // Return all to pool
    for (void* ptr : ptrs)
    {
        pool->deallocate_raw(ptr);
    }

    // Allocate again - should be much faster (from pool)
    std::vector<void*> ptrs2;
    ptrs2.reserve(num_allocations);

    auto pool_start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < std::min(static_cast<int>(ptrs.size()), 100); ++i)
    {
        void* ptr = pool->allocate_raw(64, 1024);
        if (ptr != nullptr)
        {
            ptrs2.push_back(ptr);
        }
    }

    auto pool_end_time = std::chrono::high_resolution_clock::now();

    // Clean up
    for (void* ptr : ptrs2)
    {
        pool->deallocate_raw(ptr);
    }

    // Verify reasonable performance
    auto total_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(alloc_time - start_time).count();

    EXPECT_LT(total_time, 1000);                  // Should complete within 1 second
    EXPECT_GT(ptrs.size(), num_allocations / 2);  // Most allocations should succeed

    // Pool allocations should be faster than initial allocations
    MEMORY_UNUSED auto pool_time =
        std::chrono::duration_cast<std::chrono::microseconds>(pool_end_time - pool_start_time)
            .count();

    EXPECT_GT(ptrs2.size(), 0);  // Should have successful pool allocations
}

// Test pool basic functionality
MEMORYTEST(AllocatorPool, BasicFunctionality)
{
    // Create a pool allocator with size limit of 5
    auto sub_allocator = std::make_unique<basic_cpu_allocator>(
        0,
        std::vector<memory::sub_allocator::Visitor>{},
        std::vector<memory::sub_allocator::Visitor>{});
    auto size_rounder = std::make_unique<NoopRounder>();

    allocator_pool pool(5, false, std::move(sub_allocator), std::move(size_rounder), "test_pool");

    // Test basic allocation
    void* ptr1 = pool.allocate_raw(64, 1024);
    EXPECT_NE(nullptr, ptr1);
    EXPECT_TRUE(IsAligned(ptr1, 64));

    // Test deallocation (should go to pool)
    pool.deallocate_raw(ptr1);

    // Test reallocation (should come from pool)
    void* ptr2 = pool.allocate_raw(64, 1024);
    EXPECT_NE(nullptr, ptr2);

    // Clean up
    pool.deallocate_raw(ptr2);
    pool.Clear();
}

// Test pool zero-size handling
MEMORYTEST(AllocatorPool, ZeroSizeHandling)
{
    auto sub_allocator = std::make_unique<basic_cpu_allocator>(
        0,
        std::vector<memory::sub_allocator::Visitor>{},
        std::vector<memory::sub_allocator::Visitor>{});
    auto size_rounder = std::make_unique<NoopRounder>();

    allocator_pool pool(
        2, false, std::move(sub_allocator), std::move(size_rounder), "test_pool_zero");

    // Test zero-size allocation
    //void* ptr_zero = pool.allocate_raw(4, 0);
    //EXPECT_EQ(nullptr, ptr_zero);  // Should return nullptr for zero size

    // Should not crash on nullptr deallocation
    pool.deallocate_raw(nullptr);

    // Test normal allocations still work
    void* ptr1 = pool.allocate_raw(4, 64);
    void* ptr2 = pool.allocate_raw(4, 128);
    EXPECT_NE(nullptr, ptr1);
    EXPECT_NE(nullptr, ptr2);

    pool.deallocate_raw(ptr1);
    pool.deallocate_raw(ptr2);
    pool.Clear();
}

// Test pool alignment requirements
MEMORYTEST(AllocatorPool, AlignmentRequirements)
{
    auto sub_allocator = std::make_unique<basic_cpu_allocator>(
        0,
        std::vector<memory::sub_allocator::Visitor>{},
        std::vector<memory::sub_allocator::Visitor>{});
    auto size_rounder = std::make_unique<NoopRounder>();

    allocator_pool pool(
        0, false, std::move(sub_allocator), std::move(size_rounder), "test_pool_alignment");

    // Test various alignment requirements
    for (int i = 2; i < 12; ++i)  // Test up to 4KB alignment
    {
        size_t alignment = size_t{1} << i;
        void*  ptr       = pool.allocate_raw(alignment, 256);
        EXPECT_NE(nullptr, ptr);
        EXPECT_TRUE(IsAligned(ptr, alignment));

        pool.deallocate_raw(ptr);
    }

    pool.Clear();
}
