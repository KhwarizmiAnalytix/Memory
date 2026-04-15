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

#include <cstdint>      // for uint8_t, uint64_t, uintptr_t, SIZE_MAX, UINT64_MAX, int64_t
#include <cstdlib>      // for size_t, free, malloc
#include <cstring>      // for memset
#include <functional>   // for function, _Func_class
#include <limits>       // for numeric_limits
#include <memory>       // for make_unique, unique_ptr, _Simple_types, allocator
#include <optional>     // for optional
#include <string>       // for char_traits, operator<<, string, operator+, to_string, basic...
#include <type_traits>  // for is_trivial
#include <utility>      // for move, min, max, max_element, min_element
#include <vector>       // for vector, _Vector_iterator, _Vector_const_iterator

#include "backend/allocator_bfc.h"  // for allocator_bfc
#include "backend/allocator_pool.h"  // for basic_cpu_allocator, allocator_pool, NoopRounder, round_up_i...
#include "backend/allocator_tracking.h"  // for allocator_tracking, enhanced_alloc_record, tracking_log_level
#include "common/pointer.h"              // for make_ptr_unique_mutable
#include "cpu/allocator.h"  // for sub_allocator, allocation_attributes, Allocator, allocator_m...

#if PROJECT_HAS_CUDA || PROJECT_HAS_HIP
#include "cpu/allocator_device.h"  // for allocator_device
#endif

#include "MemoryTest.h"               // for MEMORYTEST_CALL, MEMORYTEST, END_TEST, MEMORYTEST
#include "helper/memory_allocator.h"  // for free, allocate
#include "helper/process_state.h"     // for process_state
#include "profiler/unified_memory_stats.h"  // for atomic_timing_stats, unified_resource_stats, memory_fragment...

using namespace memory;
using namespace memory;

// ============================================================================
// ALLOCATOR_DEVICE TESTS
// ============================================================================
#if PROJECT_HAS_CUDA || PROJECT_HAS_HIP
MEMORYTEST(AllocatorDevice, basic_allocation)
{
    auto allocator = std::make_unique<allocator_device>();

    // Test zero allocation
    ASSERT_ANY_THROW({ allocator->allocate_raw(64, 0); });

    // Test small allocation
    void* ptr_small = allocator->allocate_raw(64, 1024);
    EXPECT_NE(nullptr, ptr_small);

    // Verify memory is writable
    if (ptr_small)
    {
        std::memset(ptr_small, 0xAA, 1024);
        EXPECT_EQ(static_cast<unsigned char*>(ptr_small)[0], 0xAA);
        EXPECT_EQ(static_cast<unsigned char*>(ptr_small)[1023], 0xAA);
        allocator->deallocate_raw(ptr_small);
    }

    // Test large allocation
    void* ptr_large = allocator->allocate_raw(64, 1024ULL);  // 1MB
    EXPECT_NE(nullptr, ptr_large);

    if (ptr_large)
    {
        // Verify memory is writable
        std::memset(ptr_large, 0x55, 1024ULL);
        EXPECT_EQ(static_cast<unsigned char*>(ptr_large)[0], 0x55);
        EXPECT_EQ(static_cast<unsigned char*>(ptr_large)[1024ULL - 1], 0x55);
        allocator->deallocate_raw(ptr_large);
    }

    END_TEST();
}

MEMORYTEST(AllocatorDevice, interface)
{
    auto allocator = std::make_unique<allocator_device>();

    // Test Name() method
    std::string name = allocator->Name();
    EXPECT_EQ("allocator_device", name);

    // Test GetMemoryType() method
    allocator_memory_enum memory_type = allocator->GetMemoryType();
    EXPECT_EQ(allocator_memory_enum::HOST_PINNED, memory_type);

    // Test tracks_allocation_sizes() - should return false by default
    EXPECT_FALSE(allocator->tracks_allocation_sizes());

    // Test AllocatesOpaqueHandle() - should return false
    EXPECT_FALSE(allocator->AllocatesOpaqueHandle());

    // Test GetStats() - should return nullopt by default
    auto stats = allocator->GetStats();
    EXPECT_FALSE(stats.has_value());

    // Test ClearStats() - should return false by default
    EXPECT_FALSE(allocator->ClearStats());

    END_TEST();
}

MEMORYTEST(AllocatorDevice, memory_type)
{
    auto allocator = std::make_unique<allocator_device>();

    // Verify memory type is HOST_PINNED
    EXPECT_EQ(allocator_memory_enum::HOST_PINNED, allocator->GetMemoryType());

    END_TEST();
}

MEMORYTEST(AllocatorDevice, error_handling)
{
    auto allocator = std::make_unique<allocator_device>();

    // Test deallocate with nullptr (should not crash)
    allocator->deallocate_raw(nullptr);

    // Test zero allocation
    ASSERT_ANY_THROW({ allocator->allocate_raw(64, 0); });

    // Test very large allocation (may fail gracefully)
    ASSERT_ANY_THROW({ allocator->allocate_raw(64, SIZE_MAX); });

    END_TEST();
}

#endif

// Test allocator stress scenarios
MEMORYTEST(AllocatorTest, StressTest)
{
#if 0
    auto base_allocator = std::make_unique<basic_cpu_allocator>(
        0, std::vector<memory::sub_allocator::Visitor>{}, std::vector<memory::sub_allocator::Visitor>{});
    auto pool = std::make_unique<allocator_pool>(
        50,
        false,
        std::move(base_allocator),
        std::make_unique<NoopRounder>(),
        "stress_pool");

    const int    num_threads            = 4;
    const int    allocations_per_thread = 100;
    const size_t max_alloc_size         = 4096;

    std::vector<std::thread>        threads;
    std::vector<std::vector<void*>> thread_ptrs(num_threads);

    // Launch multiple threads doing allocations
    for (int t = 0; t < num_threads; ++t)
    {
        threads.emplace_back(
            [&, t]()
            {
                thread_ptrs[t].reserve(allocations_per_thread);

                for (int i = 0; i < allocations_per_thread; ++i)
                {
                    size_t size      = (i % max_alloc_size) + 1;
                    size_t alignment = size_t{1} << (i % 8);  // 1, 2, 4, 8, 16, 32, 64, 128

                    void* ptr = pool->allocate_raw(alignment, size);
                    if (ptr != nullptr)
                    {
                        // Test memory access
                        memset(ptr, size, static_cast<uint8_t>(t + i));
                        thread_ptrs[t].push_back(ptr);
                    }
                }
            });
    }

    // Wait for all threads to complete
    for (auto& thread : threads)
    {
        thread.join();
    }

    // Verify memory content and deallocate
    for (int t = 0; t < num_threads; ++t)
    {
        for (size_t i = 0; i < thread_ptrs[t].size(); ++i)
        {
            void*  ptr  = thread_ptrs[t][i];
            size_t size = (i % max_alloc_size) + 1;

            // Verify memory content
            EXPECT_TRUE(ValidateMemory(ptr, size, static_cast<uint8_t>(t + i)));

            pool->deallocate_raw(ptr);
        }
    }
#endif
}

// Test error handling and edge cases
MEMORYTEST(AllocatorTest, ErrorHandling)
{
    auto base_allocator = std::make_unique<basic_cpu_allocator>(
        0,
        std::vector<memory::sub_allocator::Visitor>{},
        std::vector<memory::sub_allocator::Visitor>{});
    auto pool = std::make_unique<allocator_pool>(
        10, false, std::move(base_allocator), std::make_unique<NoopRounder>(), "error_pool");

    // Test null pointer deallocation (should not crash)
    pool->deallocate_raw(nullptr);

    // Test invalid alignment (should handle gracefully)
    void* ptr_invalid_align = pool->allocate_raw(3, 1024);  // 3 is not power of 2
    // Implementation may round up alignment or handle differently
    if (ptr_invalid_align != nullptr)
    {
        pool->deallocate_raw(ptr_invalid_align);
    }

    // Test double deallocation protection (if implemented)
    void* ptr_normal = pool->allocate_raw(64, 1024);
    if (ptr_normal != nullptr)
    {
        pool->deallocate_raw(ptr_normal);
        // Second deallocation should be handled gracefully
        pool->deallocate_raw(ptr_normal);
    }
}

// Test memory leak detection
MEMORYTEST(AllocatorTest, MemoryLeakDetection)
{
    auto base_allocator = std::make_unique<basic_cpu_allocator>(
        0,
        std::vector<memory::sub_allocator::Visitor>{},
        std::vector<memory::sub_allocator::Visitor>{});
    auto pool = std::make_unique<allocator_pool>(
        10, false, std::move(base_allocator), std::make_unique<NoopRounder>(), "leak_pool");
    auto allocator_tracking = new memory::allocator_tracking(pool.get(), true);

    // Get initial statistics
    auto initial_stats = allocator_tracking->GetStats();

    // Allocate some memory
    std::vector<void*> ptrs;
    for (int i = 0; i < 10; ++i)
    {
        void* ptr = allocator_tracking->allocate_raw(64, 1024);
        EXPECT_NE(nullptr, ptr);
        ptrs.push_back(ptr);
    }

    // Check that allocations are tracked
    auto mid_stats = allocator_tracking->GetStats();
    if (initial_stats.has_value() && mid_stats.has_value())
    {
        EXPECT_GT(mid_stats->num_allocs, initial_stats->num_allocs);
        EXPECT_GT(mid_stats->bytes_in_use, initial_stats->bytes_in_use);
    }

    // Deallocate all memory
    for (void* ptr : ptrs)
    {
        allocator_tracking->deallocate_raw(ptr);
    }

    // Check that memory is properly freed
    auto final_stats = allocator_tracking->GetStats();
    if (initial_stats.has_value() && final_stats.has_value())
    {
        EXPECT_EQ(final_stats->bytes_in_use, initial_stats->bytes_in_use);
    }

    // Properly cleanup tracking allocator by releasing reference
    allocator_tracking->GetRecordsAndUnRef();
}

// Test allocator statistics and monitoring
MEMORYTEST(AllocatorTest, StatisticsAndMonitoring)
{
    auto base_allocator = std::make_unique<basic_cpu_allocator>(
        0,
        std::vector<memory::sub_allocator::Visitor>{},
        std::vector<memory::sub_allocator::Visitor>{});
    auto pool = std::make_unique<allocator_pool>(
        10, false, std::move(base_allocator), std::make_unique<NoopRounder>(), "stats_pool");
    auto allocator_tracking = new memory::allocator_tracking(pool.get(), true);

    // Test statistics collection
    auto stats = allocator_tracking->GetStats();
    if (stats.has_value())
    {
        EXPECT_GE(stats->num_allocs, 0);
        EXPECT_GE(stats->bytes_in_use, 0);
        EXPECT_GE(stats->peak_bytes_in_use, 0);
        EXPECT_GE(stats->largest_alloc_size, 0);
    }

    // Test allocation size tracking
    if (allocator_tracking->tracks_allocation_sizes())
    {
        void* ptr = allocator_tracking->allocate_raw(64, 2048);
        EXPECT_NE(nullptr, ptr);

        size_t requested_size = allocator_tracking->RequestedSize(ptr);
        size_t allocated_size = allocator_tracking->AllocatedSize(ptr);

        EXPECT_EQ(requested_size, 2048);
        EXPECT_GE(allocated_size, 2048);

        allocator_tracking->deallocate_raw(ptr);
    }

    // Test allocator name (allocator_tracking returns the underlying allocator's name)
    EXPECT_EQ(std::string(allocator_tracking->Name()), std::string("stats_pool"));

    // Properly cleanup tracking allocator by releasing reference
    allocator_tracking->GetRecordsAndUnRef();
}

// ============================================================================
// MEMORY PORT ABSTRACTION LAYER TESTS (mem.h)
// ============================================================================

// Test memory port basic functionality
MEMORYTEST(MemoryPortTest, BasicMemoryOperations)
{
    QUARISMA_LOG_INFO("Testing memory port basic operations...");

    // Test aligned malloc/free (these are available)
    void* ptr2 = cpu::memory_allocator::allocate(2048, 64);
    EXPECT_NE(nullptr, ptr2);
    EXPECT_TRUE(IsAligned(ptr2, 64));
    cpu::memory_allocator::free(ptr2);

    QUARISMA_LOG_INFO("Memory port basic operations tests completed successfully");
}

// Test memory port alignment requirements
MEMORYTEST(MemoryPortTest, AlignmentRequirements)
{
    QUARISMA_LOG_INFO("Testing memory port alignment requirements...");

    // Test various alignment values
    std::vector<int> alignments = {8, 16, 32, 64, 128, 256, 512, 1024};

    for (int alignment : alignments)
    {
        void* ptr = cpu::memory_allocator::allocate(1024, alignment);
        EXPECT_NE(nullptr, ptr);
        EXPECT_TRUE(IsAligned(ptr, alignment));

        cpu::memory_allocator::free(ptr);
    }

    QUARISMA_LOG_INFO("Memory port alignment requirements tests completed successfully");
}

// Test memory port edge cases
MEMORYTEST(MemoryPortTest, EdgeCases)
{
    QUARISMA_LOG_INFO("Testing memory port edge cases...");

    // Test null pointer free (should not crash)
    cpu::memory_allocator::free(nullptr);

    // Test zero-size allocation
    // fixme: ASSERT_ANY_THROW({ memory::cpu::memory_allocator::allocate(0, 64); });

    QUARISMA_LOG_INFO("Memory port edge cases tests completed successfully");
}

// Test memory port system information
MEMORYTEST(MemoryPortTest, SystemInformation)
{
    QUARISMA_LOG_INFO("Testing memory port system information...");

    // Test NUMA affinity constant
    EXPECT_EQ(memory::NUMANOAFFINITY, -1);

    QUARISMA_LOG_INFO("Memory port system information tests completed successfully");
}

// ============================================================================
// ALLOCATION ATTRIBUTES TESTS (allocator.h)
// ============================================================================

// Test allocation attributes construction and behavior
MEMORYTEST(AllocationAttributesTest, ConstructionAndBehavior)
{
    QUARISMA_LOG_INFO("Testing allocation attributes construction and behavior...");

    // Test default construction
    memory::allocation_attributes default_attrs;
    EXPECT_TRUE(default_attrs.retry_on_failure);
    EXPECT_FALSE(default_attrs.allocation_will_be_logged);
    EXPECT_EQ(nullptr, default_attrs.freed_by_func);

    // Test parameterized construction
    std::function<uint64_t()>     timing_func = []() { return 12345; };
    memory::allocation_attributes custom_attrs(true, true, &timing_func);
    EXPECT_TRUE(custom_attrs.retry_on_failure);
    EXPECT_TRUE(custom_attrs.allocation_will_be_logged);
    EXPECT_EQ(&timing_func, custom_attrs.freed_by_func);

    // Test move construction
    memory::allocation_attributes moved_attrs = std::move(custom_attrs);
    EXPECT_TRUE(moved_attrs.retry_on_failure);
    EXPECT_TRUE(moved_attrs.allocation_will_be_logged);
    EXPECT_EQ(&timing_func, moved_attrs.freed_by_func);

    // Test move assignment
    memory::allocation_attributes assigned_attrs;
    assigned_attrs = std::move(moved_attrs);
    EXPECT_TRUE(assigned_attrs.retry_on_failure);
    EXPECT_TRUE(assigned_attrs.allocation_will_be_logged);
    EXPECT_EQ(&timing_func, assigned_attrs.freed_by_func);

    QUARISMA_LOG_INFO(
        "Allocation attributes construction and behavior tests completed successfully");
}

// Test allocation attributes with timing constraints
MEMORYTEST(AllocationAttributesTest, TimingConstraints)
{
    QUARISMA_LOG_INFO("Testing allocation attributes timing constraints...");

    uint64_t                  counter     = 0;
    std::function<uint64_t()> timing_func = [&counter]() { return ++counter; };

    memory::allocation_attributes attrs(true, true, &timing_func);

    // Test timing function calls
    EXPECT_EQ(&timing_func, attrs.freed_by_func);

    if (attrs.freed_by_func)
    {
        uint64_t time1 = (*attrs.freed_by_func)();
        uint64_t time2 = (*attrs.freed_by_func)();

        EXPECT_EQ(time1, 1);
        EXPECT_EQ(time2, 2);
        EXPECT_GT(time2, time1);
    }

    QUARISMA_LOG_INFO("Allocation attributes timing constraints tests completed successfully");
}

// ============================================================================
// PROCESS STATE TESTS (process_state.h) - COMMENTED OUT DUE TO LINKING ISSUES
// ============================================================================
// Note: These tests are commented out due to linking issues with FLAGS_brain_gpu_record_mem_types
// and other dependencies that are not available in the current build configuration

// Test process state singleton functionality and thread safety
MEMORYTEST(ProcessStateTest, SingletonFunctionality)
{
    QUARISMA_LOG_INFO("Testing process state singleton functionality...");

    // Test singleton access
    memory::process_state* state1 = memory::process_state::singleton();
    memory::process_state* state2 = memory::process_state::singleton();

    EXPECT_NE(nullptr, state1);
    EXPECT_EQ(state1, state2);  // Should be the same instance

    state1->TestOnlyReset();

    QUARISMA_LOG_INFO("Process state singleton functionality tests completed successfully");
}

// Test CPU allocator retrieval and management
MEMORYTEST(ProcessStateTest, CPUAllocatorManagement)
{
    QUARISMA_LOG_INFO("Testing CPU allocator retrieval and management...");

    memory::process_state* state = memory::process_state::singleton();

    // Test CPU allocator access for different NUMA nodes
    memory::Allocator* cpu_alloc1 = state->GetCPUAllocator(0);
    memory::Allocator* cpu_alloc2 = state->GetCPUAllocator(memory::NUMANOAFFINITY);

    auto ptr = cpu_alloc1->allocate_raw(64, 1000);
    cpu_alloc1->deallocate_raw(ptr);

    EXPECT_NE(nullptr, cpu_alloc1);
    EXPECT_NE(nullptr, cpu_alloc2);

    // Test allocator consistency
    memory::Allocator* cpu_alloc3 = state->GetCPUAllocator(0);
    EXPECT_EQ(cpu_alloc1, cpu_alloc3);  // Should return same allocator for same node

    QUARISMA_LOG_INFO("CPU allocator retrieval and management tests completed successfully");
}
// Test NUMA enablement and affinity handling
MEMORYTEST(ProcessStateTest, NUMAHandling)
{
    QUARISMA_LOG_INFO("Testing NUMA enablement and affinity handling...");

    memory::process_state* state = memory::process_state::singleton();

    // Test NUMA enablement (should not crash)
    state->EnableNUMA();

    // Test allocator access after NUMA enablement
    memory::Allocator* numa_alloc = state->GetCPUAllocator(0);
    EXPECT_NE(nullptr, numa_alloc);

    // Test NUMANOAFFINITY handling
    memory::Allocator* no_affinity_alloc = state->GetCPUAllocator(memory::NUMANOAFFINITY);
    EXPECT_NE(nullptr, no_affinity_alloc);

    QUARISMA_LOG_INFO("NUMA enablement and affinity handling tests completed successfully");
}

// Test memory description and pointer type detection
MEMORYTEST(ProcessStateTest, MemoryDescription)
{
    QUARISMA_LOG_INFO("Testing memory description and pointer type detection...");

    memory::process_state* state = memory::process_state::singleton();

    // Test memory description for CPU pointer
    void* cpu_ptr = std::malloc(1024);
    EXPECT_NE(nullptr, cpu_ptr);

    auto mem_desc = state->PtrType(cpu_ptr);
    EXPECT_EQ(mem_desc.loc, memory::process_state::MemDesc::CPU);
    EXPECT_EQ(mem_desc.dev_index, 0);
    EXPECT_FALSE(mem_desc.gpu_registered);
    EXPECT_FALSE(mem_desc.nic_registered);

    // Test debug string generation
    std::string debug_str = mem_desc.debug_string();
    EXPECT_FALSE(debug_str.empty());

    // Test default MemDesc construction
    memory::process_state::MemDesc default_desc;
    EXPECT_EQ(default_desc.loc, memory::process_state::MemDesc::CPU);
    EXPECT_EQ(default_desc.dev_index, 0);
    EXPECT_FALSE(default_desc.gpu_registered);
    EXPECT_FALSE(default_desc.nic_registered);

    std::free(cpu_ptr);

    state->TestOnlyReset();
    QUARISMA_LOG_INFO("Memory description and pointer type detection tests completed successfully");
}

// Test visitor registration for allocation/deallocation callbacks
MEMORYTEST(ProcessStateTest, VisitorRegistration)
{
    QUARISMA_LOG_INFO("Testing visitor registration for allocation/deallocation callbacks...");

    memory::process_state* state = memory::process_state::singleton();

    // Test CPU allocator visitor registration
    bool alloc_visitor_called = false;
    bool free_visitor_called  = false;

    auto alloc_visitor =
        [&alloc_visitor_called](
            MEMORY_UNUSED void* ptr, MEMORY_UNUSED int index, MEMORY_UNUSED size_t num_bytes)
    { alloc_visitor_called = true; };

    auto free_visitor =
        [&free_visitor_called](
            MEMORY_UNUSED void* ptr, MEMORY_UNUSED int index, MEMORY_UNUSED size_t num_bytes)
    { free_visitor_called = true; };

    // Note: These must be called before GetCPUAllocator according to the API
    // In practice, we test that the calls don't crash
    state->AddCPUAllocVisitor(alloc_visitor);
    state->AddCPUFreeVisitor(free_visitor);

    memory::Allocator* cpu_alloc = state->GetCPUAllocator(0);

    auto ptr = cpu_alloc->allocate_raw(64, 1000);
    cpu_alloc->deallocate_raw(ptr);

    QUARISMA_LOG_INFO("Visitor registration tests completed successfully");
}

// ============================================================================
// TRACKING ALLOCATOR TESTS (allocator_tracking.h)
// ============================================================================

// Test memory allocation tracking and leak detection
MEMORYTEST(AllocatorTracking, AllocationTracking)
{
    QUARISMA_LOG_INFO("Testing memory allocation tracking and leak detection...");

    // Create underlying BFC allocator for testing
    auto sub_allocator = std::make_unique<basic_cpu_allocator>(
        0,
        std::vector<memory::sub_allocator::Visitor>{},
        std::vector<memory::sub_allocator::Visitor>{});

    allocator_bfc::Options opts;
    opts.allow_growth = false;
    memory::allocator_bfc underlying_alloc(
        std::move(sub_allocator), 1024ULL * 1024ULL, "test_tracking_bfc", opts);

    // Create tracking allocator with size tracking enabled (use pointer due to protected destructor)
    auto tracker = new memory::allocator_tracking(&underlying_alloc, true);

    // Test basic allocation tracking
    void* ptr1 = tracker->allocate_raw(64, 1024);
    EXPECT_NE(nullptr, ptr1);

    // Verify tracking capabilities
    EXPECT_TRUE(tracker->tracks_allocation_sizes());
    EXPECT_EQ(tracker->RequestedSize(ptr1), 1024);
    EXPECT_GE(tracker->AllocatedSize(ptr1), 1024);

    // Test allocation ID generation
    int64_t alloc_id = tracker->AllocationId(ptr1);
    EXPECT_GT(alloc_id, 0);

    // Test multiple allocations
    void* ptr2 = tracker->allocate_raw(32, 512);
    EXPECT_NE(nullptr, ptr2);
    EXPECT_NE(ptr1, ptr2);

    // Verify different allocation IDs
    int64_t alloc_id2 = tracker->AllocationId(ptr2);
    EXPECT_GT(alloc_id2, 0);
    EXPECT_NE(alloc_id, alloc_id2);

    // Test deallocation
    tracker->deallocate_raw(ptr1);
    tracker->deallocate_raw(ptr2);

    // Properly cleanup tracking allocator by releasing reference
    tracker->GetRecordsAndUnRef();

    QUARISMA_LOG_INFO("Memory allocation tracking and leak detection tests completed successfully");
}

// Test allocation statistics collection and reporting
MEMORYTEST(AllocatorTracking, StatisticsCollection)
{
    QUARISMA_LOG_INFO("Testing allocation statistics collection and reporting...");

    // Create underlying allocator
    auto sub_allocator = std::make_unique<basic_cpu_allocator>(
        0,
        std::vector<memory::sub_allocator::Visitor>{},
        std::vector<memory::sub_allocator::Visitor>{});

    allocator_bfc::Options opts;
    opts.allow_growth = false;
    memory::allocator_bfc underlying_alloc(
        std::move(sub_allocator), 1024ULL * 1024ULL, "test_stats_bfc", opts);

    // Create tracking allocator (use pointer due to protected destructor)
    auto tracker = new memory::allocator_tracking(&underlying_alloc, true);

    // Get initial statistics
    auto initial_stats = tracker->GetStats();
    EXPECT_TRUE(initial_stats.has_value());

    // Get initial sizes
    auto [total_initial, high_initial, current_initial] = tracker->GetSizes();
    EXPECT_EQ(total_initial, 0);
    EXPECT_EQ(high_initial, 0);
    EXPECT_EQ(current_initial, 0);

    // Perform allocations
    std::vector<void*> ptrs;
    for (int i = 0; i < 5; ++i)
    {
        void* ptr = tracker->allocate_raw(64, 1024 * (i + 1));
        EXPECT_NE(nullptr, ptr);
        ptrs.push_back(ptr);
    }

    // Check updated statistics
    auto [total_after, high_after, current_after] = tracker->GetSizes();
    EXPECT_GT(total_after, 0);
    EXPECT_GT(high_after, 0);
    EXPECT_GT(current_after, 0);

    // Test statistics reset
    bool reset_success = tracker->ClearStats();
    EXPECT_TRUE(reset_success);

    // Clean up allocations
    for (void* ptr : ptrs)
    {
        tracker->deallocate_raw(ptr);
    }

    // Properly cleanup tracking allocator by releasing reference
    tracker->GetRecordsAndUnRef();

    QUARISMA_LOG_INFO(
        "Allocation statistics collection and reporting tests completed successfully");
}

// Test memory usage monitoring and bounds checking
MEMORYTEST(AllocatorTracking, MemoryUsageMonitoring)
{
    QUARISMA_LOG_INFO("Testing memory usage monitoring and bounds checking...");

    // Create underlying allocator
    auto sub_allocator = std::make_unique<basic_cpu_allocator>(
        0,
        std::vector<memory::sub_allocator::Visitor>{},
        std::vector<memory::sub_allocator::Visitor>{});

    allocator_bfc::Options opts;
    opts.allow_growth = false;
    memory::allocator_bfc underlying_alloc(
        std::move(sub_allocator), 1024ULL * 1024ULL, "test_monitoring_bfc", opts);

    // Create tracking allocator (use pointer due to protected destructor)
    auto tracker = new memory::allocator_tracking(&underlying_alloc, true);

    // Test allocation records collection
    std::vector<void*> ptrs;
    for (int i = 0; i < 3; ++i)
    {
        void* ptr = tracker->allocate_raw(64, 512 * (i + 1));
        EXPECT_NE(nullptr, ptr);
        ptrs.push_back(ptr);
    }

    // Get current allocation records
    auto current_records = tracker->GetCurrentRecords();
    EXPECT_EQ(current_records.size(), 3);

    // Verify record contents
    for (const auto& record : current_records)
    {
        EXPECT_GT(record.alloc_bytes, 0);
        EXPECT_GT(record.alloc_micros, 0);
    }

    // Test high watermark tracking
    auto [total_bytes, high_watermark, current_bytes] = tracker->GetSizes();
    EXPECT_GT(high_watermark, 0);
    EXPECT_EQ(current_bytes, high_watermark);  // Should be at peak

    // Deallocate one allocation
    tracker->deallocate_raw(ptrs[0]);
    ptrs.erase(ptrs.begin());

    // Check that current usage decreased but high watermark remained
    auto [total_after, high_after, current_after] = tracker->GetSizes();
    EXPECT_EQ(high_after, high_watermark);    // High watermark should not decrease
    EXPECT_LT(current_after, current_bytes);  // Current usage should decrease

    // Clean up remaining allocations
    for (void* ptr : ptrs)
    {
        tracker->deallocate_raw(ptr);
    }

    // Properly cleanup tracking allocator by releasing reference
    tracker->GetRecordsAndUnRef();

    QUARISMA_LOG_INFO("Memory usage monitoring and bounds checking tests completed successfully");
}

// Test integration with underlying allocator implementations
MEMORYTEST(AllocatorTracking, UnderlyingAllocatorIntegration)
{
    QUARISMA_LOG_INFO("Testing integration with underlying allocator implementations...");

    // Test with BFC allocator
    auto sub_allocator = std::make_unique<basic_cpu_allocator>(
        0,
        std::vector<memory::sub_allocator::Visitor>{},
        std::vector<memory::sub_allocator::Visitor>{});

    allocator_bfc::Options opts;
    opts.allow_growth = false;
    memory::allocator_bfc bfc_alloc(
        std::move(sub_allocator), 1024ULL, "test_integration_bfc", opts);
    auto bfc_tracker = new memory::allocator_tracking(&bfc_alloc, true);

    // Test name delegation
    std::string tracker_name = bfc_tracker->Name();
    EXPECT_FALSE(tracker_name.empty());

    // Test memory type delegation
    auto memory_type = bfc_tracker->GetMemoryType();
    EXPECT_EQ(memory_type, bfc_alloc.GetMemoryType());

    // Test allocation through tracking wrapper
    void* ptr = bfc_tracker->allocate_raw(64, 1024);
    EXPECT_NE(nullptr, ptr);

    // Verify size tracking works
    EXPECT_TRUE(bfc_tracker->tracks_allocation_sizes());
    EXPECT_EQ(bfc_tracker->RequestedSize(ptr), 1024);
    EXPECT_GE(bfc_tracker->AllocatedSize(ptr), 1024);

    bfc_tracker->deallocate_raw(ptr);

    // Test with pool allocator
    auto pool_sub_allocator = std::make_unique<basic_cpu_allocator>(
        0,
        std::vector<memory::sub_allocator::Visitor>{},
        std::vector<memory::sub_allocator::Visitor>{});
    auto size_rounder = std::make_unique<NoopRounder>();

    memory::allocator_pool pool_alloc(
        10, false, std::move(pool_sub_allocator), std::move(size_rounder), "test_integration_pool");
    auto pool_tracker = new memory::allocator_tracking(&pool_alloc, false);  // No local tracking

    void* pool_ptr = pool_tracker->allocate_raw(64, 1024);
    EXPECT_NE(nullptr, pool_ptr);

    pool_tracker->deallocate_raw(pool_ptr);

    // Properly cleanup tracking allocator by releasing reference
    bfc_tracker->GetRecordsAndUnRef();
    pool_tracker->GetRecordsAndUnRef();

    QUARISMA_LOG_INFO(
        "Integration with underlying allocator implementations tests completed successfully");
}

// Test enhanced tracking functionality with comprehensive analytics
MEMORYTEST(AllocatorTracking, EnhancedTrackingAnalytics)
{
    QUARISMA_LOG_INFO("Testing enhanced tracking analytics and performance profiling...");

    // Create underlying allocator
    auto sub_allocator = std::make_unique<basic_cpu_allocator>(
        0,
        std::vector<memory::sub_allocator::Visitor>{},
        std::vector<memory::sub_allocator::Visitor>{});

    allocator_bfc::Options opts;
    opts.allow_growth = false;
    memory::allocator_bfc underlying_alloc(
        std::move(sub_allocator), 2 * 1024ULL * 1024ULL, "test_enhanced_bfc", opts);

    // Create tracking allocator with enhanced tracking enabled
    auto tracker =
        new memory::allocator_tracking(&underlying_alloc, true, true);  // Enhanced tracking enabled

    // Test enhanced timing statistics
    auto initial_timing = tracker->GetTimingStats();
    EXPECT_EQ(initial_timing.total_allocations, 0);
    EXPECT_EQ(initial_timing.total_deallocations, 0);

    // Perform multiple allocations to generate timing data
    std::vector<void*> ptrs;
    for (int i = 0; i < 10; ++i)
    {
        void* ptr = tracker->allocate_raw(64, 1024 * (i + 1));
        EXPECT_NE(nullptr, ptr);
        ptrs.push_back(ptr);
    }

    // Test timing statistics after allocations
    auto post_alloc_timing = tracker->GetTimingStats();
    EXPECT_EQ(post_alloc_timing.total_allocations, 10);
    // Timers can report 0 µs on very fast runs or coarse clocks (e.g. Bazel sandbox).
    EXPECT_GE(post_alloc_timing.total_alloc_time_us, 0);
    EXPECT_GE(post_alloc_timing.max_alloc_time_us, 0);
    EXPECT_LT(post_alloc_timing.min_alloc_time_us, UINT64_MAX);

    // Test average allocation time calculation
    double avg_alloc_time = post_alloc_timing.average_alloc_time_us();
    EXPECT_GE(avg_alloc_time, 0.0);

    // Test enhanced allocation records
    auto enhanced_records = tracker->GetEnhancedRecords();
    EXPECT_EQ(enhanced_records.size(), 10);

    // Verify enhanced record data
    for (size_t i = 0; i < enhanced_records.size(); ++i)
    {
        const auto& record = enhanced_records[i];
        EXPECT_GT(record.requested_bytes, 0);
        EXPECT_GE(record.alloc_bytes, record.requested_bytes);
        EXPECT_GT(record.allocation_id, 0);
        EXPECT_GE(record.alloc_duration_us, 0);
        EXPECT_GE(record.alloc_micros, 0);
        EXPECT_EQ(record.alignment, 64);
    }

    // Test fragmentation metrics
    auto frag_metrics = tracker->GetFragmentationMetrics();
    EXPECT_GE(frag_metrics.fragmentation_ratio, 0.0);
    EXPECT_LE(frag_metrics.fragmentation_ratio, 1.0);
    EXPECT_GE(frag_metrics.wasted_bytes, 0);
    EXPECT_GE(frag_metrics.largest_free_block, 0);

    // Test efficiency metrics
    auto [memory_efficiency, allocation_efficiency, overall_efficiency] =
        tracker->GetEfficiencyMetrics();
    EXPECT_GE(memory_efficiency, 0.0);
    EXPECT_LE(memory_efficiency, 1.0);
    EXPECT_GE(allocation_efficiency, 0.0);
    EXPECT_LE(allocation_efficiency, 1.0);
    EXPECT_GE(overall_efficiency, 0.0);
    EXPECT_LE(overall_efficiency, 1.0);

    // Deallocate memory and test deallocation timing
    for (void* ptr : ptrs)
    {
        tracker->deallocate_raw(ptr);
    }

    // Test timing statistics after deallocations
    auto post_dealloc_timing = tracker->GetTimingStats();
    EXPECT_EQ(post_dealloc_timing.total_deallocations, 10);
    EXPECT_GE(post_dealloc_timing.total_dealloc_time_us, 0);
    EXPECT_GE(post_dealloc_timing.max_dealloc_time_us, 0);
    EXPECT_LT(post_dealloc_timing.min_dealloc_time_us, UINT64_MAX);

    // Test average deallocation time calculation
    double avg_dealloc_time = post_dealloc_timing.average_dealloc_time_us();
    EXPECT_GE(avg_dealloc_time, 0.0);

    // Verify enhanced records have deallocation timing
    auto final_records = tracker->GetEnhancedRecords();
    for (const auto& record : final_records)
    {
        EXPECT_GE(record.dealloc_duration_us, 0);
    }

    // Properly cleanup tracking allocator by releasing reference
    tracker->GetRecordsAndUnRef();

    QUARISMA_LOG_INFO(
        "Enhanced tracking analytics and performance profiling tests completed successfully");
}

// Test logging levels and comprehensive reporting
MEMORYTEST(AllocatorTracking, LoggingAndReporting)
{
    QUARISMA_LOG_INFO("Testing logging levels and comprehensive reporting...");

    // Create underlying allocator
    auto sub_allocator = std::make_unique<basic_cpu_allocator>(
        0,
        std::vector<memory::sub_allocator::Visitor>{},
        std::vector<memory::sub_allocator::Visitor>{});

    allocator_bfc::Options opts;
    opts.allow_growth = false;
    memory::allocator_bfc underlying_alloc(
        std::move(sub_allocator), 1024ULL * 1024ULL, "test_logging_bfc", opts);

    // Create tracking allocator with enhanced tracking
    auto tracker = new memory::allocator_tracking(&underlying_alloc, true, true);

    // Test logging level configuration
    tracker->SetLoggingLevel(tracking_log_level::DEBUG_LEVEL);
    EXPECT_EQ(tracker->GetLoggingLevel(), tracking_log_level::DEBUG_LEVEL);

    tracker->SetLoggingLevel(tracking_log_level::INFO);
    EXPECT_EQ(tracker->GetLoggingLevel(), tracking_log_level::INFO);

    // Perform some allocations for report generation
    std::vector<void*> ptrs;
    for (int i = 0; i < 5; ++i)
    {
        void* ptr = tracker->allocate_raw(512 * (i + 1), 32);
        EXPECT_NE(nullptr, ptr);
        ptrs.push_back(ptr);
    }

    // Test comprehensive report generation
    std::string basic_report = tracker->GenerateReport(false);
    EXPECT_FALSE(basic_report.empty());
    EXPECT_NE(basic_report.find("Memory Allocation Tracking Report"), std::string::npos);
    EXPECT_NE(basic_report.find("Performance Statistics"), std::string::npos);
    EXPECT_NE(basic_report.find("Efficiency Metrics"), std::string::npos);

    // Test detailed report with allocations
    std::string detailed_report = tracker->GenerateReport(true);
    EXPECT_FALSE(detailed_report.empty());
    EXPECT_GT(detailed_report.length(), basic_report.length());
    EXPECT_NE(detailed_report.find("Recent Allocations"), std::string::npos);

    // Test timing statistics reset
    tracker->ResetTimingStats();
    auto reset_timing = tracker->GetTimingStats();
    EXPECT_EQ(reset_timing.total_allocations, 0);
    EXPECT_EQ(reset_timing.total_deallocations, 0);
    EXPECT_EQ(reset_timing.total_alloc_time_us, 0);
    EXPECT_EQ(reset_timing.total_dealloc_time_us, 0);

    // Clean up
    for (void* ptr : ptrs)
    {
        tracker->deallocate_raw(ptr);
    }

    // Properly cleanup tracking allocator by releasing reference
    tracker->GetRecordsAndUnRef();

    QUARISMA_LOG_INFO("Logging levels and comprehensive reporting tests completed successfully");
}

// ============================================================================
// MEMORY ALLOCATOR PERFORMANCE BENCHMARK
// ============================================================================

#ifndef DEBUG
// Structure to hold benchmark results for comparison
struct BenchmarkResult
{
    std::string allocator_name;
    size_t      allocation_size;
    size_t      iterations;
    double      total_time_us;
    double      avg_time_per_op_us;
    double      throughput_ops_sec;
    size_t      peak_memory_bytes;
    double      relative_performance;  // Relative to malloc baseline
};

MEMORYTEST(AllocatorBenchmark, PerformanceBenchmark)
{
    QUARISMA_LOG_INFO("Running Comprehensive Memory Allocator Performance Benchmark...");

    // Test parameters - multiple allocation sizes for comprehensive testing
    const std::vector<size_t> test_sizes = {
        64, 1024, 65536};                // Small, Medium, Large (reduced from 65536)
    const size_t num_iterations = 2000;  // Reduced from 2000 for faster execution

    std::vector<BenchmarkResult> all_results;

    for (size_t alloc_size : test_sizes)
    {
        QUARISMA_LOG_INFO("Testing allocation size: {}  bytes", std::to_string(alloc_size));

        std::vector<BenchmarkResult> size_results;

        // ========== BASELINE: Standard malloc/free ==========
        {
            std::vector<void*> ptrs;
            ptrs.reserve(num_iterations);

            auto start = std::chrono::high_resolution_clock::now();

            // Allocation phase
            for (size_t i = 0; i < num_iterations; ++i)
            {
                void* ptr = memory::cpu::memory_allocator::allocate(alloc_size);
                if (ptr)
                {
                    ptrs.push_back(ptr);
                    memset(ptr, 0xAA, std::min(alloc_size, size_t(64)));  // Touch memory
                }
            }

            // Deallocation phase
            for (void* ptr : ptrs)
            {
                memory::cpu::memory_allocator::free(ptr);
            }

            auto end      = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            BenchmarkResult result;
            result.allocator_name       = "malloc/free";
            result.allocation_size      = alloc_size;
            result.iterations           = num_iterations;
            result.total_time_us        = duration.count();
            result.avg_time_per_op_us   = duration.count() / (num_iterations * 2.0);
            result.throughput_ops_sec   = (num_iterations * 2.0) / (duration.count() / 1e6);
            result.peak_memory_bytes    = alloc_size * num_iterations;  // Approximate
            result.relative_performance = 1.0;                          // Baseline

            size_results.push_back(result);
            EXPECT_EQ(ptrs.size(), num_iterations);
        }

        // ========== BFC ALLOCATOR ==========
        {
            std::vector<sub_allocator::Visitor> empty_visitors;
            auto                                sub_allocator =
                std::make_unique<basic_cpu_allocator>(0, empty_visitors, empty_visitors);

            allocator_bfc::Options opts;
            opts.allow_growth = true;

            size_t total_memory = std::max(
                static_cast<size_t>(64ULL << 20),
                alloc_size * num_iterations * 2);  // Adjust based on test size
            auto bfc_alloc = std::make_unique<memory::allocator_bfc>(
                std::unique_ptr<memory::sub_allocator>(sub_allocator.release()),
                total_memory,
                "benchmark_bfc_" + std::to_string(alloc_size),
                opts);

            std::vector<void*> ptrs;
            ptrs.reserve(num_iterations);

            auto start = std::chrono::high_resolution_clock::now();

            // Allocation phase
            for (size_t i = 0; i < num_iterations; ++i)
            {
                void* ptr = bfc_alloc->allocate_raw(64, alloc_size);
                if (ptr)
                {
                    ptrs.push_back(ptr);
                    memset(ptr, 0xBB, std::min(alloc_size, size_t(64)));  // Touch memory
                }
            }

            // Deallocation phase
            for (void* ptr : ptrs)
            {
                bfc_alloc->deallocate_raw(ptr);
            }

            auto end      = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            BenchmarkResult result;
            result.allocator_name     = "BFC Allocator";
            result.allocation_size    = alloc_size;
            result.iterations         = num_iterations;
            result.total_time_us      = duration.count();
            result.avg_time_per_op_us = duration.count() / (num_iterations * 2.0);
            result.throughput_ops_sec = (num_iterations * 2.0) / (duration.count() / 1e6);
            result.peak_memory_bytes  = alloc_size * num_iterations;  // Approximate
            result.relative_performance =
                size_results[0].avg_time_per_op_us / result.avg_time_per_op_us;

            size_results.push_back(result);
            EXPECT_EQ(ptrs.size(), num_iterations);
        }

        // ========== POOL ALLOCATOR ==========
        {
            std::vector<sub_allocator::Visitor> empty_visitors;
            auto                                sub_allocator =
                std::make_unique<basic_cpu_allocator>(0, empty_visitors, empty_visitors);
            auto size_rounder = std::make_unique<Pow2Rounder>();
            auto pool         = std::make_unique<allocator_pool>(
                std::max(
                    size_t(2048), alloc_size * 4),  // Adjust pool size based on allocation size
                true,                               // auto_resize
                std::move(sub_allocator),
                std::move(size_rounder),
                "benchmark_pool_" + std::to_string(alloc_size));

            std::vector<void*> ptrs;
            ptrs.reserve(num_iterations);

            auto start = std::chrono::high_resolution_clock::now();

            // Allocation phase
            for (size_t i = 0; i < num_iterations; ++i)
            {
                void* ptr = pool->allocate_raw(64, alloc_size);
                if (ptr)
                {
                    ptrs.push_back(ptr);
                    memset(ptr, 0xCC, std::min(alloc_size, size_t(64)));  // Touch memory
                }
            }

            // Deallocation phase
            for (void* ptr : ptrs)
            {
                pool->deallocate_raw(ptr);
            }

            auto end      = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            BenchmarkResult result;
            result.allocator_name     = "Pool Allocator";
            result.allocation_size    = alloc_size;
            result.iterations         = num_iterations;
            result.total_time_us      = duration.count();
            result.avg_time_per_op_us = duration.count() / (num_iterations * 2.0);
            result.throughput_ops_sec = (num_iterations * 2.0) / (duration.count() / 1e6);
            result.peak_memory_bytes  = alloc_size * num_iterations;  // Approximate
            result.relative_performance =
                size_results[0].avg_time_per_op_us / result.avg_time_per_op_us;

            size_results.push_back(result);
            EXPECT_EQ(ptrs.size(), num_iterations);
        }

        // ========== TRACKING ALLOCATOR (wrapping BFC) ==========
        {
            std::vector<sub_allocator::Visitor> empty_visitors;
            auto                                sub_allocator =
                std::make_unique<basic_cpu_allocator>(0, empty_visitors, empty_visitors);

            allocator_bfc::Options opts;
            opts.allow_growth = true;

            size_t total_memory = std::max(
                static_cast<size_t>(64ULL << 20),
                alloc_size * num_iterations * 2);  // Adjust based on test size
            auto underlying_bfc = std::make_unique<memory::allocator_bfc>(
                std::unique_ptr<memory::sub_allocator>(sub_allocator.release()),
                total_memory,
                "tracking_bfc_" + std::to_string(alloc_size),
                opts);

            // Create tracking allocator wrapping the BFC allocator
            auto tracker = new memory::allocator_tracking(underlying_bfc.get(), true);

            std::vector<void*> ptrs;
            ptrs.reserve(num_iterations);

            auto start = std::chrono::high_resolution_clock::now();

            // Allocation phase
            for (size_t i = 0; i < num_iterations; ++i)
            {
                void* ptr = tracker->allocate_raw(64, alloc_size);
                if (ptr)
                {
                    ptrs.push_back(ptr);
                    memset(ptr, 0xDD, std::min(alloc_size, size_t(64)));  // Touch memory
                }
            }

            // Deallocation phase
            for (void* ptr : ptrs)
            {
                tracker->deallocate_raw(ptr);
            }

            auto end      = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            BenchmarkResult result;
            result.allocator_name     = "Tracking Allocator";
            result.allocation_size    = alloc_size;
            result.iterations         = num_iterations;
            result.total_time_us      = duration.count();
            result.avg_time_per_op_us = duration.count() / (num_iterations * 2.0);
            result.throughput_ops_sec = (num_iterations * 2.0) / (duration.count() / 1e6);
            result.peak_memory_bytes  = alloc_size * num_iterations;  // Approximate
            result.relative_performance =
                size_results[0].avg_time_per_op_us / result.avg_time_per_op_us;

            size_results.push_back(result);
            EXPECT_EQ(ptrs.size(), num_iterations);

            // Properly cleanup tracking allocator by releasing reference
            tracker->GetRecordsAndUnRef();
        }

// ========== ALLOCATOR_DEVICE ==========
#if PROJECT_HAS_CUDA || PROJECT_HAS_HIP
        {
            auto               device_allocator = std::make_unique<allocator_device>();
            std::vector<void*> ptrs;
            ptrs.reserve(num_iterations);

            auto start = std::chrono::high_resolution_clock::now();

            // Allocation phase
            for (size_t i = 0; i < num_iterations; ++i)
            {
                void* ptr = device_allocator->allocate_raw(64, alloc_size);
                if (ptr)
                {
                    ptrs.push_back(ptr);
                    memset(ptr, 0xFF, std::min(alloc_size, size_t(64)));  // Touch memory
                }
            }

            // Deallocation phase
            for (void* ptr : ptrs)
            {
                device_allocator->deallocate_raw(ptr);
            }

            auto end      = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            BenchmarkResult result;
            result.allocator_name     = "device_option Allocator";
            result.allocation_size    = alloc_size;
            result.iterations         = num_iterations;
            result.total_time_us      = duration.count();
            result.avg_time_per_op_us = duration.count() / (num_iterations * 2.0);
            result.throughput_ops_sec = (num_iterations * 2.0) / (duration.count() / 1e6);
            result.peak_memory_bytes  = alloc_size * num_iterations;  // Approximate
            result.relative_performance =
                size_results[0].avg_time_per_op_us / result.avg_time_per_op_us;

            size_results.push_back(result);
            EXPECT_EQ(ptrs.size(), num_iterations);
        }
#endif
        // Store results for this allocation size
        all_results.insert(all_results.end(), size_results.begin(), size_results.end());
    }

    // ========== COMPREHENSIVE RESULTS PRESENTATION ==========
    QUARISMA_LOG_INFO("Generating comprehensive performance comparison report...");

    std::cout << "\n" << std::string(120, '=') << "\n";
    std::cout << "                    COMPREHENSIVE MEMORY ALLOCATOR PERFORMANCE BENCHMARK\n";
    std::cout << std::string(120, '=') << "\n\n";

    // Group results by allocation size for better presentation
    for (size_t test_size : test_sizes)
    {
        std::cout << "ALLOCATION SIZE: " << test_size << " bytes (" << num_iterations
                  << " iterations)\n";
        std::cout << std::string(120, '-') << "\n";

        // Header
        std::cout << std::left << std::setw(20) << "Allocator" << std::setw(15) << "Total Time (μs)"
                  << std::setw(18) << "Avg Time/Op (μs)" << std::setw(18) << "Throughput (ops/s)"
                  << std::setw(18) << "Peak Memory (MB)" << std::setw(15) << "Relative Perf"
                  << std::setw(10) << "vs malloc" << "\n";
        std::cout << std::string(120, '-') << "\n";

        // Find results for this allocation size
        std::vector<BenchmarkResult> size_specific_results;
        for (const auto& result : all_results)
        {
            if (result.allocation_size == test_size)
            {
                size_specific_results.push_back(result);
            }
        }

        // Display results
        for (const auto& result : size_specific_results)
        {
            double      peak_memory_mb = result.peak_memory_bytes / (1024.0 * 1024.0);
            double      perf_vs_malloc = result.relative_performance;
            std::string perf_indicator = (perf_vs_malloc > 1.0)   ? "FASTER"
                                         : (perf_vs_malloc < 1.0) ? "SLOWER"
                                                                  : "SAME";

            // Log the results to ensure they appear in test output
            std::ostringstream result_line;
            result_line << std::left << std::fixed << std::setprecision(3) << std::setw(20)
                        << result.allocator_name << std::setw(15) << result.total_time_us
                        << std::setw(18) << result.avg_time_per_op_us << std::setw(18)
                        << std::scientific << result.throughput_ops_sec << std::fixed
                        << std::setw(18) << peak_memory_mb << std::setw(15) << perf_vs_malloc << "x"
                        << std::setw(10) << perf_indicator;

            QUARISMA_LOG_INFO("{}", result_line.str());

            std::cout << std::left << std::fixed << std::setprecision(3) << std::setw(20)
                      << result.allocator_name << std::setw(15) << result.total_time_us
                      << std::setw(18) << result.avg_time_per_op_us << std::setw(18)
                      << std::scientific << result.throughput_ops_sec << std::fixed << std::setw(18)
                      << peak_memory_mb << std::setw(15) << perf_vs_malloc << "x" << std::setw(10)
                      << perf_indicator << "\n";
        }
        std::cout << "\n";
    }

    // Summary statistics
    std::cout << "PERFORMANCE SUMMARY:\n";
    std::cout << std::string(60, '-') << "\n";

    // Find best and worst performers for each size
    for (size_t test_size : test_sizes)
    {
        std::vector<BenchmarkResult> size_results;
        for (const auto& result : all_results)
        {
            if (result.allocation_size == test_size)
            {
                size_results.push_back(result);
            }
        }

        if (!size_results.empty())
        {
            auto fastest = std::max_element(
                size_results.begin(),
                size_results.end(),
                [](const BenchmarkResult& a, const BenchmarkResult& b)
                { return a.throughput_ops_sec < b.throughput_ops_sec; });

            auto slowest = std::min_element(
                size_results.begin(),
                size_results.end(),
                [](const BenchmarkResult& a, const BenchmarkResult& b)
                { return a.throughput_ops_sec < b.throughput_ops_sec; });

            std::ostringstream summary_line;
            summary_line << "Size " << test_size << "B: Fastest = " << fastest->allocator_name
                         << " (" << std::scientific << fastest->throughput_ops_sec << " ops/s), "
                         << "Slowest = " << slowest->allocator_name << " (" << std::scientific
                         << slowest->throughput_ops_sec << " ops/s)";

            QUARISMA_LOG_INFO("{}", summary_line.str());

            std::cout << "Size " << test_size << "B: Fastest = " << fastest->allocator_name << " ("
                      << std::scientific << fastest->throughput_ops_sec << " ops/s), "
                      << "Slowest = " << slowest->allocator_name << " (" << std::scientific
                      << slowest->throughput_ops_sec << " ops/s)\n";
        }
    }

    std::cout << "\nBENCHMARK METHODOLOGY:\n";
    std::cout << "- Each allocator tested with " << num_iterations
              << " allocation/deallocation cycles\n";
    std::cout << "- Memory touched after allocation to ensure realistic performance\n";
    std::cout << "- Multiple allocation sizes tested: 64B (small), 1KB (medium), 64KB (large)\n";
    std::cout << "- Relative performance calculated vs malloc/free baseline\n";
    std::cout << "- Peak memory is approximate based on allocation size × iterations\n";

    std::cout << "\n" << std::string(120, '=') << "\n";
    std::cout << "                              BENCHMARK COMPLETED SUCCESSFULLY\n";
    std::cout << std::string(120, '=') << "\n\n";

    QUARISMA_LOG_INFO(
        "Comprehensive Memory Allocator Performance Benchmark completed successfully!");
}
#endif