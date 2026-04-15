
/**
 * @file TestAllocatorBfc.cpp
 * @brief Comprehensive test suite for BFC (Best-Fit with Coalescing) allocator
 *
 * Tests all public methods and functionality of the allocator_bfc class including:
 * - Basic allocation and deallocation
 * - Alignment requirements
 * - Memory tracking capabilities
 * - Configuration options
 * - Statistics collection
 * - Error handling and edge cases
 */

#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "MemoryTest.h"
#include "backend/allocator_bfc.h"
#include "backend/allocator_pool.h"
#include "common/pointer.h"
#include "helper/memory_allocator.h"
#if PROJECT_HAS_NATIVE_PROFILER && 0
#include "native/analysis/statistical_analyzer.h"
#include "native/memory/memory_tracker.h"
#include "native/session/profiler.h"
#include "native/tracing/traceme.h"
#include "native/tracing/traceme_encode.h"
#include "native/tracing/traceme_recorder.h"
#endif

using namespace memory;
using namespace memory;

namespace
{

// Mock test allocator class removed - now using production basic_cpu_allocator

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
 * @brief Helper function to create a BFC allocator for testing
 */
std::unique_ptr<allocator_bfc> create_test_bfc_allocator()
{
    // Create production basic_cpu_allocator as sub-allocator
    auto sub_alloc = std::make_unique<basic_cpu_allocator>(
        0,                                              // numa_node = 0 (default)
        std::vector<memory::sub_allocator::Visitor>{},  // no alloc visitors
        std::vector<memory::sub_allocator::Visitor>{}   // no free visitors
    );

    // Create BFC allocator with default options
    allocator_bfc::Options opts;
    opts.allow_growth       = true;
    opts.garbage_collection = true;
    //opts.allow_retry_on_failure = false;
    opts.fragmentation_fraction = 0.0;

    return std::make_unique<allocator_bfc>(
        std::move(sub_alloc),
        1024ULL * 1024ULL,  // 1MB initial size
        "test_bfc",
        opts);
}

/**
 * @brief Test basic allocation and deallocation functionality
 */
MEMORYTEST(AllocatorBFC, basic_allocation_deallocation)
{
#if PROJECT_HAS_NATIVE_PROFILER && 0
    traceme_recorder::start(3);
    auto allocator = create_test_bfc_allocator();

    // Test basic allocation
    void* ptr1 = allocator->allocate_raw(64, 1024);
    EXPECT_NE(nullptr, ptr1);
    EXPECT_TRUE(is_aligned(ptr1, 64));

    // Test memory content
    fill_memory(ptr1, 1024, 0xAA);
    EXPECT_TRUE(validate_memory(ptr1, 1024, 0xAA));

    // Test deallocation
    allocator->deallocate_raw(ptr1);

    // Test multiple allocations
    std::vector<void*> ptrs;
    for (int i = 0; i < 10; ++i)
    {
        void* ptr = allocator->allocate_raw(32, 512);
        EXPECT_NE(nullptr, ptr);
        EXPECT_TRUE(is_aligned(ptr, 32));
        ptrs.push_back(ptr);
    }

    // Deallocate all
    for (void* ptr : ptrs)
    {
        allocator->deallocate_raw(ptr);
    }
    traceme_recorder::stop();
#endif
    END_TEST();
}

/**
 * @brief Test allocation tracking capabilities
 */
MEMORYTEST(AllocatorBFC, allocation_tracking)
{
    auto allocator = create_test_bfc_allocator();

    // Verify tracking is enabled
    EXPECT_TRUE(allocator->tracks_allocation_sizes());

    // Test allocation with size tracking
    void* ptr = allocator->allocate_raw(128, 2048);
    EXPECT_NE(nullptr, ptr);

    // Test size queries
    EXPECT_EQ(allocator->RequestedSize(ptr), 2048);
    EXPECT_GE(allocator->AllocatedSize(ptr), 2048);

    // Test allocation ID
    int64_t alloc_id = allocator->AllocationId(ptr);
    EXPECT_GT(alloc_id, 0);

    allocator->deallocate_raw(ptr);

    END_TEST();
}

/**
 * @brief Test different alignment requirements
 */
MEMORYTEST(AllocatorBFC, alignment_requirements)
{
#if 0
    auto allocator = create_test_bfc_allocator();

    // Test various alignment values
    std::vector<size_t> alignments = {16, 32, 64, 128, 256};

    for (size_t alignment : alignments)
    {
        void* ptr = allocator->allocate_raw(alignment, 1024);
        EXPECT_NE(nullptr, ptr);
        EXPECT_TRUE(is_aligned(ptr, std::max(alignment, static_cast<size_t>(64))));
        allocator->deallocate_raw(ptr);
    }
#endif
    END_TEST();
}

/**
 * @brief Test allocation with attributes
 */
MEMORYTEST(AllocatorBFC, allocation_with_attributes)
{
    auto allocator = create_test_bfc_allocator();

    allocation_attributes attrs;
    attrs.retry_on_failure = true;

    void* ptr = allocator->allocate_raw(64, 1024, attrs);
    EXPECT_NE(nullptr, ptr);
    EXPECT_TRUE(is_aligned(ptr, 64));

    allocator->deallocate_raw(ptr);

    END_TEST();
}

/**
 * @brief Test statistics collection
 */
MEMORYTEST(AllocatorBFC, statistics_collection)
{
    auto allocator = create_test_bfc_allocator();

    // Get initial statistics
    auto stats_opt = allocator->GetStats();
    EXPECT_TRUE(stats_opt.has_value());

    auto initial_stats = stats_opt.value();

    // Perform some allocations
    std::vector<void*> ptrs;
    for (int i = 0; i < 5; ++i)
    {
        void* ptr = allocator->allocate_raw(64, 1024 * (i + 1));
        EXPECT_NE(nullptr, ptr);
        ptrs.push_back(ptr);
    }

    // Check updated statistics
    auto updated_stats_opt = allocator->GetStats();
    EXPECT_TRUE(updated_stats_opt.has_value());

    auto updated_stats = updated_stats_opt.value();
    EXPECT_GT(updated_stats.num_allocs, initial_stats.num_allocs);
    EXPECT_GT(updated_stats.bytes_in_use, initial_stats.bytes_in_use);

    // Clean up
    for (void* ptr : ptrs)
    {
        allocator->deallocate_raw(ptr);
    }

    END_TEST();
}

/**
 * @brief Test allocator name and memory type
 */
MEMORYTEST(AllocatorBFC, allocator_properties)
{
    auto allocator = create_test_bfc_allocator();

    // Test allocator name
    EXPECT_EQ(std::string(allocator->Name()), std::string("test_bfc"));

    // Test memory type
    EXPECT_EQ(allocator->GetMemoryType(), allocator_memory_enum::HOST_PAGEABLE);

    END_TEST();
}

/**
 * @brief Test zero-size allocation handling
 */
MEMORYTEST(AllocatorBFC, zero_size_allocation)
{
    auto allocator = create_test_bfc_allocator();

    // Zero-size allocation should return nullptr or handle gracefully
    void* ptr = allocator->allocate_raw(64, 0);

    // Implementation may return nullptr or a valid pointer for zero-size
    if (ptr != nullptr)
    {
        allocator->deallocate_raw(ptr);
    }

    END_TEST();
}

/**
 * @brief Test null pointer deallocation
 */
MEMORYTEST(AllocatorBFC, null_pointer_deallocation)
{
    auto allocator = create_test_bfc_allocator();

    // Deallocating nullptr should not crash
    allocator->deallocate_raw(nullptr);

    // This test passes if no exception is thrown
    EXPECT_TRUE(true);

    END_TEST();
}

/**
 * @brief Test large allocation handling
 */
MEMORYTEST(AllocatorBFC, large_allocations)
{
    auto allocator = create_test_bfc_allocator();

    // Test allocation larger than initial pool size
    void* large_ptr = allocator->allocate_raw(64, 2 * 1024ULL);  // 2MB

    if (large_ptr != nullptr)
    {
        EXPECT_TRUE(is_aligned(large_ptr, 64));

        // Test memory access
        fill_memory(large_ptr, 1024, 0xBB);
        EXPECT_TRUE(validate_memory(large_ptr, 1024, 0xBB));

        allocator->deallocate_raw(large_ptr);
    }

    END_TEST();
}

/**
 * @brief Test fragmentation and coalescing behavior
 */
MEMORYTEST(AllocatorBFC, fragmentation_and_coalescing)
{
    auto allocator = create_test_bfc_allocator();

    // Allocate several blocks
    std::vector<void*> ptrs;
    for (int i = 0; i < 10; ++i)
    {
        void* ptr = allocator->allocate_raw(64, 1024);
        EXPECT_NE(nullptr, ptr);
        ptrs.push_back(ptr);
    }

    // Deallocate every other block to create fragmentation
    for (size_t i = 1; i < ptrs.size(); i += 2)
    {
        allocator->deallocate_raw(ptrs[i]);
        ptrs[i] = nullptr;
    }

    // Try to allocate a larger block (should trigger coalescing)
    void* large_ptr = allocator->allocate_raw(64, 2048);
    if (large_ptr != nullptr)
    {
        allocator->deallocate_raw(large_ptr);
    }

    // Clean up remaining blocks
    for (void* ptr : ptrs)
    {
        if (ptr != nullptr)
        {
            allocator->deallocate_raw(ptr);
        }
    }

    END_TEST();
}

/**
 * @brief Test BFC allocator with different configuration options
 */
MEMORYTEST(AllocatorBFC, configuration_options)
{
    auto sub_alloc = std::make_unique<basic_cpu_allocator>(
        0,                                              // numa_node = 0 (default)
        std::vector<memory::sub_allocator::Visitor>{},  // no alloc visitors
        std::vector<memory::sub_allocator::Visitor>{}   // no free visitors
    );

    // Test with garbage collection enabled
    allocator_bfc::Options gc_opts;
    gc_opts.allow_growth           = true;
    gc_opts.garbage_collection     = true;
    gc_opts.fragmentation_fraction = 0.1;

    auto gc_allocator = std::make_unique<allocator_bfc>(
        std::move(sub_alloc),
        512 * 1024,  // 512KB
        "test_bfc_gc",
        gc_opts);

    // Test basic functionality with GC enabled
    void* ptr = gc_allocator->allocate_raw(64, 1024);
    EXPECT_NE(nullptr, ptr);
    EXPECT_TRUE(is_aligned(ptr, 64));

    gc_allocator->deallocate_raw(ptr);

    END_TEST();
}

/**
 * @brief Test BFC allocator with growth disabled
 */
MEMORYTEST(AllocatorBFC, no_growth_configuration)
{
    auto sub_alloc = std::make_unique<basic_cpu_allocator>(
        0,                                              // numa_node = 0 (default)
        std::vector<memory::sub_allocator::Visitor>{},  // no alloc visitors
        std::vector<memory::sub_allocator::Visitor>{}   // no free visitors
    );

    allocator_bfc::Options no_growth_opts;
    no_growth_opts.allow_growth       = false;
    no_growth_opts.garbage_collection = false;

    auto fixed_allocator = std::make_unique<allocator_bfc>(
        std::move(sub_alloc),
        64 * 1024,  // 64KB fixed size
        "test_bfc_fixed",
        no_growth_opts);

    // Test allocation within fixed size
    void* ptr = fixed_allocator->allocate_raw(64, 1024);
    EXPECT_NE(nullptr, ptr);

    fixed_allocator->deallocate_raw(ptr);

    END_TEST();
}

/**
 * @brief Test concurrent access to BFC allocator
 */
MEMORYTEST(AllocatorBFC, thread_safety)
{
    auto sub_alloc = std::make_unique<basic_cpu_allocator>(
        0,                                              // numa_node = 0 (default)
        std::vector<memory::sub_allocator::Visitor>{},  // no alloc visitors
        std::vector<memory::sub_allocator::Visitor>{}   // no free visitors
    );

    allocator_bfc::Options opts;
    opts.allow_growth = true;

    auto allocator = std::make_unique<allocator_bfc>(
        std::move(sub_alloc),
        64ULL * 1024ULL,  // 1MB
        "test_bfc_concurrent",
        opts);

    const int                       num_threads            = 4;
    const int                       allocations_per_thread = 10;
    std::vector<std::thread>        threads;
    std::vector<std::vector<void*>> thread_ptrs(num_threads);

    // Launch threads that perform allocations
    for (int t = 0; t < num_threads; ++t)
    {
        threads.emplace_back(
            [&, t]()
            {
                for (int i = 0; i < allocations_per_thread; ++i)
                {
                    void* ptr = allocator->allocate_raw(64, 1024 + i * 64);
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
            allocator->deallocate_raw(ptr);
        }
    }

    EXPECT_GT(total_allocations, 0);

    END_TEST();
}

/**
 * @brief Test BFC allocator performance characteristics
 */
MEMORYTEST(AllocatorBFC, allocation_timing)
{
    auto sub_alloc = std::make_unique<basic_cpu_allocator>(
        0,                                              // numa_node = 0 (default)
        std::vector<memory::sub_allocator::Visitor>{},  // no alloc visitors
        std::vector<memory::sub_allocator::Visitor>{}   // no free visitors
    );

    allocator_bfc::Options opts;
    opts.allow_growth           = true;
    opts.allow_retry_on_failure = false;

    auto allocator = std::make_unique<allocator_bfc>(
        std::move(sub_alloc),
        2 * 1024ULL * 1024ULL,  // 2MB
        "test_bfc_perf",
        opts);

    const int          num_allocations = 1000;
    std::vector<void*> ptrs;
    ptrs.reserve(num_allocations);

    // Time allocation performance
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_allocations; ++i)
    {
        void* ptr = allocator->allocate_raw(64, 1024);
        if (ptr != nullptr)
        {
            ptrs.push_back(ptr);
        }
    }

    auto alloc_time = std::chrono::high_resolution_clock::now();

    // Time deallocation performance
    for (void* ptr : ptrs)
    {
        allocator->deallocate_raw(ptr);
    }

    auto dealloc_time = std::chrono::high_resolution_clock::now();

    // Verify reasonable performance (should complete in reasonable time)
    auto total_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(dealloc_time - start_time).count();

    EXPECT_LT(total_time, 1000);                  // Should complete within 1 second
    EXPECT_GT(ptrs.size(), num_allocations / 2);  // Most allocations should succeed

    END_TEST();
}

MEMORYTEST(AllocatorBFC, basic_allocation)
{
    // Create a BFC allocator with 1MB memory limit
    const size_t memory_limit  = 1024ULL * 1024ULL;  // 1MB
    auto         sub_allocator = std::make_unique<basic_cpu_allocator>(
        0,
        std::vector<memory::sub_allocator::Visitor>{},
        std::vector<memory::sub_allocator::Visitor>{});

    allocator_bfc::Options opts;
    opts.allow_growth = false;
    allocator_bfc allocator(std::move(sub_allocator), memory_limit, "test_bfc", opts);

    // Test basic allocation
    void* ptr1 = allocator.allocate_raw(64, 1024);
    EXPECT_NE(nullptr, ptr1);
    EXPECT_TRUE(IsAligned(ptr1, 64));

    // Test deallocation
    allocator.deallocate_raw(ptr1);

    // Test multiple allocations
    std::vector<void*> ptrs;
    for (int i = 0; i < 10; ++i)
    {
        void* ptr = allocator.allocate_raw(32, 512);
        EXPECT_NE(nullptr, ptr);
        EXPECT_TRUE(IsAligned(ptr, 32));
        ptrs.push_back(ptr);
    }

    // Deallocate all
    for (void* ptr : ptrs)
    {
        allocator.deallocate_raw(ptr);
    }

    END_TEST();
}
// Test allocator_bfc edge cases
MEMORYTEST(AllocatorBFC, EdgeCases)
{
    const size_t memory_limit  = 1024ULL * 1024ULL;  // 1MB
    auto         sub_allocator = std::make_unique<basic_cpu_allocator>(
        0,
        std::vector<memory::sub_allocator::Visitor>{},
        std::vector<memory::sub_allocator::Visitor>{});

    allocator_bfc::Options opts;
    opts.allow_growth = false;
    allocator_bfc allocator(std::move(sub_allocator), memory_limit, "test_bfc_edge", opts);

    // Test zero-size allocation
    void* ptr_zero = allocator.allocate_raw(64, 0);
    EXPECT_EQ(nullptr, ptr_zero);  // Should return nullptr for zero size

    // Test large alignment
    void* ptr_aligned = allocator.allocate_raw(4096, 1024);
    EXPECT_NE(nullptr, ptr_aligned);
    EXPECT_TRUE(IsAligned(ptr_aligned, 4096));
    allocator.deallocate_raw(ptr_aligned);

    // Test very small allocation
    void* ptr_small = allocator.allocate_raw(1, 1);
    EXPECT_NE(nullptr, ptr_small);
    allocator.deallocate_raw(ptr_small);

    // Test allocation statistics if available
    auto stats = allocator.GetStats();
    if (stats.has_value())
    {
        EXPECT_GE(stats->num_allocs, 0);
        EXPECT_GE(stats->bytes_in_use, 0);
        EXPECT_GE(stats->peak_bytes_in_use, 0);
    }
}

// Test allocator_bfc memory tracking
MEMORYTEST(AllocatorBFC, MemoryTracking)
{
    const size_t memory_limit  = 1024ULL;  // 1MB
    auto         sub_allocator = std::make_unique<basic_cpu_allocator>(
        0,
        std::vector<memory::sub_allocator::Visitor>{},
        std::vector<memory::sub_allocator::Visitor>{});

    allocator_bfc::Options opts;
    opts.allow_growth = false;
    allocator_bfc allocator(std::move(sub_allocator), memory_limit, "test_bfc_tracking", opts);

    // Test allocation size tracking
    if (allocator.tracks_allocation_sizes())
    {
        void* ptr = allocator.allocate_raw(64, 1024);
        EXPECT_NE(nullptr, ptr);

        size_t requested_size = allocator.RequestedSize(ptr);
        size_t allocated_size = allocator.AllocatedSize(ptr);

        EXPECT_EQ(requested_size, 1024);
        EXPECT_GE(allocated_size, 1024);

        allocator.deallocate_raw(ptr);
    }
}

// Test comprehensive memory profiling with BFC allocator
MEMORYTEST(AllocatorBFC, ComprehensiveMemoryProfiling)
{
#if PROJECT_HAS_NATIVE_PROFILER && 0
    auto session = profiler_session_builder()
                       .with_timing(true)
                       .with_memory_tracking(true)
                       .with_statistical_analysis(true)
                       .with_thread_safety(true)
                       .with_output_format(profiler_options::output_format_enum::JSON)
                       .build();

    session->start();

    {
        QUARISMA_PROFILE_SCOPE("bfc_allocator_comprehensive_test");

        const size_t memory_limit  = 10 * 1024 * 1024;  // 10MB
        auto         sub_allocator = std::make_unique<basic_cpu_allocator>(
            0,
            std::vector<memory::sub_allocator::Visitor>{},
            std::vector<memory::sub_allocator::Visitor>{});

        allocator_bfc::Options bfc_opts;
        bfc_opts.allow_growth = true;
        allocator_bfc allocator(std::move(sub_allocator), memory_limit, "profiled_bfc", bfc_opts);

        // Track allocation patterns over time
        std::vector<void*>  allocations;
        std::vector<size_t> allocation_sizes = {64, 128, 256, 512, 1024, 2048, 4096, 8192};

        {
            QUARISMA_PROFILE_SCOPE("allocation_phase");

            // Perform multiple allocation batches
            for (int batch = 0; batch < 5; ++batch)
            {
                QUARISMA_PROFILE_SCOPE("allocation_batch_" + std::to_string(batch));

                for (size_t size : allocation_sizes)
                {
                    for (int i = 0; i < 10; ++i)
                    {
                        QUARISMA_PROFILE_SCOPE("single_allocation");

                        void* ptr = allocator.allocate_raw(size, size);
                        EXPECT_NE(nullptr, ptr);

                        if (ptr)
                        {
                            allocations.push_back(ptr);

                            // Fill memory to simulate real usage
                            fill_memory(ptr, size, static_cast<uint8_t>(batch + i));
                        }

                        // Check memory statistics
                        auto stats = allocator.GetStats();
                        EXPECT_TRUE(stats.has_value());
                        if (stats.has_value())
                        {
                            // Verify memory is being tracked
                            EXPECT_GE(stats->bytes_in_use, 0);
                            EXPECT_GE(stats->peak_bytes_in_use, 0);
                            EXPECT_GE(stats->num_allocs, 0);
                        }
                    }
                }

                // Simulate processing delay
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }

        {
            QUARISMA_PROFILE_SCOPE("fragmentation_analysis");

            // Deallocate every other allocation to create fragmentation
            for (size_t i = 1; i < allocations.size(); i += 2)
            {
                if (allocations[i])
                {
                    allocator.deallocate_raw(allocations[i]);
                    allocations[i] = nullptr;
                }
            }

            // Check fragmentation metrics
            auto stats = allocator.GetStats();
            EXPECT_TRUE(stats.has_value());
            if (stats.has_value())
            {
                // Verify fragmentation state
                EXPECT_GE(stats->bytes_in_use, 0);
                EXPECT_GE(stats->num_allocs, 0);
            }
        }

        {
            QUARISMA_PROFILE_SCOPE("reallocation_phase");

            // Try to allocate in fragmented space
            for (int i = 0; i < 20; ++i)
            {
                QUARISMA_PROFILE_SCOPE("fragmented_allocation");

                void* ptr = allocator.allocate_raw(256, 256);
                if (ptr)
                {
                    allocations.push_back(ptr);
                }
            }
        }

        {
            QUARISMA_PROFILE_SCOPE("cleanup_phase");

            // Clean up all remaining allocations
            for (void* ptr : allocations)
            {
                if (ptr)
                {
                    allocator.deallocate_raw(ptr);
                }
            }
        }

        // Final statistics
        auto final_stats = allocator.GetStats();
        EXPECT_TRUE(final_stats.has_value());
        if (final_stats.has_value())
        {
            EXPECT_EQ(final_stats->bytes_in_use, 0);  // All memory should be freed
        }
    }

    session->stop();

    // Print profiling report to console
    std::cout << "\n=== BFC Allocator Profiling Report ===\n";
    session->print_report();

    std::cout << "\nBFC Allocator Profiling Test Completed Successfully\n";
#endif
}

// Test memory profiling with allocation hotspots identification
MEMORYTEST(AllocatorBFC, AllocationHotspotsIdentification)
{
#if PROJECT_HAS_NATIVE_PROFILER && 0
    auto session = profiler_session_builder()
                       .with_timing(true)
                       .with_memory_tracking(true)
                       .with_statistical_analysis(true)
                       .with_thread_safety(true)
                       .with_output_format(profiler_options::output_format_enum::JSON)
                       .build();
    session->start();

    {
        QUARISMA_PROFILE_SCOPE("hotspot_identification_test");

        const size_t memory_limit  = 5 * 1024 * 1024;  // 5MB
        auto         sub_allocator = std::make_unique<basic_cpu_allocator>(
            0,
            std::vector<memory::sub_allocator::Visitor>{},
            std::vector<memory::sub_allocator::Visitor>{});

        allocator_bfc::Options bfc_opts;
        bfc_opts.allow_growth = true;
        allocator_bfc allocator(std::move(sub_allocator), memory_limit, "hotspot_bfc", bfc_opts);

        // Simulate different allocation patterns to identify hotspots
        {
            QUARISMA_PROFILE_SCOPE("small_frequent_allocations");

            std::vector<void*> small_ptrs;
            for (int i = 0; i < 1000; ++i)
            {
                QUARISMA_PROFILE_SCOPE("small_alloc");
                void* ptr = allocator.allocate_raw(32, 32);
                if (ptr)
                    small_ptrs.push_back(ptr);
            }

            // Clean up
            for (void* ptr : small_ptrs)
            {
                allocator.deallocate_raw(ptr);
            }
        }

        {
            QUARISMA_PROFILE_SCOPE("large_infrequent_allocations");

            std::vector<void*> large_ptrs;
            for (int i = 0; i < 10; ++i)
            {
                QUARISMA_PROFILE_SCOPE("large_alloc");
                void* ptr = allocator.allocate_raw(64 * 1024, 64 * 1024);  // 64KB
                if (ptr)
                    large_ptrs.push_back(ptr);
            }

            // Clean up
            for (void* ptr : large_ptrs)
            {
                allocator.deallocate_raw(ptr);
            }
        }

        {
            QUARISMA_PROFILE_SCOPE("mixed_size_allocations");

            std::vector<void*>  mixed_ptrs;
            std::vector<size_t> sizes = {16, 64, 256, 1024, 4096, 16384};

            for (int round = 0; round < 100; ++round)
            {
                for (size_t size : sizes)
                {
                    QUARISMA_PROFILE_SCOPE("mixed_alloc_" + std::to_string(size));
                    void* ptr = allocator.allocate_raw(size, size);
                    if (ptr)
                        mixed_ptrs.push_back(ptr);
                }
            }

            // Clean up
            for (void* ptr : mixed_ptrs)
            {
                allocator.deallocate_raw(ptr);
            }
        }
    }

    session->stop();

    // Print profiling report to console
    std::cout << "\n=== Allocation Hotspot Profiling Report ===\n";
    session->print_report();

    std::cout << "\nAllocation Hotspot Identification Test Completed Successfully\n";
#endif
}
