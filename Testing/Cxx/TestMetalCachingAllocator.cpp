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

#include "MemoryTest.h"

#if MEMORY_HAS_METAL

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "allocator.h"
#include "gpu/caching_allocator.h"
#include "gpu/metal/metal_buffer_allocator.h"

using namespace memory;
using namespace memory::gpu;

MEMORYTEST(MetalCachingAllocator, constructs_with_valid_parameters)
{
    metal_caching_allocator allocator(0, 64 * 1024ULL);
    EXPECT_EQ(0, allocator.device());
    EXPECT_EQ(64 * 1024ULL, allocator.max_cached_bytes());
    END_TEST();
}

MEMORYTEST(MetalCachingAllocator, allocates_and_deallocates_memory)
{
    metal_caching_allocator allocator(0);

    void* ptr1 = allocator.allocate(1024);
    EXPECT_NE(nullptr, ptr1);
    allocator.deallocate(ptr1, 1024);

    std::vector<void*> ptrs;
    for (int i = 0; i < 10; ++i)
    {
        void* ptr = allocator.allocate(512 * static_cast<size_t>(i + 1));
        EXPECT_NE(nullptr, ptr);
        ptrs.push_back(ptr);
    }
    for (void* ptr : ptrs)
    {
        allocator.deallocate(ptr, 0);
    }
    END_TEST();
}

MEMORYTEST(MetalCachingAllocator, rounds_requests_to_512_byte_multiples)
{
    metal_caching_allocator allocator(0);

    void* ptr1 = allocator.allocate(1);
    void* ptr2 = allocator.allocate(512);
    ASSERT_NE(nullptr, ptr1);
    ASSERT_NE(nullptr, ptr2);

    auto stats = allocator.stats();
    EXPECT_EQ(1024U, stats.bytes_allocated.load());
    EXPECT_EQ(2U * 1024U * 1024U, stats.bytes_reserved.load());

    allocator.deallocate(ptr1, 1);
    allocator.deallocate(ptr2, 512);
    EXPECT_EQ(0U, allocator.stats().bytes_allocated.load());
    END_TEST();
}

MEMORYTEST(MetalCachingAllocator, packs_small_allocations_into_one_segment)
{
    metal_caching_allocator allocator(0);

    void* ptr1 = allocator.allocate(1024);
    void* ptr2 = allocator.allocate(1024);
    ASSERT_NE(nullptr, ptr1);
    ASSERT_NE(nullptr, ptr2);

    auto stats = allocator.stats();
    EXPECT_EQ(1U, stats.driver_allocations.load());
    EXPECT_EQ(2U * 1024U * 1024U, stats.bytes_reserved.load());
    EXPECT_EQ(2048U, stats.bytes_allocated.load());

    allocator.deallocate(ptr1, 1024);
    allocator.deallocate(ptr2, 1024);
    END_TEST();
}

MEMORYTEST(MetalCachingAllocator, reuses_cached_blocks)
{
    metal_caching_allocator allocator(0);

    void* ptr1 = allocator.allocate(1024);
    ASSERT_NE(nullptr, ptr1);
    allocator.deallocate(ptr1, 1024);

    void* ptr2 = allocator.allocate(1024);
    ASSERT_NE(nullptr, ptr2);
    EXPECT_EQ(ptr1, ptr2);

    auto stats = allocator.stats();
    EXPECT_EQ(1U, stats.driver_allocations.load());
    EXPECT_EQ(1U, stats.cache_hits.load());
    EXPECT_EQ(1U, stats.cache_misses.load());

    allocator.deallocate(ptr2, 1024);
    END_TEST();
}

MEMORYTEST(MetalCachingAllocator, splits_oversized_cached_blocks)
{
    metal_caching_allocator allocator(0);

    // Whole 20 MiB large segment (no initial split), then free and take a 4 MiB
    // slice from the same large pool so the remainder stays as inactive_split.
    void* big = allocator.allocate(20 * 1024 * 1024);
    ASSERT_NE(nullptr, big);
    allocator.deallocate(big, 20 * 1024 * 1024);

    void* small = allocator.allocate(4 * 1024 * 1024);
    ASSERT_NE(nullptr, small);
    EXPECT_EQ(big, small);

    auto stats = allocator.stats();
    EXPECT_EQ(1U, stats.driver_allocations.load());
    EXPECT_EQ(4U * 1024U * 1024U, stats.bytes_allocated.load());
    EXPECT_EQ(16U * 1024U * 1024U, stats.inactive_split_bytes.load());

    allocator.deallocate(small, 4 * 1024 * 1024);
    EXPECT_EQ(0U, allocator.stats().inactive_split_bytes.load());
    END_TEST();
}

MEMORYTEST(MetalCachingAllocator, empty_cache_releases_cached_bytes)
{
    metal_caching_allocator allocator(0);

    void* ptr = allocator.allocate(1024);
    ASSERT_NE(nullptr, ptr);
    allocator.deallocate(ptr, 1024);

    auto before = allocator.stats();
    EXPECT_GT(before.bytes_cached.load(), 0U);

    allocator.empty_cache();

    auto after = allocator.stats();
    EXPECT_EQ(0U, after.bytes_cached.load());
    END_TEST();
}

MEMORYTEST(MetalCachingAllocator, respects_max_cached_bytes_trim)
{
    metal_caching_allocator allocator(0, 0);  // no caching — trim after every free

    void* ptr = allocator.allocate(1024);
    ASSERT_NE(nullptr, ptr);
    allocator.deallocate(ptr, 1024);

    EXPECT_EQ(0U, allocator.stats().bytes_cached.load());
    END_TEST();
}

MEMORYTEST(MetalCachingAllocator, foreign_and_double_free_throw)
{
    metal_caching_allocator allocator(0);

    void* ptr = allocator.allocate(1024);
    ASSERT_NE(nullptr, ptr);

    int stack = 0;
    EXPECT_THROW(allocator.deallocate(&stack, 0), std::exception);

    allocator.deallocate(ptr, 1024);
    EXPECT_THROW(allocator.deallocate(ptr, 1024), std::exception);
    END_TEST();
}

MEMORYTEST(MetalCachingAllocator, record_stream_is_noop)
{
    metal_caching_allocator allocator(0);
    void*                   ptr = allocator.allocate(1024);
    ASSERT_NE(nullptr, ptr);
    allocator.record_stream(ptr, reinterpret_cast<void*>(1));
    allocator.deallocate(ptr, 1024);
    END_TEST();
}

MEMORYTEST(MetalCachingAllocator, mid_segment_handle_and_offset)
{
    // Use the process-wide registry so metal::mtl_buffer_* resolve the same instance.
    auto& allocator = caching_allocator_for_device(0);

    void* ptr1 = allocator.allocate(1024);
    void* ptr2 = allocator.allocate(1024);
    ASSERT_NE(nullptr, ptr1);
    ASSERT_NE(nullptr, ptr2);

    void* handle1 = metal::mtl_buffer_handle(ptr1);
    void* handle2 = metal::mtl_buffer_handle(ptr2);
    ASSERT_NE(nullptr, handle1);
    ASSERT_NE(nullptr, handle2);
    EXPECT_EQ(handle1, handle2);  // packed into one MTLBuffer

    EXPECT_EQ(0U, metal::mtl_buffer_offset(ptr1));
    EXPECT_EQ(1024U, metal::mtl_buffer_offset(ptr2));

    // Host-visible Shared storage: mid-segment pointer is writable
    auto* f2 = static_cast<float*>(ptr2);
    f2[0]    = 3.14f;
    EXPECT_FLOAT_EQ(f2[0], 3.14f);

    allocator.deallocate(ptr1, 1024);
    allocator.deallocate(ptr2, 1024);
    END_TEST();
}

MEMORYTEST(MetalCachingAllocator, allocator_t_routes_through_cache)
{
    EXPECT_TRUE(has_gpu_support());

    using alloc_t = allocator<float>;
    float* ptr    = alloc_t::allocate(256, device_enum::METAL);
    ASSERT_NE(nullptr, ptr);
    ASSERT_NE(nullptr, metal::mtl_buffer_handle(ptr));

    auto stats = caching_allocator_for_device(0).stats();
    EXPECT_GT(stats.successful_allocations.load(), 0U);

    alloc_t::free(ptr, device_enum::METAL);
    EXPECT_EQ(nullptr, ptr);
    END_TEST();
}

MEMORYTEST(MetalCachingAllocator, free_memory_callback_runs_on_miss)
{
    metal_caching_allocator allocator(0);
    bool                    called = false;
    allocator.add_free_memory_callback(
        [&]()
        {
            called = true;
            return false;
        });

    void* ptr = allocator.allocate(1024);
    ASSERT_NE(nullptr, ptr);
    // First allocation is a miss — callback runs before driver alloc
    EXPECT_TRUE(called);

    allocator.deallocate(ptr, 1024);
    allocator.clear_free_memory_callbacks();
    END_TEST();
}

MEMORYTEST(MetalCachingAllocator, supports_move_semantics)
{
    metal_caching_allocator allocator1(0);
    void*                   ptr = allocator1.allocate(1024);
    ASSERT_NE(nullptr, ptr);

    metal_caching_allocator allocator2 = std::move(allocator1);
    EXPECT_EQ(0, allocator2.device());
    allocator2.deallocate(ptr, 1024);
    END_TEST();
}

#endif  // MEMORY_HAS_METAL
