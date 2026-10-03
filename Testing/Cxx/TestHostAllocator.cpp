/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#include <cstdint>
#include <limits>
#include <list>
#include <memory_resource>
#include <vector>

#include "MemoryTest.h"
#include "common/cpu_arena.h"
#include "common/host_allocator.h"

using namespace memory;

namespace
{
bool is_aligned(void const* p, std::size_t a)
{
    return (reinterpret_cast<std::uintptr_t>(p) & (a - 1)) == 0;
}

struct alignas(128) over_aligned
{
    char bytes[128];
};
}  // namespace

MEMORYTEST(HostAllocator, VectorUsesAlignedStorage)
{
    std::vector<float, host_allocator<float>> v;
    for (int i = 0; i < 1000; ++i)
    {
        v.push_back(static_cast<float>(i));
    }
    EXPECT_TRUE(is_aligned(v.data(), MEMORY_ALIGNMENT));
    EXPECT_EQ(v[999], 999.0F);
    END_TEST();
}

MEMORYTEST(HostAllocator, RebindWorksForNodeContainers)
{
    std::list<int, host_allocator<int>> l;
    for (int i = 0; i < 100; ++i)
    {
        l.push_back(i);
    }
    EXPECT_EQ(l.size(), 100U);
    EXPECT_EQ(l.back(), 99);
    END_TEST();
}

MEMORYTEST(HostAllocator, TypeAlignmentAboveRequestedIsHonored)
{
    std::vector<over_aligned, host_allocator<over_aligned, 16>> v(3);
    EXPECT_TRUE(is_aligned(v.data(), alignof(over_aligned)));
    END_TEST();
}

MEMORYTEST(HostAllocator, ZeroCountReturnsNull)
{
    host_allocator<int> a;
    EXPECT_EQ(a.allocate(0), nullptr);
    a.deallocate(nullptr, 0);
    END_TEST();
}

MEMORYTEST(HostAllocator, OverflowThrowsBadArrayNewLength)
{
    host_allocator<std::uint64_t> a;
    EXPECT_THROW(
        (void)a.allocate(std::numeric_limits<std::size_t>::max() / 4), std::bad_array_new_length);
    END_TEST();
}

MEMORYTEST(HostAllocator, InstancesCompareEqual)
{
    host_allocator<int>   a;
    host_allocator<float> b;
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);
    END_TEST();
}

MEMORYTEST(HostMemoryResource, PmrVectorAllocatesAndFrees)
{
    std::pmr::vector<int> v(&host_memory_resource::instance());
    for (int i = 0; i < 500; ++i)
    {
        v.push_back(i);
    }
    EXPECT_EQ(v[499], 499);
    EXPECT_TRUE(host_memory_resource::instance().is_equal(host_memory_resource::instance()));
    END_TEST();
}

MEMORYTEST(HostMemoryResource, HonorsRequestedAlignment)
{
    auto& r = host_memory_resource::instance();
    void* p = r.allocate(100, 256);
    EXPECT_TRUE(is_aligned(p, 256));
    r.deallocate(p, 100, 256);
    void* z = r.allocate(0, 8);
    EXPECT_NE(z, nullptr);
    r.deallocate(z, 0, 8);
    END_TEST();
}

MEMORYTEST(ArenaMemoryResource, PmrVectorServedFromArena)
{
    cpu_arena             arena(1 << 16);
    arena_memory_resource res(arena);
    {
        std::pmr::vector<int> v(&res);
        v.reserve(256);
        for (int i = 0; i < 256; ++i)
        {
            v.push_back(i);
        }
        EXPECT_EQ(v[255], 255);
        EXPECT_GE(arena.used(), 256 * sizeof(int));
    }
    EXPECT_GT(arena.used(), 0U);  // deallocate is a no-op
    arena.reset();
    EXPECT_EQ(arena.used(), 0U);
    END_TEST();
}

MEMORYTEST(ArenaMemoryResource, ExhaustionThrowsBadAlloc)
{
    cpu_arena             arena(256);
    arena_memory_resource res(arena);
    EXPECT_THROW((void)res.allocate(1024, 16), std::bad_alloc);
    END_TEST();
}

MEMORYTEST(ArenaMemoryResource, EqualityFollowsArena)
{
    cpu_arena             a1(1024);
    cpu_arena             a2(1024);
    arena_memory_resource r1(a1);
    arena_memory_resource r1b(a1);
    arena_memory_resource r2(a2);
    EXPECT_TRUE(r1.is_equal(r1b));
    EXPECT_FALSE(r1.is_equal(r2));
    EXPECT_FALSE(r1.is_equal(host_memory_resource::instance()));
    END_TEST();
}
