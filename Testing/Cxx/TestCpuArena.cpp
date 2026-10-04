/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#include "MemoryTest.h"
#include "common/cpu_arena.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

using namespace memory;

MEMORYTEST(CpuArena, default_empty_allocates_lazily)
{
    cpu_arena arena;
    EXPECT_TRUE(arena.empty());
    float* p = arena.alloc<float>(8);
    EXPECT_NE(p, nullptr);
    EXPECT_EQ(arena.used(), 8 * sizeof(float));
    END_TEST();
}

MEMORYTEST(CpuArena, preallocated_backing)
{
    cpu_arena arena(1024);
    EXPECT_EQ(arena.capacity(), 1024U);
    EXPECT_TRUE(arena.empty());
    END_TEST();
}

MEMORYTEST(CpuArena, alloc_returns_non_null)
{
    cpu_arena arena(4096);
    int* p = arena.alloc<int>(10);
    EXPECT_NE(p, nullptr);
    EXPECT_EQ(arena.used(), 10 * sizeof(int));
    END_TEST();
}

MEMORYTEST(CpuArena, alloc_bytes_alignment)
{
    cpu_arena arena(4096, MEMORY_ALIGNMENT);
    void* p = arena.alloc_bytes(1, MEMORY_ALIGNMENT);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(p) % MEMORY_ALIGNMENT, 0U);
    END_TEST();
}

MEMORYTEST(CpuArena, multiple_allocs_do_not_overlap)
{
    cpu_arena arena(4096);
    int* a = arena.alloc<int>(4);
    int* b = arena.alloc<int>(4);
    EXPECT_NE(a, b);
    EXPECT_GE(reinterpret_cast<uintptr_t>(b), reinterpret_cast<uintptr_t>(a) + 4 * sizeof(int));
    END_TEST();
}

MEMORYTEST(CpuArena, reset_reclaims_without_freeing_backing)
{
    cpu_arena arena(4096);
    arena.alloc<char>(100);
    EXPECT_EQ(arena.used(), 100U);
    arena.reset();
    EXPECT_EQ(arena.used(), 0U);
    EXPECT_EQ(arena.capacity(), 4096U);

    // Can allocate again after reset.
    int* p = arena.alloc<int>(4);
    EXPECT_NE(p, nullptr);
    END_TEST();
}

MEMORYTEST(CpuArena, exhausted_throws_bad_alloc)
{
    cpu_arena arena(64);
    // Consume all 64 bytes.
    (void)arena.alloc_bytes(64);
    bool threw = false;
    try
    {
        (void)arena.alloc_bytes(1);
    }
    catch (std::bad_alloc const&)
    {
        threw = true;
    }
    EXPECT_TRUE(threw);
    END_TEST();
}

MEMORYTEST(CpuArena, zero_byte_alloc_returns_null)
{
    cpu_arena arena(256);
    void* p = arena.alloc_bytes(0);
    EXPECT_EQ(p, nullptr);
    EXPECT_EQ(arena.used(), 0U);
    END_TEST();
}

MEMORYTEST(CpuArena, move_transfers_ownership)
{
    cpu_arena a(1024);
    int* p = a.alloc<int>(4);
    p[0]   = 7;

    cpu_arena b = std::move(a);
    EXPECT_EQ(b.capacity(), 1024U);
    EXPECT_GT(b.used(), 0U);
    EXPECT_TRUE(a.empty());
    EXPECT_EQ(a.capacity(), 0U);
    // b still has the same data
    EXPECT_EQ(p[0], 7);
    END_TEST();
}

// --- Plan 6.1: address alignment, growth limits, overflow, reset invalidation ---

MEMORYTEST(CpuArena, alignment_is_a_property_of_the_address)
{
    // The backing is only 64-byte aligned; larger alignments must still hold in memory.
    cpu_arena arena(1U << 16, 64);
    for (size_t align : {size_t{16}, size_t{64}, size_t{256}, size_t{1024}, size_t{4096}})
    {
        (void)arena.alloc_bytes(3, 8);  // misalign the cursor first
        void* p = arena.alloc_bytes(5, align);
        ASSERT_NE(p, nullptr);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(p) % align, 0U) << "alignment " << align;
    }
    END_TEST();
}

MEMORYTEST(CpuArena, typed_allocation_honours_over_aligned_types)
{
    struct alignas(256) wide
    {
        char bytes[256];
    };
    cpu_arena arena(1U << 14, 64);
    (void)arena.alloc_bytes(1, 8);
    wide* p = arena.alloc<wide>(2);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(p) % 256, 0U);
    END_TEST();
}

MEMORYTEST(CpuArena, invalid_alignments_are_rejected)
{
    cpu_arena arena(1024);
    EXPECT_THROW((void)arena.alloc_bytes(8, 0), std::invalid_argument);
    EXPECT_THROW((void)arena.alloc_bytes(8, 3), std::invalid_argument);
    EXPECT_THROW((void)arena.alloc_bytes(8, 8192), std::invalid_argument);
    EXPECT_EQ(arena.used(), 0U);
    EXPECT_THROW(cpu_arena(1024, 3), std::invalid_argument);
    EXPECT_THROW(cpu_arena(1024, 4), std::invalid_argument);  // below sizeof(void*)
    EXPECT_THROW(cpu_arena(1024, 64, 512), std::invalid_argument);  // max < capacity
    END_TEST();
}

MEMORYTEST(CpuArena, overflowing_requests_throw_and_leave_the_arena_unchanged)
{
    cpu_arena arena(256);
    (void)arena.alloc_bytes(10);
    size_t const before = arena.used();
    EXPECT_THROW((void)arena.alloc<int>(std::numeric_limits<size_t>::max() / 2), std::bad_alloc);
    EXPECT_THROW((void)arena.alloc_bytes(std::numeric_limits<size_t>::max()), std::bad_alloc);
    EXPECT_THROW(
        (void)arena.alloc_bytes(std::numeric_limits<size_t>::max() - 8, 4096), std::bad_alloc);
    EXPECT_EQ(arena.used(), before);
    EXPECT_EQ(arena.capacity(), 256U);

    cpu_arena growable(128, 64, std::numeric_limits<size_t>::max());
    EXPECT_THROW((void)growable.alloc_bytes(std::numeric_limits<size_t>::max() - 100), std::bad_alloc);
    EXPECT_EQ(growable.capacity(), 128U);
    END_TEST();
}

MEMORYTEST(CpuArena, fixed_arena_does_not_grow)
{
    cpu_arena arena(128);
    (void)arena.alloc_bytes(100);
    EXPECT_THROW((void)arena.alloc_bytes(100), std::bad_alloc);
    EXPECT_EQ(arena.capacity(), 128U);
    END_TEST();
}

MEMORYTEST(CpuArena, growth_adds_chunks_up_to_the_limit_and_keeps_pointers_valid)
{
    cpu_arena arena(128, 64, 1024);
    char*     first = static_cast<char*>(arena.alloc_bytes(100));
    std::memset(first, 0x11, 100);
    char* second = static_cast<char*>(arena.alloc_bytes(100));  // does not fit: new chunk
    std::memset(second, 0x22, 100);
    EXPECT_GT(arena.capacity(), 128U);
    EXPECT_LE(arena.capacity(), 1024U);
    EXPECT_EQ(static_cast<unsigned char>(first[99]), 0x11U);  // the first chunk did not move

    size_t served = 200;
    bool   threw  = false;
    try
    {
        for (int i = 0; i < 64; ++i)
        {
            (void)arena.alloc_bytes(100);
            served += 100;
        }
    }
    catch (std::bad_alloc const&)
    {
        threw = true;
    }
    EXPECT_TRUE(threw);
    EXPECT_LE(arena.capacity(), 1024U);  // the limit holds
    EXPECT_GE(served, 500U);
    size_t const used = arena.used();
    EXPECT_THROW((void)arena.alloc_bytes(100), std::bad_alloc);
    EXPECT_EQ(arena.used(), used);  // a refused request changes nothing
    END_TEST();
}

MEMORYTEST(CpuArena, reset_invalidates_and_returns_extra_chunks)
{
    cpu_arena arena(128, 64, 4096);
    void*     first = arena.alloc_bytes(100);
    (void)arena.alloc_bytes(200);  // second chunk
    ASSERT_GT(arena.capacity(), 128U);
    auto const generation = arena.generation();

    arena.reset();
    EXPECT_EQ(arena.generation(), generation + 1);
    EXPECT_EQ(arena.capacity(), 128U);  // extra chunks went back
    EXPECT_TRUE(arena.empty());
#ifndef NDEBUG
    EXPECT_EQ(*static_cast<unsigned char*>(first), 0xDDU);  // stale reads are visible
#endif
    EXPECT_EQ(arena.alloc_bytes(100), first);  // the memory is reused from the same address
    END_TEST();
}

MEMORYTEST(CpuArena, lazy_first_request_larger_than_the_default_chunk)
{
    cpu_arena arena;
    void*     p = arena.alloc_bytes(cpu_arena::kDefaultLazySize * 2);
    EXPECT_NE(p, nullptr);
    EXPECT_GE(arena.capacity(), cpu_arena::kDefaultLazySize * 2);
    END_TEST();
}

MEMORYTEST(CpuArena, moved_from_arena_is_empty_and_reusable)
{
    cpu_arena a(256);
    (void)a.alloc_bytes(16);
    cpu_arena b = std::move(a);
    EXPECT_EQ(a.capacity(), 0U);  // NOLINT(bugprone-use-after-move)
    EXPECT_NE(a.alloc_bytes(8), nullptr);  // lazily re-acquires
    EXPECT_EQ(b.capacity(), 256U);
    END_TEST();
}
