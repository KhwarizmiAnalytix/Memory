/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#include "MemoryTest.h"
#include "common/cpu_arena.h"

#include <cstdint>
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
