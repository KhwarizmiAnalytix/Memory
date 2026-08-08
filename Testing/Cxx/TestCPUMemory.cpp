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

#include <cstdint>    // for uintptr_t
#include <cstdlib>    // for size_t
#include <cstring>    // for memset
#include <stdexcept>  // for runtime_error
#include <vector>     // for vector

#include "MemoryTest.h"               // for MEMORYTEST, END_TEST, IsAligned
#include "helper/memory_allocator.h"  // for free, allocate, usable_size
#include "util/memory_exception.h"    // for MEMORY_LOG_INFO

using namespace memory;

// ============================================================================
// MEMORY PORT TESTS (helper/memory_allocator.h)
// ============================================================================

// Test memory port basic operations
MEMORYTEST(MemoryPortTest, BasicMemoryOperations)
{
    MEMORY_LOG_INFO("Testing memory port basic operations...");

    // Test aligned malloc/free (these are available)
    void* ptr2 = cpu::memory_allocator::allocate(2048, 64);
    EXPECT_NE(nullptr, ptr2);
    EXPECT_TRUE(IsAligned(ptr2, 64));
    cpu::memory_allocator::free(ptr2);

    MEMORY_LOG_INFO("Memory port basic operations tests completed successfully");
}

// Test memory port alignment requirements
MEMORYTEST(MemoryPortTest, AlignmentRequirements)
{
    MEMORY_LOG_INFO("Testing memory port alignment requirements...");

    // Test various alignment values
    std::vector<int> alignments = {8, 16, 32, 64, 128, 256, 512, 1024};

    for (int alignment : alignments)
    {
        void* ptr = cpu::memory_allocator::allocate(1024, alignment);
        EXPECT_NE(nullptr, ptr);
        EXPECT_TRUE(IsAligned(ptr, alignment));

        cpu::memory_allocator::free(ptr);
    }

    MEMORY_LOG_INFO("Memory port alignment requirements tests completed successfully");
}

// Test memory port edge cases
MEMORYTEST(MemoryPortTest, EdgeCases)
{
    MEMORY_LOG_INFO("Testing memory port edge cases...");

    // Test null pointer free (should not crash)
    cpu::memory_allocator::free(nullptr);

    // Test zero-size allocation: MEMORY_CHECK rejects it via std::runtime_error.
    EXPECT_THROW({ cpu::memory_allocator::allocate(0, 64); }, std::runtime_error);

    MEMORY_LOG_INFO("Memory port edge cases tests completed successfully");
}

// Test usable_size reporting of the raw CPU allocation backend
MEMORYTEST(MemoryPortTest, UsableSize)
{
    MEMORY_LOG_INFO("Testing memory port usable_size...");

    // Null pointer must report zero
    EXPECT_EQ(cpu::memory_allocator::usable_size(nullptr), 0U);

    void* ptr = cpu::memory_allocator::allocate(1000, 64);
    EXPECT_NE(nullptr, ptr);

    // Either the backend cannot report sizes (0) or it must report at
    // least the requested size.
    const std::size_t usable = cpu::memory_allocator::usable_size(ptr);
    EXPECT_TRUE(usable == 0 || usable >= 1000);

    cpu::memory_allocator::free(ptr);

    MEMORY_LOG_INFO("Memory port usable_size tests completed successfully");
}

// Test zero-initialized allocation policy
MEMORYTEST(MemoryPortTest, ZeroInitialization)
{
    MEMORY_LOG_INFO("Testing memory port zero-initialized allocation...");

    constexpr std::size_t kBytes = 4096;
    void*                 ptr    = cpu::memory_allocator::allocate_zero(kBytes, 64);
    EXPECT_NE(nullptr, ptr);
    EXPECT_TRUE(IsAligned(ptr, 64));

    const auto* bytes = static_cast<const unsigned char*>(ptr);
    for (std::size_t i = 0; i < kBytes; ++i)
    {
        EXPECT_EQ(bytes[i], 0);
    }

    cpu::memory_allocator::free(ptr);

    MEMORY_LOG_INFO("Memory port zero-initialized allocation tests completed successfully");
}

// Test the debug-pattern initialization policy (0xCC fill)
MEMORYTEST(MemoryPortTest, PatternInitialization)
{
    MEMORY_LOG_INFO("Testing memory port pattern-initialized allocation...");

    constexpr std::size_t kBytes = 256;
    void*                 ptr =
        cpu::memory_allocator::allocate(kBytes, 64, cpu::memory_allocator::init_policy_enum::PATTERN);
    ASSERT_NE(nullptr, ptr);

#ifndef NDEBUG
    const auto* bytes = static_cast<const unsigned char*>(ptr);
    for (std::size_t i = 0; i < kBytes; ++i)
    {
        EXPECT_EQ(bytes[i], 0xCC);
    }
#endif

    cpu::memory_allocator::free(ptr);

    MEMORY_LOG_INFO("Memory port pattern-initialized allocation tests completed successfully");
}

// A request larger than the address space must fail gracefully (nullptr),
// not throw or crash -- allocate() only converts nullptr to bad_alloc one
// layer up, in memory::allocator<T>::allocate() (see TestAllocator.cpp).
MEMORYTEST(MemoryPortTest, HugeAllocationReturnsNullptr)
{
    MEMORY_LOG_INFO("Testing memory port huge allocation failure...");

    void* ptr = cpu::memory_allocator::allocate(static_cast<std::size_t>(-1) / 2, 64);
    EXPECT_EQ(ptr, nullptr);

    MEMORY_LOG_INFO("Memory port huge allocation failure tests completed successfully");
}

// Test the alignment validation helper directly (power-of-2 >= sizeof(void*))
MEMORYTEST(MemoryPortTest, IsValidAlignment)
{
    MEMORY_LOG_INFO("Testing memory port is_valid_alignment...");

    EXPECT_TRUE(cpu::memory_allocator::is_valid_alignment(sizeof(void*)));
    EXPECT_TRUE(cpu::memory_allocator::is_valid_alignment(64));
    EXPECT_TRUE(cpu::memory_allocator::is_valid_alignment(4096));
    EXPECT_FALSE(cpu::memory_allocator::is_valid_alignment(0));
    EXPECT_FALSE(cpu::memory_allocator::is_valid_alignment(3));  // not a power of 2
    EXPECT_FALSE(cpu::memory_allocator::is_valid_alignment(sizeof(void*) / 2));  // too small

    MEMORY_LOG_INFO("Memory port is_valid_alignment tests completed successfully");
}

// Test the default_alignment() accessor
MEMORYTEST(MemoryPortTest, DefaultAlignment)
{
    MEMORY_LOG_INFO("Testing memory port default_alignment...");

    EXPECT_GT(cpu::memory_allocator::default_alignment(), 0U);
    EXPECT_TRUE(cpu::memory_allocator::is_valid_alignment(cpu::memory_allocator::default_alignment()));

    MEMORY_LOG_INFO("Memory port default_alignment tests completed successfully");
}

// Test the mimalloc-specific and TBB-specific allocation entry points.
// Exactly one backend is compiled in at a time; the other degrades to a
// documented nullptr/no-op stub rather than being unreachable.
MEMORYTEST(MemoryPortTest, BackendSpecificAllocateMi)
{
    MEMORY_LOG_INFO("Testing memory port allocate_mi/free_mi...");

    void* ptr = cpu::memory_allocator::allocate_mi(1024, 64);
#if MEMORY_HAS_MIMALLOC
    EXPECT_NE(ptr, nullptr);
    EXPECT_TRUE(IsAligned(ptr, 64));
#else
    EXPECT_EQ(ptr, nullptr);
#endif
    cpu::memory_allocator::free_mi(ptr);

    MEMORY_LOG_INFO("Memory port allocate_mi/free_mi tests completed successfully");
}

MEMORYTEST(MemoryPortTest, BackendSpecificAllocateTbb)
{
    MEMORY_LOG_INFO("Testing memory port allocate_tbb/free_tbb...");

    void* ptr = cpu::memory_allocator::allocate_tbb(1024, 64);
#if MEMORY_HAS_TBB
    EXPECT_NE(ptr, nullptr);
    EXPECT_TRUE(IsAligned(ptr, 64));
#else
    EXPECT_EQ(ptr, nullptr);
#endif
    cpu::memory_allocator::free_tbb(ptr);

    MEMORY_LOG_INFO("Memory port allocate_tbb/free_tbb tests completed successfully");
}
