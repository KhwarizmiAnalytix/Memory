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

#include <cstdint>  // for uintptr_t
#include <cstdlib>  // for size_t
#include <cstring>  // for memset
#include <vector>   // for vector

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

    // Test zero-size allocation
    // fixme: ASSERT_ANY_THROW({ memory::cpu::memory_allocator::allocate(0, 64); });

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
