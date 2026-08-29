/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * This file is part of XSigma and is licensed under a dual-license model:
 *
 *   - Open-source License (GPLv3):
 *       Free for personal, academic, and research use under the terms of
 *       the GNU General Public License v3.0 or later.
 *
 *   - Commercial License:
 *       A commercial license is required for proprietary, closed-source,
 *       or SaaS usage. Contact us to obtain a commercial agreement.
 *
 * Contact: licensing@xsigma.co.uk
 * Website: https://www.xsigma.co.uk
 */

#include <cstddef>

#include "MemoryTest.h"
#include "helper/memory_allocator.h"
#include "profiler/profiled_cpu_memory_reporter.h"

using namespace memory;

class ProfiledCPUMemoryReporterTest : public ::testing::Test
{
protected:
    void TearDown() override { reporter_.reset(); }

    profiled_cpu_memory_reporter reporter_;
};

MEMORYTEST_F(ProfiledCPUMemoryReporterTest, record_allocation_zero_or_null_is_noop)
{
    reporter_.record_allocation(nullptr, 64);
    char dummy = 0;
    reporter_.record_allocation(&dummy, 0);
    EXPECT_EQ(reporter_.allocated(), 0U);
    EXPECT_EQ(reporter_.tracked_blocks(), 0U);
    END_TEST();
}

MEMORYTEST_F(ProfiledCPUMemoryReporterTest, record_allocation_and_deallocation_update_totals)
{
    char a = 0;
    char b = 0;
    reporter_.record_allocation(&a, 128);
    reporter_.record_allocation(&b, 256);
    EXPECT_EQ(reporter_.allocated(), 384U);
    EXPECT_EQ(reporter_.tracked_blocks(), 2U);

    reporter_.record_deallocation(&a);
    EXPECT_EQ(reporter_.allocated(), 256U);
    EXPECT_EQ(reporter_.tracked_blocks(), 1U);

    reporter_.record_deallocation(&b);
    EXPECT_EQ(reporter_.allocated(), 0U);
    EXPECT_EQ(reporter_.tracked_blocks(), 0U);
    END_TEST();
}

MEMORYTEST_F(ProfiledCPUMemoryReporterTest, record_deallocation_without_event_cleans_state)
{
    char dummy = 0;
    reporter_.record_allocation(&dummy, 128);
    EXPECT_EQ(reporter_.allocated(), 128U);
    EXPECT_EQ(reporter_.tracked_blocks(), 1U);
    EXPECT_TRUE(reporter_.has_tracked_blocks());

    reporter_.record_deallocation(&dummy, false);
    EXPECT_EQ(reporter_.allocated(), 0U);
    EXPECT_EQ(reporter_.tracked_blocks(), 0U);
    EXPECT_FALSE(reporter_.has_tracked_blocks());
    END_TEST();
}

MEMORYTEST_F(ProfiledCPUMemoryReporterTest, repeated_allocation_for_same_address_replaces_size)
{
    char dummy = 0;
    reporter_.record_allocation(&dummy, 128);
    reporter_.record_allocation(&dummy, 256);
    EXPECT_EQ(reporter_.allocated(), 256U);
    EXPECT_EQ(reporter_.tracked_blocks(), 1U);

    reporter_.record_deallocation(&dummy, false);
    EXPECT_EQ(reporter_.allocated(), 0U);
    EXPECT_EQ(reporter_.tracked_blocks(), 0U);
    END_TEST();
}

MEMORYTEST_F(ProfiledCPUMemoryReporterTest, record_deallocation_unknown_ptr_is_noop)
{
    char dummy = 0;
    reporter_.record_deallocation(&dummy);
    reporter_.record_deallocation(nullptr);
    EXPECT_EQ(reporter_.allocated(), 0U);
    EXPECT_EQ(reporter_.tracked_blocks(), 0U);
    END_TEST();
}

MEMORYTEST_F(ProfiledCPUMemoryReporterTest, record_out_of_memory_increments_counter)
{
    reporter_.record_out_of_memory(0);
    EXPECT_EQ(reporter_.num_ooms(), 0U);

    reporter_.record_out_of_memory(4096);
    EXPECT_EQ(reporter_.num_ooms(), 1U);
    EXPECT_EQ(reporter_.allocated(), 0U);
    END_TEST();
}

MEMORYTEST_F(ProfiledCPUMemoryReporterTest, reset_clears_table_and_counters)
{
    char dummy = 0;
    reporter_.record_allocation(&dummy, 64);
    reporter_.record_out_of_memory(1024);
    reporter_.reset();
    EXPECT_EQ(reporter_.allocated(), 0U);
    EXPECT_EQ(reporter_.tracked_blocks(), 0U);
    EXPECT_EQ(reporter_.num_ooms(), 0U);
    END_TEST();
}

MEMORYTEST(ProfiledCPUMemoryReporter, allocate_free_untracked_without_profiler_session)
{
    const std::size_t before = cpu_memory_reporter().allocated();

    void* ptr = cpu::memory_allocator::allocate(1024, 64);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(cpu_memory_reporter().allocated(), before);
    cpu::memory_allocator::free(ptr);
    EXPECT_EQ(cpu_memory_reporter().allocated(), before);
    END_TEST();
}

MEMORYTEST(ProfiledCPUMemoryReporter, huge_allocation_untracked_without_profiler_session)
{
    const std::size_t ooms_before = cpu_memory_reporter().num_ooms();

    void* ptr = cpu::memory_allocator::allocate(static_cast<std::size_t>(-1) / 2, 64);
    EXPECT_EQ(ptr, nullptr);
    EXPECT_EQ(cpu_memory_reporter().num_ooms(), ooms_before);
    END_TEST();
}

#if MEMORY_HAS_MIMALLOC
MEMORYTEST(ProfiledCPUMemoryReporter, allocate_mi_bypass_is_not_tracked)
{
    const std::size_t before = cpu_memory_reporter().allocated();

    void* ptr = cpu::memory_allocator::allocate_mi(512, 64);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(cpu_memory_reporter().allocated(), before);
    cpu::memory_allocator::free_mi(ptr);
    EXPECT_EQ(cpu_memory_reporter().allocated(), before);
    END_TEST();
}
#endif
