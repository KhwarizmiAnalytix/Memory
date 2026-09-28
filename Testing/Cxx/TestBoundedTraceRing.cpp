/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#include "MemoryTest.h"
#include "profiler/bounded_trace_ring.h"
#include "profiler/gpu_memory_snapshot.h"

#include <cstdint>
#include <vector>

using namespace memory::gpu;

MEMORYTEST(BoundedTraceRing, default_is_empty)
{
    bounded_trace_ring<int> ring{4};
    EXPECT_TRUE(ring.empty());
    EXPECT_EQ(ring.size(), 0U);
    EXPECT_EQ(ring.capacity(), 4U);
    EXPECT_EQ(ring.entries_lost(), 0U);
    END_TEST();
}

MEMORYTEST(BoundedTraceRing, push_and_copy)
{
    bounded_trace_ring<int> ring{4};
    ring.push(10);
    ring.push(20);
    ring.push(30);
    auto v = ring.copy();
    EXPECT_EQ(v.size(), 3U);
    EXPECT_EQ(v[0], 10);
    EXPECT_EQ(v[1], 20);
    EXPECT_EQ(v[2], 30);
    END_TEST();
}

MEMORYTEST(BoundedTraceRing, overflow_wraps_and_counts_lost)
{
    bounded_trace_ring<int> ring{3};
    ring.push(1);
    ring.push(2);
    ring.push(3);
    EXPECT_EQ(ring.entries_lost(), 0U);
    ring.push(4);  // overwrites 1
    EXPECT_EQ(ring.entries_lost(), 1U);
    ring.push(5);  // overwrites 2
    auto v = ring.copy();
    EXPECT_EQ(v.size(), 3U);
    EXPECT_EQ(v[0], 3);
    EXPECT_EQ(v[1], 4);
    EXPECT_EQ(v[2], 5);
    END_TEST();
}

MEMORYTEST(BoundedTraceRing, clear_resets_state)
{
    bounded_trace_ring<int> ring{3};
    ring.push(1);
    ring.push(2);
    ring.push(3);
    ring.push(4);  // causes 1 loss
    ring.clear();
    EXPECT_TRUE(ring.empty());
    EXPECT_EQ(ring.entries_lost(), 0U);
    auto v = ring.copy();
    EXPECT_TRUE(v.empty());
    END_TEST();
}

MEMORYTEST(BoundedTraceRing, resize_preserves_recent_entries)
{
    bounded_trace_ring<int> ring{5};
    for (int i = 1; i <= 5; ++i) ring.push(i);
    ring.resize(3);
    auto v = ring.copy();
    EXPECT_EQ(v.size(), 3U);
    EXPECT_EQ(v[0], 3);
    EXPECT_EQ(v[1], 4);
    EXPECT_EQ(v[2], 5);
    END_TEST();
}

MEMORYTEST(BoundedTraceRing, zero_capacity_counts_lost)
{
    bounded_trace_ring<int> ring{0};
    ring.push(1);
    EXPECT_EQ(ring.entries_lost(), 1U);
    EXPECT_TRUE(ring.empty());
    END_TEST();
}

// Tests for gpu_memory_history using the new bounded_trace_ring backend.

MEMORYTEST(GpuMemoryHistory, disabled_records_nothing)
{
    gpu_memory_history h;
    h.record(gpu_memory_trace_action::alloc, nullptr, 1024, 0, 1024, 1024, 0);
    auto v = h.copy();
    EXPECT_TRUE(v.empty());
    END_TEST();
}

MEMORYTEST(GpuMemoryHistory, enabled_records_with_new_fields)
{
    gpu_memory_history h;
    h.set_enabled(true, 10);
    void* addr = reinterpret_cast<void*>(0xDEAD);
    h.record(gpu_memory_trace_action::alloc, addr, 512, 512, 512, 1024, 0, /*alloc_id=*/42);
    auto v = h.copy();
    ASSERT_EQ(v.size(), 1U);
    EXPECT_EQ(v[0].address, addr);
    EXPECT_EQ(v[0].size, 512U);
    EXPECT_EQ(v[0].requested_size, 512U);
    EXPECT_EQ(v[0].alloc_id, 42U);
    EXPECT_EQ(v[0].sequence_num, 0U);  // first entry
    EXPECT_NE(v[0].timestamp_ns, 0);
    EXPECT_EQ(v[0].schema_version, kTraceEntrySchemaVersion);
    END_TEST();
}

MEMORYTEST(GpuMemoryHistory, sequence_numbers_are_monotonic)
{
    gpu_memory_history h;
    h.set_enabled(true, 10);
    for (int i = 0; i < 5; ++i)
    {
        h.record(gpu_memory_trace_action::alloc, nullptr, 64, 0, 64, 64, 0);
    }
    auto v = h.copy();
    ASSERT_EQ(v.size(), 5U);
    for (size_t i = 1; i < v.size(); ++i)
    {
        EXPECT_GT(v[i].sequence_num, v[i - 1].sequence_num);
    }
    END_TEST();
}

MEMORYTEST(GpuMemoryHistory, legacy_record_overload_sets_zero_requested_size)
{
    gpu_memory_history h;
    h.set_enabled(true, 10);
    h.record(gpu_memory_trace_action::alloc, nullptr, 256, 256ULL, 1024ULL, 0LL);
    auto v = h.copy();
    ASSERT_EQ(v.size(), 1U);
    EXPECT_EQ(v[0].requested_size, 0U);
    EXPECT_EQ(v[0].alloc_id, 0U);
    EXPECT_EQ(v[0].size, 256U);
    END_TEST();
}
