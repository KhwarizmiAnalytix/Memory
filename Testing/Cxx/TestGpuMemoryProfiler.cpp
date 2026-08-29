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

#include "MemoryTest.h"

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP || MEMORY_HAS_METAL

#include <cstddef>
#include <new>
#include <stdexcept>

#include "gpu/caching_allocator.h"
#include "profiler/gpu_memory_snapshot.h"

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
#include "gpu/gpu_runtime.h"
#elif MEMORY_HAS_METAL
#include "gpu/metal/metal_buffer_allocator.h"
#endif

using namespace memory;
using namespace memory::gpu;

namespace
{

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP

bool gpu_device_available()
{
    int               device_count = 0;
    const cudaError_t err          = cudaGetDeviceCount(&device_count);
    return err == cudaSuccess && device_count > 0;
}

#elif MEMORY_HAS_METAL

bool gpu_device_available() { return memory::metal::device_available(); }

#endif

size_t count_action(const gpu_memory_snapshot& snap, gpu_memory_trace_action action)
{
    size_t n = 0;
    for (const gpu_memory_trace_entry& entry : snap.device_trace)
    {
        if (entry.action == action)
        {
            ++n;
        }
    }
    return n;
}

size_t allocated_bytes(const gpu_memory_snapshot& snap)
{
    size_t n = 0;
    for (const gpu_memory_segment_info& seg : snap.segments)
    {
        n += seg.allocated_size;
    }
    return n;
}

}  // namespace

class GpuMemoryProfilerTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if (!gpu_device_available())
        {
            GTEST_SKIP() << "No GPU device available";
        }
    }
};

MEMORYTEST_F(GpuMemoryProfilerTest, snapshot_lists_live_allocation)
{
    caching_allocator alloc(0);
    void*             ptr = alloc.allocate(1024);
    ASSERT_NE(nullptr, ptr);

    gpu_memory_snapshot const snap = alloc.snapshot();
    EXPECT_EQ(1U, snap.segments.size());
    EXPECT_EQ(1024U, allocated_bytes(snap));
    EXPECT_TRUE(snap.device_trace.empty());

    bool found_live = false;
    for (const gpu_memory_block_info& block : snap.segments[0].blocks)
    {
        if (block.allocated)
        {
            EXPECT_EQ(ptr, block.address);
            EXPECT_EQ(1024U, block.size);
            EXPECT_EQ(1024U, block.requested_size);
            EXPECT_TRUE(block.active);
            found_live = true;
        }
    }
    EXPECT_TRUE(found_live);

    alloc.deallocate(ptr, 1024);
    END_TEST();
}

MEMORYTEST_F(GpuMemoryProfilerTest, history_disabled_by_default)
{
    caching_allocator alloc(0);
    void*             ptr = alloc.allocate(512);
    ASSERT_NE(nullptr, ptr);
    alloc.deallocate(ptr, 512);

    gpu_memory_snapshot const snap = alloc.snapshot();
    EXPECT_TRUE(snap.device_trace.empty());
    END_TEST();
}

MEMORYTEST_F(GpuMemoryProfilerTest, record_memory_history_captures_alloc_free_and_segment)
{
    caching_allocator alloc(0);
    alloc.record_memory_history(true, 64);

    void* ptr = alloc.allocate(1024);
    ASSERT_NE(nullptr, ptr);
    alloc.deallocate(ptr, 1024);

    gpu_memory_snapshot const snap = alloc.snapshot();
    EXPECT_GE(count_action(snap, gpu_memory_trace_action::segment_alloc), 1U);
    EXPECT_GE(count_action(snap, gpu_memory_trace_action::alloc), 1U);
    EXPECT_GE(count_action(snap, gpu_memory_trace_action::free_requested), 1U);
    EXPECT_GE(count_action(snap, gpu_memory_trace_action::free_completed), 1U);
    EXPECT_EQ(1U, count_action(snap, gpu_memory_trace_action::snapshot));

    alloc.record_memory_history(false);
    END_TEST();
}

MEMORYTEST_F(GpuMemoryProfilerTest, history_ring_drops_oldest)
{
    caching_allocator alloc(0);
    alloc.record_memory_history(true, 2);

    void* a = alloc.allocate(512);
    void* b = alloc.allocate(512);
    ASSERT_NE(nullptr, a);
    ASSERT_NE(nullptr, b);

    gpu_memory_snapshot const snap = alloc.snapshot();
    EXPECT_EQ(2U, snap.device_trace.size());

    alloc.deallocate(a, 512);
    alloc.deallocate(b, 512);
    alloc.record_memory_history(false);
    END_TEST();
}

MEMORYTEST_F(GpuMemoryProfilerTest, oom_from_memory_fraction_records_trace)
{
    caching_allocator alloc(0);
    alloc.record_memory_history(true, 32);

    size_t const total = alloc.device_total_memory();
    ASSERT_GT(total, 0U);
    alloc.set_memory_fraction(1.0 / static_cast<double>(total));

    EXPECT_THROW(alloc.allocate(1024), std::bad_alloc);
    EXPECT_GE(alloc.stats().num_ooms.load(), 1U);

    gpu_memory_snapshot const snap = alloc.snapshot();
    EXPECT_GE(count_action(snap, gpu_memory_trace_action::oom), 1U);

    alloc.set_memory_fraction(1.0);
    alloc.record_memory_history(false);
    END_TEST();
}

MEMORYTEST_F(GpuMemoryProfilerTest, process_wide_helpers_round_trip)
{
    struct guard
    {
        ~guard()
        {
            record_memory_history(false, 0, 0);
            empty_cache(0);
        }
    } const cleanup;

    record_memory_history(true, 32, 0);
    void* ptr = caching_allocator_for_device(0).allocate(256);
    ASSERT_NE(nullptr, ptr);

    gpu_memory_snapshot const snap = memory_snapshot(0);
    EXPECT_FALSE(snap.device_trace.empty());
    EXPECT_GE(allocated_bytes(snap), 256U);

    caching_allocator_for_device(0).deallocate(ptr, 256);
    END_TEST();
}

#endif  // MEMORY_HAS_CUDA || MEMORY_HAS_HIP || MEMORY_HAS_METAL
