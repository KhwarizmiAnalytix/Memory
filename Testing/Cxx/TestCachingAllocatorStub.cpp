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

// gpu/cuda_caching_allocator.cpp compiles a trivial stub Impl (its `#else`
// branch) on Metal builds: MEMORY_HAS_CUDA and MEMORY_HAS_HIP are both off,
// but gpu/*.cpp is still linked. HIP builds use the real Impl via
// gpu/gpu_runtime.h. This file is deliberately *not* named
// TestCuda*/TestGpu*/TestHip*/TestMetal* so the Testing/Cxx CMakeLists.txt /
// BUILD.bazel glob filters don't exclude it.
//
// MEMORY_GPU_BACKEND=none (the default) excludes gpu/*.cpp from the Memory
// library target entirely (see Library/Memory/CMakeLists.txt), so
// cuda_caching_allocator isn't even linked into libMemory there -- this
// file must not build in that configuration either.
//
// The real CUDA/HIP-backed Impl is covered by TestCudaCachingAllocator.cpp
// (CUDA) and by allocator/device tests under HIP.

#include "common/memory_macros.h"

#if MEMORY_HAS_METAL

#include <stdexcept>

#include "MemoryTest.h"
#include "gpu/cuda_caching_allocator.h"

using namespace memory;
using namespace memory::gpu;

MEMORYTEST(CachingAllocatorStub, ConstructAndAccessors)
{
    const cuda_caching_allocator allocator(2, 1024);
    EXPECT_EQ(allocator.device(), 2);
    EXPECT_EQ(allocator.max_cached_bytes(), 1024U);
    END_TEST();
}

MEMORYTEST(CachingAllocatorStub, DefaultConstructor)
{
    const cuda_caching_allocator allocator;
    EXPECT_EQ(allocator.device(), 0);
    END_TEST();
}

MEMORYTEST(CachingAllocatorStub, MoveConstructAndAssign)
{
    cuda_caching_allocator source(1, 2048);
    cuda_caching_allocator moved(std::move(source));
    EXPECT_EQ(moved.device(), 1);
    EXPECT_EQ(moved.max_cached_bytes(), 2048U);

    cuda_caching_allocator target;
    target = std::move(moved);
    EXPECT_EQ(target.device(), 1);
    END_TEST();
}

MEMORYTEST(CachingAllocatorStub, AllocateZeroReturnsNullptr)
{
    cuda_caching_allocator allocator;
    EXPECT_EQ(allocator.allocate(0), nullptr);
    END_TEST();
}

MEMORYTEST(CachingAllocatorStub, AllocateNonZeroThrows)
{
    cuda_caching_allocator allocator;
    // The stub Impl always throws: no CUDA driver is compiled in to serve
    // a real allocation.
    EXPECT_THROW(allocator.allocate(64), std::runtime_error);
    END_TEST();
}

MEMORYTEST(CachingAllocatorStub, DeallocateIsNoOp)
{
    cuda_caching_allocator allocator;
    EXPECT_NO_THROW({ allocator.deallocate(nullptr, 0); });
    END_TEST();
}

MEMORYTEST(CachingAllocatorStub, RecordStreamIsNoOp)
{
    cuda_caching_allocator allocator;
    EXPECT_NO_THROW({ allocator.record_stream(nullptr, nullptr); });
    END_TEST();
}

MEMORYTEST(CachingAllocatorStub, EmptyCacheIsNoOp)
{
    cuda_caching_allocator allocator;
    EXPECT_NO_THROW({ allocator.empty_cache(); });
    END_TEST();
}

MEMORYTEST(CachingAllocatorStub, SetMaxCachedBytes)
{
    cuda_caching_allocator allocator(0, 100);
    allocator.set_max_cached_bytes(500);
    EXPECT_EQ(allocator.max_cached_bytes(), 500U);
    END_TEST();
}

MEMORYTEST(CachingAllocatorStub, FreeMemoryCallbacksAreNoOps)
{
    cuda_caching_allocator allocator;
    bool                   called = false;
    EXPECT_NO_THROW({
        allocator.add_free_memory_callback(
            [&called]()
            {
                called = true;
                return true;
            });
        allocator.clear_free_memory_callbacks();
    });
    // The stub Impl never invokes registered callbacks (no cache-miss path
    // exists without a real driver behind it).
    EXPECT_FALSE(called);
    END_TEST();
}

MEMORYTEST(CachingAllocatorStub, StatsReturnsDefault)
{
    const cuda_caching_allocator allocator;
    const unified_cache_stats    stats = allocator.stats();
    EXPECT_EQ(stats.cache_hits.load(), 0U);
    EXPECT_EQ(stats.cache_misses.load(), 0U);
    END_TEST();
}

#endif  // MEMORY_HAS_METAL
