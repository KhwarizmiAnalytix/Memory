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

// gpu/caching_allocator_config.h is header-only and shared by the CUDA/HIP
// and Metal caching allocators, but has no driver dependency itself, so this
// file (deliberately not named TestCuda*/TestGpu*/TestHip*/TestMetal*, see
// Testing/Cxx/CMakeLists.txt's glob filters) builds and runs under every
// MEMORY_GPU_BACKEND, including "none".

#include <limits>

#include "MemoryTest.h"
#include "gpu/caching_allocator_config.h"

using namespace memory::gpu::caching_config;

MEMORYTEST(CachingAllocatorConfig, RoundRequestSizeBelowMinRoundsUpToMin)
{
    EXPECT_EQ(kMinBlockSize, round_request_size(0));
    EXPECT_EQ(kMinBlockSize, round_request_size(1));
    EXPECT_EQ(kMinBlockSize, round_request_size(kMinBlockSize));
    END_TEST();
}

MEMORYTEST(CachingAllocatorConfig, RoundRequestSizeRoundsUpToMultiple)
{
    EXPECT_EQ(kMinBlockSize * 2, round_request_size(kMinBlockSize + 1));
    EXPECT_EQ(kMinBlockSize * 3, round_request_size(kMinBlockSize * 2 + 200));
    END_TEST();
}

MEMORYTEST(CachingAllocatorConfig, SegmentSizeForSizeClasses)
{
    EXPECT_EQ(kSmallBuffer, segment_size_for(1));
    EXPECT_EQ(kSmallBuffer, segment_size_for(kSmallSize));
    EXPECT_EQ(kLargeBuffer, segment_size_for(kSmallSize + 1));
    EXPECT_EQ(kLargeBuffer, segment_size_for(kMinLargeAlloc - 1));
    EXPECT_EQ(kMinLargeAlloc, segment_size_for(kMinLargeAlloc));
    EXPECT_EQ(kRoundLarge * 6, segment_size_for(kRoundLarge * 5 + 1));
    END_TEST();
}

// Regression: round_request_size(SIZE_MAX) and segment_size_for(SIZE_MAX)
// used to compute `size + unit - 1` unchecked, wrapping past SIZE_MAX back
// to a value near zero -- silently turning an unserviceable request into one
// that looked tiny and cheap. Both must now saturate to SIZE_MAX instead.
MEMORYTEST(CachingAllocatorConfig, RoundRequestSizeAtSizeMaxSaturates)
{
    constexpr size_t kMax = std::numeric_limits<size_t>::max();
    EXPECT_EQ(kMax, round_request_size(kMax));
    EXPECT_EQ(kMax, round_request_size(kMax - kMinBlockSize + 2));
    END_TEST();
}

MEMORYTEST(CachingAllocatorConfig, SegmentSizeForAtSizeMaxSaturates)
{
    constexpr size_t kMax = std::numeric_limits<size_t>::max();
    EXPECT_EQ(kMax, segment_size_for(kMax));
    EXPECT_EQ(kMax, segment_size_for(kMax - kRoundLarge + 2));
    END_TEST();
}

MEMORYTEST(CachingAllocatorConfig, RoundUpSaturatingExactMultipleIsUnchanged)
{
    EXPECT_EQ(4096U, round_up_saturating(4096U, 512U));
    EXPECT_EQ(512U, round_up_saturating(1U, 512U));
    END_TEST();
}
