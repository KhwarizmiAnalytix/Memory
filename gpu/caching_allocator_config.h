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

#pragma once

// Shared size-class policy for CUDA/HIP and Metal caching allocators (PyTorch
// CUDACachingAllocator constants). Kept header-only so both TUs share one
// definition without pulling in driver headers.

#include <cstddef>

namespace memory::gpu::caching_config
{
// Requests are rounded to 512-byte multiples; small requests (<= 1 MiB) pack
// into 2 MiB segments, 1-10 MiB into 20 MiB segments, larger requests round up
// to 2 MiB multiples — one driver allocation per segment.
constexpr size_t kMinBlockSize  = 512;
constexpr size_t kSmallSize     = 1048576;   // 1 MiB
constexpr size_t kSmallBuffer   = 2097152;   // 2 MiB
constexpr size_t kMinLargeAlloc = 10485760;  // 10 MiB
constexpr size_t kLargeBuffer   = 20971520;  // 20 MiB
constexpr size_t kRoundLarge    = 2097152;   // 2 MiB

inline size_t round_request_size(size_t size)
{
    if (size < kMinBlockSize)
    {
        return kMinBlockSize;
    }
    return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
}

inline size_t segment_size_for(size_t size)
{
    if (size <= kSmallSize)
    {
        return kSmallBuffer;
    }
    if (size < kMinLargeAlloc)
    {
        return kLargeBuffer;
    }
    return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
}

}  // namespace memory::gpu::caching_config
