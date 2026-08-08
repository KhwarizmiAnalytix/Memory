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

// Phase 1 milestone test for the Metal GPU backend: validates that
// memory::allocator<float>::allocate(..., device_enum::METAL) round-trips through a
// real MTLBuffer, independent of tensors/expressions (those come in later phases).
// This file is plain C++ (no Objective-C) — metal_buffer_allocator.h's surface never
// leaks id<MTLBuffer> across the boundary.

#include "MemoryTest.h"

#if MEMORY_HAS_METAL

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "allocator.h"
#include "common/device.h"
#include "gpu/metal/metal_buffer_allocator.h"

MEMORYTEST(MetalBufferAllocator, DeviceAvailable)
{
    EXPECT_TRUE(memory::metal::device_available());
    END_TEST();
}

MEMORYTEST(MetalBufferAllocator, AllocateWriteReadFree)
{
    using alloc_t          = memory::allocator<float>;
    constexpr size_t count = 1024;

    float* ptr = alloc_t::allocate(count, memory::device_enum::METAL);
    ASSERT_NE(ptr, nullptr);

    // MTLResourceStorageModeShared: the pointer is directly host-writable — this is the
    // load-bearing unified-memory assumption the whole Metal backend design rests on.
    for (size_t i = 0; i < count; ++i)
    {
        ptr[i] = static_cast<float>(i);
    }
    for (size_t i = 0; i < count; ++i)
    {
        EXPECT_FLOAT_EQ(ptr[i], static_cast<float>(i));
    }

    alloc_t::free(ptr, memory::device_enum::METAL);
    EXPECT_EQ(ptr, nullptr);
    END_TEST();
}

MEMORYTEST(MetalBufferAllocator, CopyRoundTrip)
{
    using alloc_t          = memory::allocator<float>;
    constexpr size_t count = 256;

    std::vector<float> host_in(count);
    for (size_t i = 0; i < count; ++i)
    {
        host_in[i] = static_cast<float>(i) * 1.5f;
    }

    float* metal_ptr = alloc_t::allocate(count, memory::device_enum::METAL);
    ASSERT_NE(metal_ptr, nullptr);

    alloc_t::copy(
        host_in.data(), count, metal_ptr, memory::device_enum::CPU, memory::device_enum::METAL);

    std::vector<float> host_out(count, 0.0f);
    alloc_t::copy(
        metal_ptr, count, host_out.data(), memory::device_enum::METAL, memory::device_enum::CPU);

    for (size_t i = 0; i < count; ++i)
    {
        EXPECT_FLOAT_EQ(host_out[i], host_in[i]);
    }

    alloc_t::free(metal_ptr, memory::device_enum::METAL);
    END_TEST();
}

MEMORYTEST(MetalBufferAllocator, DoubleAllocationThrows)
{
    using alloc_t = memory::allocator<double>;
    EXPECT_THROW(alloc_t::allocate(8, memory::device_enum::METAL), std::invalid_argument);
    END_TEST();
}

MEMORYTEST(MetalBufferAllocator, OptimalAlignmentAndIsGpuDevice)
{
    EXPECT_EQ(memory::optimal_alignment(memory::device_enum::METAL), 256U);
    EXPECT_TRUE(memory::is_gpu_device(memory::device_enum::METAL));
    EXPECT_TRUE(memory::has_gpu_support());
    END_TEST();
}

#endif  // MEMORY_HAS_METAL
