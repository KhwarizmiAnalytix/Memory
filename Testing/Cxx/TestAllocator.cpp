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

// Exercises the memory::allocator<T> CPU path (allocate/free/copy and the
// alignment helpers) plus the free functions at the bottom of allocator.h.
// The Metal-specific device_enum::METAL path is covered separately in
// TestMetalBufferAllocator.cpp.

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "MemoryTest.h"
#include "allocator.h"
#include "common/data_ptr.h"
#include "common/data_view.h"
#include "common/device.h"

using namespace memory;

MEMORYTEST(Allocator, AllocateZeroElementsReturnsNullptr)
{
    using alloc_t = allocator<float>;
    EXPECT_EQ(alloc_t::allocate(0), nullptr);
    END_TEST();
}

MEMORYTEST(Allocator, AllocateAndFreeCpu)
{
    using alloc_t           = allocator<float>;
    constexpr size_t kCount = 256;

    float* ptr = alloc_t::allocate(kCount, device_enum::CPU);
    ASSERT_NE(ptr, nullptr);

    for (size_t i = 0; i < kCount; ++i)
    {
        ptr[i] = static_cast<float>(i);
    }
    for (size_t i = 0; i < kCount; ++i)
    {
        EXPECT_FLOAT_EQ(ptr[i], static_cast<float>(i));
    }

    alloc_t::free(ptr, device_enum::CPU);
    EXPECT_EQ(ptr, nullptr);
    END_TEST();
}

MEMORYTEST(Allocator, AllocateDefaultsToCpuDevice)
{
    using alloc_t = allocator<double>;

    double* ptr = alloc_t::allocate(16);
    ASSERT_NE(ptr, nullptr);
    alloc_t::free(ptr);
    EXPECT_EQ(ptr, nullptr);
    END_TEST();
}

MEMORYTEST(Allocator, FreeNullptrIsNoOp)
{
    using alloc_t = allocator<float>;
    float* ptr    = nullptr;
    EXPECT_NO_THROW({ alloc_t::free(ptr, device_enum::CPU); });
    EXPECT_EQ(ptr, nullptr);
    END_TEST();
}

MEMORYTEST(Allocator, AllocateHugeSizeThrowsBadAlloc)
{
    using alloc_t = allocator<char>;
    // Larger than any real system can satisfy -- forces the underlying
    // backend to return nullptr, which allocate() converts to bad_alloc.
    EXPECT_THROW(alloc_t::allocate(static_cast<size_t>(-1) / 2, device_enum::CPU), std::bad_alloc);
    END_TEST();
}

MEMORYTEST(Allocator, AllocateUnsupportedDeviceThrows)
{
    using alloc_t = allocator<float>;
#if !MEMORY_HAS_CUDA && !MEMORY_HAS_HIP
    // HIP reuses the CUDA device enum via gpu_runtime.h (is_active_gpu_device).
    // On a CPU/Metal-only build, CUDA is not a compiled-in device path.
    EXPECT_THROW(alloc_t::allocate(4, device_enum::CUDA), std::invalid_argument);
#endif
    END_TEST();
}

MEMORYTEST(Allocator, FreeUnsupportedDeviceThrows)
{
    using alloc_t = allocator<float>;
#if !MEMORY_HAS_CUDA && !MEMORY_HAS_HIP
    float* ptr = alloc_t::allocate(4, device_enum::CPU);
    ASSERT_NE(ptr, nullptr);
    EXPECT_THROW(alloc_t::free(ptr, device_enum::CUDA), std::invalid_argument);
    // The throwing branch never reaches `ptr = nullptr`, so free it for real
    // to avoid leaking in the test process.
    alloc_t::free(ptr, device_enum::CPU);
#endif
    END_TEST();
}

MEMORYTEST(Allocator, CopyCpuToCpu)
{
    using alloc_t           = allocator<int>;
    constexpr size_t kCount = 64;

    std::vector<int> src(kCount);
    for (size_t i = 0; i < kCount; ++i)
    {
        src[i] = static_cast<int>(i * 2);
    }
    std::vector<int> dst(kCount, 0);

    alloc_t::copy(src.data(), kCount, dst.data(), device_enum::CPU, device_enum::CPU);
    EXPECT_EQ(src, dst);
    END_TEST();
}

MEMORYTEST(Allocator, CopyWithNullptrOrZeroCountIsNoOp)
{
    using alloc_t = allocator<int>;
    std::vector<int> dst(4, -1);

    EXPECT_NO_THROW({ alloc_t::copy(nullptr, 4, dst.data(), device_enum::CPU, device_enum::CPU); });
    EXPECT_NO_THROW({
        int src = 0;
        alloc_t::copy(&src, 4, nullptr, device_enum::CPU, device_enum::CPU);
    });
    EXPECT_NO_THROW({
        int src = 0;
        alloc_t::copy(&src, 0, dst.data(), device_enum::CPU, device_enum::CPU);
    });

    // Untouched by the no-op copies above.
    for (int value : dst)
    {
        EXPECT_EQ(value, -1);
    }
    END_TEST();
}

#if !MEMORY_HAS_CUDA && !MEMORY_HAS_HIP
MEMORYTEST(Allocator, CopyUnsupportedCombinationThrows)
{
    using alloc_t = allocator<float>;
    float src     = 1.0F;
    float dst     = 0.0F;
    // HIP is a valid device_enum value but is not compiled in on this build
    // (Metal/none), and is never handled by the Metal copy branch — it must
    // fall through to the final throw.
    EXPECT_THROW(
        alloc_t::copy(&src, 1, &dst, device_enum::HIP, device_enum::HIP), std::invalid_argument);
    END_TEST();
}
#endif

MEMORYTEST(Allocator, FirstAlignedAlreadyAligned)
{
    using alloc_t = allocator<float, 64>;
    alignas(64) float buffer[16];
    EXPECT_EQ(alloc_t::first_aligned(buffer, 16), 0U);
    END_TEST();
}

MEMORYTEST(Allocator, FirstAlignedClampedToSize)
{
    using alloc_t = allocator<float, 64>;
    alignas(64) float buffer[16];
    // Requesting a first-aligned index over a tiny window must clamp to the
    // window size rather than return an out-of-range offset.
    const size_t first = alloc_t::first_aligned(buffer, 1);
    EXPECT_LE(first, 1U);
    END_TEST();
}

MEMORYTEST(Allocator, FirstAlignedScalarMisalignedPointerReturnsSize)
{
    using alloc_t = allocator<float, 64>;
    alignas(64) unsigned char raw[128];
    // Offsetting by one byte breaks alignment to sizeof(float) itself (not
    // just to the SIMD alignment), which first_aligned() detects and bails
    // out on by returning `size` rather than scanning for an aligned index.
    const auto* misaligned = reinterpret_cast<const float*>(raw + 1);
    EXPECT_EQ(alloc_t::first_aligned(misaligned, 16), 16U);
    END_TEST();
}

MEMORYTEST(Allocator, LastAligned)
{
    using alloc_t = allocator<float>;
    // 10 elements starting at 0, SIMD stride 4 -> last full stride ends at 8.
    EXPECT_EQ(alloc_t::last_aligned(0, 10, 4), 8U);
    END_TEST();
}

MEMORYTEST(Allocator, ScalarAndAlignmentConstants)
{
    using alloc_t = allocator<float, 64>;
    EXPECT_EQ(alloc_t::scalar_size, sizeof(float));
    EXPECT_EQ(alloc_t::alignment_size, 64U / sizeof(float));
    END_TEST();
}

MEMORYTEST(Allocator, OptimalAlignment)
{
    EXPECT_EQ(optimal_alignment(device_enum::CPU), MEMORY_ALIGNMENT);
    EXPECT_EQ(optimal_alignment(device_enum::CUDA), 256U);
    EXPECT_EQ(optimal_alignment(device_enum::HIP), 256U);
    EXPECT_EQ(optimal_alignment(device_enum::METAL), 256U);
    EXPECT_EQ(optimal_alignment(device_enum::PrivateUse1), 32U);
    END_TEST();
}

MEMORYTEST(Allocator, IsGpuDevice)
{
    EXPECT_FALSE(is_gpu_device(device_enum::CPU));
    EXPECT_TRUE(is_gpu_device(device_enum::CUDA));
    EXPECT_TRUE(is_gpu_device(device_enum::HIP));
    EXPECT_TRUE(is_gpu_device(device_enum::METAL));
    EXPECT_FALSE(is_gpu_device(device_enum::PrivateUse1));
    END_TEST();
}

MEMORYTEST(Allocator, HasGpuSupport)
{
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP || MEMORY_HAS_METAL
    EXPECT_TRUE(has_gpu_support());
#else
    EXPECT_FALSE(has_gpu_support());
#endif
    END_TEST();
}

MEMORYTEST(Allocator, IsActiveGpuDevice)
{
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
    EXPECT_TRUE(is_active_gpu_device(device_enum::CUDA));
    EXPECT_TRUE(is_active_gpu_device(device_enum::HIP));
    EXPECT_FALSE(is_active_gpu_device(device_enum::METAL));
#elif MEMORY_HAS_METAL
    EXPECT_FALSE(is_active_gpu_device(device_enum::CUDA));
    EXPECT_FALSE(is_active_gpu_device(device_enum::HIP));
    EXPECT_TRUE(is_active_gpu_device(device_enum::METAL));
#else
    EXPECT_FALSE(is_active_gpu_device(device_enum::CUDA));
    EXPECT_FALSE(is_active_gpu_device(device_enum::HIP));
    EXPECT_FALSE(is_active_gpu_device(device_enum::METAL));
#endif
    EXPECT_FALSE(is_active_gpu_device(device_enum::CPU));
    END_TEST();
}

MEMORYTEST(DataPtr, stores_device_index_and_device)
{
    data_ptr<float> ptr(8, device_enum::CPU, 0);
    EXPECT_EQ(0, ptr.device_index());
    EXPECT_EQ(device_enum::CPU, ptr.device());
    EXPECT_EQ(8U, ptr.size());
    ASSERT_NE(nullptr, ptr.data());
    END_TEST();
}

MEMORYTEST(DataPtr, copy_assign_releases_previous_storage)
{
    data_ptr<int> first(4, device_enum::CPU);
    data_ptr<int> second(4, device_enum::CPU);
    first.data()[0]  = 1;
    second.data()[0] = 2;

    first = second;
    EXPECT_EQ(2, first.data()[0]);
    EXPECT_EQ(2, second.data()[0]);
    EXPECT_NE(first.data(), second.data());

    for (int i = 0; i < 64; ++i)
    {
        data_ptr<int> tmp(4, device_enum::CPU);
        tmp.data()[0] = i;
        first         = tmp;
        EXPECT_EQ(i, first.data()[0]);
    }
    END_TEST();
}

MEMORYTEST(DataPtr, move_assign_releases_previous_storage)
{
    data_ptr<int> first(4, device_enum::CPU);
    data_ptr<int> second(4, device_enum::CPU);
    first.data()[0]  = 1;
    second.data()[0] = 2;
    int* const kept  = second.data();

    first = std::move(second);
    EXPECT_EQ(kept, first.data());
    EXPECT_EQ(2, first.data()[0]);
    EXPECT_EQ(nullptr, second.data());
    END_TEST();
}

MEMORYTEST(DataPtr, default_stream_is_null)
{
    data_ptr<float> ptr(4, device_enum::CPU);
    EXPECT_EQ(nullptr, ptr.stream());
    END_TEST();
}

MEMORYTEST(DataPtr, stores_and_moves_stream)
{
    using stream_t       = data_ptr<int>::stream_t;
    stream_t const dummy = reinterpret_cast<stream_t>(static_cast<std::uintptr_t>(0x11));

    data_ptr<int> ptr(4, device_enum::CPU, 0, dummy);
    EXPECT_EQ(dummy, ptr.stream());

    data_ptr<int> moved = std::move(ptr);
    EXPECT_EQ(dummy, moved.stream());
    EXPECT_EQ(nullptr, ptr.stream());
    EXPECT_EQ(0U, ptr.size());
    END_TEST();
}

MEMORYTEST(DataPtr, copy_clones_storage)
{
    data_ptr<int> src(4, device_enum::CPU);
    src.data()[0] = 7;
    data_ptr<int> dst(src);
    EXPECT_NE(src.data(), dst.data());
    EXPECT_EQ(7, dst.data()[0]);
    dst.data()[0] = 9;
    EXPECT_EQ(7, src.data()[0]);
    END_TEST();
}

MEMORYTEST(DataPtr, clones_from_view)
{
    int            raw[4] = {1, 2, 3, 4};
    data_view<int> view   = data_view<int>::borrow(raw, 4, device_enum::CPU);
    data_ptr<int>  owned(view);
    ASSERT_NE(nullptr, owned.data());
    EXPECT_NE(owned.data(), raw);
    EXPECT_EQ(1, owned.data()[0]);
    owned.data()[0] = 8;
    EXPECT_EQ(1, raw[0]);
    END_TEST();
}
