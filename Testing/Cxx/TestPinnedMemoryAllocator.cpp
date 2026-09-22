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
#include "common/pinned_buffer.h"

#include <cstdint>
#include <limits>
#include <utility>

using memory::cpu::pinned_memory_allocator;

MEMORYTEST(PinnedMemory, empty_and_overflow)
{
    memory::pinned_buffer<float> empty;
    EXPECT_EQ(nullptr, empty.data());
    EXPECT_EQ(0U, empty.size());
    EXPECT_TRUE(empty.reset());
    EXPECT_THROW(
        (void)memory::pinned_buffer<double>(std::numeric_limits<std::size_t>::max()),
        std::length_error);
    pinned_memory_allocator allocator;
    EXPECT_EQ(nullptr, allocator.allocate(0));
    EXPECT_TRUE(allocator.deallocate(nullptr));
    EXPECT_THROW(pinned_memory_allocator(-1), std::invalid_argument);
}

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP

class PinnedMemoryRuntime : public ::testing::Test
{
    void SetUp() override
    {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
            GTEST_SKIP() << "No CUDA/HIP device available";
        ASSERT_EQ(cudaSuccess, cudaSetDevice(0));
    }
};

MEMORYTEST_F(PinnedMemoryRuntime, reuse_alignment_and_limits)
{
    pinned_memory_allocator allocator;
    void*                   first = allocator.allocate(1000);
    EXPECT_EQ(0U, reinterpret_cast<std::uintptr_t>(first) % allocator.alignment);
    EXPECT_TRUE(allocator.deallocate(first));
    void* second = allocator.allocate(1000);
    EXPECT_EQ(first, second);
    EXPECT_EQ(1U, allocator.stats().cache_hits);
    EXPECT_TRUE(allocator.deallocate(second));
    EXPECT_FALSE(allocator.deallocate(second));
    allocator.set_max_cached_bytes(0);
    EXPECT_EQ(0U, allocator.stats().bytes_reserved);
}

MEMORYTEST_F(PinnedMemoryRuntime, asynchronous_round_trip_and_move)
{
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    void* device = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&device, 4096));
    {
        memory::pinned_buffer<int> source(1024), result(1024);
        for (int i = 0; i < 1024; ++i)
            source.data()[i] = i * 7;
        source.copy_to_device_async(static_cast<int*>(device), stream);
        result.copy_from_device_async(static_cast<int*>(device), stream);
        memory::pinned_buffer<int> moved(std::move(source));
        EXPECT_EQ(nullptr, source.data());
        EXPECT_TRUE(moved.reset());  // H2D may still be in flight.
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
        for (int i = 0; i < 1024; ++i)
            EXPECT_EQ(i * 7, result.data()[i]);
    }
    memory::cpu::pinned_allocator_for_device().empty_cache();
    EXPECT_EQ(cudaSuccess, cudaFree(device));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

#else

MEMORYTEST(PinnedMemory, unsupported_backends_do_not_return_pageable_memory)
{
    EXPECT_FALSE(pinned_memory_allocator::supported());
    pinned_memory_allocator allocator;
    EXPECT_THROW(allocator.allocate(64), std::runtime_error);
    EXPECT_THROW((void)memory::pinned_buffer<float>(16), std::runtime_error);
    int foreign = 0;
    EXPECT_FALSE(allocator.deallocate(&foreign));
    EXPECT_THROW(allocator.record_stream(&foreign, nullptr), std::runtime_error);
    allocator.empty_cache();
    allocator.set_max_cached_bytes(1024);
    EXPECT_EQ(1024U, allocator.max_cached_bytes());
    EXPECT_EQ(0U, allocator.stats().bytes_reserved);
}
#endif
