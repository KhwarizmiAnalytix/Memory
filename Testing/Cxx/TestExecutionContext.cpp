/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#include "MemoryTest.h"
#include "common/execution_context.h"

using namespace memory;

MEMORYTEST(ExecutionContext, default_is_cpu)
{
    execution_context ctx{};
    EXPECT_EQ(ctx.device_type, device_enum::CPU);
    EXPECT_EQ(ctx.device_index, 0);
    EXPECT_EQ(ctx.stream, nullptr);
    EXPECT_FALSE(ctx.is_gpu());
    END_TEST();
}

MEMORYTEST(ExecutionContext, cpu_factory)
{
    auto ctx = execution_context::cpu();
    EXPECT_EQ(ctx.device_type, device_enum::CPU);
    EXPECT_EQ(ctx.device_index, 0);
    EXPECT_FALSE(ctx.is_gpu());
    END_TEST();
}

MEMORYTEST(ExecutionContext, cuda_factory)
{
    auto ctx = execution_context::cuda(2, nullptr);
    EXPECT_EQ(ctx.device_type, device_enum::CUDA);
    EXPECT_EQ(ctx.device_index, 2);
    EXPECT_EQ(ctx.stream, nullptr);
    EXPECT_TRUE(ctx.is_gpu());
    END_TEST();
}

MEMORYTEST(ExecutionContext, hip_factory)
{
    auto ctx = execution_context::hip(1);
    EXPECT_EQ(ctx.device_type, device_enum::HIP);
    EXPECT_EQ(ctx.device_index, 1);
    EXPECT_TRUE(ctx.is_gpu());
    END_TEST();
}

MEMORYTEST(ExecutionContext, metal_factory)
{
    auto ctx = execution_context::metal();
    EXPECT_EQ(ctx.device_type, device_enum::METAL);
    EXPECT_EQ(ctx.device_index, 0);
    EXPECT_TRUE(ctx.is_gpu());
    END_TEST();
}

MEMORYTEST(ExecutionContext, equality)
{
    auto a = execution_context::cuda(0, nullptr);
    auto b = execution_context::cuda(0, nullptr);
    auto c = execution_context::cuda(1, nullptr);
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
    EXPECT_TRUE(a != c);
    END_TEST();
}

MEMORYTEST(ExecutionContext, cpu_is_not_equal_to_cuda)
{
    auto cpu  = execution_context::cpu();
    auto cuda = execution_context::cuda(0, nullptr);
    EXPECT_FALSE(cpu == cuda);
    EXPECT_TRUE(cpu != cuda);
    END_TEST();
}
