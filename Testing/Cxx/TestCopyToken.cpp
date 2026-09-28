/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#include "MemoryTest.h"
#include "common/copy_token.h"
#include "common/execution_context.h"

using namespace memory;

MEMORYTEST(CopyToken, default_is_cpu_immediately_complete)
{
    copy_token token{};
    EXPECT_TRUE(token.ready());
    END_TEST();
}

MEMORYTEST(CopyToken, cpu_context_is_ready)
{
    copy_token token{execution_context::cpu()};
    EXPECT_TRUE(token.ready());
    END_TEST();
}

MEMORYTEST(CopyToken, wait_on_cpu_is_noop)
{
    copy_token token{};
    token.wait();  // must not throw or block
    EXPECT_TRUE(token.ready());
    END_TEST();
}

MEMORYTEST(CopyToken, ctx_is_preserved)
{
    auto const ctx = execution_context::cuda(2, nullptr);
    copy_token token{ctx};
    EXPECT_EQ(token.ctx().device_type, device_enum::CUDA);
    EXPECT_EQ(token.ctx().device_index, 2);
    END_TEST();
}

MEMORYTEST(CopyToken, copy_and_move)
{
    copy_token a{execution_context::cuda(1, nullptr)};
    copy_token b{a};
    EXPECT_EQ(b.ctx().device_index, 1);
    copy_token c{std::move(a)};
    EXPECT_EQ(c.ctx().device_index, 1);
    END_TEST();
}
