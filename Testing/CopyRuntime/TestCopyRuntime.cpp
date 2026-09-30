/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#include <gtest/gtest.h>
#include <memory>

// Include fake runtime first
#include "fake_runtime.h"

// Then include Memory headers
#include "common/copy_token.h"
#include "common/execution_context.h"
#include "common/retained_operation_service.h"

using namespace memory;

class CopyTokenTest : public ::testing::Test
{
protected:
    void SetUp() override { fake_runtime::reset(); }
    void TearDown() override { fake_runtime::reset(); }
};

// Test: token reports complete for CPU operations
TEST_F(CopyTokenTest, CPUTokenAlwaysComplete)
{
    execution_context ctx;
    ctx.device_type = device_enum::CPU;
    copy_token token(ctx);
    EXPECT_EQ(token.state(), completion_state::complete);
    EXPECT_TRUE(token.ready());
}

// Test: token reports complete for a ready GPU stream
TEST_F(CopyTokenTest, GPUTokenReadyStream)
{
    auto stream = reinterpret_cast<void*>(1);
    fake_runtime::set_stream_ready(stream, true);

    execution_context ctx;
    ctx.device_type  = device_enum::CUDA;
    ctx.device_index = 0;
    ctx.stream       = stream;
    copy_token token(ctx);

    EXPECT_EQ(token.state(), completion_state::complete);
    EXPECT_TRUE(token.ready());
}

// Test: token reports pending for a not-ready GPU stream
TEST_F(CopyTokenTest, GPUTokenNotReadyStream)
{
    auto stream = reinterpret_cast<void*>(1);
    fake_runtime::set_stream_ready(stream, false);

    execution_context ctx;
    ctx.device_type  = device_enum::CUDA;
    ctx.device_index = 0;
    ctx.stream       = stream;
    copy_token token(ctx);

    EXPECT_EQ(token.state(), completion_state::pending);
    EXPECT_FALSE(token.ready());
}

// Test: token reports failed for a stream in error state
TEST_F(CopyTokenTest, GPUTokenErrorStream)
{
    auto stream = reinterpret_cast<void*>(1);
    fake_runtime::set_stream_error(stream);

    execution_context ctx;
    ctx.device_type  = device_enum::CUDA;
    ctx.device_index = 0;
    ctx.stream       = stream;
    copy_token token(ctx);

    EXPECT_EQ(token.state(), completion_state::failed);
    EXPECT_FALSE(token.ready());
}

// Test: token.wait() blocks until stream is ready
TEST_F(CopyTokenTest, GPUTokenWaitBlocks)
{
    auto stream = reinterpret_cast<void*>(1);
    fake_runtime::set_stream_ready(stream, false);

    execution_context ctx;
    ctx.device_type  = device_enum::CUDA;
    ctx.device_index = 0;
    ctx.stream       = stream;
    copy_token token(ctx);

    EXPECT_FALSE(token.ready());
    fake_runtime::set_stream_ready(stream, true);
    EXPECT_NO_THROW(token.wait());
    EXPECT_TRUE(token.ready());
}

// Test: token.wait() throws on stream error
TEST_F(CopyTokenTest, GPUTokenWaitThrowsOnError)
{
    auto stream = reinterpret_cast<void*>(1);
    fake_runtime::set_stream_error(stream);

    execution_context ctx;
    ctx.device_type  = device_enum::CUDA;
    ctx.device_index = 0;
    ctx.stream       = stream;
    copy_token token(ctx);

    EXPECT_THROW(token.wait(), std::runtime_error);
}



// Test: retained_operation_service enqueues a token
TEST_F(CopyTokenTest, ServiceEnqueueToken)
{
    execution_context ctx;
    ctx.device_type  = device_enum::CUDA;
    ctx.device_index = 0;
    ctx.stream       = reinterpret_cast<void*>(1);
    fake_runtime::set_stream_ready(ctx.stream, false);

    auto& service = retained_operation_service::instance();
    service.reset();
    service.set_max_pending(10);

    copy_token token(ctx);
    EXPECT_NO_THROW(service.enqueue(token, 0, false));
    EXPECT_EQ(service.pending_count(), 1);

    service.reset();
}

// Test: service.poll() completes ready tokens
TEST_F(CopyTokenTest, ServicePollCompletesReady)
{
    auto& service = retained_operation_service::instance();
    service.reset();
    service.set_max_pending(10);

    execution_context ctx;
    ctx.device_type  = device_enum::CUDA;
    ctx.device_index = 0;
    ctx.stream       = reinterpret_cast<void*>(1);

    fake_runtime::set_stream_ready(ctx.stream, false);
    copy_token token1(ctx);
    service.enqueue(token1, 0, false);
    EXPECT_EQ(service.pending_count(), 1);

    fake_runtime::set_stream_ready(ctx.stream, true);
    size_t completed = service.poll();
    EXPECT_EQ(completed, 1);
    EXPECT_EQ(service.pending_count(), 0);

    service.reset();
}

// Test: service backpressure with max_pending
TEST_F(CopyTokenTest, ServiceBackpressure)
{
    auto& service = retained_operation_service::instance();
    service.reset();
    service.set_max_pending(2);

    execution_context ctx;
    ctx.device_type  = device_enum::CUDA;
    ctx.device_index = 0;
    ctx.stream       = reinterpret_cast<void*>(1);
    fake_runtime::set_stream_ready(ctx.stream, false);

    copy_token token(ctx);
    service.enqueue(token, 0, false);
    service.enqueue(token, 0, false);

    // Third enqueue should throw (non-blocking, limit reached)
    EXPECT_THROW(service.enqueue(token, 0, false), std::runtime_error);

    service.reset();
}

// Test: service.drain() completes all pending
TEST_F(CopyTokenTest, ServiceDrain)
{
    auto& service = retained_operation_service::instance();
    service.reset();
    service.set_max_pending(10);

    execution_context ctx;
    ctx.device_type  = device_enum::CUDA;
    ctx.device_index = 0;
    ctx.stream       = reinterpret_cast<void*>(1);
    fake_runtime::set_stream_ready(ctx.stream, false);

    copy_token token(ctx);
    service.enqueue(token, 0, false);
    service.enqueue(token, 0, false);

    fake_runtime::set_stream_ready(ctx.stream, true);
    size_t remaining = service.drain(std::chrono::milliseconds(100));
    EXPECT_EQ(remaining, 0);

    service.reset();
}

// Test: ready token is not enqueued
TEST_F(CopyTokenTest, ServiceSkipsReadyToken)
{
    auto& service = retained_operation_service::instance();
    service.reset();
    service.set_max_pending(10);

    execution_context ctx;
    ctx.device_type  = device_enum::CUDA;
    ctx.device_index = 0;
    ctx.stream       = reinterpret_cast<void*>(1);
    fake_runtime::set_stream_ready(ctx.stream, true);

    copy_token token(ctx);
    service.enqueue(token, 0, false);
    EXPECT_EQ(service.pending_count(), 0);  // Ready tokens are dropped

    service.reset();
}
