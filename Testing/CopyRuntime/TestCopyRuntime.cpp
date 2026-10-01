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
#if MEMORY_HAS_CUDA
#include "allocator.h"
#endif

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

TEST_F(CopyTokenTest, BlockingEnqueuePollsForItsOwnCapacity)
{
    auto& service = retained_operation_service::instance();
    service.reset();
    service.set_max_pending(1);

    auto first_stream = reinterpret_cast<void*>(11);
    auto second_stream = reinterpret_cast<void*>(12);
    fake_runtime::set_stream_ready(first_stream, false);
    fake_runtime::set_stream_ready(second_stream, false);
    execution_context first_ctx;
    first_ctx.device_type = device_enum::CUDA;
    first_ctx.stream = first_stream;
    execution_context second_ctx = first_ctx;
    second_ctx.stream = second_stream;
    copy_token first(first_ctx);
    copy_token second(second_ctx);
    service.enqueue(first, 0, false);

    fake_runtime::set_stream_ready(first_stream, true);
    EXPECT_NO_THROW(service.enqueue(second, 0, true));
    EXPECT_EQ(service.pending_count(), 1);

    fake_runtime::set_stream_ready(second_stream, true);
    EXPECT_EQ(service.poll(), 1);
    service.reset();
}

TEST_F(CopyTokenTest, ShutdownRejectsNewWorkAndResetReopensService)
{
    auto& service = retained_operation_service::instance();
    service.reset();
    EXPECT_EQ(service.shutdown(std::chrono::milliseconds(10)), 0);

    execution_context ctx;
    ctx.device_type = device_enum::CUDA;
    ctx.stream = reinterpret_cast<void*>(13);
    fake_runtime::set_stream_ready(ctx.stream, false);
    copy_token token(ctx);
    EXPECT_THROW(service.enqueue(token, 0, false), std::runtime_error);

    service.reset();
    EXPECT_NO_THROW(service.enqueue(token, 0, false));
    fake_runtime::set_stream_ready(ctx.stream, true);
    EXPECT_EQ(service.poll(), 1);
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

TEST_F(CopyTokenTest, OperationEventIgnoresLaterStreamWork)
{
    auto stream = reinterpret_cast<void*>(1);
    fake_runtime::set_stream_ready(stream, true);

    execution_context ctx;
    ctx.device_type = device_enum::CUDA;
    ctx.stream = stream;
    copy_token token(ctx);
    token.prepare_event();
    token.record_event();

    fake_runtime::set_stream_ready(stream, false);
    EXPECT_EQ(token.state(), completion_state::complete);
    EXPECT_EQ(fake_runtime::stream_queries, 0);
    EXPECT_EQ(fake_runtime::event_queries, 1);
}

TEST_F(CopyTokenTest, OperationEventWaitSynchronizesOnlyTheEvent)
{
    auto stream = reinterpret_cast<void*>(2);
    fake_runtime::set_stream_ready(stream, false);
    execution_context ctx;
    ctx.device_type = device_enum::CUDA;
    ctx.stream = stream;
    copy_token token(ctx);
    token.prepare_event();
    token.record_event();

    EXPECT_NO_THROW(token.wait());
    EXPECT_TRUE(token.ready());
    EXPECT_EQ(fake_runtime::event_syncs, 1);
    EXPECT_EQ(fake_runtime::stream_syncs, 0);
    EXPECT_EQ(fake_runtime::device_syncs, 0);
}

#if MEMORY_HAS_CUDA
TEST_F(CopyTokenTest, RetainedCopyValidatesExtentAndSupportsSlices)
{
    using T = float;
    auto stream = reinterpret_cast<void*>(6);
    fake_runtime::set_stream_ready(stream, true);
    execution_context ctx;
    ctx.device_type = device_enum::CUDA;
    ctx.stream = stream;
    auto source = allocator<T>::allocate_adopted(
        new T[6]{1, 2, 3, 4, 5, 6}, 6, ctx,
        [](T* ptr, size_t, execution_context const&) { delete[] ptr; });
    auto destination = allocator<T>::allocate_adopted(
        new T[6]{}, 6, ctx,
        [](T* ptr, size_t, execution_context const&) { delete[] ptr; });
    auto source_slice = source.slice(2, 3);
    auto destination_slice = destination.slice(1, 3);

    auto token = allocator<T>::copy_async_retained(source_slice, destination_slice, stream);
    EXPECT_TRUE(token.ready());
    EXPECT_EQ(destination.data()[1], 3);
    EXPECT_EQ(destination.data()[2], 4);
    EXPECT_EQ(destination.data()[3], 5);

    auto short_destination = destination.slice(0, 2);
    EXPECT_THROW(
        allocator<T>::copy_async_retained(source_slice, short_destination),
        std::invalid_argument);
    EXPECT_EQ(fake_runtime::copies, 1);
    EXPECT_EQ(fake_runtime::event_records, 1);
}

TEST_F(CopyTokenTest, RetainedCopySurvivesTokenDiscardUntilEventCompletes)
{
    using T = float;
    auto& service = retained_operation_service::instance();
    service.reset();
    auto stream = reinterpret_cast<void*>(7);
    fake_runtime::set_stream_ready(stream, false);
    execution_context ctx;
    ctx.device_type = device_enum::CUDA;
    ctx.stream = stream;

    int releases = 0;
    {
        auto source = allocator<T>::allocate_adopted(
            new T[2]{8, 9}, 2, ctx,
            [&releases](T* ptr, size_t, execution_context const&) {
                ++releases;
                delete[] ptr;
            });
        auto destination = allocator<T>::allocate_adopted(
            new T[2]{}, 2, ctx,
            [&releases](T* ptr, size_t, execution_context const&) {
                ++releases;
                delete[] ptr;
            });
        (void)allocator<T>::copy_async_retained(source, destination, stream);
    }

    EXPECT_EQ(releases, 0);
    EXPECT_EQ(service.pending_count(), 1);
    EXPECT_EQ(service.poll(), 0);
    EXPECT_EQ(releases, 0);
    EXPECT_EQ(service.shutdown(std::chrono::milliseconds(1)), 1);
    EXPECT_EQ(releases, 0);

    fake_runtime::set_stream_ready(stream, true);
    EXPECT_EQ(service.shutdown(std::chrono::milliseconds(100)), 0);
    EXPECT_EQ(releases, 2);
}

TEST_F(CopyTokenTest, FailedRetainedCopyIsQuarantined)
{
    using T = float;
    auto& service = retained_operation_service::instance();
    service.reset();
    auto stream = reinterpret_cast<void*>(8);
    fake_runtime::set_stream_error(stream);
    execution_context ctx;
    ctx.device_type = device_enum::CUDA;
    ctx.stream = stream;

    int releases = 0;
    {
        auto source = allocator<T>::allocate_adopted(
            new T[1]{8}, 1, ctx,
            [&releases](T* ptr, size_t, execution_context const&) {
                ++releases;
                delete[] ptr;
            });
        auto destination = allocator<T>::allocate_adopted(
            new T[1]{}, 1, ctx,
            [&releases](T* ptr, size_t, execution_context const&) {
                ++releases;
                delete[] ptr;
            });
        (void)allocator<T>::copy_async_retained(source, destination, stream);
    }

    EXPECT_EQ(service.poll(), 1);
    EXPECT_EQ(service.pending_count(), 0);
    EXPECT_EQ(service.failed_count(), 1);
    EXPECT_EQ(releases, 0);
}
#endif
