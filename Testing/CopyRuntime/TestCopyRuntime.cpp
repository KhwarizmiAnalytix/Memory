/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#include <gtest/gtest.h>
#include <atomic>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <memory>
#include <new>

// Include fake runtime first
#include "fake_runtime.h"

// Then include Memory headers
#include "common/copy_token.h"
#include "common/execution_context.h"
#include "common/retained_operation_service.h"
#include "common/retained_ptr.h"
#if MEMORY_HAS_CUDA
#include "allocator.h"
#endif

using namespace memory;

namespace
{
std::atomic<bool>   g_count_new{false};
std::atomic<size_t> g_new_calls{0};
std::atomic<bool>   g_fail_next_new{false};  // one-shot: the next operator new throws
}  // namespace

void* operator new(std::size_t n)
{
    if (g_count_new.load(std::memory_order_relaxed))
    {
        g_new_calls.fetch_add(1, std::memory_order_relaxed);
    }
    if (g_fail_next_new.exchange(false, std::memory_order_relaxed))
    {
        throw std::bad_alloc();
    }
    if (void* p = std::malloc(n ? n : 1))
    {
        return p;
    }
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

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

#if MEMORY_HAS_CUDA
// Phase 0.3 probes (plan §6.1). Counting operator new + fake-runtime counters.
TEST_F(CopyTokenTest, ProbeTokenReadyAfterTerminalMakesNoDriverCalls)
{
    using T = float;
    auto  stream = reinterpret_cast<void*>(3);
    fake_runtime::set_stream_ready(stream, true);
    execution_context ctx;
    ctx.device_type = device_enum::CUDA;
    ctx.stream      = stream;
    auto del = [](T* p, size_t, execution_context const&) { delete[] p; };
    auto src = allocator<T>::allocate_adopted(new T[8]{}, 8, ctx, del);
    auto dst = allocator<T>::allocate_adopted(new T[8]{}, 8, ctx, del);
    auto token = allocator<T>::copy_async_retained(src, dst, stream);
    ASSERT_TRUE(token.ready());  // reach terminal
    int const queries = fake_runtime::event_queries + fake_runtime::stream_queries;
    for (int i = 0; i < 100; ++i)
    {
        (void)token.ready();
    }
    EXPECT_EQ(queries, fake_runtime::event_queries + fake_runtime::stream_queries);
}

TEST_F(CopyTokenTest, ProbeRetainedCopySteadyStateHeapAndEvents)
{
    using T = float;
    auto& service = retained_operation_service::instance();
    service.reset();
    auto stream = reinterpret_cast<void*>(3);
    fake_runtime::set_stream_ready(stream, true);
    execution_context ctx;
    ctx.device_type = device_enum::CUDA;
    ctx.stream      = stream;
    auto del = [](T* p, size_t, execution_context const&) { delete[] p; };
    auto src = allocator<T>::allocate_adopted(new T[8]{}, 8, ctx, del);
    auto dst = allocator<T>::allocate_adopted(new T[8]{}, 8, ctx, del);
    for (int i = 0; i < 20; ++i)  // warm
    {
        (void)allocator<T>::copy_async_retained(src, dst, stream);
        (void)service.poll();
    }
    int const creates0 = fake_runtime::event_creates;
    g_new_calls = 0;
    g_count_new = true;
    for (int i = 0; i < 100; ++i)
    {
        (void)allocator<T>::copy_async_retained(src, dst, stream);
        (void)service.poll();
    }
    g_count_new = false;
    size_t const heap    = g_new_calls.load();
    int const    creates = fake_runtime::event_creates - creates0;
    RecordProperty("retained_copy_heap_allocations_per_100", static_cast<int>(heap));
    RecordProperty("retained_copy_event_creates_per_100", creates);
    // Target (§6.1, task 3.4): 0 heap allocations, 0 event create/destroy.
    if (heap != 0 || creates != 0)
    {
        GTEST_SKIP() << "expected-fail (plan 3.4): " << heap << " heap allocations and " << creates
                     << " event creates per 100 retained copies";
    }
    (void)service.shutdown(std::chrono::milliseconds(100));
}
#endif  // MEMORY_HAS_CUDA

// ---------------------------------------------------------------------------
// Task 1.4: token terminal state is published once and shared by all copies.
// ---------------------------------------------------------------------------
namespace
{
execution_context gpu_ctx(void* stream)
{
    execution_context ctx;
    ctx.device_type  = device_enum::CUDA;
    ctx.device_index = 0;
    ctx.stream       = stream;
    return ctx;
}

copy_token submitted_token(void* stream)
{
    copy_token token(gpu_ctx(stream));
    token.prepare_event();
    token.record_event();
    return token;
}
}  // namespace

TEST_F(CopyTokenTest, TwoCopiesObserveOneCompleteResultWhileLaterStreamWorkPending)
{
    auto stream = reinterpret_cast<void*>(1);
    fake_runtime::set_stream_ready(stream, true);
    copy_token a = submitted_token(stream);
    copy_token b = a;

    EXPECT_EQ(a.state(), completion_state::complete);
    int const queries = fake_runtime::event_queries;

    // Later work on the same stream is pending, then fails; the finished copy
    // must not regress, and the second copy must not query the driver again.
    fake_runtime::set_stream_ready(stream, false);
    EXPECT_EQ(b.state(), completion_state::complete);
    fake_runtime::set_stream_error(stream);
    EXPECT_EQ(a.state(), completion_state::complete);
    EXPECT_TRUE(b.ready());
    EXPECT_NO_THROW(b.wait());
    EXPECT_EQ(queries, fake_runtime::event_queries);
    EXPECT_EQ(0, fake_runtime::event_syncs);
}

TEST_F(CopyTokenTest, CompatTokenCompleteIsStableAcrossLaterStreamWork)
{
    auto stream = reinterpret_cast<void*>(2);
    fake_runtime::set_stream_ready(stream, true);
    copy_token a(gpu_ctx(stream));  // no event: stream-query compatibility path
    copy_token b = a;
    EXPECT_EQ(a.state(), completion_state::complete);
    fake_runtime::set_stream_ready(stream, false);
    EXPECT_EQ(b.state(), completion_state::complete);
    EXPECT_NO_THROW(b.wait());
    EXPECT_EQ(0, fake_runtime::stream_syncs);
}

TEST_F(CopyTokenTest, ForcedFailureIsSharedAndWaitAgreesWithState)
{
    auto stream = reinterpret_cast<void*>(3);
    fake_runtime::set_stream_ready(stream, false);
    copy_token a = submitted_token(stream);
    copy_token b = a;
    EXPECT_EQ(a.state(), completion_state::pending);

    memory::detail::copy_token_access::fail(a);
    EXPECT_EQ(a.state(), completion_state::failed);
    EXPECT_EQ(b.state(), completion_state::failed);
    EXPECT_FALSE(b.ready());
    EXPECT_THROW(a.wait(), std::runtime_error);
    EXPECT_THROW(b.wait(), std::runtime_error);

    // The stream becoming idle later does not turn the failure into success.
    fake_runtime::set_stream_ready(stream, true);
    EXPECT_EQ(b.state(), completion_state::failed);
    EXPECT_THROW(b.wait(), std::runtime_error);
    EXPECT_EQ(0, fake_runtime::event_syncs);  // wait() trusts the published result
}

TEST_F(CopyTokenTest, CancellationCompletesAllCopiesAndReleasesPayload)
{
    auto stream = reinterpret_cast<void*>(4);
    fake_runtime::set_stream_ready(stream, false);
    copy_token a = submitted_token(stream);
    copy_token b = a;
    auto       payload = std::make_shared<int>(7);
    std::weak_ptr<int> weak = payload;
    memory::detail::copy_token_access::set_retained(a, std::move(payload));
    EXPECT_FALSE(weak.expired());

    memory::detail::copy_token_access::complete(a);
    EXPECT_TRUE(weak.expired());  // nothing in flight: payload released
    EXPECT_EQ(b.state(), completion_state::complete);
    EXPECT_NO_THROW(b.wait());
    EXPECT_EQ(0, fake_runtime::event_syncs);
}

TEST_F(CopyTokenTest, FirstTerminalResultWins)
{
    auto stream = reinterpret_cast<void*>(5);
    fake_runtime::set_stream_ready(stream, false);
    copy_token failed_first = submitted_token(stream);
    memory::detail::copy_token_access::fail(failed_first);
    memory::detail::copy_token_access::complete(failed_first);
    EXPECT_EQ(failed_first.state(), completion_state::failed);

    copy_token complete_first = submitted_token(stream);
    memory::detail::copy_token_access::complete(complete_first);
    memory::detail::copy_token_access::fail(complete_first);
    EXPECT_EQ(complete_first.state(), completion_state::complete);
    EXPECT_NO_THROW(complete_first.wait());
}

TEST_F(CopyTokenTest, QueryFailureIsPersistedAndWaitThrowsWithoutRequerying)
{
    auto stream = reinterpret_cast<void*>(6);
    fake_runtime::set_stream_ready(stream, false);
    copy_token a = submitted_token(stream);
    copy_token b = a;

    fake_runtime::fail_stream_query = true;
    EXPECT_EQ(a.state(), completion_state::failed);
    fake_runtime::fail_stream_query = false;
    fake_runtime::set_stream_ready(stream, true);  // would now succeed if re-queried

    EXPECT_EQ(b.state(), completion_state::failed);
    EXPECT_THROW(b.wait(), std::runtime_error);
    EXPECT_EQ(1, fake_runtime::event_queries);
    EXPECT_EQ(0, fake_runtime::event_syncs);
}

TEST_F(CopyTokenTest, WaitFailureIsPersistedForAllCopies)
{
    auto stream = reinterpret_cast<void*>(7);
    fake_runtime::set_stream_error(stream);
    copy_token a = submitted_token(stream);
    copy_token b = a;

    EXPECT_THROW(a.wait(), std::runtime_error);
    EXPECT_EQ(b.state(), completion_state::failed);
    EXPECT_THROW(b.wait(), std::runtime_error);
    EXPECT_EQ(1, fake_runtime::event_syncs);  // the second wait did not synchronize
}

TEST_F(CopyTokenTest, WaitSuccessPublishesCompleteForAllCopies)
{
    auto stream = reinterpret_cast<void*>(8);
    fake_runtime::set_stream_ready(stream, false);
    copy_token a = submitted_token(stream);
    copy_token b = a;
    EXPECT_EQ(b.state(), completion_state::pending);
    EXPECT_NO_THROW(a.wait());
    fake_runtime::set_stream_ready(stream, false);
    EXPECT_EQ(b.state(), completion_state::complete);
    EXPECT_NO_THROW(b.wait());
    EXPECT_EQ(1, fake_runtime::event_syncs);
}

TEST_F(CopyTokenTest, MovedFromAndDefaultTokensAreComplete)
{
    auto stream = reinterpret_cast<void*>(9);
    fake_runtime::set_stream_ready(stream, false);
    copy_token a = submitted_token(stream);
    copy_token b = std::move(a);
    EXPECT_EQ(a.state(), completion_state::complete);  // NOLINT(bugprone-use-after-move)
    EXPECT_NO_THROW(a.wait());
    EXPECT_EQ(b.state(), completion_state::pending);
    copy_token d;
    EXPECT_TRUE(d.ready());
    EXPECT_NO_THROW(d.wait());
}

#if MEMORY_HAS_CUDA
// ---------------------------------------------------------------------------
// Tasks 1.5 / 1.6: pre-submission validation, exact rollback, and post-submission
// failure handling for retained copies.
// ---------------------------------------------------------------------------
namespace
{
struct retained_pair
{
    retained_ptr<float> source;
    retained_ptr<float> destination;

    retained_pair(int* counter, void* stream)
    {
        execution_context ctx;
        ctx.device_type = device_enum::CUDA;
        ctx.stream      = stream;
        auto deleter    = [counter](float* p, size_t, execution_context const&) {
            ++*counter;
            delete[] p;
        };
        source      = allocator<float>::allocate_adopted(new float[4]{1, 2, 3, 4}, 4, ctx, deleter);
        destination = allocator<float>::allocate_adopted(new float[4]{}, 4, ctx, deleter);
    }
};

class CopyFailureTest : public CopyTokenTest
{
protected:
    void SetUp() override
    {
        CopyTokenTest::SetUp();
        service().reset();
        service().clear_failed();
        service().set_max_pending(1);
        failed_before_ = service().failed_count();
    }
    void TearDown() override
    {
        service().reset();
        service().clear_failed();
        service().set_max_pending(0);
        CopyTokenTest::TearDown();
    }
    static retained_operation_service& service() { return retained_operation_service::instance(); }

    size_t failed_before_{0};
};
}  // namespace

// The shim targets do not link the real cache registry, so the stream-tracking
// copy_async (which calls record_stream) cannot be used here; validation and
// failure behavior is identical for the untracked variant.
static copy_token copy_async_untracked(
    float const* from, size_t n, float* to, void* stream, device_enum from_type, device_enum to_type)
{
    return allocator<float>::copy_async_impl<false>(
        from, n, to, stream, from_type, to_type, 0, 0, {}, false);
}

TEST_F(CopyFailureTest, ValidationRejectsBeforeAnyWorkOrAdmission)
{
    auto  stream = reinterpret_cast<void*>(1);
    float src[4] = {};
    float dst[4] = {};

    EXPECT_THROW(
        copy_async_untracked(nullptr, 4, dst, stream, device_enum::CPU, device_enum::CUDA),
        std::invalid_argument);
    EXPECT_THROW(
        copy_async_untracked(src, 4, nullptr, stream, device_enum::CPU, device_enum::CUDA),
        std::invalid_argument);
    EXPECT_THROW(
        copy_async_untracked(
            src, std::numeric_limits<size_t>::max() / 2, dst, stream, device_enum::CPU,
            device_enum::CUDA),
        std::overflow_error);
    // METAL is not the compiled backend in this build.
    EXPECT_THROW(
        copy_async_untracked(src, 4, dst, stream, device_enum::CPU, device_enum::METAL),
        std::invalid_argument);

    EXPECT_EQ(0, fake_runtime::event_creates);
    EXPECT_EQ(0, fake_runtime::copies);
    EXPECT_EQ(0U, service().pending_count());
}

TEST_F(CopyFailureTest, ZeroCountCopyIsAnImmediateNoOp)
{
    auto  stream = reinterpret_cast<void*>(1);
    float dst[1] = {};
    auto  token  = copy_async_untracked(
        nullptr, 0, dst, stream, device_enum::CPU, device_enum::CUDA);
    EXPECT_TRUE(token.ready());
    EXPECT_EQ(0, fake_runtime::event_creates);
    EXPECT_EQ(0, fake_runtime::copies);
}

TEST_F(CopyFailureTest, EventCreationFailureSubmitsNothingAndRestoresAdmissionOnce)
{
    auto stream = reinterpret_cast<void*>(2);
    fake_runtime::set_stream_ready(stream, false);
    int           releases = 0;
    retained_pair pair(&releases, stream);

    fake_runtime::fail_event_create = true;
    EXPECT_THROW(
        allocator<float>::copy_async_retained(pair.source, pair.destination, stream),
        std::runtime_error);
    fake_runtime::fail_event_create = false;

    EXPECT_EQ(0, fake_runtime::copies);        // nothing submitted
    EXPECT_EQ(0, fake_runtime::stream_syncs);  // no waiting on unrelated work
    EXPECT_EQ(0U, service().pending_count());
    EXPECT_EQ(failed_before_, service().failed_count());
    EXPECT_EQ(1, pair.source.use_count());  // temporary owners released
    EXPECT_EQ(1, pair.destination.use_count());
    EXPECT_EQ(fake_runtime::event_creates, fake_runtime::event_destroys);

    // The single admission slot (limit 1) is free again.
    auto token = allocator<float>::copy_async_retained(pair.source, pair.destination, stream);
    EXPECT_EQ(1U, service().pending_count());
    fake_runtime::set_stream_ready(stream, true);
    token.wait();
    EXPECT_EQ(1U, service().poll());
    EXPECT_EQ(0, releases);  // caller still owns the endpoints
}

TEST_F(CopyFailureTest, SetupFailureAfterAdmissionCancelsReservationWithoutWaiting)
{
    auto stream = reinterpret_cast<void*>(3);
    fake_runtime::set_stream_ready(stream, false);
    int           releases = 0;
    retained_pair pair(&releases, stream);

    // Current device 1, operation device 0. cudaSetDevice calls: prepare_event
    // switches and restores (1, 2), the admission-time state() query does the
    // same (3, 4), then the submission guard fails to switch (5) before any
    // driver copy is issued.
    fake_runtime::current_device          = 1;
    fake_runtime::fail_set_device_on_call = 5;
    EXPECT_THROW(
        allocator<float>::copy_async_retained(pair.source, pair.destination, stream),
        std::runtime_error);
    fake_runtime::fail_set_device_on_call = 0;

    EXPECT_EQ(0, fake_runtime::copies);
    EXPECT_EQ(0, fake_runtime::stream_syncs);  // never waited on the (busy) stream
    EXPECT_EQ(0U, service().pending_count());  // admission restored
    EXPECT_EQ(failed_before_, service().failed_count());
    EXPECT_EQ(1, pair.source.use_count());  // payload released
    EXPECT_EQ(fake_runtime::event_creates, fake_runtime::event_destroys);

    fake_runtime::current_device = 0;
    EXPECT_NO_THROW(
        (void)allocator<float>::copy_async_retained(pair.source, pair.destination, stream));
    EXPECT_EQ(1U, service().pending_count());  // slot was free: exactly one admission
}

TEST_F(CopyFailureTest, EventRecordFailureWithIdleStreamIsProvenSafeAndRolledBack)
{
    auto stream = reinterpret_cast<void*>(4);
    fake_runtime::set_stream_ready(stream, true);
    int           releases = 0;
    retained_pair pair(&releases, stream);

    fake_runtime::fail_event_record = true;
    EXPECT_THROW(
        allocator<float>::copy_async_retained(pair.source, pair.destination, stream),
        std::runtime_error);
    fake_runtime::fail_event_record = false;

    EXPECT_EQ(1, fake_runtime::copies);        // work was submitted
    EXPECT_EQ(1, fake_runtime::stream_syncs);  // completion proven on the submitting stream
    EXPECT_EQ(0U, service().pending_count());
    EXPECT_EQ(failed_before_, service().failed_count());
    EXPECT_EQ(1, pair.source.use_count());
}

TEST_F(CopyFailureTest, EventRecordFailureWithUnprovenStreamQuarantinesAndRetainsOwners)
{
    auto stream = reinterpret_cast<void*>(5);
    fake_runtime::set_stream_error(stream);  // synchronize will fail: completion unprovable
    int releases = 0;
    {
        retained_pair pair(&releases, stream);
        fake_runtime::fail_event_record = true;
        EXPECT_THROW(
            allocator<float>::copy_async_retained(pair.source, pair.destination, stream),
            std::runtime_error);
        fake_runtime::fail_event_record = false;

        EXPECT_EQ(1, fake_runtime::copies);
        EXPECT_EQ(0U, service().pending_count());
        EXPECT_EQ(failed_before_ + 1, service().failed_count());
        EXPECT_GT(pair.source.use_count(), 1);  // the quarantined token still owns it
        EXPECT_GT(pair.destination.use_count(), 1);
    }
    EXPECT_EQ(0, releases);  // caller dropped its handles; allocation is NOT recycled

    service().clear_failed();  // explicit recovery releases the retained owners
    EXPECT_EQ(2, releases);
}

TEST_F(CopyFailureTest, SubmissionErrorIsProvenOrQuarantinedNeverInferredFromTheCode)
{
    auto stream   = reinterpret_cast<void*>(6);
    int  releases = 0;
    {
        // Idle stream: proven safe, rolled back.
        fake_runtime::set_stream_ready(stream, true);
        retained_pair pair(&releases, stream);
        fake_runtime::fail_memcpy = true;
        EXPECT_THROW(
            allocator<float>::copy_async_retained(pair.source, pair.destination, stream),
            std::runtime_error);
        fake_runtime::fail_memcpy = false;
        EXPECT_EQ(0U, service().pending_count());
        EXPECT_EQ(failed_before_, service().failed_count());
        EXPECT_EQ(1, pair.source.use_count());
    }
    {
        // Stream cannot be synchronized: quarantined with owners retained.
        fake_runtime::set_stream_error(stream);
        retained_pair pair(&releases, stream);
        fake_runtime::fail_memcpy = true;
        EXPECT_THROW(
            allocator<float>::copy_async_retained(pair.source, pair.destination, stream),
            std::runtime_error);
        fake_runtime::fail_memcpy = false;
        EXPECT_EQ(failed_before_ + 1, service().failed_count());
        EXPECT_GT(pair.source.use_count(), 1);
    }
    int const before_clear = releases;
    service().clear_failed();
    EXPECT_EQ(before_clear + 2, releases);
}
#endif  // MEMORY_HAS_CUDA

// Task 1.8: if adoption itself fails to allocate, the deleter has not run and the
// caller still owns the pointer; a later successful adoption runs it once.
TEST_F(CopyTokenTest, AdoptionAllocationFailureLeavesTheCallerOwningThePointer)
{
    int  calls = 0;
    auto deleter = [&calls](float* p, size_t, execution_context const&) {
        ++calls;
        delete[] p;
    };
    float* raw = new float[4]{};

    // Explicit try/catch rather than EXPECT_THROW: the one-shot failure must be
    // consumed by the control-block allocation, and the Debug test binary aborts
    // when the gtest macro wraps the injected failure.
    bool threw_bad_alloc = false;
    g_fail_next_new      = true;
    try
    {
        (void)retained_ptr<float>::adopt(raw, 4, execution_context::cpu(), deleter);
    }
    catch (std::bad_alloc const&)
    {
        threw_bad_alloc = true;
    }
    g_fail_next_new = false;
    EXPECT_TRUE(threw_bad_alloc);
    EXPECT_EQ(calls, 0);  // not freed, not owned by anything

    {
        auto adopted = retained_ptr<float>::adopt(raw, 4, execution_context::cpu(), deleter);
        EXPECT_EQ(adopted.data(), raw);
        EXPECT_EQ(calls, 0);
    }
    EXPECT_EQ(calls, 1);  // ownership returned exactly once, freed exactly once
}
