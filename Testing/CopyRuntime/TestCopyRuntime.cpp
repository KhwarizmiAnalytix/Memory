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
#include <chrono>
#include <memory>
#include <new>
#include <thread>

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

// Counters defined by the shim's gpu_dispatch stub (gpu_dispatch_stub.cpp).
namespace memory::gpu::test
{
extern std::atomic<int>         record_stream_use_calls;
extern std::atomic<void const*> foreign_pointer;
}  // namespace memory::gpu::test

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

namespace
{
// Every event the library created is accounted for: pooled events (plan 3.4) are
// released first, so a leaked event shows as creates != destroys.
bool no_event_leaked()
{
    memory::detail::release_token_event_pool();
    return fake_runtime::event_creates == fake_runtime::event_destroys;
}
}  // namespace

class CopyTokenTest : public ::testing::Test
{
protected:
    // Pooled token events outlive a token; drop them so each test starts clean.
    void SetUp() override
    {
        memory::detail::release_token_event_pool();
        fake_runtime::reset();
    }
    void TearDown() override
    {
        memory::detail::release_token_event_pool();
        fake_runtime::reset();
    }
};

// Test: token reports complete for CPU operations
TEST_F(CopyTokenTest, CPUTokenAlwaysComplete)
{
    execution_context ctx;
    ctx.dev.type = device_enum::CPU;
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
    ctx.dev.type  = device_enum::CUDA;
    ctx.dev.index = 0;
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
    ctx.dev.type  = device_enum::CUDA;
    ctx.dev.index = 0;
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
    ctx.dev.type  = device_enum::CUDA;
    ctx.dev.index = 0;
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
    ctx.dev.type  = device_enum::CUDA;
    ctx.dev.index = 0;
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
    ctx.dev.type  = device_enum::CUDA;
    ctx.dev.index = 0;
    ctx.stream       = stream;
    copy_token token(ctx);

    EXPECT_THROW(token.wait(), std::runtime_error);
}



// Test: retained_operation_service enqueues a token
TEST_F(CopyTokenTest, ServiceEnqueueToken)
{
    execution_context ctx;
    ctx.dev.type  = device_enum::CUDA;
    ctx.dev.index = 0;
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

// Test: service.poll().total() completes ready tokens
TEST_F(CopyTokenTest, ServicePollCompletesReady)
{
    auto& service = retained_operation_service::instance();
    service.reset();
    service.set_max_pending(10);

    execution_context ctx;
    ctx.dev.type  = device_enum::CUDA;
    ctx.dev.index = 0;
    ctx.stream       = reinterpret_cast<void*>(1);

    fake_runtime::set_stream_ready(ctx.stream, false);
    copy_token token1(ctx);
    service.enqueue(token1, 0, false);
    EXPECT_EQ(service.pending_count(), 1);

    fake_runtime::set_stream_ready(ctx.stream, true);
    size_t completed = service.poll().total();
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
    ctx.dev.type  = device_enum::CUDA;
    ctx.dev.index = 0;
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
    first_ctx.dev.type = device_enum::CUDA;
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
    EXPECT_EQ(service.poll().total(), 1);
    service.reset();
}

TEST_F(CopyTokenTest, ShutdownRejectsNewWorkAndResetReopensService)
{
    auto& service = retained_operation_service::instance();
    service.reset();
    EXPECT_EQ(service.shutdown(std::chrono::milliseconds(10)), 0);

    execution_context ctx;
    ctx.dev.type = device_enum::CUDA;
    ctx.stream = reinterpret_cast<void*>(13);
    fake_runtime::set_stream_ready(ctx.stream, false);
    copy_token token(ctx);
    EXPECT_THROW(service.enqueue(token, 0, false), std::runtime_error);

    service.reset();
    EXPECT_NO_THROW(service.enqueue(token, 0, false));
    fake_runtime::set_stream_ready(ctx.stream, true);
    EXPECT_EQ(service.poll().total(), 1);
}

// Test: service.drain() completes all pending
TEST_F(CopyTokenTest, ServiceDrain)
{
    auto& service = retained_operation_service::instance();
    service.reset();
    service.set_max_pending(10);

    execution_context ctx;
    ctx.dev.type  = device_enum::CUDA;
    ctx.dev.index = 0;
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
    ctx.dev.type  = device_enum::CUDA;
    ctx.dev.index = 0;
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
    ctx.dev.type = device_enum::CUDA;
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
    ctx.dev.type = device_enum::CUDA;
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
    ctx.dev.type = device_enum::CUDA;
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
    ctx.dev.type = device_enum::CUDA;
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
    EXPECT_EQ(service.poll().total(), 0);
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
    ctx.dev.type = device_enum::CUDA;
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

    EXPECT_EQ(service.poll().total(), 1);
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
    ctx.dev.type = device_enum::CUDA;
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
    ctx.dev.type = device_enum::CUDA;
    ctx.stream      = stream;
    auto del = [](T* p, size_t, execution_context const&) { delete[] p; };
    auto src = allocator<T>::allocate_adopted(new T[8]{}, 8, ctx, del);
    auto dst = allocator<T>::allocate_adopted(new T[8]{}, 8, ctx, del);
    for (int i = 0; i < 20; ++i)  // warm
    {
        (void)allocator<T>::copy_async_retained(src, dst, stream);
        (void)service.poll();
    }
    int const creates0  = fake_runtime::event_creates;
    int const destroys0 = fake_runtime::event_destroys;
    g_new_calls = 0;
    g_count_new = true;
    for (int i = 0; i < 100; ++i)
    {
        (void)allocator<T>::copy_async_retained(src, dst, stream);
        (void)service.poll();
    }
    g_count_new = false;
    size_t const heap     = g_new_calls.load();
    int const    creates  = fake_runtime::event_creates - creates0;
    int const    destroys = fake_runtime::event_destroys - destroys0;
    RecordProperty("retained_copy_heap_allocations_per_100", static_cast<int>(heap));
    RecordProperty("retained_copy_event_creates_per_100", creates);
    // §6.1, task 3.4: steady state allocates nothing and creates/destroys no event.
    EXPECT_EQ(0u, heap);
    EXPECT_EQ(0, creates);
    EXPECT_EQ(0, destroys);
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
    ctx.dev.type  = device_enum::CUDA;
    ctx.dev.index = 0;
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
        ctx.dev.type = device_enum::CUDA;
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
        service().abandon_quarantined();
        service().set_max_pending(1);
        failed_before_ = service().failed_count();
    }
    void TearDown() override
    {
        service().reset();
        service().abandon_quarantined();
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
    EXPECT_TRUE(no_event_leaked());

    // The single admission slot (limit 1) is free again.
    auto token = allocator<float>::copy_async_retained(pair.source, pair.destination, stream);
    EXPECT_EQ(1U, service().pending_count());
    fake_runtime::set_stream_ready(stream, true);
    token.wait();
    EXPECT_EQ(1U, service().poll().total());
    EXPECT_EQ(0, releases);  // caller still owns the endpoints
}

TEST_F(CopyFailureTest, SetupFailureAfterAdmissionCancelsReservationWithoutWaiting)
{
    auto stream = reinterpret_cast<void*>(3);
    fake_runtime::set_stream_ready(stream, false);
    int           releases = 0;
    retained_pair pair(&releases, stream);

    // Current device 1, operation device 0. cudaSetDevice calls: prepare_event
    // switches and restores (1, 2); the admission-time state() query returns
    // pending before the event is recorded, without touching the device; then
    // the submission guard fails to switch (3) before any driver copy is issued.
    fake_runtime::current_device          = 1;
    fake_runtime::fail_set_device_on_call = 3;
    EXPECT_THROW(
        allocator<float>::copy_async_retained(pair.source, pair.destination, stream),
        std::runtime_error);
    fake_runtime::fail_set_device_on_call = 0;

    EXPECT_EQ(0, fake_runtime::copies);
    EXPECT_EQ(0, fake_runtime::stream_syncs);  // never waited on the (busy) stream
    EXPECT_EQ(0U, service().pending_count());  // admission restored
    EXPECT_EQ(failed_before_, service().failed_count());
    EXPECT_EQ(1, pair.source.use_count());  // payload released
    EXPECT_TRUE(no_event_leaked());

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

    service().abandon_quarantined();  // explicit recovery releases the retained owners
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
    service().abandon_quarantined();
    EXPECT_EQ(before_clear + 2, releases);
}

// ---------------------------------------------------------------------------
// Phase 5 (5.1-5.6): admission, accounting, release outside the lock, failure path
// without allocation, ordered shutdown, endpoint lifetimes.
// ---------------------------------------------------------------------------
namespace
{
execution_context cuda_ctx_on(void* stream)
{
    execution_context ctx;
    ctx.dev.type = device_enum::CUDA;
    ctx.stream   = stream;
    return ctx;
}
}  // namespace

TEST_F(CopyFailureTest, AdmissionIsFiniteByDefault)
{
    service().reset();
    EXPECT_EQ(retained_operation_service::default_max_pending, service().max_pending());
    EXPECT_GT(service().max_pending(), 0U);
    EXPECT_GT(service().max_quarantined(), 0U);
}

TEST_F(CopyFailureTest, FullQueueFailsFastWhenAskedAndSubmitsNothing)
{
    auto stream = reinterpret_cast<void*>(21);
    fake_runtime::set_stream_ready(stream, false);
    int           releases = 0;
    retained_pair first(&releases, stream);
    retained_pair second(&releases, stream);

    auto token = allocator<float>::copy_async_retained(first.source, first.destination, stream);
    EXPECT_EQ(1U, service().pending_count());
    int const copies = fake_runtime::copies;

    EXPECT_THROW(
        allocator<float>::copy_async_retained(second.source, second.destination, stream, false),
        std::runtime_error);
    EXPECT_EQ(copies, fake_runtime::copies);  // nothing was submitted
    EXPECT_EQ(1U, service().pending_count());
    EXPECT_EQ(1, second.source.use_count());  // the refused attempt kept no owner
    EXPECT_EQ(1, second.destination.use_count());
    token.wait();
    // The service's copy of the payload goes now, while `releases` is still alive.
    EXPECT_EQ(1U, service().poll().completed);
    EXPECT_EQ(0U, service().pending_count());
    EXPECT_EQ(0, releases);  // the handles in this scope still own the storage
}

TEST_F(CopyFailureTest, BlockedAdmissionIsWokenByShutdownAndByALimitChange)
{
    auto stream_a = reinterpret_cast<void*>(22);
    auto stream_b = reinterpret_cast<void*>(23);
    auto stream_c = reinterpret_cast<void*>(24);
    fake_runtime::set_stream_ready(stream_a, false);
    fake_runtime::set_stream_ready(stream_b, false);
    fake_runtime::set_stream_ready(stream_c, false);
    copy_token filler(cuda_ctx_on(stream_a));
    copy_token waiter(cuda_ctx_on(stream_b));
    copy_token second_waiter(cuda_ctx_on(stream_c));
    ASSERT_TRUE(service().enqueue(filler, 0, false));  // limit is 1: the queue is full

    // The fake runtime is only read while a waiter polls; the main thread sleeps
    // and then calls the service.
    std::atomic<bool> started{false};
    std::atomic<bool> threw{false};
    std::atomic<bool> admitted{false};
    std::thread       thread([&] {
        started = true;
        try
        {
            admitted = service().enqueue(waiter, 0, true);
        }
        catch (std::runtime_error const&)
        {
            threw = true;
        }
    });
    while (!started)
    {
        std::this_thread::yield();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    EXPECT_FALSE(threw);
    EXPECT_FALSE(admitted);  // still waiting for capacity

    service().set_max_pending(2);  // a limit change wakes it and it is admitted
    thread.join();
    EXPECT_TRUE(admitted);
    EXPECT_EQ(2U, service().pending_count());

    started  = false;
    threw    = false;
    admitted = false;
    std::thread second([&] {
        started = true;
        try
        {
            admitted = service().enqueue(second_waiter, 0, true);
        }
        catch (std::runtime_error const&)
        {
            threw = true;
        }
    });
    while (!started)
    {
        std::this_thread::yield();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    EXPECT_FALSE(threw);
    (void)service().shutdown(std::chrono::milliseconds(1));  // shutdown wakes it; it throws
    second.join();
    EXPECT_TRUE(threw);
    EXPECT_FALSE(admitted);
    EXPECT_EQ(2U, service().pending_count());  // nothing was dropped
}

TEST_F(CopyFailureTest, StatsCountPendingAndQuarantinedOperationsAndBytes)
{
    service().set_max_pending(8);
    auto good = reinterpret_cast<void*>(25);
    auto bad  = reinterpret_cast<void*>(26);
    fake_runtime::set_stream_ready(good, false);
    fake_runtime::set_stream_error(bad);
    int           releases = 0;
    retained_pair a(&releases, good);
    retained_pair b(&releases, bad);

    auto ta = allocator<float>::copy_async_retained(a.source, a.destination, good);
    auto tb = allocator<float>::copy_async_retained(b.source, b.destination, bad);
    constexpr size_t kBytes = 4 * sizeof(float);
    auto             s      = service().stats();
    EXPECT_EQ(2U, s.pending_ops);
    EXPECT_EQ(2 * kBytes, s.pending_bytes);
    EXPECT_EQ(0U, s.quarantined_ops);

    fake_runtime::set_stream_ready(good, true);
    auto r = service().poll();
    EXPECT_EQ(1U, r.completed);  // completed and failed are reported apart
    EXPECT_EQ(1U, r.failed);
    s = service().stats();
    EXPECT_EQ(0U, s.pending_ops);
    EXPECT_EQ(0U, s.pending_bytes);
    EXPECT_EQ(1U, s.quarantined_ops);
    EXPECT_EQ(kBytes, s.quarantined_bytes);
    EXPECT_EQ(1U, service().failed_count());
    EXPECT_EQ(1U, service().abandon_quarantined());  // the service lets go before `releases` dies
    EXPECT_EQ(0U, service().failed_count());
    (void)ta;
    (void)tb;
}

TEST_F(CopyFailureTest, QuarantineBudgetRefusesAdmissionButNeverDropsAnOwner)
{
    service().set_max_pending(8);
    auto bad = reinterpret_cast<void*>(27);
    fake_runtime::set_stream_error(bad);
    int           releases = 0;
    retained_pair a(&releases, bad);
    retained_pair b(&releases, bad);
    (void)allocator<float>::copy_async_retained(a.source, a.destination, bad);
    EXPECT_EQ(1U, service().poll().failed);

    service().set_max_quarantined(1);  // at the limit: admission is refused
    int const copies = fake_runtime::copies;
    EXPECT_THROW(
        allocator<float>::copy_async_retained(b.source, b.destination, bad, false),
        std::runtime_error);
    EXPECT_EQ(copies, fake_runtime::copies);
    EXPECT_EQ(1U, service().failed_count());
    EXPECT_GT(a.source.use_count(), 1);  // lowering and refusing released nothing
    EXPECT_EQ(0, releases);

    service().set_max_quarantined(0);  // 0 = unlimited again
    EXPECT_NO_THROW(
        (void)allocator<float>::copy_async_retained(b.source, b.destination, bad, false));
    service().poll();
    service().abandon_quarantined();
}

TEST_F(CopyFailureTest, PayloadDeleterRunsOutsideTheServiceLock)
{
    service().set_max_pending(8);
    auto stream = reinterpret_cast<void*>(28);
    fake_runtime::set_stream_ready(stream, false);
    int  releases = 0;
    auto deleter  = [&releases](float* p, size_t, execution_context const&) {
        // Re-enters the service from inside the release. Under the service mutex
        // this would deadlock.
        (void)retained_operation_service::instance().poll();
        (void)retained_operation_service::instance().stats();
        ++releases;
        delete[] p;
    };
    {
        auto ctx = cuda_ctx_on(stream);
        auto src = allocator<float>::allocate_adopted(new float[2]{1, 2}, 2, ctx, deleter);
        auto dst = allocator<float>::allocate_adopted(new float[2]{}, 2, ctx, deleter);
        (void)allocator<float>::copy_async_retained(src, dst, stream);
    }
    EXPECT_EQ(0, releases);
    fake_runtime::set_stream_ready(stream, true);
    EXPECT_EQ(1U, service().poll().completed);
    EXPECT_EQ(2, releases);
    EXPECT_EQ(0U, service().pending_count());
}

TEST_F(CopyFailureTest, FailurePathAllocatesNothingAndKeepsTheOwner)
{
    service().set_max_pending(8);
    auto stream = reinterpret_cast<void*>(29);
    fake_runtime::set_stream_error(stream);
    int releases = 0;
    {
        retained_pair pair(&releases, stream);
        (void)allocator<float>::copy_async_retained(pair.source, pair.destination, stream);
    }
    g_fail_next_new  = true;  // any allocation on the failure path would throw
    auto       r     = service().poll();
    bool const armed = g_fail_next_new.exchange(false);
    EXPECT_EQ(1U, r.failed);
    EXPECT_TRUE(armed);  // nothing allocated: the quarantine is a flag on the entry
    EXPECT_EQ(1U, service().failed_count());
    EXPECT_EQ(0, releases);

    // Checked recovery releases only what it can prove idle.
    EXPECT_EQ(0U, service().recover_quarantined());
    EXPECT_EQ(0, releases);
    EXPECT_EQ(1U, service().failed_count());
    fake_runtime::set_stream_ready(stream, true);
    EXPECT_EQ(1U, service().recover_quarantined());
    EXPECT_EQ(2, releases);
    EXPECT_EQ(0U, service().failed_count());
}

TEST_F(CopyFailureTest, ResetWaitsForPendingWorkAndQuarantinesAFailureInsteadOfDroppingIt)
{
    service().set_max_pending(8);
    auto ok  = reinterpret_cast<void*>(30);
    auto bad = reinterpret_cast<void*>(31);
    fake_runtime::set_stream_ready(ok, false);
    fake_runtime::set_stream_ready(bad, false);
    int releases = 0;
    {
        retained_pair a(&releases, ok);
        retained_pair b(&releases, bad);
        (void)allocator<float>::copy_async_retained(a.source, a.destination, ok);
        (void)allocator<float>::copy_async_retained(b.source, b.destination, bad);
    }
    fake_runtime::set_stream_error(bad);  // the wait on this one will fail
    service().reset();
    EXPECT_EQ(2, releases);                   // the successful one released its two owners
    EXPECT_EQ(1U, service().failed_count());  // the failed one is still owned
    EXPECT_EQ(0U, service().pending_count());
    EXPECT_EQ(retained_operation_service::default_max_pending, service().max_pending());
    service().abandon_quarantined();
    EXPECT_EQ(4, releases);
}

TEST_F(CopyFailureTest, LifetimeHoldsPageableHostAndAdoptedDeviceEndpointsUntilCompletion)
{
    service().set_max_pending(8);
    auto stream = reinterpret_cast<void*>(32);
    fake_runtime::set_stream_ready(stream, false);
    int  host_releases   = 0;
    int  device_releases = 0;
    {
        auto src = allocator<float>::allocate_adopted(
            new float[3]{4, 5, 6}, 3, execution_context::cpu(),
            [&](float* p, size_t, execution_context const&) {
                ++host_releases;
                delete[] p;
            });
        auto dst = allocator<float>::allocate_adopted(
            new float[3]{}, 3, cuda_ctx_on(stream),
            [&](float* p, size_t, execution_context const&) {
                ++device_releases;
                delete[] p;
            });
        (void)allocator<float>::copy_async_retained(src, dst, stream);
    }  // every user handle is gone while the copy is still in flight
    EXPECT_EQ(0, host_releases);
    EXPECT_EQ(0, device_releases);
    EXPECT_EQ(0U, service().poll().total());
    EXPECT_EQ(0, host_releases);
    fake_runtime::set_stream_ready(stream, true);
    EXPECT_EQ(1U, service().poll().completed);
    EXPECT_EQ(1, host_releases);  // each foreign deleter ran exactly once, after completion
    EXPECT_EQ(1, device_releases);
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

// ===========================================================================
// Phase 4: transfer completion and context semantics (plan 4.1-4.7)
// ===========================================================================
namespace
{
execution_context cuda_ctx(void* stream, int device = 0)
{
    execution_context ctx;
    ctx.dev.type  = device_enum::CUDA;
    ctx.dev.index = static_cast<std::int16_t>(device);
    ctx.stream    = stream;
    return ctx;
}
}  // namespace

// 4.1: a failed device activation is an error, never a result. Before this task a
// query whose device switch failed went ahead on the wrong device and could report
// a copy complete.
TEST_F(CopyTokenTest, ActivationFailureDuringQueryFailsTheTokenAndKeepsTheDevice)
{
    auto stream = reinterpret_cast<void*>(5);
    fake_runtime::set_stream_ready(stream, true);  // the event IS complete
    copy_token token(cuda_ctx(stream, 0));
    token.prepare_event();
    token.record_event();

    fake_runtime::current_device          = 1;  // the query must switch to device 0 ...
    fake_runtime::fail_set_device_on_call = fake_runtime::set_device_calls + 1;  // ... and fails
    EXPECT_EQ(token.state(), completion_state::failed);
    EXPECT_FALSE(token.ready());
    EXPECT_EQ(fake_runtime::current_device, 1) << "the caller's device must be left alone";
    fake_runtime::fail_set_device_on_call = 0;

    // The failure is published once and is stable: later queries and waits agree.
    EXPECT_EQ(token.state(), completion_state::failed);
    EXPECT_THROW(token.wait(), std::runtime_error);
}

TEST_F(CopyTokenTest, ActivationFailureDuringWaitFailsTheTokenAndNeverSynchronizes)
{
    auto stream = reinterpret_cast<void*>(6);
    fake_runtime::set_stream_ready(stream, false);
    copy_token token(cuda_ctx(stream, 0));
    token.prepare_event();
    token.record_event();

    fake_runtime::current_device          = 2;
    fake_runtime::fail_set_device_on_call = fake_runtime::set_device_calls + 1;
    EXPECT_THROW(token.wait(), std::runtime_error);
    fake_runtime::fail_set_device_on_call = 0;
    EXPECT_EQ(fake_runtime::event_syncs, 0) << "nothing may be waited on in the wrong context";
    EXPECT_EQ(fake_runtime::current_device, 2);
    EXPECT_EQ(token.state(), completion_state::failed);
}

TEST_F(CopyTokenTest, SuccessfulCrossDeviceQueryRestoresThePreviousDevice)
{
    auto stream = reinterpret_cast<void*>(7);
    fake_runtime::set_stream_ready(stream, true);
    copy_token token(cuda_ctx(stream, 0));
    token.prepare_event();
    token.record_event();
    fake_runtime::current_device = 3;
    EXPECT_EQ(token.state(), completion_state::complete);
    EXPECT_EQ(fake_runtime::current_device, 3);
}

#if MEMORY_HAS_CUDA
// 4.2: copy_sync returns when ITS copy completed: it waits on the copy's own
// event, on the stream it was given, with no stream or device synchronize.
TEST_F(CopyTokenTest, CopySyncWaitsOnItsOwnEventOnTheGivenStream)
{
    auto  stream = reinterpret_cast<void*>(3);
    auto  other  = reinterpret_cast<void*>(4);
    float src[4] = {1, 2, 3, 4};
    float dst[4] = {};
    fake_runtime::set_stream_ready(stream, false);  // earlier work on the stream is pending
    fake_runtime::set_stream_ready(other, false);   // unrelated work elsewhere stays pending

    allocator<float>::copy_sync(src, 4, dst, device_enum::CPU, device_enum::CUDA, 0, 0, stream);

    EXPECT_EQ(dst[3], 4.0F);
    EXPECT_EQ(fake_runtime::event_records, 1);
    EXPECT_EQ(fake_runtime::event_syncs, 1) << "waits on its own event";
    EXPECT_EQ(fake_runtime::stream_syncs, 0) << "not on the whole stream";
    EXPECT_EQ(fake_runtime::device_syncs, 0) << "never device-wide";
    EXPECT_EQ(fake_runtime::get_stream_state(other), fake_runtime::STREAM_NOT_READY)
        << "unrelated streams are untouched";
    EXPECT_TRUE(no_event_leaked());
}

TEST_F(CopyTokenTest, CopySyncNeverReportsCompletionForAFailedCopyOrWait)
{
    auto  stream = reinterpret_cast<void*>(3);
    float src[4] = {1, 2, 3, 4};
    float dst[4] = {};

    fake_runtime::fail_memcpy = true;  // submission fails: nothing in flight
    EXPECT_THROW(
        allocator<float>::copy_sync(src, 4, dst, device_enum::CPU, device_enum::CUDA, 0, 0, stream),
        std::runtime_error);
    fake_runtime::fail_memcpy = false;
    EXPECT_EQ(dst[0], 0.0F);

    fake_runtime::set_stream_error(stream);  // the wait on the copy's event fails
    EXPECT_THROW(
        allocator<float>::copy_sync(src, 4, dst, device_enum::CPU, device_enum::CUDA, 0, 0, stream),
        std::runtime_error);
}

// 4.3: a null stream is only meaningful when the caller and the library agree on
// which default stream it names, and a per-thread default stream cannot be a cache
// identity. Ambiguity is rejected before anything is acquired or submitted.
namespace
{
struct library_mode_guard
{
    explicit library_mode_guard(memory::detail::default_stream_mode mode)
    {
        memory::detail::set_library_default_stream_mode_for_testing(mode);
    }
    ~library_mode_guard()
    {
        memory::detail::set_library_default_stream_mode_for_testing(
            memory::detail::default_stream_mode::legacy);
    }
};
}  // namespace

TEST_F(CopyTokenTest, AmbiguousNullStreamIsRejectedBeforeAnySubmission)
{
    using memory::detail::default_stream_mode;
    float src[4] = {1, 2, 3, 4};
    float dst[4] = {};
    {
        library_mode_guard const per_thread(default_stream_mode::per_thread);  // caller is legacy
        EXPECT_THROW(
            allocator<float>::copy_async(src, 4, dst, nullptr, device_enum::CPU, device_enum::CUDA),
            std::invalid_argument);
        EXPECT_THROW(
            allocator<float>::copy_sync(src, 4, dst, device_enum::CPU, device_enum::CUDA),
            std::invalid_argument);
        EXPECT_THROW(
            allocator<float>::record_stream(dst, device_enum::CUDA, 0, nullptr),
            std::invalid_argument);
        EXPECT_EQ(fake_runtime::copies, 0);
        EXPECT_EQ(fake_runtime::event_creates, 0) << "rejected before any reservation";

        // An explicit stream names one stream in either mode: the supported path.
        auto stream = reinterpret_cast<void*>(3);
        fake_runtime::set_stream_ready(stream, true);
        EXPECT_NO_THROW(
            allocator<float>::copy_async(src, 4, dst, stream, device_enum::CPU, device_enum::CUDA)
                .wait());
    }
    // Legacy on both sides: the null stream is unambiguous and accepted.
    fake_runtime::set_stream_ready(nullptr, true);
    EXPECT_NO_THROW(
        allocator<float>::copy_async(src, 4, dst, nullptr, device_enum::CPU, device_enum::CUDA)
            .wait());
}

TEST_F(CopyTokenTest, StreamHandleValidationMatrix)
{
    using memory::detail::default_stream_mode;
    using memory::detail::validate_stream_handle;
    auto explicit_stream = reinterpret_cast<void*>(3);
    {
        library_mode_guard const library(default_stream_mode::legacy);
        EXPECT_NO_THROW(validate_stream_handle(nullptr, default_stream_mode::legacy));
        EXPECT_THROW(
            validate_stream_handle(nullptr, default_stream_mode::per_thread), std::invalid_argument);
        EXPECT_NO_THROW(validate_stream_handle(explicit_stream, default_stream_mode::per_thread));
    }
    {
        library_mode_guard const library(default_stream_mode::per_thread);
        // Agreeing is not enough: one cache identity cannot name every thread's stream.
        EXPECT_THROW(
            validate_stream_handle(nullptr, default_stream_mode::per_thread), std::invalid_argument);
        EXPECT_THROW(
            validate_stream_handle(nullptr, default_stream_mode::legacy), std::invalid_argument);
        EXPECT_NO_THROW(validate_stream_handle(explicit_stream, default_stream_mode::per_thread));
    }
}

// 4.4: record_stream is reuse-only. It keeps the block from being recycled until the
// consumer's stream catches up; it never makes that stream wait for the producer.
// stream_wait is the ordering call.
TEST_F(CopyTokenTest, RecordStreamAloneDoesNotOrderAConsumerButStreamWaitDoes)
{
    auto  producer = reinterpret_cast<void*>(3);
    auto  consumer = reinterpret_cast<void*>(4);
    float src[4]   = {1, 2, 3, 4};
    float dst[4]   = {};
    fake_runtime::set_stream_ready(producer, false);  // the copy is still running

    copy_token token =
        allocator<float>::copy_async(src, 4, dst, producer, device_enum::CPU, device_enum::CUDA);
    allocator<float>::record_stream(dst, device_enum::CUDA, 0, consumer);
    EXPECT_EQ(token.state(), completion_state::pending);
    EXPECT_TRUE(fake_runtime::stream_waits.empty())
        << "record_stream alone leaves the consumer free to run ahead of the producer";

    token.stream_wait(consumer);
    ASSERT_EQ(fake_runtime::stream_waits.size(), 1U);
    EXPECT_EQ(fake_runtime::stream_waits[0].first, consumer);

    // A finished operation needs no device-side wait.
    token.wait();
    fake_runtime::stream_waits.clear();
    token.stream_wait(consumer);
    EXPECT_TRUE(fake_runtime::stream_waits.empty());
}

TEST_F(CopyTokenTest, StreamWaitFailureAndFailedTokensThrow)
{
    auto producer = reinterpret_cast<void*>(3);
    auto consumer = reinterpret_cast<void*>(4);
    fake_runtime::set_stream_ready(producer, false);
    copy_token pending(cuda_ctx(producer));
    pending.prepare_event();
    pending.record_event();

    fake_runtime::fail_stream_wait = true;
    EXPECT_THROW(pending.stream_wait(consumer), std::runtime_error);
    fake_runtime::fail_stream_wait = false;

    copy_token failed(cuda_ctx(producer));
    memory::detail::copy_token_access::fail(failed);
    EXPECT_THROW(failed.stream_wait(consumer), std::runtime_error);

    copy_token no_event(cuda_ctx(producer));  // never submitted: nothing to wait on
    EXPECT_THROW(no_event.stream_wait(consumer), std::runtime_error);

    copy_token cpu_token;
    EXPECT_NO_THROW(cpu_token.stream_wait(consumer));
}

// 4.5: an unsupported peer copy is rejected before any reservation, stream-use
// record or driver call. The copy only checks peer access; it never grants it.
TEST_F(CopyTokenTest, UnsupportedPeerCopyIsRejectedBeforeAnySubmission)
{
    auto  stream = reinterpret_cast<void*>(3);
    float src[4] = {1, 2, 3, 4};
    float dst[4] = {};
    fake_runtime::set_stream_ready(stream, true);
    fake_runtime::peer_denied[1][0] = true;  // device 1 cannot reach device 0
    int const uses_before           = memory::gpu::test::record_stream_use_calls.load();

    EXPECT_THROW(
        allocator<float>::copy_async(src, 4, dst, stream, device_enum::CUDA, device_enum::CUDA, 0, 1),
        std::invalid_argument);
    EXPECT_GT(fake_runtime::peer_access_queries, 0);
    EXPECT_EQ(fake_runtime::peer_copies, 0);
    EXPECT_EQ(fake_runtime::copies, 0);
    EXPECT_EQ(fake_runtime::event_creates, 0);
    EXPECT_EQ(fake_runtime::event_records, 0);
    EXPECT_EQ(memory::gpu::test::record_stream_use_calls.load(), uses_before);

    // The opposite direction is supported and the copy goes through.
    EXPECT_NO_THROW(
        allocator<float>::copy_async(
            src, 4, dst, stream, device_enum::CUDA, device_enum::CUDA, 1, 0)
            .wait());
    EXPECT_EQ(fake_runtime::peer_copies, 1);

    // An out-of-range device cannot be queried and is rejected the same way.
    EXPECT_THROW(
        allocator<float>::copy_async(src, 4, dst, stream, device_enum::CUDA, device_enum::CUDA, 0, 7),
        std::invalid_argument);

    // Same-device copies never consult peer access.
    int const queries = fake_runtime::peer_access_queries;
    EXPECT_NO_THROW(
        allocator<float>::copy_async(
            src, 4, dst, stream, device_enum::CUDA, device_enum::CUDA, 2, 2)
            .wait());
    EXPECT_EQ(fake_runtime::peer_access_queries, queries);
}

// 4.7: an interior, freed or foreign GPU pointer cannot be tracked by the cache, so
// the copy is refused before either endpoint's stream use is recorded or any work
// is submitted (it used to fail partway, after recording the first endpoint).
TEST_F(CopyTokenTest, InteriorOrForeignGpuPointerIsRejectedBeforeSubmission)
{
    auto  stream   = reinterpret_cast<void*>(3);
    float base[8]  = {};
    float other[8] = {};
    fake_runtime::set_stream_ready(stream, true);
    memory::gpu::test::foreign_pointer = base + 1;  // the cache does not own this address
    int const uses_before              = memory::gpu::test::record_stream_use_calls.load();

    EXPECT_THROW(
        allocator<float>::copy_async(
            other, 4, base + 1, stream, device_enum::CUDA, device_enum::CUDA),
        std::invalid_argument);  // interior destination
    EXPECT_THROW(
        allocator<float>::copy_async(
            base + 1, 4, other, stream, device_enum::CUDA, device_enum::CUDA),
        std::invalid_argument);  // interior source
    EXPECT_EQ(fake_runtime::copies, 0);
    EXPECT_EQ(fake_runtime::event_records, 0);
    EXPECT_EQ(memory::gpu::test::record_stream_use_calls.load(), uses_before)
        << "the valid endpoint must not be recorded when the other is refused";

    memory::gpu::test::foreign_pointer = nullptr;
    EXPECT_NO_THROW(
        allocator<float>::copy_async(
            other, 4, base, stream, device_enum::CUDA, device_enum::CUDA)
            .wait());
    EXPECT_EQ(fake_runtime::copies, 1);
}
#endif  // MEMORY_HAS_CUDA
