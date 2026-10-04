/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

// Phase 4 (plan 4.2, 4.4, 4.6, 4.7) on a real CUDA/HIP device. The deterministic
// shim covers the same contracts without hardware (TestCopyRuntime.cpp); these
// tests need a device and skip without one. A host callback holds a stream so the
// ordering of "enqueued" and "complete" is observable instead of racy.

#include <gtest/gtest.h>

#include "common/memory_macros.h"

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP

#include <atomic>
#include <cstdio>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "allocator.h"
#include "common/copy_token.h"
#include "common/data_ptr.h"
#include "common/retained_operation_service.h"
#include "common/pinned_buffer.h"
#include "common/pinned_staging_ring.h"
#include "common/retained_ptr.h"
#include "gpu/device_guard.h"
#include "gpu/gpu_runtime.h"
#include "gpu/gpu_workspace.h"
#include "helper/pinned_memory_allocator.h"

using namespace memory;
using namespace memory::gpu;

namespace
{
using clock_type = std::chrono::steady_clock;

bool device_available()
{
    int               count = 0;
    const cudaError_t err   = cudaGetDeviceCount(&count);
    return err == cudaSuccess && count > 0;
}

class test_stream
{
public:
    test_stream()
    {
        throw_on_cuda_error(
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    }
    ~test_stream() { (void)cudaStreamDestroy(stream); }
    test_stream(const test_stream&)            = delete;
    test_stream& operator=(const test_stream&) = delete;
    cudaStream_t stream{nullptr};
};

// Holds a stream until released (or a 10 s deadline, so a regression that makes
// the stream wait on itself fails instead of hanging).
class stream_blocker
{
public:
    explicit stream_blocker(cudaStream_t stream) : stream_(stream) {}
    ~stream_blocker() { (void)release(); }
    stream_blocker(const stream_blocker&)            = delete;
    stream_blocker& operator=(const stream_blocker&) = delete;

    cudaError_t start()
    {
        auto       context = std::make_unique<std::shared_ptr<callback_state>>(state_);
        const auto result  = cudaLaunchHostFunc(stream_, wait, context.get());
        active_            = result == cudaSuccess;
        if (active_)
        {
            (void)context.release();
        }
        return result;
    }
    // Does not synchronize the stream: the caller may be the thread that waits on it.
    void open() { state_->released.store(true, std::memory_order_release); }
    cudaError_t release()
    {
        open();
        if (!active_)
        {
            return cudaSuccess;
        }
        const auto result = cudaStreamSynchronize(stream_);
        if (result == cudaSuccess)
        {
            active_ = false;
        }
        return result;
    }
    bool timed_out() const { return state_->timed_out.load(std::memory_order_acquire); }

private:
    struct callback_state
    {
        std::atomic<bool> released{false};
        std::atomic<bool> timed_out{false};
    };
    static void CUDART_CB wait(void* data)
    {
        const std::unique_ptr<std::shared_ptr<callback_state>> context(
            static_cast<std::shared_ptr<callback_state>*>(data));
        auto&      self     = **context;
        const auto deadline = clock_type::now() + std::chrono::seconds(10);
        while (!self.released.load(std::memory_order_acquire))
        {
            if (clock_type::now() >= deadline)
            {
                self.timed_out.store(true, std::memory_order_release);
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    cudaStream_t                    stream_;
    bool                            active_{false};
    std::shared_ptr<callback_state> state_{std::make_shared<callback_state>()};
};

template <typename T>
class pinned_host
{
public:
    explicit pinned_host(size_t count) : count_(count)
    {
        throw_on_cuda_error(
            cudaHostAlloc(reinterpret_cast<void**>(&data_), count * sizeof(T), 0), "cudaHostAlloc");
    }
    ~pinned_host() { (void)cudaFreeHost(data_); }
    pinned_host(const pinned_host&)            = delete;
    pinned_host& operator=(const pinned_host&) = delete;
    T*           data() { return data_; }
    size_t       size() const { return count_; }

private:
    T*     data_{nullptr};
    size_t count_;
};

constexpr size_t kCount = 4096;

void fill_device(data_ptr<float>& buffer, float value, cudaStream_t stream)
{
    pinned_host<float> host(buffer.size());
    for (size_t i = 0; i < host.size(); ++i)
    {
        host.data()[i] = value;
    }
    ASSERT_EQ(
        cudaMemcpyAsync(
            buffer.data(), host.data(), buffer.size() * sizeof(float), cudaMemcpyHostToDevice, stream),
        cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
}

class Phase4Hardware : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if (!device_available())
        {
            GTEST_SKIP() << "no CUDA/HIP device";
        }
    }
};
}  // namespace

// 4.6: clone() hands back a COMPLETE copy. The stream is held, so a clone that only
// enqueued its copy would return at once; this one must wait for the copy behind
// the hold, and the stream must be idle when it returns.
TEST_F(Phase4Hardware, CloneIsCompleteOnReturn)
{
    test_stream             s;
    execution_context const ctx{device_enum::CUDA, 0, s.stream};
    data_ptr<float>         source(kCount, ctx);
    fill_device(source, 5.0F, s.stream);
    {
        data_ptr<float> warm(kCount, ctx);  // pre-warm the cache: no cudaMalloc while held
    }

    stream_blocker blocker(s.stream);
    ASSERT_EQ(blocker.start(), cudaSuccess);
    std::thread releaser(
        [&]
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
            blocker.open();
        });
    const auto      started = clock_type::now();
    data_ptr<float> copy    = source.clone();
    const auto      waited  = clock_type::now() - started;
    releaser.join();

    EXPECT_FALSE(blocker.timed_out());
    EXPECT_GE(waited, std::chrono::milliseconds(100))
        << "clone returned before the copy behind the held stream could have run";
    EXPECT_EQ(cudaStreamQuery(s.stream), cudaSuccess) << "the clone's copy must be finished";

    pinned_host<float> host(kCount);
    ASSERT_EQ(
        cudaMemcpy(host.data(), copy.data(), kCount * sizeof(float), cudaMemcpyDeviceToHost),
        cudaSuccess);
    EXPECT_EQ(host.data()[0], 5.0F);
    EXPECT_EQ(host.data()[kCount - 1], 5.0F);
}

// 4.2: copy_sync on an explicit stream returns only after its own copy.
TEST_F(Phase4Hardware, CopySyncOnAnExplicitStreamWaitsForItsOwnCopy)
{
    test_stream             s;
    execution_context const ctx{device_enum::CUDA, 0, s.stream};
    data_ptr<float>         source(kCount, ctx);
    data_ptr<float>         destination(kCount, ctx);
    fill_device(source, 3.0F, s.stream);
    fill_device(destination, 0.0F, s.stream);

    stream_blocker blocker(s.stream);
    ASSERT_EQ(blocker.start(), cudaSuccess);
    std::thread releaser(
        [&]
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
            blocker.open();
        });
    const auto started = clock_type::now();
    allocator<float>::copy_sync(
        source.data(), kCount, destination.data(), device_enum::CUDA, device_enum::CUDA, 0, 0, s.stream);
    const auto waited = clock_type::now() - started;
    releaser.join();

    EXPECT_FALSE(blocker.timed_out());
    EXPECT_GE(waited, std::chrono::milliseconds(100));
    pinned_host<float> host(kCount);
    ASSERT_EQ(
        cudaMemcpy(host.data(), destination.data(), kCount * sizeof(float), cudaMemcpyDeviceToHost),
        cudaSuccess);
    EXPECT_EQ(host.data()[0], 3.0F);
}

// 4.4: record_stream alone leaves a consumer racing the producer; stream_wait
// orders it. The producer is held so the difference is deterministic.
TEST_F(Phase4Hardware, StreamWaitOrdersAConsumerThatRecordStreamAloneDoesNot)
{
    test_stream             producer;
    test_stream             consumer;
    execution_context const ctx{device_enum::CUDA, 0, producer.stream};
    data_ptr<float>         source(kCount, ctx);
    data_ptr<float>         destination(kCount, ctx);
    fill_device(source, 7.0F, producer.stream);
    pinned_host<float> host(kCount);

    auto const read_on_consumer = [&]
    {
        host.data()[0] = -1.0F;
        ASSERT_EQ(
            cudaMemcpyAsync(
                host.data(), destination.data(), sizeof(float), cudaMemcpyDeviceToHost, consumer.stream),
            cudaSuccess);
    };

    {  // record_stream only: the consumer reads before the held producer copy ran.
        fill_device(destination, 0.0F, producer.stream);
        stream_blocker blocker(producer.stream);
        ASSERT_EQ(blocker.start(), cudaSuccess);
        copy_token token = copy_async(source, destination, producer.stream);
        destination.record_stream(consumer.stream);
        read_on_consumer();
        ASSERT_EQ(cudaStreamSynchronize(consumer.stream), cudaSuccess);
        EXPECT_EQ(host.data()[0], 0.0F) << "record_stream must not order the consumer";
        ASSERT_EQ(blocker.release(), cudaSuccess);
        token.wait();
    }
    {  // stream_wait: the consumer runs after the producer's copy.
        fill_device(destination, 0.0F, producer.stream);
        stream_blocker blocker(producer.stream);
        ASSERT_EQ(blocker.start(), cudaSuccess);
        copy_token token = copy_async(source, destination, producer.stream);
        token.stream_wait(consumer.stream);
        read_on_consumer();
        std::thread releaser(
            [&]
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                blocker.open();
            });
        ASSERT_EQ(cudaStreamSynchronize(consumer.stream), cudaSuccess);
        releaser.join();
        EXPECT_EQ(host.data()[0], 7.0F) << "stream_wait must order the consumer after the copy";
        token.wait();
    }
}

// 4.7: interior, borrowed and slice sources are rejected before anything is
// submitted, using the allocation identity the view carries.
TEST_F(Phase4Hardware, InteriorAndBorrowedGpuPointersAreRejected)
{
    test_stream             s;
    execution_context const ctx{device_enum::CUDA, 0, s.stream};
    data_ptr<float>         a(256, ctx);
    data_ptr<float>         b(256, ctx);

    EXPECT_THROW(
        allocator<float>::copy_async(
            a.data() + 1, 8, b.data(), s.stream, device_enum::CUDA, device_enum::CUDA, 0, 0),
        std::invalid_argument);
    EXPECT_THROW(
        allocator<float>::copy_async(
            a.data(), 8, b.data() + 1, s.stream, device_enum::CUDA, device_enum::CUDA, 0, 0),
        std::invalid_argument);

    data_view<float> interior = a.view(4, 8);
    EXPECT_THROW((void)data_ptr<float>(interior), std::invalid_argument);

    data_view<float> borrowed =
        data_view<float>::borrow(a.data(), 8, device_enum::CUDA, 0, s.stream);
    EXPECT_THROW((void)data_ptr<float>(borrowed), std::invalid_argument);

    // The whole allocation (and a prefix starting at its base) is still accepted.
    data_view<float> whole = a.view();
    fill_device(a, 2.0F, s.stream);
    data_ptr<float> from_view(whole);
    pinned_host<float> host(256);
    ASSERT_EQ(
        cudaMemcpy(host.data(), from_view.data(), 256 * sizeof(float), cudaMemcpyDeviceToHost),
        cudaSuccess);
    EXPECT_EQ(host.data()[255], 2.0F);
    data_view<float> prefix = a.view(0, 16);
    EXPECT_NO_THROW((void)data_ptr<float>(prefix));
}

// 4.3 on the real library: the legacy null stream keeps working, a legacy-mode
// caller and library agree, and the explicit per-thread handle is refused.
TEST_F(Phase4Hardware, DefaultStreamModesAreExplicit)
{
    float* device_ptr = allocator<float>::allocate(64, device_enum::CUDA, 0, nullptr);
    ASSERT_NE(device_ptr, nullptr);
    allocator<float>::free(device_ptr, device_enum::CUDA, 0, 64, nullptr);

#ifdef cudaStreamPerThread
    EXPECT_THROW(
        (void)allocator<float>::allocate(
            64, device_enum::CUDA, 0, static_cast<void*>(cudaStreamPerThread)),
        std::invalid_argument);
#endif
}

// 5.5: ordered teardown. With a retained copy still in flight the runtime teardown
// reports it and touches nothing else; once it completes, teardown succeeds.
TEST_F(Phase4Hardware, ShutdownRuntimeWaitsForRetainedWorkBeforeTearingDownCaches)
{
    auto& service = retained_operation_service::instance();
    service.reset();
    test_stream             s;
    execution_context const ctx{device_enum::CUDA, 0, s.stream};
    stream_blocker          blocker(s.stream);
    ASSERT_EQ(blocker.start(), cudaSuccess);
    {
        auto source      = make_retained<float>(kCount, ctx);
        auto destination = make_retained<float>(kCount, ctx);
        (void)allocator<float>::copy_async_retained(source, destination, s.stream);
    }
    EXPECT_EQ(shutdown_runtime(std::chrono::milliseconds(30)), 1U);  // reported, not torn down
    EXPECT_EQ(service.pending_count(), 1U);
    EXPECT_THROW(
        (void)service.enqueue(copy_token(ctx), 0, false), std::runtime_error);  // admission closed

    blocker.open();
    EXPECT_EQ(shutdown_runtime(std::chrono::seconds(5)), 0U);
    EXPECT_FALSE(blocker.timed_out());
    service.reset();  // reopen admission for the tests that follow
}

// 5.6 on the real cache: managed GPU storage behind a retained copy stays allocated
// (not returned to the cache for reuse) after every user handle is dropped, until the
// copy behind a held stream completes; then the service releases it.
TEST_F(Phase4Hardware, RetainedManagedGpuCopySurvivesDroppedHandlesUntilCompletion)
{
    auto& service = retained_operation_service::instance();
    service.reset();
    test_stream             s;
    execution_context const ctx{device_enum::CUDA, 0, s.stream};
    {
        auto warm = make_retained<float>(kCount, ctx);  // pre-warm: no cudaMalloc while held
    }
    size_t const baseline = allocator<float>::memory_allocated(0);

    stream_blocker blocker(s.stream);
    ASSERT_EQ(blocker.start(), cudaSuccess);
    {
        auto source      = make_retained<float>(kCount, ctx);
        auto destination = make_retained<float>(kCount, ctx);
        (void)allocator<float>::copy_async_retained(source, destination, s.stream);
    }  // every user handle is gone; the stream is still held
    EXPECT_EQ(service.pending_count(), 1U);
    EXPECT_GT(allocator<float>::memory_allocated(0), baseline)
        << "the blocks must stay allocated while the copy is in flight";
    EXPECT_EQ(service.poll().total(), 0U);

    blocker.open();
    EXPECT_EQ(service.wait_all(std::chrono::seconds(5)), 0U);
    EXPECT_FALSE(blocker.timed_out());
    EXPECT_EQ(allocator<float>::memory_allocated(0), baseline)
        << "after completion the service must have released both blocks";
    EXPECT_EQ(service.failed_count(), 0U);
}

// ---------------------------------------------------------------------------
// Phase 6 on a real device (plan 6.2-6.4). Shim and CPU coverage: TestCpuArena.cpp
// and the PinnedRuntime shim; these need a device and a held stream.
// ---------------------------------------------------------------------------
namespace
{
class Phase6Hardware : public Phase4Hardware
{
};
}  // namespace

// 6.2: several slices may be live at once; reset() and rebind() refuse while any is;
// moving to another stream keeps the slab only if the old stream is idle now.
TEST_F(Phase6Hardware, WorkspaceFailsClosedOnLiveSlicesAndOnABusyPreviousStream)
{
    test_stream             a;
    test_stream             b;
    execution_context const ctx_a{device_enum::CUDA, 0, a.stream};
    execution_context const ctx_b{device_enum::CUDA, 0, b.stream};
    gpu_workspace           ws{1U << 16, ctx_a};

    auto* p1 = static_cast<char*>(ws.acquire(300));
    auto* p2 = static_cast<char*>(ws.acquire(300));
    ASSERT_NE(p1, nullptr);
    ASSERT_NE(p2, nullptr);
    EXPECT_GE(p2 - p1, 300) << "slices must not overlap";
    EXPECT_EQ(reinterpret_cast<uintptr_t>(p1) % 256, 0U);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(p2) % 256, 0U);
    EXPECT_THROW(ws.reset(), logging::exception);
    EXPECT_THROW(ws.rebind(ctx_b), logging::exception);
    EXPECT_EQ(ws.used(), 600U + 212U);  // the refusals changed nothing (padding to 256)

    ws.release();
    stream_blocker blocker(a.stream);
    ASSERT_EQ(blocker.start(), cudaSuccess);
    EXPECT_THROW(ws.rebind(ctx_b), std::runtime_error) << "stream a still has work queued";
    EXPECT_EQ(ws.ctx().stream, a.stream);  // unchanged on refusal
    EXPECT_NO_THROW(ws.rebind(ctx_a));     // same stream: ordered by the stream, no proof

    ASSERT_EQ(blocker.release(), cudaSuccess);  // opens the hold and drains the stream
    EXPECT_FALSE(blocker.timed_out());
    EXPECT_NO_THROW(ws.rebind(ctx_b));
    EXPECT_EQ(ws.ctx().stream, b.stream);
    EXPECT_NO_THROW(ws.reset());
    EXPECT_EQ(ws.capacity(), 0U);
}

// 6.3: a pinned endpoint adopted into a retained copy stays allocated in its pool
// after the user handle is dropped, until the held transfer completes; then it goes
// back to the pool.
TEST_F(Phase6Hardware, RetainedPinnedCopySurvivesDroppedHandlesUntilCompletion)
{
    auto& service = retained_operation_service::instance();
    service.reset();
    auto& pool = cpu::pinned_allocator_for_device(0);
    pool.empty_cache();
    test_stream             s;
    execution_context const ctx{device_enum::CUDA, 0, s.stream};
    auto                    destination = make_retained<float>(kCount, ctx);
    {
        pinned_buffer<float> warm(kCount);  // pre-warm the pool: no cudaHostAlloc while held
    }
    pool.poll();
    size_t const live_before = pool.stats().bytes_allocated;

    stream_blocker blocker(s.stream);
    ASSERT_EQ(blocker.start(), cudaSuccess);
    {
        pinned_buffer<float> host(kCount);
        for (size_t i = 0; i < kCount; ++i)
        {
            host.data()[i] = 3.0F;
        }
        auto source = std::move(host).into_retained();
        EXPECT_EQ(host.data(), nullptr);  // NOLINT(bugprone-use-after-move)
        (void)allocator<float>::copy_async_retained(source, destination, s.stream);
    }  // the only user handle to the pinned memory is gone; the transfer is held
    EXPECT_EQ(service.pending_count(), 1U);
    EXPECT_GT(pool.stats().bytes_allocated, live_before)
        << "the pinned block must stay live while the transfer is in flight";

    blocker.open();
    EXPECT_EQ(service.wait_all(std::chrono::seconds(5)), 0U);
    EXPECT_FALSE(blocker.timed_out());
    EXPECT_EQ(pool.stats().bytes_allocated, live_before)
        << "after completion the service must have returned the block to the pool";
    EXPECT_EQ(service.failed_count(), 0U);

    pinned_host<float> check(kCount);
    ASSERT_EQ(
        cudaMemcpy(
            check.data(), destination.data(), kCount * sizeof(float), cudaMemcpyDeviceToHost),
        cudaSuccess);
    EXPECT_EQ(check.data()[0], 3.0F);
    EXPECT_EQ(check.data()[kCount - 1], 3.0F);
}

// 6.4: a slot comes back only when the transfer submitted for it has completed.
TEST_F(Phase6Hardware, StagingRingReusesASlotOnlyAfterItsTransferCompletes)
{
    test_stream             s;
    execution_context const ctx{device_enum::CUDA, 0, s.stream};
    data_ptr<float>         device_buffer(kCount, ctx);
    pinned_staging_ring<float> ring(2, kCount);
    EXPECT_EQ(ring.idle_count(), 2U);

    stream_blocker blocker(s.stream);
    ASSERT_EQ(blocker.start(), cudaSuccess);
    std::vector<size_t> used;
    for (int i = 0; i < 2; ++i)
    {
        auto slot = ring.acquire(std::chrono::milliseconds(100));
        used.push_back(slot.index());
        for (size_t j = 0; j < slot.size(); ++j)
        {
            slot.data()[j] = static_cast<float>(i + 1);
        }
        ring.submit(
            slot,
            allocator<float>::copy_async(
                slot.data(), kCount, device_buffer.data(), s.stream, device_enum::CPU,
                device_enum::CUDA, 0, 0));
    }
    EXPECT_NE(used[0], used[1]);
    EXPECT_EQ(ring.in_flight_count(), 2U);
    EXPECT_FALSE(ring.try_acquire().has_value()) << "no slot may be reused while its copy is held";
    EXPECT_THROW((void)ring.acquire(std::chrono::milliseconds(20)), std::runtime_error);

    blocker.open();
    auto again = ring.acquire(std::chrono::seconds(5));
    EXPECT_FALSE(blocker.timed_out());
    EXPECT_LT(again.index(), 2U);
    ring.release(again);
    EXPECT_THROW(ring.release(again), std::invalid_argument);  // not acquired any more
    EXPECT_EQ(ring.quarantined_count(), 0U);
}

// 6.4: a failed transfer quarantines its slot for good; with every slot gone the
// wait fails at once instead of running out its timeout.
TEST_F(Phase6Hardware, StagingRingQuarantinesTheSlotOfAFailedTransfer)
{
    pinned_staging_ring<float> ring(1, 16);
    auto                       slot = ring.acquire(std::chrono::milliseconds(100));
    execution_context          ctx{device_enum::CUDA, 0, nullptr};
    copy_token                 token(ctx);
    memory::detail::copy_token_access::fail(token);  // as a driver failure would
    ring.submit(slot, token);
    EXPECT_FALSE(ring.try_acquire().has_value());
    EXPECT_EQ(ring.quarantined_count(), 1U);
    const auto started = clock_type::now();
    EXPECT_THROW((void)ring.acquire(std::chrono::seconds(5)), std::runtime_error);
    EXPECT_LT(clock_type::now() - started, std::chrono::seconds(1));
}

// 5.8 admission under pressure on a real device: with the stream held and the limit
// at 2, the third retained copy is refused without submitting anything (try mode) or
// waits until a completion frees a slot (wait mode); nothing is dropped or leaked.
TEST_F(Phase4Hardware, AdmissionUnderPressureRefusesOrWaitsAndLeaksNothing)
{
    auto& service = retained_operation_service::instance();
    service.reset();
    service.set_max_pending(2);
    test_stream             s;
    execution_context const ctx{device_enum::CUDA, 0, s.stream};
    auto                    src = make_retained<float>(kCount, ctx);
    auto                    dst = make_retained<float>(kCount, ctx);
    size_t const            baseline = allocator<float>::memory_allocated(0);

    stream_blocker blocker(s.stream);
    ASSERT_EQ(blocker.start(), cudaSuccess);
    (void)allocator<float>::copy_async_retained(src, dst, s.stream);
    (void)allocator<float>::copy_async_retained(src, dst, s.stream);
    ASSERT_EQ(service.pending_count(), 2U);

    EXPECT_THROW(
        (void)allocator<float>::copy_async_retained(src, dst, s.stream, /*wait_for_admission=*/false),
        std::runtime_error);
    EXPECT_EQ(service.pending_count(), 2U) << "a refused copy must not have been submitted";

    std::atomic<bool> admitted{false};
    std::thread       waiter(
        [&]
        {
            (void)allocator<float>::copy_async_retained(src, dst, s.stream, true);
            admitted = true;
        });
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    EXPECT_FALSE(admitted.load()) << "the waiter must block while the queue is full";
    blocker.open();
    waiter.join();  // polls for its own capacity once the stream drains
    EXPECT_TRUE(admitted.load());
    EXPECT_EQ(service.wait_all(std::chrono::seconds(5)), 0U);
    EXPECT_FALSE(blocker.timed_out());
    EXPECT_EQ(service.failed_count(), 0U);
    EXPECT_EQ(allocator<float>::memory_allocated(0), baseline);
    service.set_max_pending(retained_operation_service::default_max_pending);
}

// 5.7 on the real runtime: per-thread streams and pre-made buffers, so the only shared
// state is the retained-operation service and the token event pool. Prints ops/s per
// thread count; the decision about sharding is recorded in the plan from these numbers.
TEST_F(Phase4Hardware, ServiceContentionMeasurement)
{
    auto& service = retained_operation_service::instance();
    service.reset();
    constexpr int kIters = 3000;
    for (int threads : {1, 2, 8, 32})
    {
        std::vector<std::thread> pool;
        std::atomic<int>         ready{0};
        std::atomic<bool>        go{false};
        std::atomic<int>         errors{0};
        for (int t = 0; t < threads; ++t)
        {
            pool.emplace_back(
                [&]
                {
                    test_stream             st;
                    execution_context const ctx{device_enum::CUDA, 0, st.stream};
                    auto                    a = make_retained<float>(64, ctx);
                    auto                    b = make_retained<float>(64, ctx);
                    ready++;
                    while (!go.load()) {}
                    for (int i = 0; i < kIters; ++i)
                    {
                        try
                        {
                            auto token = allocator<float>::copy_async_retained(a, b, st.stream);
                            token.wait();
                            (void)service.poll();
                        }
                        catch (...) { errors++; }
                    }
                });
        }
        while (ready.load() != threads) {}
        auto const t0 = clock_type::now();
        go            = true;
        for (auto& th : pool) { th.join(); }
        double const sec = std::chrono::duration<double>(clock_type::now() - t0).count();
        std::printf(
            "CONTENTION threads=%d ops=%d seconds=%.3f ops_per_sec=%.0f per_thread_us=%.2f errors=%d\n",
            threads, threads * kIters, sec, threads * kIters / sec, sec * 1e6 / kIters, errors.load());
        EXPECT_EQ(0, errors.load());
        EXPECT_EQ(service.wait_all(std::chrono::seconds(10)), 0U);

        // Control: the same shape with the driver only (no token, no service).
        std::vector<std::thread> raw;
        std::atomic<int>         rready{0};
        std::atomic<bool>        rgo{false};
        for (int t = 0; t < threads; ++t)
        {
            raw.emplace_back(
                [&]
                {
                    test_stream st;
                    float*      a = nullptr;
                    float*      b = nullptr;
                    (void)cudaMalloc(&a, 64 * sizeof(float));
                    (void)cudaMalloc(&b, 64 * sizeof(float));
                    rready++;
                    while (!rgo.load()) {}
                    for (int i = 0; i < kIters; ++i)
                    {
                        (void)cudaMemcpyAsync(b, a, 64 * sizeof(float), cudaMemcpyDeviceToDevice, st.stream);
                        (void)cudaStreamSynchronize(st.stream);
                    }
                    (void)cudaFree(a);
                    (void)cudaFree(b);
                });
        }
        while (rready.load() != threads) {}
        auto const r0 = clock_type::now();
        rgo           = true;
        for (auto& th : raw) { th.join(); }
        double const rsec = std::chrono::duration<double>(clock_type::now() - r0).count();
        std::printf(
            "CONTROL    threads=%d ops=%d seconds=%.3f ops_per_sec=%.0f (raw memcpyAsync+sync)\n",
            threads, threads * kIters, rsec, threads * kIters / rsec);
    }
}

#endif  // MEMORY_HAS_CUDA || MEMORY_HAS_HIP
