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
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "allocator.h"
#include "common/copy_token.h"
#include "common/data_ptr.h"
#include "gpu/device_guard.h"
#include "gpu/gpu_runtime.h"

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

#endif  // MEMORY_HAS_CUDA || MEMORY_HAS_HIP
