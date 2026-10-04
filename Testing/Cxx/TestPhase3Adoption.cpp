/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * Phase 3: Adoption and Retained Operations Tests
 */

#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <stdexcept>

#include "allocator.h"
#include "common/copy_token.h"
#include "common/device.h"
#include "common/retained_operation_service.h"
#include "common/retained_ptr.h"

namespace memory
{

// --- Adoption Factory Tests ---

class Phase3Adoption : public ::testing::Test
{
protected:
    using T = float;
};

TEST_F(Phase3Adoption, allocate_adopted_basic)
{
    T*     raw_ptr = new T[100];
    size_t count   = 100;

    auto adopted = allocator<T>::allocate_adopted(
        raw_ptr, count, execution_context::cpu(),
        [](T* p, size_t, execution_context const&) { delete[] p; });

    EXPECT_EQ(adopted.data(), raw_ptr);
    EXPECT_EQ(adopted.size(), count);
    EXPECT_EQ(adopted.use_count(), 1);
}

TEST_F(Phase3Adoption, allocate_adopted_null_pointer_throws)
{
    auto no_op = [](T*, size_t, execution_context const&) {};
    EXPECT_THROW(
        allocator<T>::allocate_adopted(nullptr, 100, execution_context::cpu(), no_op),
        std::invalid_argument);
}

TEST_F(Phase3Adoption, allocate_adopted_zero_count_throws)
{
    T*   raw_ptr = new T[100];
    auto no_op   = [](T*, size_t, execution_context const&) {};
    EXPECT_THROW(
        allocator<T>::allocate_adopted(raw_ptr, 0, execution_context::cpu(), no_op),
        std::invalid_argument);
    delete[] raw_ptr;
}

TEST_F(Phase3Adoption, allocate_adopted_null_deleter_throws)
{
    // P1.8: explicit deleter is required; null deleter must throw.
    T* raw_ptr = new T[10];
    EXPECT_THROW(
        allocator<T>::allocate_adopted(
            raw_ptr, 10, execution_context::cpu(),
            std::function<void(T*, size_t, execution_context const&)>{}),
        std::invalid_argument);
    delete[] raw_ptr;
}

TEST_F(Phase3Adoption, allocate_adopted_with_custom_deleter)
{
    static int deleter_called = 0;
    T*         raw_ptr        = new T[50];

    {
        auto adopted = allocator<T>::allocate_adopted(
            raw_ptr, 50, execution_context::cpu(),
            [](T* p, size_t, execution_context const&) {
                ++deleter_called;
                delete[] p;
            });
    }

    EXPECT_EQ(deleter_called, 1);
}

TEST_F(Phase3Adoption, allocate_adopted_with_noop_deleter)
{
    // Caller takes ownership of raw_ptr; no-op deleter signals intent explicitly.
    T* raw_ptr = new T[30];
    {
        auto adopted = allocator<T>::allocate_adopted(
            raw_ptr, 30, execution_context::cpu(),
            [](T*, size_t, execution_context const&) { /* caller manages lifetime */ });
        EXPECT_EQ(adopted.size(), 30);
    }
    delete[] raw_ptr;
}

TEST_F(Phase3Adoption, allocate_adopted_copy_increments_refcount)
{
    T*     raw_ptr = new T[20];
    size_t count   = 20;

    auto adopted1 = allocator<T>::allocate_adopted(
        raw_ptr, count, execution_context::cpu(),
        [](T* p, size_t, execution_context const&) { delete[] p; });

    EXPECT_EQ(adopted1.use_count(), 1);

    auto adopted2 = adopted1;

    EXPECT_EQ(adopted1.use_count(), 2);
    EXPECT_EQ(adopted2.use_count(), 2);
}

TEST_F(Phase3Adoption, allocate_adopted_move_transfers_ownership)
{
    T*     raw_ptr = new T[15];
    size_t count   = 15;

    auto adopted1 = allocator<T>::allocate_adopted(
        raw_ptr, count, execution_context::cpu(),
        [](T* p, size_t, execution_context const&) { delete[] p; });

    auto adopted2 = std::move(adopted1);

    EXPECT_TRUE(adopted1.empty());
    EXPECT_EQ(adopted2.data(), raw_ptr);
    EXPECT_EQ(adopted2.size(), count);
    EXPECT_EQ(adopted2.use_count(), 1);
}

TEST_F(Phase3Adoption, allocate_adopted_allocation_id_unique)
{
    T* raw_ptr1 = new T[10];
    T* raw_ptr2 = new T[10];

    auto adopted1 = allocator<T>::allocate_adopted(
        raw_ptr1, 10, execution_context::cpu(),
        [](T* p, size_t, execution_context const&) { delete[] p; });

    auto adopted2 = allocator<T>::allocate_adopted(
        raw_ptr2, 10, execution_context::cpu(),
        [](T* p, size_t, execution_context const&) { delete[] p; });

    EXPECT_NE(adopted1.identity().alloc_id, adopted2.identity().alloc_id);
}

TEST_F(Phase3Adoption, allocate_adopted_preserves_context)
{
    T* raw_ptr = new T[8];

    execution_context ctx = execution_context::cpu();
    auto               adopted =
        allocator<T>::allocate_adopted(raw_ptr, 8, ctx,
                                       [](T* p, size_t, execution_context const&) { delete[] p; });

    EXPECT_EQ(adopted.ctx().device_type(), ctx.device_type());
    EXPECT_EQ(adopted.ctx().device_index(), ctx.device_index());
}

// --- Retained Transfer Tests ---

class Phase3RetainedTransfer : public ::testing::Test
{
protected:
    using T = float;
};

TEST_F(Phase3RetainedTransfer, copy_async_retained_basic)
{
    auto from = allocator<T>::allocate(10, execution_context::cpu());
    auto to   = allocator<T>::allocate(10, execution_context::cpu());

    auto cpu_free = [](T* p, size_t, execution_context const&) {
        allocator<T>::free(p, device_enum::CPU);
    };
    auto from_retained = allocator<T>::allocate_adopted(
        from, 10, execution_context::cpu(), cpu_free);
    auto to_retained = allocator<T>::allocate_adopted(
        to, 10, execution_context::cpu(), cpu_free);

    std::fill(from_retained.data(), from_retained.end(), 42.0f);

    copy_token token = allocator<T>::copy_async_retained(from_retained, to_retained);

    EXPECT_TRUE(token.ready());

    for (size_t i = 0; i < 10; ++i)
    {
        EXPECT_EQ(to_retained.data()[i], 42.0f);
    }
    // retained_ptr deleter handles cleanup; no explicit free needed
}

TEST_F(Phase3RetainedTransfer, copy_async_retained_null_source_throws)
{
    auto empty_ptr = retained_ptr<T>();
    auto to        = allocator<T>::allocate_adopted(
        new T[5], 5, execution_context::cpu(),
        [](T* p, size_t, execution_context const&) { delete[] p; });

    EXPECT_THROW(allocator<T>::copy_async_retained(empty_ptr, to),
                 std::invalid_argument);
}

TEST_F(Phase3RetainedTransfer, copy_async_retained_null_destination_throws)
{
    auto from      = allocator<T>::allocate_adopted(
        new T[5], 5, execution_context::cpu(),
        [](T* p, size_t, execution_context const&) { delete[] p; });
    auto empty_ptr = retained_ptr<T>();

    EXPECT_THROW(allocator<T>::copy_async_retained(from, empty_ptr),
                 std::invalid_argument);
}

TEST_F(Phase3RetainedTransfer, copy_async_retained_large_buffer)
{
    size_t count = 10000;

    auto from = allocator<T>::allocate(count, execution_context::cpu());
    auto to   = allocator<T>::allocate(count, execution_context::cpu());

    auto cpu_free = [](T* p, size_t, execution_context const&) {
        allocator<T>::free(p, device_enum::CPU);
    };
    auto from_retained = allocator<T>::allocate_adopted(
        from, count, execution_context::cpu(), cpu_free);
    auto to_retained = allocator<T>::allocate_adopted(
        to, count, execution_context::cpu(), cpu_free);

    for (size_t i = 0; i < count; ++i)
    {
        from_retained.data()[i] = static_cast<T>(i);
    }

    copy_token token = allocator<T>::copy_async_retained(from_retained, to_retained);

    EXPECT_TRUE(token.ready());

    for (size_t i = 0; i < count; ++i)
    {
        EXPECT_EQ(to_retained.data()[i], static_cast<T>(i));
    }
    // retained_ptr deleter handles cleanup; no explicit free needed
}

// --- Retained Operation Service Tests ---

class Phase3RetainedService : public ::testing::Test
{
protected:
    void SetUp() override
    {
        auto& service = retained_operation_service::instance();
        service.reset();
    }

    void TearDown() override
    {
        auto& service = retained_operation_service::instance();
        service.reset();
    }

    using T = float;
};

TEST_F(Phase3RetainedService, service_singleton)
{
    auto& service1 = retained_operation_service::instance();
    auto& service2 = retained_operation_service::instance();

    EXPECT_EQ(&service1, &service2);
}

TEST_F(Phase3RetainedService, service_enqueue_already_complete)
{
    auto& service = retained_operation_service::instance();

    copy_token token;
    service.enqueue(std::move(token));

    EXPECT_EQ(service.pending_count(), 0);
}

TEST_F(Phase3RetainedService, service_poll_empty)
{
    auto& service = retained_operation_service::instance();

    size_t completed = service.poll().total();

    EXPECT_EQ(completed, 0);
    EXPECT_EQ(service.pending_count(), 0);
}

TEST_F(Phase3RetainedService, service_pending_count)
{
    auto& service = retained_operation_service::instance();

    EXPECT_EQ(service.pending_count(), 0);
}

TEST_F(Phase3RetainedService, service_max_pending_is_finite_by_default_and_unlimited_when_asked)
{
    auto& service = retained_operation_service::instance();

    EXPECT_EQ(service.max_pending(), retained_operation_service::default_max_pending);

    service.set_max_pending(10);
    EXPECT_EQ(service.max_pending(), 10);

    service.set_max_pending(0);
    EXPECT_EQ(service.max_pending(), 0);
}

TEST_F(Phase3RetainedService, service_failed_count_empty)
{
    auto& service = retained_operation_service::instance();

    EXPECT_EQ(service.failed_count(), 0);
}

TEST_F(Phase3RetainedService, service_drain_empty)
{
    auto& service = retained_operation_service::instance();

    size_t discarded = service.drain(std::chrono::milliseconds(100));

    EXPECT_EQ(discarded, 0);
}

TEST_F(Phase3RetainedService, service_integration_with_allocate_adopted)
{
    auto& service = retained_operation_service::instance();

    T*   raw_ptr = new T[20];
    auto adopted = allocator<T>::allocate_adopted(
        raw_ptr, 20, execution_context::cpu(),
        [](T* p, size_t, execution_context const&) { delete[] p; });

    std::fill(adopted.data(), adopted.end(), 99.0f);

    EXPECT_EQ(service.pending_count(), 0);
}

TEST_F(Phase3RetainedService, service_reset_clears_state)
{
    auto& service = retained_operation_service::instance();

    service.set_max_pending(5);
    service.reset();

    EXPECT_EQ(service.pending_count(), 0);
    EXPECT_EQ(service.max_pending(), retained_operation_service::default_max_pending);
}

// --- Integration Tests ---

class Phase3Integration : public ::testing::Test
{
protected:
    using T = float;
};

TEST_F(Phase3Integration, adopt_copy_and_verify)
{
    const T      expected = 3.14f;
    T*           raw_src  = new T[5];
    T*           raw_dst  = new T[5];
    std::fill(raw_src, raw_src + 5, expected);

    auto src_adopted = allocator<T>::allocate_adopted(
        raw_src, 5, execution_context::cpu(),
        [](T* p, size_t, execution_context const&) { delete[] p; });

    auto dst_adopted = allocator<T>::allocate_adopted(
        raw_dst, 5, execution_context::cpu(),
        [](T* p, size_t, execution_context const&) { delete[] p; });

    copy_token token = allocator<T>::copy_async_retained(src_adopted, dst_adopted);

    token.wait();

    for (size_t i = 0; i < 5; ++i)
    {
        EXPECT_EQ(dst_adopted.data()[i], expected);
    }
}

TEST_F(Phase3Integration, adopt_multiple_copies_share_ownership)
{
    T* raw_ptr = new T[10];
    std::fill(raw_ptr, raw_ptr + 10, 7.0f);

    auto adopted1 = allocator<T>::allocate_adopted(
        raw_ptr, 10, execution_context::cpu(),
        [](T* p, size_t, execution_context const&) { delete[] p; });

    EXPECT_EQ(adopted1.use_count(), 1);

    auto adopted2 = adopted1;
    EXPECT_EQ(adopted1.use_count(), 2);
    EXPECT_EQ(adopted2.use_count(), 2);

    adopted1 = {};
    EXPECT_EQ(adopted2.use_count(), 1);
}

}  // namespace memory
