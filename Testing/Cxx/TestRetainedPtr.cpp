/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#include "MemoryTest.h"
#include "common/retained_ptr.h"
#include "common/execution_context.h"

#include <string>

using namespace memory;

// Helper: allocate raw heap memory and return via retained_ptr with a
// simple delete[] deleter.
template <typename T>
static retained_ptr<T> make_retained(size_t count)
{
    T* data = new T[count]{};
    return retained_ptr<T>::adopt(
        data,
        count,
        execution_context::cpu(),
        [](T* p, size_t /*n*/, execution_context const&) { delete[] p; });
}

MEMORYTEST(RetainedPtr, default_is_null)
{
    retained_ptr<int> rp;
    EXPECT_TRUE(rp.empty());
    EXPECT_EQ(rp.data(), nullptr);
    EXPECT_EQ(rp.size(), 0U);
    EXPECT_EQ(rp.use_count(), 0);
    END_TEST();
}

MEMORYTEST(RetainedPtr, adopt_single_owner)
{
    auto rp = make_retained<int>(4);
    EXPECT_FALSE(rp.empty());
    EXPECT_NE(rp.data(), nullptr);
    EXPECT_EQ(rp.size(), 4U);
    EXPECT_EQ(rp.use_count(), 1);
    EXPECT_TRUE(rp.identity().valid());
    END_TEST();
}

MEMORYTEST(RetainedPtr, adopt_null_yields_empty)
{
    retained_ptr<int> rp = retained_ptr<int>::adopt(
        nullptr, 4, execution_context::cpu(), nullptr);
    EXPECT_TRUE(rp.empty());
    END_TEST();
}

MEMORYTEST(RetainedPtr, copy_increments_ref_count)
{
    auto a = make_retained<int>(2);
    auto b = a;
    EXPECT_EQ(a.use_count(), 2);
    EXPECT_EQ(b.use_count(), 2);
    EXPECT_EQ(a.data(), b.data());
    END_TEST();
}

MEMORYTEST(RetainedPtr, move_transfers_ownership)
{
    auto a = make_retained<int>(3);
    int* p = a.data();
    auto b = std::move(a);
    EXPECT_TRUE(a.empty());
    EXPECT_EQ(b.data(), p);
    EXPECT_EQ(b.use_count(), 1);
    END_TEST();
}

MEMORYTEST(RetainedPtr, last_owner_calls_deleter)
{
    bool deleted = false;
    {
        int*              raw = new int[5]{};
        retained_ptr<int> rp  = retained_ptr<int>::adopt(
            raw, 5, execution_context::cpu(),
            [&deleted](int* p, size_t, execution_context const&)
            {
                delete[] p;
                deleted = true;
            });
    }
    EXPECT_TRUE(deleted);
    END_TEST();
}

MEMORYTEST(RetainedPtr, slice_shares_control_block)
{
    auto original = make_retained<int>(10);
    original.data()[3] = 99;

    auto slice = original.slice(2, 4);
    EXPECT_EQ(slice.size(), 4U);
    EXPECT_EQ(slice.data()[1], 99);  // original[3] == slice[1]
    EXPECT_EQ(original.use_count(), 2);
    EXPECT_EQ(slice.identity().alloc_id, original.identity().alloc_id);
    EXPECT_EQ(slice.base(), original.base());  // same allocation base
    END_TEST();
}

MEMORYTEST(RetainedPtr, slice_out_of_range_yields_empty)
{
    auto rp = make_retained<int>(5);
    auto s1 = rp.slice(10, 1);   // offset >= size
    auto s2 = rp.slice(3, 100);  // count clamped
    EXPECT_TRUE(s1.empty());
    EXPECT_EQ(s2.size(), 2U);    // 5 - 3 = 2
    END_TEST();
}

MEMORYTEST(RetainedPtr, deleter_called_after_last_slice_released)
{
    bool deleted = false;
    {
        int* raw = new int[8]{};
        retained_ptr<int> rp = retained_ptr<int>::adopt(
            raw, 8, execution_context::cpu(),
            [&deleted](int* p, size_t, execution_context const&)
            {
                delete[] p;
                deleted = true;
            });
        auto s = rp.slice(0, 4);
        rp     = {};          // release original; count goes to 1 (slice holds it)
        EXPECT_FALSE(deleted);
    }  // slice destroyed here; count goes to 0 → deleter fires
    EXPECT_TRUE(deleted);
    END_TEST();
}

MEMORYTEST(RetainedPtr, ctx_is_preserved)
{
    auto ctx = execution_context::cuda(3, nullptr);
    int* raw = new int[2]{};
    auto rp  = retained_ptr<int>::adopt(
        raw, 2, ctx,
        [](int* p, size_t, execution_context const&) { delete[] p; });
    EXPECT_EQ(rp.ctx().device_type, device_enum::CUDA);
    EXPECT_EQ(rp.ctx().device_index, 3);
    END_TEST();
}

MEMORYTEST(RetainedPtr, assign_copy_and_move)
{
    auto a = make_retained<int>(3);
    retained_ptr<int> b;
    b = a;
    EXPECT_EQ(b.use_count(), 2);
    retained_ptr<int> c;
    c = std::move(b);
    EXPECT_TRUE(b.empty());
    EXPECT_EQ(c.use_count(), 2);
    END_TEST();
}
