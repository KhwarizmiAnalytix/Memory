/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * This file is part of XSigma and is licensed under a dual-license model:
 *
 *   - Open-source License (GPLv3):
 *       Free for personal, academic, and research use under the terms of
 *       the GNU General Public License v3.0 or later.
 *
 *   - Commercial License:
 *       A commercial license is required for proprietary, closed-source,
 *       or SaaS usage. Contact us to obtain a commercial agreement.
 *
 * Contact: licensing@xsigma.co.uk
 * Website: https://www.xsigma.co.uk
 */

#include "MemoryTest.h"
#include "common/data_ptr.h"
#include "common/data_view.h"
#include "common/device.h"
#include "common/execution_context.h"
#include "common/retained_ptr.h"

using namespace memory;

MEMORYTEST(DataView, views_data_ptr)
{
    data_ptr<int> owned(3, device_enum::CPU);
    owned.data()[0] = 11;
    data_view<int> view(owned);
    EXPECT_EQ(owned.data(), view.data());
    EXPECT_EQ(3U, view.size());
    EXPECT_EQ(11, view.data()[0]);
    EXPECT_EQ(owned.device(), view.device());
    EXPECT_EQ(owned.stream(), view.stream());
    END_TEST();
}

MEMORYTEST(DataView, copy_aliases_owner_buffer)
{
    data_ptr<int> owned(2, device_enum::CPU);
    owned.data()[0] = 1;
    data_view<int> first(owned);
    data_view<int> second(first);
    EXPECT_EQ(first.data(), second.data());
    second.data()[0] = 9;
    EXPECT_EQ(9, owned.data()[0]);
    END_TEST();
}

MEMORYTEST(DataView, subview_is_slice_of_data_ptr)
{
    data_ptr<int> owned(4, device_enum::CPU);
    owned.data()[0] = 1;
    owned.data()[1] = 2;
    owned.data()[2] = 3;
    owned.data()[3] = 4;

    data_view<int> slice(owned, 1, 2);
    EXPECT_EQ(owned.data() + 1, slice.data());
    EXPECT_EQ(2U, slice.size());
    EXPECT_EQ(2, slice.data()[0]);
    EXPECT_EQ(3, slice.data()[1]);

    data_view<int> via_view = owned.view(1, 2);
    EXPECT_EQ(slice.data(), via_view.data());
    EXPECT_EQ(slice.size(), via_view.size());
    END_TEST();
}

MEMORYTEST(DataView, destructor_does_not_free)
{
    data_ptr<int> owned(2, device_enum::CPU);
    owned.data()[0] = 3;
    int* const kept = owned.data();
    {
        data_view<int> view = owned.view();
        EXPECT_EQ(kept, view.data());
    }
    EXPECT_EQ(3, owned.data()[0]);
    END_TEST();
}

MEMORYTEST(DataView, borrow_wraps_foreign_buffer)
{
    int            raw[4] = {4, 5, 6, 7};
    data_view<int> view   = data_view<int>::borrow(raw, 4, device_enum::CPU);
    EXPECT_EQ(raw, view.data());
    EXPECT_EQ(4U, view.size());
    EXPECT_EQ(4, view.data()[0]);
    END_TEST();
}

// Regression for the finding that a sliced/windowed view forwarded its own
// (offset) data() to record_stream instead of the allocation's base pointer,
// which the CUDA/HIP caching allocator tracks live blocks by. base() must
// keep naming the original data_ptr allocation through any depth of slicing.
MEMORYTEST(DataView, subview_base_is_original_allocation)
{
    data_ptr<int> owned(8, device_enum::CPU);

    data_view<int> full(owned);
    EXPECT_EQ(owned.data(), full.base());

    data_view<int> slice = owned.view(2, 4);
    EXPECT_EQ(owned.data() + 2, slice.data());
    EXPECT_EQ(owned.data(), slice.base());

    data_view<int> nested = slice.subview(1, 2);
    EXPECT_EQ(owned.data() + 3, nested.data());
    EXPECT_EQ(owned.data(), nested.base());
    END_TEST();
}

MEMORYTEST(DataView, borrow_base_is_the_borrowed_pointer)
{
    int            raw[4] = {4, 5, 6, 7};
    data_view<int> view   = data_view<int>::borrow(raw, 4, device_enum::CPU);
    EXPECT_EQ(raw, view.base());

    data_view<int> slice = view.subview(1, 2);
    EXPECT_EQ(raw + 1, slice.data());
    EXPECT_EQ(raw, slice.base());
    END_TEST();
}

MEMORYTEST(DataView, default_constructed_has_null_base)
{
    data_view<int> view;
    EXPECT_EQ(nullptr, view.base());
    END_TEST();
}

// Task 2.5: a view carries storage_ref {base, id, device}; slices keep all three.
MEMORYTEST(DataView, slice_of_slice_keeps_base_id_and_device)
{
    data_ptr<int> owned(8, device_enum::CPU);
    EXPECT_TRUE(owned.id().valid());

    data_view<int> nested = owned.view(2, 4).subview(1, 2);
    EXPECT_EQ(owned.data(), nested.storage().base);
    EXPECT_EQ(owned.id(), nested.id());
    EXPECT_EQ(device::cpu(), nested.storage().dev);
    END_TEST();
}

MEMORYTEST(DataView, borrow_has_invalid_id)
{
    int            raw[4] = {0, 1, 2, 3};
    data_view<int> view   = data_view<int>::borrow(raw, 4, device_enum::CPU);
    EXPECT_FALSE(view.id().valid());
    EXPECT_FALSE(view.subview(1, 2).id().valid());
    END_TEST();
}

MEMORYTEST(DataView, default_constructed_has_invalid_id)
{
    data_view<int> view;
    EXPECT_FALSE(view.id().valid());
    END_TEST();
}

MEMORYTEST(DataView, data_ptr_view_is_allocated_provenance)
{
    data_ptr<int>  owned(4, device_enum::CPU);
    data_view<int> view(owned);
    EXPECT_EQ(view.provenance(), storage_provenance::allocated);
    END_TEST();
}

MEMORYTEST(DataView, borrow_is_adopted_provenance)
{
    int            raw[4] = {0, 1, 2, 3};
    data_view<int> view   = data_view<int>::borrow(raw, 4, device_enum::CPU);
    EXPECT_EQ(view.provenance(), storage_provenance::adopted);
    END_TEST();
}

MEMORYTEST(DataView, default_constructed_is_adopted_provenance)
{
    data_view<int> view;
    EXPECT_EQ(view.provenance(), storage_provenance::adopted);
    END_TEST();
}

// Vectorization design plan §6: a retained_ref leaf's data_view must carry the
// retained_ptr's own allocation identity and provenance, not borrow()'s
// always-invalid/adopted defaults, so the overlap classifier can take the
// `allocated` shortcut for materialized product/temporary leaves too.
MEMORYTEST(DataView, views_retained_ptr_allocated)
{
    auto rp      = memory::make_retained<int>(4, execution_context::cpu());
    rp.data()[0] = 7;
    data_view<int> view(rp);
    EXPECT_EQ(rp.data(), view.data());
    EXPECT_EQ(4U, view.size());
    EXPECT_EQ(7, view.data()[0]);
    EXPECT_EQ(rp.base(), view.base());
    EXPECT_EQ(rp.identity().alloc_id, view.id().value);
    EXPECT_EQ(view.provenance(), storage_provenance::allocated);
    END_TEST();
}

MEMORYTEST(DataView, views_retained_ptr_adopted)
{
    int* raw = new int[4]{};
    auto rp  = retained_ptr<int>::adopt(
        raw,
        4,
        execution_context::cpu(),
        [](int* p, size_t, execution_context const&) { delete[] p; });
    data_view<int> view(rp);
    EXPECT_EQ(rp.data(), view.data());
    EXPECT_EQ(view.provenance(), storage_provenance::adopted);
    END_TEST();
}

MEMORYTEST(DataView, views_retained_ptr_slice_keeps_allocation_base_and_id)
{
    auto           rp    = memory::make_retained<int>(8, execution_context::cpu());
    auto           slice = rp.slice(2, 4);
    data_view<int> view(slice);
    EXPECT_EQ(slice.data(), view.data());  // the slice's own window
    EXPECT_EQ(rp.base(), view.base());     // the allocation's base, not the slice's
    EXPECT_EQ(rp.identity().alloc_id, view.id().value);
    EXPECT_EQ(view.provenance(), storage_provenance::allocated);
    END_TEST();
}

MEMORYTEST(DataView, views_empty_retained_ptr)
{
    retained_ptr<int> rp;
    data_view<int>    view(rp);
    EXPECT_EQ(view.data(), nullptr);
    EXPECT_EQ(view.size(), 0U);
    EXPECT_FALSE(view.id().valid());
    END_TEST();
}
