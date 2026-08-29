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
