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

#include <sstream>

#include "MemoryTest.h"
#include "common/device.h"

using namespace memory;

MEMORYTEST(Device, FactoriesSetTypeAndIndex)
{
    EXPECT_EQ(device::cuda(2).type, device_enum::CUDA);
    EXPECT_EQ(device::cuda(2).index, 2);
    EXPECT_EQ(device::cpu().type, device_enum::CPU);
    EXPECT_EQ(device::hip(1).type, device_enum::HIP);
    EXPECT_EQ(device::metal(3).type, device_enum::METAL);
    END_TEST();
}

MEMORYTEST(Device, EqualityOperator)
{
    EXPECT_TRUE(device::cpu() == device::cpu());
    EXPECT_FALSE(device::cpu() == (device{device_enum::CPU, 1}));
    EXPECT_TRUE(device::cpu() != device::cuda(0));
    END_TEST();
}

MEMORYTEST(Device, IsGpuCoversAllGpuBackends)
{
    EXPECT_FALSE(device::cpu().is_gpu());
    EXPECT_TRUE(device::cuda().is_gpu());
    EXPECT_TRUE(device::hip().is_gpu());
    EXPECT_TRUE(device::metal().is_gpu());
    EXPECT_FALSE((device{device_enum::PrivateUse1, 0}).is_gpu());
    END_TEST();
}

MEMORYTEST(Device, StreamInsertionDeviceEnum)
{
    std::ostringstream oss;
    oss << device_enum::CUDA;
    EXPECT_NE(oss.str().find("device type"), std::string::npos);
    END_TEST();
}

MEMORYTEST(Device, StreamInsertionDevice)
{
    std::ostringstream oss;
    oss << device::metal(5);
    EXPECT_NE(oss.str().find("index 5"), std::string::npos);
    END_TEST();
}
