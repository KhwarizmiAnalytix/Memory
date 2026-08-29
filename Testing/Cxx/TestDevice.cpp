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

MEMORYTEST(DeviceOption, ConstructWithInt16Index)
{
    const device_option opt(device_enum::CUDA, device_option::int_t{2});
    EXPECT_EQ(opt.type(), device_enum::CUDA);
    EXPECT_EQ(opt.index(), 2);
    END_TEST();
}

MEMORYTEST(DeviceOption, ConstructWithIntIndex)
{
    const device_option opt(device_enum::CPU, 3);
    EXPECT_EQ(opt.type(), device_enum::CPU);
    EXPECT_EQ(opt.index(), 3);
    END_TEST();
}

MEMORYTEST(DeviceOption, EqualityOperator)
{
    const device_option a(device_enum::CPU, 0);
    const device_option b(device_enum::CPU, 0);
    const device_option c(device_enum::CPU, 1);
    const device_option d(device_enum::CUDA, 0);

    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
    EXPECT_FALSE(a == d);
    END_TEST();
}

MEMORYTEST(DeviceOption, AllDeviceEnumValues)
{
    const device_option cpu(device_enum::CPU, 0);
    const device_option cuda(device_enum::CUDA, 0);
    const device_option hip(device_enum::HIP, 0);
    const device_option private_use(device_enum::PrivateUse1, 0);
    const device_option metal(device_enum::METAL, 0);

    EXPECT_EQ(cpu.type(), device_enum::CPU);
    EXPECT_EQ(cuda.type(), device_enum::CUDA);
    EXPECT_EQ(hip.type(), device_enum::HIP);
    EXPECT_EQ(private_use.type(), device_enum::PrivateUse1);
    EXPECT_EQ(metal.type(), device_enum::METAL);
    END_TEST();
}

MEMORYTEST(DeviceOption, StreamInsertionDeviceEnum)
{
    std::ostringstream oss;
    oss << device_enum::CUDA;
    EXPECT_NE(oss.str().find("device_option type"), std::string::npos);
    END_TEST();
}

MEMORYTEST(DeviceOption, StreamInsertionDeviceOption)
{
    const device_option opt(device_enum::METAL, 5);
    std::ostringstream  oss;
    oss << opt;
    EXPECT_NE(oss.str().find("index 5"), std::string::npos);
    END_TEST();
}
