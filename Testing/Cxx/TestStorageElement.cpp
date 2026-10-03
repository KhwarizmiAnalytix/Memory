/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#include <cstddef>
#include <string>
#include <vector>

#include "MemoryTest.h"
#include "common/data_ptr.h"
#include "common/retained_ptr.h"
#include "common/storage_element.h"

using namespace memory;

namespace
{
struct pod
{
    int   a;
    float b;
};
struct alignas(256) over_aligned
{
    char c;
};
struct has_dtor
{
    ~has_dtor() {}
};
}  // namespace

// Task 2.9: the documented constraint, checked at compile time.
static_assert(is_storage_element_v<float, 64>);
static_assert(is_storage_element_v<std::byte, 64>);
static_assert(is_storage_element_v<pod, 64>);
static_assert(!is_storage_element_v<std::string, 64>, "non-trivially copyable");
static_assert(!is_storage_element_v<std::vector<int>, 64>, "non-trivially copyable");
static_assert(!is_storage_element_v<has_dtor, 64>, "non-trivially destructible");
static_assert(!is_storage_element_v<float const, 64>, "const element");
static_assert(!is_storage_element_v<float volatile, 64>, "volatile element");
static_assert(!is_storage_element_v<over_aligned, 64>, "alignof above the allocation alignment");
static_assert(is_storage_element_v<over_aligned, 256>);

MEMORYTEST(StorageElement, OwnersAcceptTrivialTypes)
{
    data_ptr<pod> d(4, device_enum::CPU);
    EXPECT_EQ(d.size(), 4U);
    retained_ptr<pod> r(std::move(d));
    EXPECT_EQ(r.size(), 4U);
    END_TEST();
}
