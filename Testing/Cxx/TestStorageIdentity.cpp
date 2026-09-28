/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#include "MemoryTest.h"
#include "common/storage_identity.h"

using namespace memory;

MEMORYTEST(StorageIdentity, default_is_invalid)
{
    storage_identity id{};
    EXPECT_FALSE(id.valid());
    EXPECT_EQ(id.alloc_id, 0U);
    EXPECT_EQ(id.base, nullptr);
    EXPECT_EQ(id.capacity, 0U);
    END_TEST();
}

MEMORYTEST(StorageIdentity, next_id_is_nonzero)
{
    uint64_t const id = storage_identity::next_id();
    EXPECT_NE(id, 0U);
    END_TEST();
}

MEMORYTEST(StorageIdentity, next_id_is_unique)
{
    uint64_t const a = storage_identity::next_id();
    uint64_t const b = storage_identity::next_id();
    uint64_t const c = storage_identity::next_id();
    EXPECT_NE(a, b);
    EXPECT_NE(b, c);
    EXPECT_NE(a, c);
    END_TEST();
}

MEMORYTEST(StorageIdentity, next_id_is_monotonic)
{
    uint64_t const a = storage_identity::next_id();
    uint64_t const b = storage_identity::next_id();
    EXPECT_LT(a, b);
    END_TEST();
}

MEMORYTEST(StorageIdentity, valid_when_alloc_id_set)
{
    int dummy = 42;
    storage_identity id;
    id.alloc_id = storage_identity::next_id();
    id.base     = &dummy;
    id.capacity = sizeof(int);
    EXPECT_TRUE(id.valid());
    END_TEST();
}

MEMORYTEST(StorageIdentity, equality)
{
    int dummy = 0;
    storage_identity a, b;
    a.alloc_id = 7;
    a.base     = &dummy;
    b.alloc_id = 7;
    b.base     = &dummy;
    EXPECT_TRUE(a == b);
    b.alloc_id = 8;
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    END_TEST();
}
