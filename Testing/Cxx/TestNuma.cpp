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

// Exercises common/numa.{h,cpp}. On builds without MEMORY_HAS_NUMA (the
// default outside Linux+libnuma configs), every function below degrades to a
// documented no-op / sentinel value (false, -1) rather than being unreachable
// -- that fallback behavior is what these tests pin down. When
// MEMORY_HAS_NUMA is on, the same calls exercise the real libnuma path.

#include <cstddef>
#include <stdexcept>

#include "MemoryTest.h"
#include "common/numa.h"
#include "helper/memory_allocator.h"

using namespace memory;

MEMORYTEST(Numa, IsNUMAEnabledDoesNotThrow)
{
    EXPECT_NO_THROW({ (void)IsNUMAEnabled(); });
    END_TEST();
}

MEMORYTEST(Numa, GetNumNUMANodes)
{
    const int nodes = GetNumNUMANodes();
    if (IsNUMAEnabled())
    {
        EXPECT_GT(nodes, 0);
    }
    else
    {
        EXPECT_EQ(nodes, -1);
    }
    END_TEST();
}

MEMORYTEST(Numa, GetCurrentNUMANode)
{
    const int node = GetCurrentNUMANode();
    if (!IsNUMAEnabled())
    {
        EXPECT_EQ(node, -1);
    }
    END_TEST();
}

MEMORYTEST(Numa, GetNUMANodeForAllocatedPointer)
{
    void* ptr = cpu::memory_allocator::allocate(4096, 64);
    ASSERT_NE(ptr, nullptr);

    const int node = GetNUMANode(ptr);
    if (!IsNUMAEnabled())
    {
        EXPECT_EQ(node, -1);
    }

    cpu::memory_allocator::free(ptr);
    END_TEST();
}

// Without MEMORY_HAS_NUMA, GetNUMANode() returns -1 before ever validating
// ptr (the libnuma path's null check is compiled out entirely), so nullptr
// must not throw there. With MEMORY_HAS_NUMA on and NUMA actually available
// at runtime, the real libnuma path does validate ptr and must throw.
MEMORYTEST(Numa, GetNUMANodeNullptr)
{
#if MEMORY_HAS_NUMA
    if (IsNUMAEnabled())
    {
        EXPECT_THROW(GetNUMANode(nullptr), std::runtime_error);
    }
    else
    {
        EXPECT_EQ(GetNUMANode(nullptr), -1);
    }
#else
    EXPECT_EQ(GetNUMANode(nullptr), -1);
#endif
    END_TEST();
}

MEMORYTEST(Numa, NUMABindNegativeNodeIsNoOp)
{
    // A negative node id is documented as a no-op regardless of NUMA
    // availability -- must not throw or crash.
    EXPECT_NO_THROW({ NUMABind(-1); });
    END_TEST();
}

MEMORYTEST(Numa, NUMABindNodeZero)
{
    EXPECT_NO_THROW({ NUMABind(0); });
    END_TEST();
}

MEMORYTEST(Numa, NUMAMoveNegativeNodeIsNoOp)
{
    void* ptr = cpu::memory_allocator::allocate(4096, 64);
    ASSERT_NE(ptr, nullptr);

    EXPECT_NO_THROW({ NUMAMove(ptr, 4096, -1); });

    cpu::memory_allocator::free(ptr);
    END_TEST();
}

MEMORYTEST(Numa, NUMAMoveNodeZero)
{
    void* ptr = cpu::memory_allocator::allocate(4096, 64);
    ASSERT_NE(ptr, nullptr);

    EXPECT_NO_THROW({ NUMAMove(ptr, 4096, 0); });

    cpu::memory_allocator::free(ptr);
    END_TEST();
}
