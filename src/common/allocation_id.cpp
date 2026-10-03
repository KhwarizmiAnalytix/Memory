/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#include <atomic>
#include <cstdint>

#include "common/storage_identity.h"

namespace memory
{

namespace
{
// Constant-initialized, so it is usable before any dynamic initialization.
std::atomic<std::uint64_t> g_next_allocation_id{1};
}  // namespace

// The one generator per process (plan §5.1). Header-inline code in client
// binaries calls this exported function instead of owning a counter, so IDs
// minted inside the library and in client translation units never collide.
// Never wraps in practice (2^64 allocations) and is never reset; zero stays
// reserved as the invalid sentinel.
allocation_id next_allocation_id() noexcept
{
    return allocation_id(g_next_allocation_id.fetch_add(1, std::memory_order_relaxed));
}

}  // namespace memory
