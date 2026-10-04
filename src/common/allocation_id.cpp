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
//
// Each thread takes a block of kBlock IDs from the shared counter and hands them
// out locally, so the shared cache line is touched once per kBlock allocations
// instead of on every one (a per-allocation fetch_add cost data_ptr about 10x
// its raw allocation at 32 threads, plan 3.6). IDs are unique, not globally
// ordered: a thread's unused tail is skipped when the thread exits.
allocation_id next_allocation_id() noexcept
{
    constexpr std::uint64_t kBlock = 1024;
    thread_local std::uint64_t next = 0;
    thread_local std::uint64_t end  = 0;
    if (next == end)
    {
        next = g_next_allocation_id.fetch_add(kBlock, std::memory_order_relaxed);
        end  = next + kBlock;
    }
    return allocation_id(next++);
}

}  // namespace memory
