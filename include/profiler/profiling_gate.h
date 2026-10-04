/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#pragma once

#include <atomic>
#include <cstdint>

namespace memory::detail
{

/**
 * Cheap front gate for `profiler::memory_profiling_active()` on allocator hot
 * paths (plan 3.6, R8).
 *
 * The profiler's query takes a process-wide mutex (its global-state manager) on
 * every call, which made the CPU allocate/free facade about 300x slower than raw
 * mimalloc at 32 threads with profiling OFF. While no session has been seen, a
 * thread asks the profiler only on its first call and then once per
 * `kStride` calls; every other call is a thread-local decrement and one relaxed
 * load. While a session is active every call asks the profiler, as before.
 *
 * Trade-off, stated so it is not mistaken for exactness: when a session starts,
 * a thread can miss up to `kStride - 1` allocation events before it notices.
 * Frees of blocks the session never saw are already ignored by the reporter.
 * Removing the trade-off needs a lock-free `GlobalStateManager::get()` in the
 * Profiler repository; this gate can then be deleted.
 */
class profiling_gate
{
public:
    static constexpr std::uint32_t kStride = 256;

    /// @p query returns whether a memory-profiling session is active.
    template <typename Query>
    static bool active(Query&& query) noexcept(noexcept(query()))
    {
        if (seen_active_.load(std::memory_order_relaxed))
        {
            bool const now = query();
            if (!now)
            {
                seen_active_.store(false, std::memory_order_relaxed);
            }
            return now;
        }
        if (countdown() != 0)
        {
            --countdown();
            return false;
        }
        countdown() = kStride - 1;
        bool const now = query();
        if (now)
        {
            seen_active_.store(true, std::memory_order_relaxed);
        }
        return now;
    }

    /// Test hook: forget what this thread and the process have seen.
    static void reset_for_testing() noexcept
    {
        seen_active_.store(false, std::memory_order_relaxed);
        countdown() = 0;
    }

private:
    static std::uint32_t& countdown() noexcept
    {
        static thread_local std::uint32_t value = 0;
        return value;
    }

    static inline std::atomic<bool> seen_active_{false};
};

}  // namespace memory::detail
