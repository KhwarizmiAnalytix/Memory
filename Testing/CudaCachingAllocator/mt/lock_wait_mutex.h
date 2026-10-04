#pragma once

// Instrumented build only (force-included with -include into
// gpu/cuda_caching_allocator.cpp by the MemoryGpuShimContentionPhase8Lockwait target).
// Replaces std::recursive_mutex with a drop-in that records, per thread, how many
// acquisitions there were, how many found the mutex held, and how long those waited.
// Counters are thread_local plain integers so the instrumentation does not add a
// shared cache line of its own; each worker reads its own counters after the run.
// Declaring a type in namespace std is formally undefined behaviour; this is a
// measurement build and the library sources are not changed.

#include <chrono>
#include <cstdint>
#include <mutex>

namespace std
{
struct bench_lock_stats
{
    std::uint64_t acquisitions{0};
    std::uint64_t contended{0};
    std::uint64_t wait_ns{0};
};
inline thread_local bench_lock_stats g_bench_lock_stats;

class bench_timed_recursive_mutex
{
public:
    void lock()
    {
        auto& s = g_bench_lock_stats;
        ++s.acquisitions;
        if (m_.try_lock())
        {
            return;
        }
        ++s.contended;
        auto const t0 = std::chrono::steady_clock::now();
        m_.lock();
        s.wait_ns += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t0)
                .count());
    }
    bool try_lock() { return m_.try_lock(); }
    void unlock() { m_.unlock(); }

private:
    std::recursive_mutex m_;
};
}  // namespace std

#define recursive_mutex bench_timed_recursive_mutex
