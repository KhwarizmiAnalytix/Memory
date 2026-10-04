#pragma once

// Instrumented build only (force-included with -include into
// gpu/cuda_caching_allocator.cpp by the MemoryGpuShimContentionPhase8Lockwait target).
// Replaces std::recursive_mutex with a drop-in that records, per thread, how many
// acquisitions there were, how many found the mutex held, how long those waited, and
// (outermost acquisitions only) how long the lock was held, in TSC cycles.
// Counters are thread_local plain integers so the instrumentation does not add a
// shared cache line of its own; each worker reads its own counters after the run.
// Hold time includes the two TSC reads (the bench measures that floor and reports it).
// Declaring a type in namespace std is formally undefined behaviour; this is a
// measurement build and the library sources are not changed.

#include <chrono>
#include <cstdint>
#include <mutex>

#if defined(__x86_64__) || defined(__i386__)
#    include <x86intrin.h>
#    define BENCH_TSC() __rdtsc()
#else
#    define BENCH_TSC() 0ULL
#endif

namespace std
{
struct bench_lock_stats
{
    std::uint64_t acquisitions{0};
    std::uint64_t contended{0};
    std::uint64_t wait_ns{0};
    std::uint64_t outer_acquisitions{0};
    std::uint64_t hold_cycles{0};
};
inline thread_local bench_lock_stats g_bench_lock_stats;

class bench_timed_recursive_mutex
{
public:
    void lock()
    {
        auto& s = g_bench_lock_stats;
        ++s.acquisitions;
        if (!m_.try_lock())
        {
            ++s.contended;
            auto const t0 = std::chrono::steady_clock::now();
            m_.lock();
            s.wait_ns += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - t0)
                    .count());
        }
        if (depth_++ == 0)
        {
            t_acquired_ = BENCH_TSC();
        }
    }
    bool try_lock()
    {
        if (!m_.try_lock())
        {
            return false;
        }
        if (depth_++ == 0)
        {
            t_acquired_ = BENCH_TSC();
        }
        return true;
    }
    void unlock()
    {
        if (--depth_ == 0)
        {
            auto& s = g_bench_lock_stats;
            s.hold_cycles += BENCH_TSC() - t_acquired_;
            ++s.outer_acquisitions;
        }
        m_.unlock();
    }

private:
    std::recursive_mutex m_;
    unsigned             depth_{0};  // touched only by the owning thread
    std::uint64_t        t_acquired_{0};
};
}  // namespace std

#define recursive_mutex bench_timed_recursive_mutex
