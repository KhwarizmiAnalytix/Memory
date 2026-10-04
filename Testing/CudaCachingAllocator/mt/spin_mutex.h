#pragma once

// Task 8.7 experiment only (force-included into gpu/cuda_caching_allocator.cpp by the
// MemoryGpuShimContentionPhase8Spin target): std::recursive_mutex replaced by a
// spin-then-yield recursive lock, to bound how much of the 8.1 collapse is the cost of
// parking/waking threads on a futex versus the length of the critical section itself.
// Not proposed for the library as-is: unbounded yielding is unfair, oversubscription and
// priority inversion are untested. Same measurement-only caveat about namespace std as
// lock_wait_mutex.h.

#include <atomic>
#include <mutex>
#include <thread>

namespace std
{
class bench_spin_recursive_mutex
{
public:
    void lock()
    {
        void const* const me = self();
        if (owner_.load(std::memory_order_relaxed) == me)
        {
            ++depth_;
            return;
        }
        for (unsigned spins = 0;; ++spins)
        {
            void const* expected = nullptr;
            if (owner_.load(std::memory_order_relaxed) == nullptr &&
                owner_.compare_exchange_weak(expected, me, std::memory_order_acquire, std::memory_order_relaxed))
            {
                depth_ = 1;
                return;
            }
            if (spins < 2000)
            {
#if defined(__x86_64__) || defined(__i386__)
                __builtin_ia32_pause();
#endif
            }
            else
            {
                std::this_thread::yield();
            }
        }
    }
    bool try_lock()
    {
        void const* const me = self();
        if (owner_.load(std::memory_order_relaxed) == me)
        {
            ++depth_;
            return true;
        }
        void const* expected = nullptr;
        if (owner_.compare_exchange_strong(expected, me, std::memory_order_acquire, std::memory_order_relaxed))
        {
            depth_ = 1;
            return true;
        }
        return false;
    }
    void unlock()
    {
        if (--depth_ == 0)
        {
            owner_.store(nullptr, std::memory_order_release);
        }
    }

private:
    static void const* self()
    {
        static thread_local char tag;
        return &tag;
    }
    std::atomic<void const*> owner_{nullptr};
    unsigned                 depth_{0};
};
}  // namespace std

#define recursive_mutex bench_spin_recursive_mutex
