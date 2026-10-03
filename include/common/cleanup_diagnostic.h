/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

#include "common/memory_export.h"

namespace memory
{

// Which cleanup path reported a failure.
enum class cleanup_source : std::uint8_t
{
    unknown = 0,
    data_ptr,
    retained_ptr,
    pinned_buffer,
    gpu_cache,
    count_  // number of sources; not a valid source
};

// Non-allocating diagnostic channel for destructor-path failures (plan §5.2:
// "A swallowed exception is not a successful cleanup").
//
// Failures during free (data_ptr, retained_ptr, pinned_buffer destructors and
// the GPU cache deleter) must not throw, must not allocate, and must not take
// any lock. record_failure() is a handful of relaxed atomic operations plus an
// optional handler call.
//
// Handler contract:
//   - A plain function pointer; it must stay valid until replaced or cleared
//     with set_handler(nullptr), and in practice for the life of the process
//     because a thread that loaded the old pointer may still be running it
//     after set_handler() returns.
//   - It is invoked on the failing thread, possibly concurrently, and possibly
//     while the caller holds unrelated locks. It must be noexcept, must not
//     allocate and must not call back into the allocator or cleanup_diagnostic.
//   - The allocator never calls it while holding its own lock.
//
// Counters are monotonic. reset_count() exists only in non-NDEBUG builds for
// test teardown; tests in Release compare deltas.
class MEMORY_VISIBILITY cleanup_diagnostic
{
public:
    using handler_fn = void (*)(cleanup_source source) noexcept;

    // Count one failed cleanup. Never throws; never allocates; never locks.
    static void record_failure(cleanup_source source = cleanup_source::unknown) noexcept
    {
        failures_.fetch_add(1, std::memory_order_relaxed);
        per_source_[index_of(source)].fetch_add(1, std::memory_order_relaxed);
        handler_fn const handler = handler_.load(std::memory_order_acquire);
        if (handler != nullptr)
        {
            handler(source);
        }
    }

    // Total failures across all sources.
    static std::uint64_t failure_count() noexcept
    {
        return failures_.load(std::memory_order_relaxed);
    }

    // Failures reported by one source.
    static std::uint64_t failure_count(cleanup_source source) noexcept
    {
        return per_source_[index_of(source)].load(std::memory_order_relaxed);
    }

    // Install (or clear with nullptr) the failure handler; returns the previous one.
    static handler_fn set_handler(handler_fn handler) noexcept
    {
        return handler_.exchange(handler, std::memory_order_acq_rel);
    }

#ifndef NDEBUG
    // Reset for use in test teardown.  Not safe in production.
    static void reset_count() noexcept
    {
        failures_.store(0, std::memory_order_relaxed);
        for (auto& c : per_source_)
        {
            c.store(0, std::memory_order_relaxed);
        }
    }
#endif

private:
    static constexpr std::size_t kSourceCount = static_cast<std::size_t>(cleanup_source::count_);

    static constexpr std::size_t index_of(cleanup_source source) noexcept
    {
        auto const i = static_cast<std::size_t>(source);
        return i < kSourceCount ? i : 0;
    }

    static inline std::atomic<std::uint64_t> failures_{0};
    static inline std::atomic<std::uint64_t> per_source_[kSourceCount]{};
    static inline std::atomic<handler_fn>    handler_{nullptr};
};

}  // namespace memory
