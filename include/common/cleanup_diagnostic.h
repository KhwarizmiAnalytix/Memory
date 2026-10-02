/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#pragma once

#include <atomic>
#include <cstdint>

#include "common/memory_export.h"

namespace memory
{

// Non-allocating diagnostic channel for destructor-path failures.
//
// Failures during free (data_ptr, retained_ptr, pinned_buffer destructors)
// must not throw, must not allocate, and must not hold any allocator lock.
// They increment this counter so the failure is observable without escaping
// the destructor or crashing the process.
//
// Read via cleanup_diagnostic::failure_count() after a test that injects a
// free failure; reset via cleanup_diagnostic::reset_count() in test teardown.
// Neither operation is safe to call concurrently with live destructors.
class MEMORY_VISIBILITY cleanup_diagnostic
{
public:
    // Increment the failure counter.  Never throws; never allocates.
    static void record_failure() noexcept
    {
        failures_.fetch_add(1, std::memory_order_relaxed);
    }

    // Return the current failure count.
    static std::uint64_t failure_count() noexcept
    {
        return failures_.load(std::memory_order_relaxed);
    }

#ifndef NDEBUG
    // Reset for use in test teardown.  Not safe in production.
    static void reset_count() noexcept
    {
        failures_.store(0, std::memory_order_relaxed);
    }
#endif

private:
    static inline std::atomic<std::uint64_t> failures_{0};
};

}  // namespace memory
