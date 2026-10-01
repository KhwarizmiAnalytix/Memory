/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>

#include "common/copy_token.h"
#include "common/memory_export.h"

namespace memory
{

/**
 * @brief Manages lifetime of pending async operations with bounded queue.
 *
 * Retained async operations are registered before submission. The service
 * polls operation-specific completion markers and keeps failed operations
 * quarantined because their storage is not known to be safe to reuse.
 *
 * Usage:
 * ```cpp
 * auto token = allocator<T>::copy_async_retained(from, to);
 * token.wait();  // Or let the service retain it after the token is discarded
 * ```
 *
 * Benefits:
 * - Prevents unbounded growth of pending operations
 * - Token destruction does not crash or leak
 * - Blocking admission polls for its own capacity; no separate polling thread is required
 * - Diagnostic access to pending operation count
 *
 * Thread-safety: instance() is thread-safe; enqueue/poll use internal mutex.
 *
 * shutdown() stops admission and drains; unresolved operations remain retained
 * by the process-lifetime service after a timeout.
 */
class MEMORY_API retained_operation_service
{
public:
    // Get singleton instance (thread-safe).
    static retained_operation_service& instance() noexcept;

    // Enqueue a token for automatic cleanup when complete.
    // If max_pending is reached and blocking=true, waits for space.
    // If max_pending is reached and blocking=false, throws std::runtime_error.
    // No-op if token is already complete (ready() returns true).
    void enqueue(copy_token const& token, size_t priority = 0, bool blocking = true);

    // Poll pending operations; return count of newly completed.
    // Does NOT wait; returns immediately with completion count.
    // Removes completed tokens from queue.
    size_t poll();

    // Block until all pending complete or timeout expires.
    // Returns count of completed operations.
    // If timeout is zero, waits indefinitely.
    size_t wait_all(std::chrono::milliseconds timeout = {});

    // Current count of pending operations in the queue.
    size_t pending_count() const noexcept;

    // Set maximum allowed pending operations (backpressure control).
    // If queue grows beyond limit, enqueue() waits or throws.
    // Default: unlimited (0).
    void set_max_pending(size_t limit) noexcept;

    // Get current max_pending limit (0 = unlimited).
    size_t max_pending() const noexcept;

    // Diagnostics: return count of operations that failed (state == failed).
    size_t failed_count() const noexcept;

    // Release all quarantined failed operations. Only call when you can accept
    // that the underlying storage may be in an unknown state (e.g., after a
    // device reset or process teardown).
    void clear_failed() noexcept;

    // Drain all pending operations (called at shutdown).
    // Blocks until all complete or timeout expires.
    // Returns count of discarded operations (if timeout).
    size_t drain(std::chrono::milliseconds timeout = {});

    // Stop accepting new operations and drain. Timed-out operations remain
    // retained by the service; failed operations remain quarantined.
    // Returns the number of still-pending operations.
    size_t shutdown(std::chrono::milliseconds timeout = {});

    // Wait for pending work and reopen admission (for testing). Failed
    // operations remain quarantined and are not cleared.
    void reset() noexcept;

private:
    retained_operation_service() = default;

    struct pending_op
    {
        copy_token token;
        size_t     priority;
    };

    mutable std::mutex           mu_;
    std::condition_variable      cv_;          // Signals when space available or ops complete
    std::deque<pending_op>       pending_;
    std::deque<copy_token>       failed_;      // Quarantine failures with owners retained
    size_t                       max_pending_{0};  // 0 = unlimited
    bool                         stopping_{false};
};

}  // namespace memory
