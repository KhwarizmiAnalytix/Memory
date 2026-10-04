/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

#include "common/copy_token.h"
#include "common/memory_export.h"

namespace memory
{

/// Counts and bytes held by the service (plan 5.2). Bytes are the payload sizes
/// the submitter declared at admission (0 when it declared none).
struct retained_service_stats
{
    size_t pending_ops{0};
    size_t pending_bytes{0};
    size_t quarantined_ops{0};
    size_t quarantined_bytes{0};
};

/// What one poll() found (plan 5.5). `completed` operations released their owners;
/// `failed` ones moved to quarantine and keep them.
struct retained_poll_result
{
    size_t completed{0};
    size_t failed{0};
    size_t total() const noexcept { return completed + failed; }
};

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
 * Rules (plan 5.1-5.5):
 * - Admission is finite by default (`default_max_pending`); 0 means unlimited,
 *   explicitly. A full queue makes `enqueue(..., blocking=true)` wait (polling for
 *   its own capacity, since nothing polls in the background) and
 *   `blocking=false` throw. `shutdown()`, `reset()` and `set_max_pending()` wake
 *   waiters.
 * - Pending and quarantined operations and their declared bytes are accounted
 *   (`stats()`). A full quarantine budget refuses admission; lowering a limit
 *   never releases an owner.
 * - Quarantine is a flag on the operation's existing entry, so the failure path
 *   allocates nothing and cannot lose an owner to `bad_alloc`.
 * - Payloads (and the custom deleters they run) are released after the service
 *   mutex is dropped, so a deleter may call back into the service.
 * - Quarantined owners are released only by `recover_quarantined()` (the stream is
 *   proven idle) or `abandon_quarantined()` (the caller accepts the risk).
 *
 * Thread-safety: instance() is thread-safe; the other members use an internal mutex.
 */
class MEMORY_API retained_operation_service
{
public:
    /// Default admission limit and quarantine budget.
    static constexpr size_t default_max_pending     = 4096;
    static constexpr size_t default_max_quarantined = 1024;

    // Get singleton instance (thread-safe).
    static retained_operation_service& instance() noexcept;

    // Admit a token whose payload is @p bytes large. No-op (returns false) if the
    // token is already complete. Returns true when admitted; the entry must then
    // be released exactly once, by completing, cancel() or quarantine().
    // If the queue is full: blocking=true waits for capacity, blocking=false throws
    // std::runtime_error. A full quarantine budget and a shutting-down service
    // throw in both modes.
    bool enqueue(copy_token const& token, size_t bytes = 0, bool blocking = true);

    // Roll back an admission whose operation never started or was proven idle:
    // removes the token's pending entry without waiting. Returns false if it is
    // not pending (already reaped), so a rollback happens at most once.
    bool cancel(copy_token const& token) noexcept;

    // Move an admitted token whose completion could not be proven straight to
    // quarantine, keeping its owners. Allocates nothing. Returns false only if
    // the token is not pending.
    bool quarantine(copy_token const& token) noexcept;

    // Poll pending operations. Does NOT wait. Completed operations are removed
    // (owners released outside the lock); failed ones are quarantined.
    retained_poll_result poll();

    // Block until all pending complete or timeout expires.
    // Returns the number still pending. If timeout is zero, waits indefinitely.
    size_t wait_all(std::chrono::milliseconds timeout = {});

    // Operations currently pending (not quarantined).
    size_t pending_count() const noexcept;

    // Pending/quarantined operation and byte counts.
    retained_service_stats stats() const noexcept;

    // Admission limit (0 = unlimited). Lowering it never drops an owner; it only
    // stops new admission until the queue drains. Wakes blocked enqueuers.
    void   set_max_pending(size_t limit) noexcept;
    size_t max_pending() const noexcept;

    // Quarantine budget: admission is refused while quarantined operations (or
    // bytes, when non-zero) are at or above the limit. 0 ops = unlimited.
    void   set_max_quarantined(size_t ops, size_t bytes = 0) noexcept;
    size_t max_quarantined() const noexcept;
    size_t max_quarantined_bytes() const noexcept;

    // Operations quarantined after failing (owners retained).
    size_t failed_count() const noexcept;

    // Checked recovery: for each quarantined operation, synchronize its stream and
    // release its owners only if that succeeds (the stream is proven idle). The
    // others stay quarantined. Blocks on the driver. Returns the number released.
    size_t recover_quarantined() noexcept;

    // Unchecked recovery: release every quarantined owner. Only for when the
    // storage may be in an unknown state and that is acceptable (device reset,
    // process teardown). Returns the number released.
    size_t abandon_quarantined() noexcept;

    // Wait for pending work. Returns the number still pending at the timeout.
    size_t drain(std::chrono::milliseconds timeout = {});

    // Stop accepting new operations and drain. Timed-out operations remain
    // retained by the service; failed operations remain quarantined. Wakes blocked
    // enqueuers (they throw). Returns the number of still-pending operations.
    size_t shutdown(std::chrono::milliseconds timeout = {});

    // Wait for every pending operation (one that fails is quarantined, never
    // dropped), restore the default limits and reopen admission. For tests.
    void reset() noexcept;

private:
    retained_operation_service() = default;

    struct op_entry
    {
        copy_token token;
        size_t     bytes{0};
        bool       quarantined{false};
    };

    static constexpr size_t kReleaseBatch = 8;

    bool quarantine_full_locked() const noexcept;
    void take_locked(size_t index, copy_token& out) noexcept;
    void mark_quarantined_locked(op_entry& op) noexcept;
    size_t release_quarantined(bool check_idle) noexcept;

    mutable std::mutex      mu_;
    std::condition_variable cv_;
    std::vector<op_entry>   ops_;  // pending and quarantined; capacity retained
    size_t                  pending_n_{0};
    size_t                  pending_bytes_{0};
    size_t                  quarantined_n_{0};
    size_t                  quarantined_bytes_{0};
    size_t                  max_pending_{default_max_pending};
    size_t                  max_quarantined_{default_max_quarantined};
    size_t                  max_quarantined_bytes_{0};
    bool                    stopping_{false};
};

/// Ordered runtime teardown (plan 5.5): stop admission and wait for retained
/// operations; only if none remain, drop the token event pool and flush the GPU
/// caches. Returns the number of operations still pending (then nothing else is
/// torn down: their owners may hold cache blocks). Streams and devices belong to
/// the caller and must outlive this call.
MEMORY_API size_t shutdown_runtime(std::chrono::milliseconds timeout = {});

}  // namespace memory
