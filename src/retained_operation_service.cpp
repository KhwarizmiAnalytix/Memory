/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#include "common/retained_operation_service.h"

#include <algorithm>
#include <array>
#include <condition_variable>
#include <exception>
#include <new>
#include <stdexcept>
#include <thread>

#include "common/transfer.h"

namespace memory
{

retained_operation_service& retained_operation_service::instance() noexcept
{
    // condition_variable construction can throw system_error. nothrow new only
    // turns allocation failure into nullptr. Either failure terminates.
    try
    {
        static auto* const s_instance = new (std::nothrow) retained_operation_service();
        if (s_instance == nullptr)
        {
            std::terminate();
        }
        return *s_instance;
    }
    catch (...)
    {
        std::terminate();
    }
}

bool retained_operation_service::quarantine_full_locked() const noexcept
{
    if (max_quarantined_ != 0 && quarantined_n_ >= max_quarantined_)
    {
        return true;
    }
    return max_quarantined_bytes_ != 0 && quarantined_bytes_ >= max_quarantined_bytes_;
}

void retained_operation_service::mark_quarantined_locked(op_entry& op) noexcept
{
    op.quarantined = true;
    --pending_n_;
    pending_bytes_ -= op.bytes;
    ++quarantined_n_;
    quarantined_bytes_ += op.bytes;
}

void retained_operation_service::take_locked(size_t index, copy_token& out) noexcept
{
    op_entry& op = ops_[index];
    if (op.quarantined)
    {
        --quarantined_n_;
        quarantined_bytes_ -= op.bytes;
    }
    else
    {
        --pending_n_;
        pending_bytes_ -= op.bytes;
    }
    out = std::move(op.token);
    ops_.erase(ops_.begin() + static_cast<std::ptrdiff_t>(index));
}

bool retained_operation_service::enqueue(copy_token const& token, size_t bytes, bool blocking)
{
    if (token.ready())
    {
        return false;
    }

    std::unique_lock<std::mutex> lock(mu_);
    while (true)
    {
        if (stopping_)
        {
            throw std::runtime_error("retained_operation_service: service is shutting down");
        }
        if (quarantine_full_locked())
        {
            throw std::runtime_error(
                "retained_operation_service: quarantine budget exhausted; "
                "recover_quarantined() or raise set_max_quarantined()");
        }
        if (max_pending_ == 0 || pending_n_ < max_pending_)
        {
            break;
        }
        if (!blocking)
        {
            throw std::runtime_error(
                "retained_operation_service: max pending operations reached; "
                "call poll() to drain or set_max_pending(0) for unlimited");
        }
        // Nothing polls in the background: reap our own capacity, then sleep until
        // a limit change, shutdown or a short interval.
        lock.unlock();
        (void)poll();
        lock.lock();
        if (!stopping_ && max_pending_ != 0 && pending_n_ >= max_pending_)
        {
            cv_.wait_for(lock, std::chrono::milliseconds(1));
        }
    }

    ops_.push_back({token, bytes, false});  // may throw bad_alloc: nothing admitted
    ++pending_n_;
    pending_bytes_ += bytes;
    return true;
}

bool retained_operation_service::cancel(copy_token const& token) noexcept
{
    // Declared before the lock scope so the token copy, which may hold the last
    // reference to the retained payload, is destroyed after the mutex is released.
    copy_token released;
    bool       found = false;
    {
        std::lock_guard<std::mutex> const lock(mu_);
        for (size_t i = 0; i < ops_.size(); ++i)
        {
            if (!ops_[i].quarantined && ops_[i].token.same_operation(token))
            {
                take_locked(i, released);
                cv_.notify_all();
                found = true;
                break;
            }
        }
    }
    return found;
}

bool retained_operation_service::quarantine(copy_token const& token) noexcept
{
    std::lock_guard<std::mutex> const lock(mu_);
    for (auto& op : ops_)
    {
        if (!op.quarantined && op.token.same_operation(token))
        {
            mark_quarantined_locked(op);
            cv_.notify_all();
            return true;
        }
    }
    return false;
}

retained_poll_result retained_operation_service::poll()
{
    retained_poll_result result;
    while (true)
    {
        // A bounded batch of completed tokens is moved out under the lock and
        // destroyed after it is dropped: releasing a payload runs user deleters,
        // which may call back into the service.
        std::array<copy_token, kReleaseBatch> released;
        size_t                                taken = 0;
        bool                                  more  = false;
        {
            std::lock_guard<std::mutex> const lock(mu_);
            size_t                            write   = 0;
            bool                              changed = false;
            for (size_t read = 0; read < ops_.size(); ++read)
            {
                op_entry& op = ops_[read];
                if (!op.quarantined)
                {
                    if (taken == kReleaseBatch)
                    {
                        more = true;
                    }
                    else
                    {
                        completion_state const state = op.token.state();
                        if (state == completion_state::complete)
                        {
                            released[taken++] = std::move(op.token);
                            --pending_n_;
                            pending_bytes_ -= op.bytes;
                            ++result.completed;
                            changed = true;
                            continue;
                        }
                        if (state == completion_state::failed)
                        {
                            mark_quarantined_locked(op);  // flag only: no allocation
                            ++result.failed;
                            changed = true;
                        }
                    }
                }
                if (write != read)
                {
                    ops_[write] = std::move(op);
                }
                ++write;
            }
            ops_.erase(ops_.begin() + static_cast<std::ptrdiff_t>(write), ops_.end());
            if (changed)
            {
                cv_.notify_all();
            }
        }
        if (!more)
        {
            return result;
        }
    }
}

size_t retained_operation_service::wait_all(std::chrono::milliseconds timeout)
{
    auto deadline = std::chrono::steady_clock::now() + timeout;

    while (true)
    {
        (void)poll();

        std::unique_lock<std::mutex> lock(mu_);
        if (pending_n_ == 0)
        {
            return 0;
        }

        std::chrono::milliseconds wait_timeout(10);
        if (timeout.count() != 0)
        {
            auto now = std::chrono::steady_clock::now();
            if (now >= deadline)
            {
                return pending_n_;
            }
            auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
            wait_timeout   = std::min(remaining, std::chrono::milliseconds(10));
        }
        cv_.wait_for(lock, wait_timeout, [this]() { return pending_n_ == 0; });
    }
}

size_t retained_operation_service::pending_count() const noexcept
{
    std::lock_guard<std::mutex> const lock(mu_);
    return pending_n_;
}

retained_service_stats retained_operation_service::stats() const noexcept
{
    std::lock_guard<std::mutex> const lock(mu_);
    return {pending_n_, pending_bytes_, quarantined_n_, quarantined_bytes_};
}

void retained_operation_service::set_max_pending(size_t limit) noexcept
{
    {
        std::lock_guard<std::mutex> const lock(mu_);
        max_pending_ = limit;
    }
    cv_.notify_all();
}

size_t retained_operation_service::max_pending() const noexcept
{
    std::lock_guard<std::mutex> const lock(mu_);
    return max_pending_;
}

void retained_operation_service::set_max_quarantined(size_t ops, size_t bytes) noexcept
{
    {
        std::lock_guard<std::mutex> const lock(mu_);
        max_quarantined_       = ops;
        max_quarantined_bytes_ = bytes;
    }
    cv_.notify_all();
}

size_t retained_operation_service::max_quarantined() const noexcept
{
    std::lock_guard<std::mutex> const lock(mu_);
    return max_quarantined_;
}

size_t retained_operation_service::max_quarantined_bytes() const noexcept
{
    std::lock_guard<std::mutex> const lock(mu_);
    return max_quarantined_bytes_;
}

size_t retained_operation_service::failed_count() const noexcept
{
    std::lock_guard<std::mutex> const lock(mu_);
    return quarantined_n_;
}

size_t retained_operation_service::release_quarantined(bool check_idle) noexcept
{
    size_t total = 0;
    while (true)
    {
        std::array<copy_token, kReleaseBatch> released;
        size_t                                taken = 0;
        bool                                  more  = false;
        {
            std::lock_guard<std::mutex> const lock(mu_);
            size_t                            write = 0;
            for (size_t read = 0; read < ops_.size(); ++read)
            {
                op_entry& op = ops_[read];
                if (op.quarantined)
                {
                    if (taken == kReleaseBatch)
                    {
                        more = true;
                    }
                    else if (!check_idle || detail::stream_proven_idle(op.token.ctx()))
                    {
                        released[taken++] = std::move(op.token);
                        --quarantined_n_;
                        quarantined_bytes_ -= op.bytes;
                        ++total;
                        continue;
                    }
                }
                if (write != read)
                {
                    ops_[write] = std::move(op);
                }
                ++write;
            }
            ops_.erase(ops_.begin() + static_cast<std::ptrdiff_t>(write), ops_.end());
            if (taken != 0)
            {
                cv_.notify_all();
            }
        }
        if (!more)
        {
            return total;
        }
    }
}

size_t retained_operation_service::recover_quarantined() noexcept
{
    return release_quarantined(true);
}

size_t retained_operation_service::abandon_quarantined() noexcept
{
    return release_quarantined(false);
}

size_t retained_operation_service::drain(std::chrono::milliseconds timeout)
{
    auto deadline = std::chrono::steady_clock::now() + timeout;

    while (true)
    {
        (void)poll();

        {
            std::lock_guard<std::mutex> const lock(mu_);
            if (pending_n_ == 0)
            {
                return 0;
            }
        }

        if (timeout.count() == 0)
        {
            std::this_thread::yield();
            continue;
        }

        auto now = std::chrono::steady_clock::now();
        if (now >= deadline)
        {
            std::lock_guard<std::mutex> const lock(mu_);
            return pending_n_;
        }

        auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
        std::this_thread::sleep_for(std::min(remaining, std::chrono::milliseconds(10)));
    }
}

size_t retained_operation_service::shutdown(std::chrono::milliseconds timeout)
{
    {
        std::lock_guard<std::mutex> const lock(mu_);
        stopping_ = true;
    }
    cv_.notify_all();  // blocked enqueuers throw
    return drain(timeout);
}

void retained_operation_service::reset() noexcept
{
    // Every pending operation is waited for in place: the entry stays in the
    // service (owners retained) until its wait returns, and a failed wait
    // quarantines it. Nothing here allocates.
    while (true)
    {
        copy_token op_token;
        {
            std::lock_guard<std::mutex> const lock(mu_);
            auto const it = std::find_if(
                ops_.begin(), ops_.end(), [](op_entry const& op) { return !op.quarantined; });
            if (it == ops_.end())
            {
                break;
            }
            op_token = it->token;
        }

        bool waited = true;
        try
        {
            op_token.wait();
        }
        catch (...)
        {
            waited = false;
        }

        copy_token released;  // destroyed after the lock is dropped
        {
            std::lock_guard<std::mutex> const lock(mu_);
            for (size_t i = 0; i < ops_.size(); ++i)
            {
                if (!ops_[i].quarantined && ops_[i].token.same_operation(op_token))
                {
                    if (waited)
                    {
                        take_locked(i, released);
                    }
                    else
                    {
                        mark_quarantined_locked(ops_[i]);
                    }
                    break;
                }
            }
        }
    }

    {
        std::lock_guard<std::mutex> const lock(mu_);
        max_pending_           = default_max_pending;
        max_quarantined_       = default_max_quarantined;
        max_quarantined_bytes_ = 0;
        stopping_              = false;
    }
    cv_.notify_all();
}

}  // namespace memory
