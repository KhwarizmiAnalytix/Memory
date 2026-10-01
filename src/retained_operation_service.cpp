/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#include "common/retained_operation_service.h"

#include <condition_variable>
#include <exception>
#include <new>
#include <stdexcept>
#include <thread>

namespace memory
{

retained_operation_service& retained_operation_service::instance() noexcept
{
    static auto* s_instance = new (std::nothrow) retained_operation_service();
    if (s_instance == nullptr)
    {
        std::terminate();
    }
    return *s_instance;
}

void retained_operation_service::enqueue(copy_token const& token, size_t priority, bool blocking)
{
    std::unique_lock<std::mutex> lock(mu_);

    if (token.ready())
    {
        return;
    }
    if (stopping_)
    {
        throw std::runtime_error("retained_operation_service: service is shutting down");
    }

    while (max_pending_ > 0 && pending_.size() >= max_pending_)
    {
        if (!blocking)
        {
            throw std::runtime_error(
                "retained_operation_service: max pending operations reached; "
                "call poll() to drain or set_max_pending(0) for unlimited");
        }
        lock.unlock();
        poll();
        lock.lock();
        if (stopping_)
        {
            throw std::runtime_error("retained_operation_service: service is shutting down");
        }
        if (max_pending_ > 0 && pending_.size() >= max_pending_)
        {
            cv_.wait_for(lock, std::chrono::milliseconds(1));
        }
    }

    pending_.push_back({token, priority});
}

size_t retained_operation_service::poll()
{
    std::unique_lock<std::mutex> lock(mu_);
    size_t                        completed = 0;

    for (auto it = pending_.begin(); it != pending_.end();)
    {
        completion_state state = it->token.state();
        if (state == completion_state::complete)
        {
            ++completed;
            it = pending_.erase(it);
            // Notify waiters that space may be available
            cv_.notify_one();
        }
        else if (state == completion_state::failed)
        {
            ++completed;
            failed_.push_back(it->token);
            it = pending_.erase(it);
            // Notify waiters that space may be available
            cv_.notify_one();
        }
        else
        {
            ++it;
        }
    }

    return completed;
}

size_t retained_operation_service::wait_all(std::chrono::milliseconds timeout)
{
    auto deadline = std::chrono::steady_clock::now() + timeout;

    while (true)
    {
        poll();  // Poll once to reap any completed operations

        std::unique_lock<std::mutex> lock(mu_);
        if (pending_.empty())
        {
            return 0;  // All pending operations completed
        }

        // Calculate remaining time
        std::chrono::milliseconds wait_timeout;
        if (timeout.count() == 0)
        {
            wait_timeout = std::chrono::milliseconds(10);  // Indefinite: wait with short timeout
        }
        else
        {
            auto now = std::chrono::steady_clock::now();
            if (now >= deadline)
            {
                return pending_.size();  // Timeout expired, return remaining count
            }
            auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
            wait_timeout = std::chrono::milliseconds(
                std::min(remaining.count(), static_cast<decltype(remaining.count())>(10)));
        }

        // Wait for completion or timeout
        cv_.wait_for(lock, wait_timeout, [this]() { return pending_.empty(); });
    }
}

size_t retained_operation_service::pending_count() const noexcept
{
    std::unique_lock<std::mutex> lock(mu_);
    return pending_.size();
}

void retained_operation_service::set_max_pending(size_t limit) noexcept
{
    std::unique_lock<std::mutex> lock(mu_);
    max_pending_ = limit;
}

size_t retained_operation_service::max_pending() const noexcept
{
    std::unique_lock<std::mutex> lock(mu_);
    return max_pending_;
}

size_t retained_operation_service::failed_count() const noexcept
{
    std::unique_lock<std::mutex> lock(mu_);
    return failed_.size();
}

void retained_operation_service::clear_failed() noexcept
{
    std::deque<copy_token> to_release;
    {
        std::unique_lock<std::mutex> lock(mu_);
        to_release.swap(failed_);
    }
    // Tokens released here outside the lock.
}

size_t retained_operation_service::drain(std::chrono::milliseconds timeout)
{
    auto deadline = std::chrono::steady_clock::now() + timeout;

    while (true)
    {
        poll();

        {
            std::unique_lock<std::mutex> lock(mu_);
            if (pending_.empty())
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
            std::unique_lock<std::mutex> lock(mu_);
            return pending_.size();
        }

        auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
        std::this_thread::sleep_for(std::chrono::milliseconds(
            std::min(remaining.count(), static_cast<decltype(remaining.count())>(10))));
    }
}

size_t retained_operation_service::shutdown(std::chrono::milliseconds timeout)
{
    {
        std::unique_lock<std::mutex> lock(mu_);
        stopping_ = true;
    }
    return drain(timeout);
}

void retained_operation_service::reset() noexcept
{
    std::deque<pending_op> to_drain;
    {
        std::unique_lock<std::mutex> lock(mu_);
        to_drain.swap(pending_);
        max_pending_ = 0;
        stopping_ = false;
    }

    for (auto& op : to_drain)
    {
        try
        {
            op.token.wait();
        }
        catch (...)
        {
            try
            {
                std::unique_lock<std::mutex> lock(mu_);
                failed_.push_back(op.token);
            }
            catch (...)  // NOLINT(bugprone-empty-catch)
            {
                // Silence exceptions during quarantine push; reset() must not throw
            }
        }
    }
}

}  // namespace memory
