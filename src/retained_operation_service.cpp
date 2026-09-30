/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#include "common/retained_operation_service.h"

#include <condition_variable>
#include <stdexcept>
#include <thread>

namespace memory
{

retained_operation_service& retained_operation_service::instance() noexcept
{
    static retained_operation_service s_instance;
    return s_instance;
}

void retained_operation_service::enqueue(copy_token token, size_t priority, bool blocking)
{
    std::unique_lock<std::mutex> lock(mu_);

    if (token.ready())
    {
        return;
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
        std::this_thread::yield();
        lock.lock();
    }

    pending_.push_back({token, priority});
}

size_t retained_operation_service::poll() noexcept
{
    std::unique_lock<std::mutex> lock(mu_);
    size_t                        completed = 0;

    for (auto it = pending_.begin(); it != pending_.end();)
    {
        if (it->token.ready())
        {
            ++completed;
            it = pending_.erase(it);
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
        size_t completed = poll();
        if (completed > 0)
        {
            return completed;
        }

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
            return 0;
        }

        auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
        std::this_thread::sleep_for(std::chrono::milliseconds(
            std::min(remaining.count(), static_cast<long long>(10))));
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
    size_t                        count = 0;
    for (auto const& op : pending_)
    {
        if (op.token.state() == completion_state::failed)
        {
            ++count;
        }
    }
    return count;
}

size_t retained_operation_service::drain(std::chrono::milliseconds timeout) noexcept
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
            std::min(remaining.count(), static_cast<long long>(10))));
    }
}

void retained_operation_service::reset() noexcept
{
    std::unique_lock<std::mutex> lock(mu_);
    pending_.clear();
    max_pending_ = 0;
}

}  // namespace memory
