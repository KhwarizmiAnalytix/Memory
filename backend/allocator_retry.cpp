/*
 * Quarisma: High-Performance Computational Library
 *
 * Original work Copyright 2015 The TensorFlow Authors
 * Modified work Copyright 2025 Quarisma Contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * This file contains code modified from TensorFlow (Apache 2.0 licensed)
 * and is part of Quarisma, licensed under a dual-license model:
 *
 *   - Open-source License (GPLv3):
 *       Free for personal, academic, and research use under the terms of
 *       the GNU General Public License v3.0 or later.
 *
 *   - Commercial License:
 *       A commercial license is required for proprietary, closed-source,
 *       or SaaS usage. Contact us to obtain a commercial agreement.
 *
 * MODIFICATIONS FROM ORIGINAL:
 * - Adapted for Quarisma quantitative computing requirements
 * - Added high-performance memory allocation optimizations
 * - Integrated NUMA-aware allocation strategies
 *
 * Contact: licensing@quarisma.co.uk
 * Website: https://www.quarisma.co.uk
 */

#include "backend/allocator_retry.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <functional>
#include <optional>

#include "common/memory_macros.h"

namespace memory
{
namespace
{

// Simple counter class for metrics
template <size_t NumLabels = 0>
class Counter
{
public:
    static Counter* New(const char* name, const char* description)
    {
        return new Counter(name, description);
    }

    class Cell
    {
    public:
        void IncrementBy(int64_t value) { value_.fetch_add(value, std::memory_order_relaxed); }

    private:
        friend class Counter;
        std::atomic<int64_t> value_{0};
    };

    Cell* GetCell() { return &cell_; }

private:
    Counter(const char* name, const char* description) : name_(name), description_(description) {}

    const std::string name_;
    const std::string description_;
    Cell              cell_;
};

Counter<0>* get_allocator_bfc_delay_counter()
{
    static auto* const counter = Counter<0>::New(
        "/memory/cpu/allocator_bfc_delay",
        "The total time spent running each graph "
        "optimization pass in microseconds.");
    return counter;
}

void update_allocator_bfc_delay_time(const uint64_t delay_usecs)
{
    static auto* allocator_bfc_delay_cell = get_allocator_bfc_delay_counter()->GetCell();
    if (delay_usecs > 0)
    {
        allocator_bfc_delay_cell->IncrementBy(static_cast<int64_t>(delay_usecs));
    }
}

class scoped_time_tracker
{
public:
    MEMORY_DELETE_COPY(scoped_time_tracker);

    scoped_time_tracker()                                 = default;
    scoped_time_tracker(scoped_time_tracker&&)            = delete;
    scoped_time_tracker& operator=(scoped_time_tracker&&) = delete;

    // Start tracking time if not already tracking
    void enable()
    {
        if (!start_us_)
        {  // Only override start_us when not set yet
            start_us_ = std::chrono::steady_clock::now();
        }
    }

    // Destructor updates metrics with elapsed time
    ~scoped_time_tracker()
    {
        if (start_us_)
        {
            const auto end_us = std::chrono::steady_clock::now();
            const auto elapsed_us =
                std::chrono::duration_cast<std::chrono::microseconds>(end_us - *start_us_).count();

            update_allocator_bfc_delay_time(elapsed_us);
        }
    }

private:
    std::optional<std::chrono::steady_clock::time_point> start_us_;
};
}  // namespace

allocator_retry::allocator_retry() = default;

allocator_retry::~allocator_retry()
{
    // Lock the mutex to make sure that all memory effects are safely published
    // and available to a thread running the destructor.
    std::unique_lock<std::mutex> const l(mu_);
}

void* allocator_retry::allocate_raw(
    std::function<void*(size_t alignment, size_t num_bytes, bool verbose_failure)> alloc_func,
    int    max_millis_to_wait,
    size_t alignment,
    size_t num_bytes)
{
    if (num_bytes == 0)
    {
        return nullptr;
    }

    scoped_time_tracker tracker;

    using Clock         = std::chrono::steady_clock;
    const auto deadline = Clock::now() + std::chrono::milliseconds(max_millis_to_wait);

    void* ptr = nullptr;  //NOLINT

    while (ptr == nullptr)
    {
        // Try allocation without holding the lock
        ptr = alloc_func(alignment, num_bytes, /*verbose_failure=*/false);

        if (ptr == nullptr)
        {
            auto now = Clock::now();

            if (now < deadline)
            {
                tracker.enable();
                // Wait for memory to become available
                std::unique_lock<std::mutex> lock(mu_);

                memory_returned_.wait_until(lock, deadline);
            }
            else
            {
                // Final attempt with verbose failure
                return alloc_func(alignment, num_bytes, /*verbose_failure=*/true);
            }
        }
    }

    return ptr;
}
}  // namespace memory
