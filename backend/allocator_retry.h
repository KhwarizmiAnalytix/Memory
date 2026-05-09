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

#pragma once

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>

#include "common/memory_macros.h"

namespace memory
{
class allocator_retry
{
public:
    allocator_retry();
    ~allocator_retry();

    // Call 'alloc_func' to obtain memory.  On first call,
    // 'verbose_failure' will be false.  If return value is nullptr,
    // then wait up to 'max_millis_to_wait' milliseconds, retrying each
    // time a call to deallocate_raw() is detected, until either a good
    // pointer is returned or the deadline is exhausted.  If the
    // deadline is exhausted, try one more time with 'verbose_failure'
    // set to true.  The value returned is either the first good pointer
    // obtained from 'alloc_func' or nullptr.
    void* allocate_raw(
        std::function<void*(size_t alignment, size_t num_bytes, bool verbose_failure)> alloc_func,
        int    max_millis_to_wait,
        size_t alignment,
        size_t bytes);

    // Called to notify clients that some memory was returned.
    void NotifyDealloc();

private:
    std::mutex mu_;

    std::condition_variable memory_returned_ MEMORY_GUARDED_BY(mu_);
};

// Implementation details below
inline void allocator_retry::NotifyDealloc()
{
    std::unique_lock<std::mutex> const l(mu_);
    memory_returned_.notify_all();
}
}  // namespace memory
