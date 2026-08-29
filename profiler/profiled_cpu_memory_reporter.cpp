/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * This file is part of XSigma and is licensed under a dual-license model:
 *
 *   - Open-source License (GPLv3):
 *       Free for personal, academic, and research use under the terms of
 *       the GNU General Public License v3.0 or later.
 *
 *   - Commercial License:
 *       A commercial license is required for proprietary, closed-source,
 *       or SaaS usage. Contact us to obtain a commercial agreement.
 *
 * Contact: licensing@xsigma.co.uk
 * Website: https://www.xsigma.co.uk
 */

#include "profiler/profiled_cpu_memory_reporter.h"

#include <cstdint>

#include "util/memory_exception.h"

#if MEMORY_HAS_PROFILER
#include "common/instrumentation.h"
#endif

namespace memory
{
namespace
{
constexpr int16_t kCpuDeviceType  = 0;  // profiler::device_enum::CPU
constexpr int16_t kCpuDeviceIndex = -1;
}  // namespace

profiled_cpu_memory_reporter& cpu_memory_reporter()
{
    static profiled_cpu_memory_reporter reporter;
    return reporter;
}

void profiled_cpu_memory_reporter::record_allocation(void* ptr, std::size_t nbytes)
{
    if (ptr == nullptr || nbytes == 0)
    {
        return;
    }

    std::size_t allocated = 0;
    {
        std::scoped_lock const lock(mutex_);
        const auto             it = size_table_.find(ptr);
        if (it != size_table_.end())
        {
            allocated_ -= it->second;
        }
        else
        {
            tracked_blocks_fast_.fetch_add(1, std::memory_order_relaxed);
        }
        size_table_[ptr] = nbytes;
        allocated_ += nbytes;
        allocated = allocated_;
    }

#if MEMORY_HAS_PROFILER
    profiler::report_memory_usage(
        ptr,
        static_cast<int64_t>(nbytes),
        allocated,
        /*total_reserved=*/0,
        kCpuDeviceType,
        kCpuDeviceIndex);
#else
    (void)allocated;
#endif
}

void profiled_cpu_memory_reporter::record_deallocation(void* ptr, bool emit_event)
{
    if (ptr == nullptr)
    {
        return;
    }

    std::size_t nbytes    = 0;
    std::size_t allocated = 0;
    {
        std::scoped_lock const lock(mutex_);
        const auto             it = size_table_.find(ptr);
        if (it != size_table_.end())
        {
            allocated_ -= it->second;
            allocated = allocated_;
            nbytes    = it->second;
            size_table_.erase(it);
            tracked_blocks_fast_.fetch_sub(1, std::memory_order_relaxed);
        }
        else if (emit_event && unknown_free_log_count_++ % 1000 == 0)
        {
            MEMORY_LOG_WARNING(
                "CPU memory block of unknown size was allocated before profiling "
                "started; deallocation event is omitted");
        }
    }

    if (nbytes == 0)
    {
        return;
    }

#if MEMORY_HAS_PROFILER
    if (emit_event)
    {
        profiler::report_memory_usage(
            ptr,
            -static_cast<int64_t>(nbytes),
            allocated,
            /*total_reserved=*/0,
            kCpuDeviceType,
            kCpuDeviceIndex);
    }
#else
    (void)allocated;
    (void)emit_event;
#endif
}

void profiled_cpu_memory_reporter::record_out_of_memory(std::size_t nbytes)
{
    if (nbytes == 0)
    {
        return;
    }

    std::size_t allocated = 0;
    {
        std::scoped_lock const lock(mutex_);
        ++num_ooms_;
        allocated = allocated_;
    }

#if MEMORY_HAS_PROFILER
    profiler::report_out_of_memory(
        static_cast<int64_t>(nbytes),
        allocated,
        /*total_reserved=*/0,
        kCpuDeviceType,
        kCpuDeviceIndex);
#else
    (void)allocated;
#endif
}

std::size_t profiled_cpu_memory_reporter::allocated() const
{
    std::scoped_lock const lock(mutex_);
    return allocated_;
}

std::size_t profiled_cpu_memory_reporter::tracked_blocks() const
{
    std::scoped_lock const lock(mutex_);
    return size_table_.size();
}

bool profiled_cpu_memory_reporter::has_tracked_blocks() const noexcept
{
    return tracked_blocks_fast_.load(std::memory_order_relaxed) != 0;
}

std::size_t profiled_cpu_memory_reporter::num_ooms() const
{
    std::scoped_lock const lock(mutex_);
    return num_ooms_;
}

void profiled_cpu_memory_reporter::reset()
{
    std::scoped_lock const lock(mutex_);
    size_table_.clear();
    tracked_blocks_fast_.store(0, std::memory_order_relaxed);
    allocated_              = 0;
    num_ooms_               = 0;
    unknown_free_log_count_ = 0;
}

}  // namespace memory
