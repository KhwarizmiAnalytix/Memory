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

#include "backend/allocator_tracking.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <ios>
#include <iterator>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

//#include "logger/logger.h"
#include "cpu/allocator.h"
#include "profiler/unified_memory_stats.h"
#include "util/exception.h"

namespace memory
{

allocator_tracking::allocator_tracking(
    Allocator* allocator, bool track_sizes, bool enable_enhanced_tracking)
    : allocator_(allocator),
      track_sizes_locally_(
          (track_sizes && !allocator_->tracks_allocation_sizes()) || enable_enhanced_tracking),
      next_allocation_id_(0),
      enhanced_tracking_enabled_(enable_enhanced_tracking)
{
    // Initialize timing stats
    timing_stats_.reset();

    // Reserve space for enhanced records to minimize reallocations
    if (enhanced_tracking_enabled_)
    {
        enhanced_records_.reserve(1000);  // Reserve space for 1000 initial records
    }

    // Log initialization if enabled
    if (log_level_.load(std::memory_order_relaxed) >= tracking_log_level::INFO)
    {
        MEMORY_LOG_INFO(
            "allocator_tracking initialized: track_sizes={}, enhanced={}, underlying={}",
            track_sizes_locally_,
            enhanced_tracking_enabled_,
            allocator_->Name());
    }
}

void* allocator_tracking::allocate_raw(
    size_t alignment, size_t num_bytes, const allocation_attributes& allocation_attr)
{
    // Start timing for performance analysis
    auto start_time = std::chrono::steady_clock::now();

    // Log allocation attempt if enabled (simplified logging)
    auto current_log_level = log_level_.load(std::memory_order_relaxed);
    if (current_log_level >= tracking_log_level::DEBUG_LEVEL)
    {
        // Simplified logging to avoid macro issues
        // MEMORY_LOG_INFO("allocator_tracking::allocate_raw: "
        //                + std::to_string(num_bytes) + " bytes, alignment=" + std::to_string(alignment));
    }

    void* ptr = allocator_->allocate_raw(alignment, num_bytes, allocation_attr);  //NOLINT

    // Calculate allocation timing
    auto end_time = std::chrono::steady_clock::now();
    auto duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    auto duration_us_unsigned = std::max<uint64_t>(0ULL, duration_us);

    // Update timing statistics
    timing_stats_.total_allocations.fetch_add(1, std::memory_order_relaxed);
    timing_stats_.total_alloc_time_us.fetch_add(duration_us_unsigned, std::memory_order_relaxed);

    // Update min/max timing
    auto current_min = timing_stats_.min_alloc_time_us.load(std::memory_order_relaxed);
    while (duration_us_unsigned < current_min &&
           !timing_stats_.min_alloc_time_us.compare_exchange_weak(
               current_min, duration_us_unsigned, std::memory_order_relaxed))
    {
        // Retry if another thread updated min_alloc_time_us
    }

    auto current_max = timing_stats_.max_alloc_time_us.load(std::memory_order_relaxed);
    while (duration_us_unsigned > current_max &&
           !timing_stats_.max_alloc_time_us.compare_exchange_weak(
               current_max, duration_us_unsigned, std::memory_order_relaxed))
    {
        // Retry if another thread updated max_alloc_time_us
    }

    // If memory is exhausted allocate_raw returns nullptr, and we should
    // pass this through to the caller
    if (nullptr == ptr)
    {
        if (current_log_level >= tracking_log_level::WARNING)
        {
            MEMORY_LOG_WARNING(
                "allocator_tracking::allocate_raw failed: {} bytes, alignment={}",
                num_bytes,
                alignment);
        }
        return ptr;
    }

    // Log successful allocation (simplified logging)
    if (current_log_level >= tracking_log_level::TRACE)
    {
        // Simplified logging to avoid macro issues
        // MEMORY_LOG_INFO("allocator_tracking::allocate_raw success: ptr="
        //                + std::to_string(reinterpret_cast<uintptr_t>(ptr)) + ", "
        //                + std::to_string(num_bytes) + " bytes, time="
        //                + std::to_string(duration_us) + "μs");
    }
    if (allocator_->tracks_allocation_sizes())
    {
        size_t const allocated_bytes = allocator_->AllocatedSize(ptr);
        int64_t      allocation_id   = 0;

        {
            std::unique_lock<std::mutex> const lock(mu_);
            allocated_ += allocated_bytes;
            high_watermark_ = std::max(high_watermark_, allocated_);
            total_bytes_ += allocated_bytes;

            int64_t const tmp = std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count();
            allocations_.emplace_back(allocated_bytes, tmp);

            // Generate allocation ID for enhanced tracking
            next_allocation_id_ += 1;
            allocation_id = next_allocation_id_;

            // Populate in_use_ map if we're tracking sizes locally (needed for RequestedSize)
            if (track_sizes_locally_)
            {
                Chunk const chunk = {num_bytes, allocated_bytes, allocation_id};
                in_use_.emplace(std::make_pair(ptr, chunk));
            }

            ++ref_;
        }

        // Add enhanced record if enabled
        if (enhanced_tracking_enabled_)
        {
            std::unique_lock<std::shared_mutex> const enhanced_lock(shared_mu_);
            enhanced_records_.emplace_back(
                num_bytes,        // requested_size
                allocated_bytes,  // actual_size
                alignment,        // alignment
                std::chrono::duration_cast<std::chrono::microseconds>(end_time.time_since_epoch())
                    .count(),
                allocation_id,  // allocation_id
                "",             // tag (empty for now)
                nullptr,        // source_file
                0,              // source_line
                nullptr         // function_name
            );
            enhanced_records_.back().alloc_duration_us = duration_us;
        }
    }
    else if (track_sizes_locally_)
    {
        // Call the underlying allocator to try to get the allocated size
        // whenever possible, even when it might be slow. If this fails,
        // use the requested size as an approximation.
        size_t allocated_bytes = allocator_->AllocatedSizeSlow(ptr);
        allocated_bytes        = std::max(num_bytes, allocated_bytes);
        int64_t allocation_id  = 0;

        {
            std::unique_lock<std::mutex> const lock(mu_);
            next_allocation_id_ += 1;
            allocation_id     = next_allocation_id_;
            Chunk const chunk = {num_bytes, allocated_bytes, allocation_id};
            in_use_.emplace(std::make_pair(ptr, chunk));
            allocated_ += allocated_bytes;
            high_watermark_ = std::max(high_watermark_, allocated_);
            total_bytes_ += allocated_bytes;
            int64_t const tmp = std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count();
            allocations_.emplace_back(allocated_bytes, tmp);
            ++ref_;
        }

        // Add enhanced record if enabled
        if (enhanced_tracking_enabled_)
        {
            std::unique_lock<std::shared_mutex> const enhanced_lock(shared_mu_);
            enhanced_records_.emplace_back(
                num_bytes,        // requested_size
                allocated_bytes,  // actual_size
                alignment,        // alignment
                std::chrono::duration_cast<std::chrono::microseconds>(end_time.time_since_epoch())
                    .count(),
                allocation_id,  // allocation_id
                "",             // tag (empty for now)
                nullptr,        // source_file
                0,              // source_line
                nullptr         // function_name
            );
            enhanced_records_.back().alloc_duration_us = duration_us;
        }
    }
    else
    {
        int64_t allocation_id = 0;
        {
            std::unique_lock<std::mutex> const lock(mu_);
            total_bytes_ += num_bytes;
            int64_t const tmp = std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count();
            allocations_.emplace_back(num_bytes, tmp);
            next_allocation_id_ += 1;
            allocation_id = next_allocation_id_;
            ++ref_;
        }

        // Add enhanced record if enabled (even without size tracking)
        if (enhanced_tracking_enabled_)
        {
            std::unique_lock<std::shared_mutex> const enhanced_lock(shared_mu_);
            enhanced_records_.emplace_back(
                num_bytes,  // requested_size
                num_bytes,  // actual_size (estimate)
                alignment,  // alignment
                std::chrono::duration_cast<std::chrono::microseconds>(end_time.time_since_epoch())
                    .count(),
                allocation_id,  // allocation_id
                "",             // tag (empty for now)
                nullptr,        // source_file
                0,              // source_line
                nullptr         // function_name
            );
            enhanced_records_.back().alloc_duration_us = duration_us;
        }
    }
    return ptr;
}

void allocator_tracking::deallocate_raw(void* ptr)
{
    // freeing a null ptr is a no-op
    if (ptr == nullptr)
    {
        return;
    }

    // Start timing for performance analysis
    auto start_time = std::chrono::steady_clock::now();

    // Log deallocation attempt if enabled (simplified logging)
    auto current_log_level = log_level_.load(std::memory_order_relaxed);
    if (current_log_level >= tracking_log_level::TRACE)
    {
        // Simplified logging to avoid macro issues
        // MEMORY_LOG_INFO("allocator_tracking::deallocate_raw: ptr="
        //                + std::to_string(reinterpret_cast<uintptr_t>(ptr)));
    }

    bool should_delete;
    // fetch the following outside the lock in case the call to
    // AllocatedSize is slow
    bool    tracks_allocation_sizes = allocator_->tracks_allocation_sizes();
    size_t  allocated_bytes         = 0;
    int64_t allocation_id           = 0;

    // Always check local tracking for allocation_id when enhanced tracking is enabled
    if (track_sizes_locally_)
    {
        std::unique_lock<std::mutex> const lock(mu_);
        auto                               itr = in_use_.find(ptr);
        if (itr != in_use_.end())
        {
            allocated_bytes         = (*itr).second.allocated_size;
            allocation_id           = (*itr).second.allocation_id;
            tracks_allocation_sizes = true;
            in_use_.erase(itr);
        }
    }
    else if (tracks_allocation_sizes)
    {
        allocated_bytes = allocator_->AllocatedSize(ptr);
    }
    Allocator* allocator = allocator_;
    {
        std::unique_lock<std::mutex> const lock(mu_);
        if (tracks_allocation_sizes)
        {
            MEMORY_CHECK_DEBUG(allocated_ >= allocated_bytes);
            allocated_ -= allocated_bytes;
            int64_t const tmp = std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count();
            allocations_.emplace_back(-static_cast<int64_t>(allocated_bytes), tmp);
        }
        should_delete = UnRef();
    }

    // Perform actual deallocation
    allocator->deallocate_raw(ptr);

    // Calculate deallocation timing
    auto end_time = std::chrono::steady_clock::now();
    auto duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    auto duration_us_unsigned = std::max<uint64_t>(0ULL, duration_us);

    // Update timing statistics
    timing_stats_.total_deallocations.fetch_add(1, std::memory_order_relaxed);
    timing_stats_.total_dealloc_time_us.fetch_add(duration_us_unsigned, std::memory_order_relaxed);

    // Update min/max timing
    auto current_min = timing_stats_.min_dealloc_time_us.load(std::memory_order_relaxed);
    while (duration_us_unsigned < current_min &&
           !timing_stats_.min_dealloc_time_us.compare_exchange_weak(
               current_min, duration_us_unsigned, std::memory_order_relaxed))
    {
        // Retry if another thread updated min_dealloc_time_us
    }

    auto current_max = timing_stats_.max_dealloc_time_us.load(std::memory_order_relaxed);
    while (duration_us_unsigned > current_max &&
           !timing_stats_.max_dealloc_time_us.compare_exchange_weak(
               current_max, duration_us_unsigned, std::memory_order_relaxed))
    {
        // Retry if another thread updated max_dealloc_time_us
    }

    // Update enhanced record with deallocation timing if enabled and allocation_id is valid
    if (enhanced_tracking_enabled_ && allocation_id > 0)
    {
        std::unique_lock<std::shared_mutex> const enhanced_lock(shared_mu_);
        // Find the corresponding allocation record and update deallocation time
        auto record_it = std::find_if(
            enhanced_records_.begin(),
            enhanced_records_.end(),
            [allocation_id](const auto& record) { return record.allocation_id == allocation_id; });

        if (record_it != enhanced_records_.end())
        {
            record_it->dealloc_duration_us = duration_us;
        }
    }

    // Log successful deallocation (simplified logging)
    if (current_log_level >= tracking_log_level::TRACE)
    {
        // Simplified logging to avoid macro issues
        // MEMORY_LOG_INFO("allocator_tracking::deallocate_raw success: ptr="
        //                + std::to_string(reinterpret_cast<uintptr_t>(ptr)) + ", time="
        //                + std::to_string(duration_us) + "μs");
    }

    if (should_delete)
    {
        if (current_log_level >= tracking_log_level::INFO)
        {
            MEMORY_LOG_INFO("allocator_tracking self-destructing: ref_count=0");
        }
        delete this;
    }
}

bool allocator_tracking::tracks_allocation_sizes() const noexcept
{
    return track_sizes_locally_ || allocator_->tracks_allocation_sizes();
}

size_t allocator_tracking::RequestedSize(const void* ptr) const
{
    if (track_sizes_locally_)
    {
        std::unique_lock<std::mutex> const lock(mu_);
        auto                               it = in_use_.find(ptr);
        if (it != in_use_.end())
        {
            return (*it).second.requested_size;
        }
        return 0;
    }

    return allocator_->RequestedSize(ptr);
}

size_t allocator_tracking::AllocatedSize(const void* ptr) const
{
    if (track_sizes_locally_)
    {
        std::unique_lock<std::mutex> const lock(mu_);
        auto                               it = in_use_.find(ptr);
        if (it != in_use_.end())
        {
            return (*it).second.allocated_size;
        }
        return 0;
    }

    return allocator_->AllocatedSize(ptr);
}

int64_t allocator_tracking::AllocationId(const void* ptr) const
{
    if (track_sizes_locally_)
    {
        std::unique_lock<std::mutex> const lock(mu_);
        auto                               it = in_use_.find(ptr);
        if (it != in_use_.end())
        {
            return (*it).second.allocation_id;
        }
        return 0;
    }

    return allocator_->AllocationId(ptr);
}

std::optional<allocator_stats> allocator_tracking::GetStats() const
{
    return allocator_->GetStats();
}

bool allocator_tracking::ClearStats()
{
    return allocator_->ClearStats();
}

std::tuple<size_t, size_t, size_t> allocator_tracking::GetSizes() const
{
    size_t high_watermark;
    size_t total_bytes;
    size_t still_live_bytes;
    {
        std::unique_lock<std::mutex> const lock(mu_);
        high_watermark   = high_watermark_;
        total_bytes      = total_bytes_;
        still_live_bytes = allocated_;
    }
    return std::make_tuple(total_bytes, high_watermark, still_live_bytes);
}

std::vector<alloc_record> allocator_tracking::GetRecordsAndUnRef()
{
    bool                      should_delete;
    std::vector<alloc_record> allocations;
    {
        std::unique_lock<std::mutex> const lock(mu_);
        allocations.swap(allocations_);
        should_delete = UnRef();
    }
    if (should_delete)
    {
        delete this;
    }
    return allocations;
}

std::vector<alloc_record> allocator_tracking::GetCurrentRecords()
{
    std::vector<alloc_record> allocations;
    {
        std::unique_lock<std::mutex> const lock(mu_);

        std::copy(allocations_.begin(), allocations_.end(), std::back_inserter(allocations));
    }
    return allocations;
}

bool allocator_tracking::UnRef()
{
    MEMORY_CHECK_DEBUG(ref_ > 0);
    --ref_;
    return (ref_ == 0);
}

// ========== Enhanced Analytics Implementation ==========

memory_fragmentation_metrics allocator_tracking::GetFragmentationMetrics() const
{
    std::shared_lock<std::shared_mutex> lock(shared_mu_);

    // Check if cached metrics are still valid (updated within last 5 seconds)
    auto now = std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
                   .count();

    if (now - last_fragmentation_update_.load(std::memory_order_relaxed) < 5000000)  // 5 seconds
    {
        return cached_fragmentation_;
    }

    // Need to recalculate - upgrade to exclusive lock
    lock.unlock();
    std::unique_lock<std::shared_mutex> const exclusive_lock(shared_mu_);

    // Double-check pattern - another thread might have updated while we waited
    if (now - last_fragmentation_update_.load(std::memory_order_relaxed) < 5000000)
    {
        return cached_fragmentation_;
    }

    // Get current allocation statistics
    size_t total_allocated;
    size_t total_requested;
    {
        std::scoped_lock const stats_lock(mu_);
        total_allocated = allocated_;
        total_requested = 0;

        // Calculate total requested bytes from local tracking
        if (track_sizes_locally_)
        {
            for (const auto& [ptr, chunk] : in_use_)
            {
                total_requested += chunk.requested_size;
            }
        }
        else
        {
            total_requested = total_allocated;  // Best estimate when not tracking locally
        }
    }

    // For now, we can't get free block information from the underlying allocator
    // This would require extending the Allocator interface
    std::vector<size_t> const free_blocks;  // Empty for now

    cached_fragmentation_ =
        memory_fragmentation_metrics::calculate(total_allocated, total_requested, free_blocks);
    last_fragmentation_update_.store(now, std::memory_order_relaxed);

    return cached_fragmentation_;
}

atomic_timing_stats allocator_tracking::GetTimingStats() const noexcept
{
    return timing_stats_;
}

std::vector<enhanced_alloc_record> allocator_tracking::GetEnhancedRecords() const
{
    if (!enhanced_tracking_enabled_)
    {
        return {};  // Return empty vector if enhanced tracking is disabled
    }

    std::shared_lock<std::shared_mutex> const lock(shared_mu_);
    return enhanced_records_;  // Return copy
}

void allocator_tracking::SetLoggingLevel(tracking_log_level level) noexcept
{
    log_level_.store(level, std::memory_order_relaxed);

    if (level >= tracking_log_level::INFO)
    {
        MEMORY_LOG_INFO("allocator_tracking logging level changed to: {}", static_cast<int>(level));
    }
}

tracking_log_level allocator_tracking::GetLoggingLevel() const noexcept
{
    return log_level_.load(std::memory_order_relaxed);
}

void allocator_tracking::ResetTimingStats() noexcept
{
    timing_stats_.reset();

    if (log_level_.load(std::memory_order_relaxed) >= tracking_log_level::INFO)
    {
        MEMORY_LOG_INFO("allocator_tracking timing statistics reset");
    }
}

std::tuple<double, double, double> allocator_tracking::GetEfficiencyMetrics() const
{
    std::scoped_lock const lock(mu_);

    if (allocated_ == 0)
    {
        return std::make_tuple(1.0, 0.0, 1.0);  // Perfect efficiency when no allocations
    }

    double utilization_ratio = 1.0;
    double overhead_ratio    = 0.0;

    if (track_sizes_locally_)
    {
        size_t total_requested = 0;
        // Calculate actual utilization from tracked data
        for (const auto& [ptr, chunk] : in_use_)
        {
            total_requested += chunk.requested_size;
        }

        if (total_requested > 0)
        {
            utilization_ratio = static_cast<double>(total_requested) / allocated_;
            overhead_ratio    = static_cast<double>(allocated_ - total_requested) / allocated_;
        }
    }

    // Calculate efficiency score (weighted combination of metrics)
    double const efficiency_score = (utilization_ratio * 0.7) + ((1.0 - overhead_ratio) * 0.3);

    return std::make_tuple(utilization_ratio, overhead_ratio, efficiency_score);
}

std::string allocator_tracking::GenerateReport(bool include_allocations) const
{
    std::ostringstream report;

    // Header
    report << "=== Memory Memory Allocation Tracking Report ===\n";
    report << "Underlying Allocator: " << allocator_->Name() << "\n";
    report << "Enhanced Tracking: " << (enhanced_tracking_enabled_ ? "Enabled" : "Disabled")
           << "\n";
    report << "Local Size Tracking: " << (track_sizes_locally_ ? "Enabled" : "Disabled") << "\n";
    report << "Logging Level: " << static_cast<int>(log_level_.load(std::memory_order_relaxed))
           << "\n\n";

    // Memory Usage Summary
    auto [total_bytes, high_watermark, current_bytes] = GetSizes();
    report << "--- Memory Usage Summary ---\n";
    report << "Current Allocated: " << current_bytes << " bytes\n";
    report << "Peak Usage (High Watermark): " << high_watermark << " bytes\n";
    report << "Total Allocated (Cumulative): " << total_bytes << " bytes\n\n";

    // Performance Statistics
    auto timing = GetTimingStats();
    report << "--- Performance Statistics ---\n";
    report << "Total Allocations: " << timing.total_allocations << "\n";
    report << "Total Deallocations: " << timing.total_deallocations << "\n";
    report << "Average Allocation Time: " << std::fixed << std::setprecision(2)
           << timing.average_alloc_time_us() << " μs\n";
    report << "Average Deallocation Time: " << std::fixed << std::setprecision(2)
           << timing.average_dealloc_time_us() << " μs\n";

    auto min_alloc = timing.min_alloc_time_us.load(std::memory_order_relaxed);
    auto max_alloc = timing.max_alloc_time_us.load(std::memory_order_relaxed);
    if (min_alloc != UINT64_MAX)
    {
        report << "Min/Max Allocation Time: " << min_alloc << "/" << max_alloc << " μs\n";
    }

    auto min_dealloc = timing.min_dealloc_time_us.load(std::memory_order_relaxed);
    auto max_dealloc = timing.max_dealloc_time_us.load(std::memory_order_relaxed);
    if (min_dealloc != UINT64_MAX)
    {
        report << "Min/Max Deallocation Time: " << min_dealloc << "/" << max_dealloc << " μs\n";
    }
    report << "\n";

    // Efficiency Metrics
    auto [utilization, overhead, efficiency] = GetEfficiencyMetrics();
    report << "--- Efficiency Metrics ---\n";
    report << "Memory Utilization: " << std::fixed << std::setprecision(1) << (utilization * 100.0)
           << "%\n";
    report << "Overhead Ratio: " << std::fixed << std::setprecision(1) << (overhead * 100.0)
           << "%\n";
    report << "Overall Efficiency Score: " << std::fixed << std::setprecision(1)
           << (efficiency * 100.0) << "%\n\n";

    // Fragmentation Analysis
    auto fragmentation = GetFragmentationMetrics();
    report << "--- Fragmentation Analysis ---\n";
    report << "Total Free Blocks: " << fragmentation.total_free_blocks << "\n";
    report << "Largest Free Block: " << fragmentation.largest_free_block << " bytes\n";
    report << "Fragmentation Ratio: " << std::fixed << std::setprecision(1)
           << (fragmentation.fragmentation_ratio * 100.0) << "%\n";
    report << "External Fragmentation: " << std::fixed << std::setprecision(1)
           << fragmentation.external_fragmentation << "%\n";
    report << "Internal Fragmentation: " << std::fixed << std::setprecision(1)
           << fragmentation.internal_fragmentation << "%\n";
    report << "Wasted Bytes: " << fragmentation.wasted_bytes << " bytes\n\n";

    // Recommendations
    report << "--- Optimization Recommendations ---\n";
    if (efficiency < 0.8)
    {
        report << "• Low efficiency detected - consider reviewing allocation patterns\n";
    }
    if (overhead > 0.2)
    {
        report << "• High overhead ratio - consider using larger allocation sizes\n";
    }
    if (fragmentation.fragmentation_ratio > 0.3)
    {
        report << "• High fragmentation - consider using memory pools or different allocation "
                  "strategy\n";
    }
    if (timing.average_alloc_time_us() > 100.0)
    {
        report << "• Slow allocation performance - consider caching or pre-allocation strategies\n";
    }
    report << "\n";

    // Individual Allocation Details (if requested)
    if (include_allocations && enhanced_tracking_enabled_)
    {
        auto records = GetEnhancedRecords();
        report << "--- Individual Allocation Details ---\n";
        report << "Total Records: " << records.size() << "\n";

        if (!records.empty())
        {
            report << "Recent Allocations (last 10):\n";
            size_t const start = records.size() > 10 ? records.size() - 10 : 0;

            for (size_t i = start; i < records.size(); ++i)
            {
                const auto& record = records[i];
                report << "  [" << record.allocation_id << "] " << record.requested_bytes << "/"
                       << record.alloc_bytes << " bytes, " << "align=" << record.alignment << ", "
                       << "time=" << record.alloc_duration_us << "μs";

                if (!record.tag.empty())
                {
                    report << ", tag=" << record.tag;
                }

                if (record.source_file != nullptr)
                {
                    report << ", " << record.source_file << ":" << record.source_line;
                }

                report << "\n";
            }
        }
    }

    report << "=== End Report ===\n";
    return report.str();
}

}  // namespace memory
