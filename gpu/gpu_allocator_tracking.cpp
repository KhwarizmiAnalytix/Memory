/*
 * Quarisma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * This file is part of Quarisma and is licensed under a dual-license model:
 *
 *   - Open-source License (GPLv3):
 *       Free for personal, academic, and research use under the terms of
 *       the GNU General Public License v3.0 or later.
 *
 *   - Commercial License:
 *       A commercial license is required for proprietary, closed-source,
 *       or SaaS usage. Contact us to obtain a commercial agreement.
 *
 * Original TensorFlow code:
 * Copyright 2015 The TensorFlow Authors. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0.
 *
 * Modifications for Quarisma:
 * - Adapted for Quarisma quantitative computing requirements
 * - Added high-performance GPU memory allocation optimizations
 * - Integrated CUDA-aware allocation strategies
 *
 * Contact: licensing@quarisma.co.uk
 * Website: https://www.quarisma.co.uk
 */

#include "gpu/gpu_allocator_tracking.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <utility>

#include "util/memory_exception.h"

namespace memory
{
namespace gpu
{

// ========== CUDA Error Info Implementation ==========

#if MEMORY_HAS_CUDA
cuda_error_info::cuda_error_info(
    cudaError_t cuda_error, const char* function_name, size_t size, int device) noexcept
    : error_code(static_cast<int>(cuda_error)),
      error_message(cudaGetErrorString(cuda_error)),
      cuda_function((function_name != nullptr) ? function_name : "unknown"),
      timestamp_us(std::chrono::duration_cast<std::chrono::microseconds>(
                       std::chrono::steady_clock::now().time_since_epoch())
                       .count()),
      attempted_size(size),
      device_index(device)
{
}
#endif

// ========== GPU Timing Stats Implementation ==========
// gpu_timing_stats methods are now provided by unified_memory_stats.h

// ========== Enhanced GPU Allocation Record Implementation ==========

enhanced_gpu_alloc_record::enhanced_gpu_alloc_record(
    size_t      requested_size,
    size_t      actual_size,
    size_t      align,
    device_enum dev_type,
    int         dev_index,
    int64_t     alloc_time,
    int64_t     alloc_id,
    std::string allocation_tag,
    const char* file,
    int         line,
    const char* func,
    void*       stream) noexcept
    : requested_bytes(requested_size),
      allocated_bytes(actual_size),
      alignment(align),
      device_type(dev_type),
      device_index(dev_index),
      allocation_id(alloc_id),
      alloc_timestamp_us(alloc_time),
      tag(std::move(allocation_tag)),
      source_file(file),
      source_line(line),
      function_name(func),
      cuda_stream(stream)
{
}

// ========== GPU Bandwidth Metrics Implementation ==========

gpu_bandwidth_metrics gpu_bandwidth_metrics::calculate(
    const gpu_device_info&                device_info,
    const std::vector<gpu_transfer_info>& transfer_records) noexcept
{
    gpu_bandwidth_metrics metrics;

    if (transfer_records.empty())
    {
        return metrics;  // Return default-initialized metrics
    }

    // Calculate basic transfer statistics
    metrics.total_bytes_transferred = 0;
    metrics.total_transfer_time_us  = 0;

    for (const auto& transfer : transfer_records)
    {
        metrics.total_bytes_transferred += transfer.bytes_transferred;
        metrics.total_transfer_time_us += transfer.duration_us;
    }

    // Calculate effective bandwidth
    if (metrics.total_transfer_time_us > 0)
    {
        double const total_time_seconds =
            static_cast<double>(metrics.total_transfer_time_us) / 1000000.0;
        double const total_gb =
            static_cast<double>(metrics.total_bytes_transferred) / (1024.0 * 1024.0 * 1024.0);
        metrics.effective_bandwidth_gbps = total_gb / total_time_seconds;
    }

    // Get theoretical peak bandwidth from device info
    metrics.peak_bandwidth_gbps = device_info.memory_bandwidth_gb_per_sec;

    // Calculate utilization percentage
    if (metrics.peak_bandwidth_gbps > 0.0)
    {
        metrics.utilization_percentage =
            (metrics.effective_bandwidth_gbps / metrics.peak_bandwidth_gbps) * 100.0;
        metrics.utilization_percentage = std::min(metrics.utilization_percentage, 100.0);
    }

    // For now, assume perfect coalescing (would need detailed memory access analysis)
    metrics.coalesced_accesses   = transfer_records.size();
    metrics.uncoalesced_accesses = 0;
    metrics.memory_efficiency    = 1.0;

    return metrics;
}

// ========== GPU Allocator Tracking Implementation ==========

gpu_allocator_tracking::gpu_allocator_tracking(
    device_enum device_type,
    int         device_index,
    bool        enable_enhanced_tracking,
    bool        enable_bandwidth_tracking)
    : device_type_(device_type),
      device_index_(device_index),
      enhanced_tracking_enabled_(enable_enhanced_tracking),
      bandwidth_tracking_enabled_(enable_bandwidth_tracking)
{
    // Initialize GPU timing stats
    gpu_timing_stats_.reset();

    // Get device information
    auto& device_manager = gpu_device_manager::instance();
    auto  devices        = device_manager.get_available_devices();

    // Find our target device
    auto device_it = std::find_if(
        devices.begin(),
        devices.end(),
        [this](const auto& device)
        { return device.device_type == device_type_ && device.device_index == device_index_; });

    if (device_it != devices.end())
    {
        device_info_ = *device_it;
    }
    else
    {
        MEMORY_THROW(
            "GPU device not found: type={}, index={}",
            static_cast<int>(device_type_),
            device_index_);
    }

    // Reserve space for GPU records to minimize reallocations
    if (enhanced_tracking_enabled_)
    {
        gpu_records_.reserve(1000);  // Reserve space for 1000 initial records
    }

    // Initialize CUDA events if CUDA is enabled
    InitializeCUDAEvents();

    // Log initialization if enabled
    if (gpu_log_level_.load(std::memory_order_relaxed) >= gpu_tracking_log_level::INFO)
    {
        MEMORY_LOG_INFO(
            "gpu_allocator_tracking initialized: device={}, enhanced={}, bandwidth={}",
            device_info_.name,
            enhanced_tracking_enabled_,
            bandwidth_tracking_enabled_);
    }
}

gpu_allocator_tracking::~gpu_allocator_tracking()
{
    // Log destruction if enabled
    if (gpu_log_level_.load(std::memory_order_relaxed) >= gpu_tracking_log_level::INFO)
    {
        MEMORY_LOG_INFO("gpu_allocator_tracking destroying: device={}", device_info_.name);
    }

    // Cleanup CUDA events
    CleanupCUDAEvents();
}

void gpu_allocator_tracking::InitializeCUDAEvents()
{
#if MEMORY_HAS_CUDA
    if (device_type_ == device_enum::CUDA)
    {
        // Set device context
        cudaError_t result = cudaSetDevice(device_index_);
        if (result == cudaSuccess)
        {
            // Create CUDA events for timing
            result = cudaEventCreate(&start_event_);
            if (result == cudaSuccess)
            {
                result = cudaEventCreate(&end_event_);
                if (result == cudaSuccess)
                {
                    cuda_events_initialized_ = true;
                }
                else
                {
                    cudaEventDestroy(start_event_);
                }
            }
        }

        if (!cuda_events_initialized_)
        {
            MEMORY_LOG_WARNING(
                "Failed to initialize CUDA events for timing: {}", cudaGetErrorString(result));
        }
    }
#endif
}

void gpu_allocator_tracking::CleanupCUDAEvents() noexcept
{
#if MEMORY_HAS_CUDA
    if (cuda_events_initialized_)
    {
        cudaEventDestroy(start_event_);
        cudaEventDestroy(end_event_);
        cuda_events_initialized_ = false;
    }
#endif
}

// ========== Core GPU Allocation Methods ==========

void* gpu_allocator_tracking::allocate_raw(
    size_t                           bytes,
    size_t                           alignment,
    std::shared_ptr<gpu_memory_pool> pool,
    const std::string&               tag,
    void*                            stream)
{
    if (bytes == 0)
    {
        return nullptr;
    }

    // Start timing for performance analysis
    auto start_time = std::chrono::steady_clock::now();

    // Log allocation attempt if enabled
    auto current_log_level = gpu_log_level_.load(std::memory_order_relaxed);
    if (current_log_level >= gpu_tracking_log_level::TRACE)
    {
        MEMORY_LOG_INFO_DEBUG(
            "gpu_allocator_tracking::allocate_raw: bytes={}, alignment={}, device={}, tag={}",
            bytes,
            alignment,
            device_index_,
            tag);
    }

#if MEMORY_HAS_CUDA
    // Record CUDA event for GPU-side timing if available
    if (cuda_events_initialized_ && (stream != nullptr))
    {
        cudaEventRecord(start_event_, static_cast<cudaStream_t>(stream));
    }
#endif

    // Perform actual GPU allocation using the appropriate allocator
    void*           ptr = nullptr;  //NOLINT
    cuda_error_info error_info;

    try
    {
        if (pool)
        {
            // Use memory pool allocation
            auto block = pool->allocate(bytes, device_type_, device_index_);
            ptr        = block.ptr;
        }
        else
        {
            // Use direct CUDA allocation (replacing gpu_allocator)
#if MEMORY_HAS_CUDA
            if (device_type_ == device_enum::CUDA || device_type_ == device_enum::HIP)
            {
                cudaError_t result = cudaSetDevice(device_index_);
                if (result == cudaSuccess)
                {
                    result = cudaMalloc(&ptr, bytes);
                    if (result != cudaSuccess)
                    {
                        ptr = nullptr;
                    }
                }
            }
#endif
        }
    }
    catch (const std::exception& e)
    {
        // Capture allocation failure information
#if MEMORY_HAS_CUDA
        if (device_type_ == device_enum::CUDA)
        {
            cudaError_t const cuda_error = cudaGetLastError();
            error_info =
                cuda_error_info(cuda_error, "gpu_allocator::allocate", bytes, device_index_);
        }
#endif

        MEMORY_LOG_ERROR(
            "GPU allocation failed: {}, bytes={}, device={}", e.what(), bytes, device_index_);

        // Update failure statistics
        gpu_timing_stats_.total_allocations.fetch_add(1, std::memory_order_relaxed);

        return nullptr;
    }

    // Calculate allocation timing
    auto end_time = std::chrono::steady_clock::now();
    auto duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

#if MEMORY_HAS_CUDA
    // Get GPU-side timing if CUDA events are available
    if (cuda_events_initialized_ && (stream != nullptr))
    {
        cudaEventRecord(end_event_, static_cast<cudaStream_t>(stream));
        cudaEventSynchronize(end_event_);

        float             gpu_time_ms = 0.0f;
        cudaError_t const result = cudaEventElapsedTime(&gpu_time_ms, start_event_, end_event_);
        if (result == cudaSuccess)
        {
            // Use GPU timing if available (more accurate for GPU operations)
            duration_us = static_cast<uint64_t>(gpu_time_ms * 1000.0f);
        }
    }
#endif

    // Update timing statistics
    gpu_timing_stats_.total_allocations.fetch_add(1, std::memory_order_relaxed);
    gpu_timing_stats_.total_alloc_time_us.fetch_add(duration_us, std::memory_order_relaxed);

    // Update min/max timing with lock-free compare-and-swap
    auto current_min = gpu_timing_stats_.min_alloc_time_us.load(std::memory_order_relaxed);
    while (duration_us < current_min && !gpu_timing_stats_.min_alloc_time_us.compare_exchange_weak(
                                            current_min, duration_us, std::memory_order_relaxed))
    {
        // Retry if another thread updated min_alloc_time_us
    }

    auto current_max = gpu_timing_stats_.max_alloc_time_us.load(std::memory_order_relaxed);
    while (duration_us > current_max && !gpu_timing_stats_.max_alloc_time_us.compare_exchange_weak(
                                            current_max, duration_us, std::memory_order_relaxed))
    {
        // Retry if another thread updated max_alloc_time_us
    }

    // Update memory usage statistics
    UpdateMemoryUsage(bytes, true);

    // Add enhanced record if enabled
    if (enhanced_tracking_enabled_ && (ptr != nullptr))
    {
        std::unique_lock<std::shared_mutex> const enhanced_lock(gpu_shared_mu_);
        int64_t const                             allocation_id =
            next_gpu_allocation_id_.fetch_add(1, std::memory_order_relaxed);

        gpu_records_.emplace_back(
            bytes,          // requested_bytes
            bytes,          // allocated_bytes (assume same for now)
            alignment,      // alignment
            device_type_,   // device_type
            device_index_,  // device_index
            std::chrono::duration_cast<std::chrono::microseconds>(end_time.time_since_epoch())
                .count(),
            allocation_id,  // allocation_id
            tag,            // tag
            nullptr,        // source_file (would be filled by macro)
            0,              // source_line
            nullptr,        // function_name
            stream          // cuda_stream
        );
        gpu_records_.back().alloc_duration_us = duration_us;
        gpu_records_.back().error_info        = error_info;
    }

    // Log successful allocation
    if (current_log_level >= gpu_tracking_log_level::TRACE)
    {
        MEMORY_LOG_INFO_DEBUG(
            "gpu_allocator_tracking::allocate_raw success: ptr={}, bytes={}, time={}μs",
            ptr,
            bytes,
            duration_us);
    }

    return ptr;
}

void gpu_allocator_tracking::deallocate_raw(void* ptr, size_t bytes, void* stream)
{
    // Deallocating a null ptr is a no-op
    if (ptr == nullptr)
    {
        return;
    }

    // Start timing for performance analysis
    auto start_time = std::chrono::steady_clock::now();

    // Log deallocation attempt if enabled
    auto current_log_level = gpu_log_level_.load(std::memory_order_relaxed);
    if (current_log_level >= gpu_tracking_log_level::TRACE)
    {
        MEMORY_LOG_INFO_DEBUG(
            "gpu_allocator_tracking::deallocate_raw: ptr={}, bytes={}, device={}",
            ptr,
            bytes,
            device_index_);
    }

#if MEMORY_HAS_CUDA
    // Record CUDA event for GPU-side timing if available
    if (cuda_events_initialized_ && (stream != nullptr))
    {
        cudaEventRecord(start_event_, static_cast<cudaStream_t>(stream));
    }
#endif

    // Find allocation record for enhanced tracking
    int64_t allocation_id = 0;
    if (enhanced_tracking_enabled_)
    {
        std::shared_lock<std::shared_mutex> const enhanced_lock(gpu_shared_mu_);
        auto                                      record_it = std::find_if(
            gpu_records_.begin(),
            gpu_records_.end(),
            [](const auto& record)
            {
                return record.dealloc_timestamp_us == 0;  // Not yet deallocated
            });

        if (record_it != gpu_records_.end())
        {
            allocation_id = record_it->allocation_id;
        }
    }

    // Perform actual GPU deallocation using direct CUDA calls
#if MEMORY_HAS_CUDA
    if (device_type_ == device_enum::CUDA || device_type_ == device_enum::HIP)
    {
        cudaError_t result = cudaSetDevice(device_index_);
        if (result != cudaSuccess)
        {
            throw std::runtime_error(
                "Failed to set CUDA device: " + std::string(cudaGetErrorString(result)));
        }
        result = cudaFree(ptr);
        if (result != cudaSuccess)
        {
            throw std::runtime_error(
                "Failed to free GPU memory: " + std::string(cudaGetErrorString(result)));
        }
    }
    else
#endif
    {
        throw std::runtime_error("GPU deallocation not supported without CUDA");
    }

    // Calculate deallocation timing
    auto end_time = std::chrono::steady_clock::now();
    auto duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

#if MEMORY_HAS_CUDA
    // Get GPU-side timing if CUDA events are available
    if (cuda_events_initialized_ && (stream != nullptr))
    {
        cudaEventRecord(end_event_, static_cast<cudaStream_t>(stream));
        cudaEventSynchronize(end_event_);

        float             gpu_time_ms = 0.0f;
        cudaError_t const result = cudaEventElapsedTime(&gpu_time_ms, start_event_, end_event_);
        if (result == cudaSuccess)
        {
            duration_us = static_cast<uint64_t>(gpu_time_ms * 1000.0f);
        }
    }
#endif

    // Update timing statistics
    gpu_timing_stats_.total_deallocations.fetch_add(1, std::memory_order_relaxed);
    gpu_timing_stats_.total_dealloc_time_us.fetch_add(duration_us, std::memory_order_relaxed);

    // Update min/max timing
    auto current_min = gpu_timing_stats_.min_dealloc_time_us.load(std::memory_order_relaxed);
    while (duration_us < current_min &&
           !gpu_timing_stats_.min_dealloc_time_us.compare_exchange_weak(
               current_min, duration_us, std::memory_order_relaxed))
    {
        // Retry if another thread updated min_dealloc_time_us
    }

    auto current_max = gpu_timing_stats_.max_dealloc_time_us.load(std::memory_order_relaxed);
    while (duration_us > current_max &&
           !gpu_timing_stats_.max_dealloc_time_us.compare_exchange_weak(
               current_max, duration_us, std::memory_order_relaxed))
    {
        // Retry if another thread updated max_dealloc_time_us
    }

    // Update memory usage statistics
    UpdateMemoryUsage(bytes, false);

    // Update enhanced record with deallocation timing if enabled and allocation_id is valid
    if (enhanced_tracking_enabled_ && allocation_id > 0)
    {
        std::unique_lock<std::shared_mutex> const enhanced_lock(gpu_shared_mu_);
        auto                                      record_it = std::find_if(
            gpu_records_.begin(),
            gpu_records_.end(),
            [allocation_id](const auto& record)
            { return record.allocation_id == allocation_id && record.dealloc_timestamp_us == 0; });

        if (record_it != gpu_records_.end())
        {
            record_it->dealloc_duration_us = duration_us;
            record_it->dealloc_timestamp_us =
                std::chrono::duration_cast<std::chrono::microseconds>(end_time.time_since_epoch())
                    .count();
        }
    }

    // Log successful deallocation
    if (current_log_level >= gpu_tracking_log_level::TRACE)
    {
        MEMORY_LOG_INFO_DEBUG(
            "gpu_allocator_tracking::deallocate_raw success: ptr={}, time={}μs", ptr, duration_us);
    }
}

// ========== GPU Analytics and Metrics Methods ==========

gpu_bandwidth_metrics gpu_allocator_tracking::GetBandwidthMetrics() const
{
    std::shared_lock<std::shared_mutex> const lock(gpu_shared_mu_);

    // For now, return basic metrics - would need detailed transfer tracking for full implementation
    gpu_bandwidth_metrics metrics;
    metrics.peak_bandwidth_gbps = device_info_.memory_bandwidth_gb_per_sec;

    // Calculate basic effective bandwidth from allocation timing
    auto total_allocs  = gpu_timing_stats_.total_allocations.load(std::memory_order_relaxed);
    auto total_time_us = gpu_timing_stats_.total_alloc_time_us.load(std::memory_order_relaxed);

    if (total_allocs > 0 && total_time_us > 0)
    {
        // Estimate based on allocation patterns (simplified)
        size_t const estimated_bytes    = total_device_memory_.load(std::memory_order_relaxed);
        double const total_time_seconds = static_cast<double>(total_time_us) / 1000000.0;
        double const total_gb = static_cast<double>(estimated_bytes) / (1024.0 * 1024.0 * 1024.0);
        metrics.effective_bandwidth_gbps = total_gb / total_time_seconds;

        if (metrics.peak_bandwidth_gbps > 0.0)
        {
            metrics.utilization_percentage =
                (metrics.effective_bandwidth_gbps / metrics.peak_bandwidth_gbps) * 100.0;
            metrics.utilization_percentage = std::min(metrics.utilization_percentage, 100.0);
        }
    }

    return metrics;
}

atomic_timing_stats gpu_allocator_tracking::GetGPUTimingStats() const noexcept
{
    // Copy constructor will handle atomic value loading safely
    return gpu_timing_stats_;
}

std::vector<enhanced_gpu_alloc_record> gpu_allocator_tracking::GetEnhancedGPURecords() const
{
    std::shared_lock<std::shared_mutex> const lock(gpu_shared_mu_);
    return gpu_records_;  // Return copy
}

void gpu_allocator_tracking::SetGPULoggingLevel(gpu_tracking_log_level level) noexcept
{
    gpu_log_level_.store(level, std::memory_order_relaxed);
}

gpu_tracking_log_level gpu_allocator_tracking::GetGPULoggingLevel() const noexcept
{
    return gpu_log_level_.load(std::memory_order_relaxed);
}

void gpu_allocator_tracking::ResetGPUTimingStats() noexcept
{
    gpu_timing_stats_.reset();
}

std::tuple<double, double, double> gpu_allocator_tracking::GetGPUEfficiencyMetrics() const
{
    std::shared_lock<std::shared_mutex> const lock(gpu_shared_mu_);

    // Calculate memory coalescing efficiency (simplified - would need detailed access pattern analysis)
    double const coalescing_efficiency = 0.85;  // Assume reasonable coalescing for now

    // Calculate memory utilization
    size_t const total_allocated = total_device_memory_.load(std::memory_order_relaxed) +
                                   total_unified_memory_.load(std::memory_order_relaxed) +
                                   total_pinned_memory_.load(std::memory_order_relaxed);

    double memory_utilization = 0.0;
    if (device_info_.total_memory_bytes > 0)
    {
        memory_utilization = static_cast<double>(total_allocated) /
                             static_cast<double>(device_info_.total_memory_bytes);
        memory_utilization = std::min(memory_utilization, 1.0);
    }

    // Calculate overall GPU efficiency score
    double const gpu_efficiency_score = (coalescing_efficiency + memory_utilization) / 2.0;

    return std::make_tuple(coalescing_efficiency, memory_utilization, gpu_efficiency_score);
}

std::string gpu_allocator_tracking::GenerateGPUReport(
    bool include_allocations, bool include_cuda_info) const
{
    std::shared_lock<std::shared_mutex> const lock(gpu_shared_mu_);
    std::ostringstream                        report;

    report << "=== GPU Memory Allocation Tracking Report ===\n\n";

    // Device Information
    report << "GPU Device Information:\n";
    report << "  Name: " << device_info_.name << "\n";
    report << "  Type: " << (device_type_ == device_enum::CUDA ? "CUDA" : "Other") << "\n";
    report << "  Index: " << device_index_ << "\n";
    report << "  Total Memory: " << (device_info_.total_memory_bytes / (1024ULL)) << " MB\n";
    report << "  Memory Bandwidth: " << device_info_.memory_bandwidth_gb_per_sec << " GB/s\n\n";

    // Memory Usage Summary
    auto device_mem  = total_device_memory_.load(std::memory_order_relaxed);
    auto unified_mem = total_unified_memory_.load(std::memory_order_relaxed);
    auto pinned_mem  = total_pinned_memory_.load(std::memory_order_relaxed);

    report << "Memory Usage Summary:\n";
    report << "  Device Memory: " << (device_mem / (1024ULL)) << " MB\n";
    report << "  Unified Memory: " << (unified_mem / (1024ULL)) << " MB\n";
    report << "  Pinned Memory: " << (pinned_mem / (1024ULL)) << " MB\n";
    report << "  Total Allocated: " << ((device_mem + unified_mem + pinned_mem) / (1024ULL))
           << " MB\n\n";

    // Performance Statistics
    auto stats = GetGPUTimingStats();
    report << "Performance Statistics:\n";
    report << "  Total Allocations: " << stats.total_allocations.load(std::memory_order_relaxed)
           << "\n";
    report << "  Total Deallocations: " << stats.total_deallocations.load(std::memory_order_relaxed)
           << "\n";
    report << "  Average Allocation Time: " << std::fixed << std::setprecision(2)
           << stats.average_alloc_time_us() << " μs\n";
    report << "  Average Deallocation Time: " << std::fixed << std::setprecision(2)
           << stats.average_dealloc_time_us() << " μs\n\n";

    // Efficiency Metrics
    auto [coalescing, utilization, efficiency] = GetGPUEfficiencyMetrics();
    report << "Efficiency Metrics:\n";
    report << "  Memory Coalescing: " << std::fixed << std::setprecision(1) << (coalescing * 100.0)
           << "%\n";
    report << "  Memory Utilization: " << std::fixed << std::setprecision(1)
           << (utilization * 100.0) << "%\n";
    report << "  Overall GPU Efficiency: " << std::fixed << std::setprecision(1)
           << (efficiency * 100.0) << "%\n\n";

    // CUDA-specific information
    if (include_cuda_info && device_type_ == device_enum::CUDA)
    {
        report << "CUDA-Specific Information:\n";
#if MEMORY_HAS_CUDA
        report << "  CUDA Events Initialized: " << (cuda_events_initialized_ ? "Yes" : "No")
               << "\n";
#else
        report << "  CUDA Events Initialized: No (CUDA disabled)\n";
#endif
        report << "  Enhanced Tracking: " << (enhanced_tracking_enabled_ ? "Enabled" : "Disabled")
               << "\n";
        report << "  Bandwidth Tracking: " << (bandwidth_tracking_enabled_ ? "Enabled" : "Disabled")
               << "\n\n";
    }

    // Individual allocation details
    if (include_allocations && !gpu_records_.empty())
    {
        report << "Recent Allocations (last " << std::min(size_t(10), gpu_records_.size())
               << "):\n";
        size_t count = 0;
        for (auto it = gpu_records_.rbegin(); it != gpu_records_.rend() && count < 10;
             ++it, ++count)
        {
            const auto& record = *it;
            report << "  [" << record.allocation_id << "] " << (record.allocated_bytes / 1024)
                   << " KB, " << record.alloc_duration_us << " μs";
            if (!record.tag.empty())
            {
                report << ", tag: " << record.tag;
            }
            report << "\n";
        }
        report << "\n";
    }

    return report.str();
}

gpu_device_info gpu_allocator_tracking::GetDeviceInfo() const noexcept
{
    return device_info_;
}

std::tuple<size_t, size_t, size_t> gpu_allocator_tracking::GetGPUMemoryUsage() const
{
    return std::make_tuple(
        total_device_memory_.load(std::memory_order_relaxed),
        total_unified_memory_.load(std::memory_order_relaxed),
        total_pinned_memory_.load(std::memory_order_relaxed));
}

// ========== Private Helper Methods ==========

void gpu_allocator_tracking::UpdateMemoryUsage(size_t bytes, bool is_allocation)
{
    if (is_allocation)
    {
        // For now, assume all allocations are device memory
        // In a full implementation, would need to detect memory type
        auto new_device_mem =
            total_device_memory_.fetch_add(bytes, std::memory_order_relaxed) + bytes;

        // Update peak usage
        auto current_peak = peak_device_memory_.load(std::memory_order_relaxed);
        while (new_device_mem > current_peak &&
               !peak_device_memory_.compare_exchange_weak(
                   current_peak, new_device_mem, std::memory_order_relaxed))
        {
            // Retry if another thread updated peak
        }
    }
    else
    {
        // Deallocation
        total_device_memory_.fetch_sub(bytes, std::memory_order_relaxed);
    }
}

void gpu_allocator_tracking::LogGPUOperation(
    MEMORY_UNUSED const std::string& operation, MEMORY_UNUSED const std::string& details) const
{
    auto current_log_level = gpu_log_level_.load(std::memory_order_relaxed);
    if (current_log_level >= gpu_tracking_log_level::DEBUG_LEVEL)
    {
        MEMORY_LOG_INFO_DEBUG("GPU {}: {}", operation, details);
    }
}

}  // namespace gpu
}  // namespace memory
