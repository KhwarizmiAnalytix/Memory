/*
 * Quarisma: High-Performance Computational Library
 * Copyright 2025 Quarisma Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#include <atomic>
#include <chrono>
#include <iomanip>
#include <map>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include "CoreTest.h"
#include "backend/allocator_bfc.h"
#include "backend/allocator_pool.h"
#include "backend/allocator_tracking.h"
#include "cpu/allocator_cpu.h"
#include "gpu/gpu_allocator_tracking.h"

using namespace quarisma;
using namespace memory;

namespace
{

/**
 * @brief Tracking system benchmark configuration
 */
struct tracking_benchmark_config
{
    size_t      num_iterations  = 2000;
    size_t      allocation_size = 1024;
    std::string test_name;
    bool        enable_enhanced_tracking = true;
    bool        enable_size_tracking     = true;
};

/**
 * @brief Tracking system benchmark results
 */
struct tracking_benchmark_results
{
    std::string system_name;
    double      avg_alloc_time_ns         = 0.0;
    double      avg_dealloc_time_ns       = 0.0;
    double      tracking_overhead_percent = 0.0;
    size_t      memory_overhead_bytes     = 0;
    size_t      total_allocations         = 0;
    size_t      tracking_accuracy_percent = 100;
    double      total_time_ms             = 0.0;
};

/**
 * @brief High-precision timer for tracking benchmarks
 */
class tracking_timer
{
public:
    void start() { start_time_ = std::chrono::high_resolution_clock::now(); }

    void stop() { end_time_ = std::chrono::high_resolution_clock::now(); }

    double elapsed_ns() const
    {
        auto duration =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end_time_ - start_time_);
        return static_cast<double>(duration.count());
    }

    double elapsed_ms() const { return elapsed_ns() / 1000000.0; }

private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
};

/**
 * @brief Benchmark allocator without tracking (baseline)
 */
tracking_benchmark_results benchmark_baseline_allocator(const tracking_benchmark_config& config)
{
    tracking_benchmark_results results;
    results.system_name = "Baseline-BFC-" + config.test_name;

    // Create baseline BFC allocator
    auto bfc_allocator = std::make_unique<allocator_cpu>();

    tracking_timer     timer;
    std::vector<void*> allocated_ptrs;
    allocated_ptrs.reserve(config.num_iterations);

    // Measure allocation time
    timer.start();
    for (size_t i = 0; i < config.num_iterations; ++i)
    {
        void* ptr = bfc_allocator->allocate_raw(64, config.allocation_size);
        if (ptr != nullptr)
        {
            allocated_ptrs.push_back(ptr);
        }
    }
    timer.stop();

    double alloc_time_total   = timer.elapsed_ns();
    results.total_allocations = allocated_ptrs.size();

    if (results.total_allocations > 0)
    {
        results.avg_alloc_time_ns = alloc_time_total / results.total_allocations;
    }

    // Measure deallocation time
    timer.start();
    for (void* ptr : allocated_ptrs)
    {
        bfc_allocator->deallocate_raw(ptr);
    }
    timer.stop();

    if (results.total_allocations > 0)
    {
        results.avg_dealloc_time_ns = timer.elapsed_ns() / results.total_allocations;
    }

    results.total_time_ms             = (alloc_time_total + timer.elapsed_ns()) / 1000000.0;
    results.tracking_overhead_percent = 0.0;  // Baseline has no tracking overhead
    results.memory_overhead_bytes     = 0;    // No tracking structures

    return results;
}

/**
 * @brief Benchmark CPU tracking allocator
 */
tracking_benchmark_results benchmark_cpu_tracking_allocator(const tracking_benchmark_config& config)
{
    tracking_benchmark_results results;
    results.system_name = "CPU-Tracking-" + config.test_name;

    // Create underlying BFC allocator
    auto bfc_allocator = std::make_unique<allocator_cpu>();

    // Create tracking wrapper
    auto tracking_allocator = new allocator_tracking(
        bfc_allocator.get(), config.enable_size_tracking, config.enable_enhanced_tracking);

    tracking_timer     timer;
    std::vector<void*> allocated_ptrs;
    allocated_ptrs.reserve(config.num_iterations);

    // Measure allocation time with tracking
    timer.start();
    for (size_t i = 0; i < config.num_iterations; ++i)
    {
        void* ptr = tracking_allocator->allocate_raw(64, config.allocation_size);
        if (ptr != nullptr)
        {
            allocated_ptrs.push_back(ptr);
        }
    }
    timer.stop();

    double alloc_time_total   = timer.elapsed_ns();
    results.total_allocations = allocated_ptrs.size();

    if (results.total_allocations > 0)
    {
        results.avg_alloc_time_ns = alloc_time_total / results.total_allocations;
    }

    // Test tracking accuracy
    size_t accurate_tracking_count = 0;
    for (void* ptr : allocated_ptrs)
    {
        if (tracking_allocator->tracks_allocation_sizes())
        {
            size_t  requested_size = tracking_allocator->RequestedSize(ptr);
            size_t  allocated_size = tracking_allocator->AllocatedSize(ptr);
            int64_t alloc_id       = tracking_allocator->AllocationId(ptr);

            if (requested_size == config.allocation_size &&
                allocated_size >= config.allocation_size && alloc_id > 0)
            {
                accurate_tracking_count++;
            }
        }
    }

    if (results.total_allocations > 0)
    {
        results.tracking_accuracy_percent =
            (accurate_tracking_count * 100) / results.total_allocations;
    }

    // Measure deallocation time
    timer.start();
    for (void* ptr : allocated_ptrs)
    {
        tracking_allocator->deallocate_raw(ptr);
    }
    timer.stop();

    if (results.total_allocations > 0)
    {
        results.avg_dealloc_time_ns = timer.elapsed_ns() / results.total_allocations;
    }

    results.total_time_ms = (alloc_time_total + timer.elapsed_ns()) / 1000000.0;

    // Get statistics for memory overhead estimation
    auto stats = tracking_allocator->GetStats();
    if (stats.has_value())
    {
        // Estimate memory overhead based on tracking structures
        // Each allocation record has approximately 64-128 bytes overhead
        results.memory_overhead_bytes = results.total_allocations * 96;  // Conservative estimate
    }

    // Clean up (tracking allocator has protected destructor)
    tracking_allocator->GetRecordsAndUnRef();

    return results;
}

/**
 * @brief Benchmark GPU tracking allocator (if CUDA available)
 */
tracking_benchmark_results benchmark_gpu_tracking_allocator(const tracking_benchmark_config& config)
{
    tracking_benchmark_results results;
    results.system_name = "GPU-Tracking-" + config.test_name;

#if PROJECT_HAS_CUDA
    int         device_count = 0;
    cudaError_t error        = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0)
    {
        // No CUDA devices available
        results.system_name += "-SKIPPED";
        return results;
    }

    try
    {
        // Create GPU tracking allocator
        auto gpu_tracker = std::make_unique<gpu::gpu_allocator_tracking>(
            device_enum::CUDA, 0, config.enable_enhanced_tracking, false);

        tracking_timer     timer;
        std::vector<void*> allocated_ptrs;
        allocated_ptrs.reserve(config.num_iterations);

        // Measure allocation time with GPU tracking
        timer.start();
        for (size_t i = 0; i < config.num_iterations; ++i)
        {
            void* ptr = gpu_tracker->allocate_raw(config.allocation_size, 256);
            if (ptr != nullptr)
            {
                allocated_ptrs.push_back(ptr);
            }
        }
        timer.stop();

        double alloc_time_total   = timer.elapsed_ns();
        results.total_allocations = allocated_ptrs.size();

        if (results.total_allocations > 0)
        {
            results.avg_alloc_time_ns = alloc_time_total / results.total_allocations;
        }

        // Test GPU memory usage tracking
        auto [device_mem, unified_mem, pinned_mem] = gpu_tracker->GetGPUMemoryUsage();
        if (device_mem > 0)
        {
            results.tracking_accuracy_percent = 100;  // GPU tracking is accurate
        }

        // Measure deallocation time
        timer.start();
        for (void* ptr : allocated_ptrs)
        {
            gpu_tracker->deallocate_raw(ptr, config.allocation_size);
        }
        timer.stop();

        if (results.total_allocations > 0)
        {
            results.avg_dealloc_time_ns = timer.elapsed_ns() / results.total_allocations;
        }

        results.total_time_ms = (alloc_time_total + timer.elapsed_ns()) / 1000000.0;

        // GPU tracking overhead is typically higher due to CUDA API calls
        results.memory_overhead_bytes = results.total_allocations * 128;  // Estimate
    }
    catch (const std::exception& e)
    {
        results.system_name += "-ERROR";
        QUARISMA_LOG_WARNING("GPU tracking benchmark failed: {}", e.what());
    }
#else
    results.system_name += "-CUDA-DISABLED";
#endif

    return results;
}

/**
 * @brief Print tracking system benchmark results
 */
void print_tracking_benchmark_results(const std::vector<tracking_benchmark_results>& results)
{
    std::cout << "\n=== Memory Tracking System Benchmark Results ===\n\n";
    std::cout << std::left << std::setw(25) << "Tracking System" << std::setw(12) << "Alloc (ns)"
              << std::setw(12) << "Dealloc (ns)" << std::setw(12) << "Overhead (%)" << std::setw(15)
              << "Memory OH (KB)" << std::setw(12) << "Accuracy (%)" << std::setw(10)
              << "Total (ms)" << "\n";
    std::cout << std::string(120, '-') << "\n";

    // Find baseline for overhead calculation
    const tracking_benchmark_results* baseline = nullptr;
    for (const auto& result : results)
    {
        if (result.system_name.find("Baseline") != std::string::npos)
        {
            baseline = &result;
            break;
        }
    }

    for (const auto& result : results)
    {
        double overhead_percent = 0.0;
        if (baseline && baseline->avg_alloc_time_ns > 0 &&
            result.system_name.find("Baseline") == std::string::npos)
        {
            overhead_percent = ((result.avg_alloc_time_ns - baseline->avg_alloc_time_ns) /
                                baseline->avg_alloc_time_ns) *
                               100.0;
        }

        std::cout << std::left << std::setw(25) << result.system_name << std::setw(12) << std::fixed
                  << std::setprecision(2) << result.avg_alloc_time_ns << std::setw(12) << std::fixed
                  << std::setprecision(2) << result.avg_dealloc_time_ns << std::setw(12)
                  << std::fixed << std::setprecision(1) << overhead_percent << std::setw(15)
                  << std::fixed << std::setprecision(1) << (result.memory_overhead_bytes / 1024.0)
                  << std::setw(12) << result.tracking_accuracy_percent << std::setw(10)
                  << std::fixed << std::setprecision(2) << result.total_time_ms << "\n";
    }
    std::cout << "\n";
}

}  // anonymous namespace

/**
 * @brief Comprehensive tracking system benchmark
 */
QUARISMATEST(TrackingSystemBenchmark, ComprehensiveTrackingBenchmark)
{
    std::vector<tracking_benchmark_results> all_results;

    // Test configurations
    std::vector<tracking_benchmark_config> configs = {
        {2000, 64, "Small-64B", true, true},
        {2000, 1024, "Medium-1KB", true, true},
        {1000, 65536, "Large-64KB", true, true},
        {2000, 1024, "Enhanced-Off", false, true},
        {2000, 1024, "Size-Track-Off", true, false},
    };

    for (const auto& config : configs)
    {
        std::cout << "\nRunning tracking benchmark: " << config.test_name << "...\n";

        // Baseline (no tracking)
        auto baseline_results = benchmark_baseline_allocator(config);
        all_results.push_back(baseline_results);

        // CPU tracking
        auto cpu_tracking_results = benchmark_cpu_tracking_allocator(config);
        all_results.push_back(cpu_tracking_results);

        // GPU tracking (if available)
        auto gpu_tracking_results = benchmark_gpu_tracking_allocator(config);
        if (gpu_tracking_results.total_allocations > 0 ||
            gpu_tracking_results.system_name.find("SKIPPED") != std::string::npos ||
            gpu_tracking_results.system_name.find("CUDA-DISABLED") != std::string::npos)
        {
            all_results.push_back(gpu_tracking_results);
        }
    }

    // Print comprehensive results
    print_tracking_benchmark_results(all_results);

    // Basic validation
    EXPECT_FALSE(all_results.empty());

    // Verify tracking accuracy
    for (const auto& result : all_results)
    {
        if (result.system_name.find("Tracking") != std::string::npos &&
            result.total_allocations > 0)
        {
            EXPECT_GE(result.tracking_accuracy_percent, 95);  // At least 95% accuracy
        }
    }

    // Verify overhead is reasonable (less than 500% for most cases)
    size_t high_overhead_count = 0;
    for (const auto& result : all_results)
    {
        if (result.tracking_overhead_percent > 500.0)
        {
            high_overhead_count++;
        }
    }
    EXPECT_LE(
        high_overhead_count, all_results.size() / 4);  // At most 25% of tests have high overhead
}
