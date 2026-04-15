/*
 * Quarisma: High-Performance Computational Library
 * Copyright 2025 Quarisma Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */
#if 1
#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#include "MemoryTest.h"
#include "gpu/allocator_gpu.h"
#include "gpu/cuda_caching_allocator.h"

#if PROJECT_HAS_CUDA
#include <cuda_runtime.h>
#endif

using namespace memory;
using namespace memory::gpu;

namespace
{

/**
 * @brief GPU benchmark configuration
 */
struct gpu_benchmark_config
{
    size_t              num_threads      = 1;
    size_t              num_iterations   = 5000;
    size_t              min_alloc_size   = 16;
    size_t              max_alloc_size   = 16 * 1024 * 1024;  // 16MB
    double              allocation_ratio = 0.6;  // More allocations than deallocations initially
    std::string         test_name;
    std::vector<size_t> size_classes = {16, 256, 4096, 65536, 1048576, 16777216};  // 16B to 16MB
};

/**
 * @brief GPU benchmark results with detailed metrics
 */
struct gpu_benchmark_results
{
    std::string              allocator_name;
    double                   total_time_ms              = 0.0;
    double                   avg_alloc_time_ns          = 0.0;
    double                   avg_dealloc_time_ns        = 0.0;
    double                   throughput_mb_per_sec      = 0.0;
    size_t                   total_allocations          = 0;
    size_t                   total_deallocations        = 0;
    size_t                   peak_memory_usage          = 0;
    double                   memory_fragmentation_ratio = 0.0;
    std::map<size_t, double> size_class_performance;  // ns per allocation by size class
    size_t                   allocation_failures = 0;
};

/**
 * @brief High-precision GPU timer using CUDA events
 */
class gpu_timer
{
public:
    gpu_timer()
    {
#if PROJECT_HAS_CUDA
        cudaEventCreate(&start_event_);
        cudaEventCreate(&stop_event_);
#endif
    }

    ~gpu_timer()
    {
#if PROJECT_HAS_CUDA
        cudaEventDestroy(start_event_);
        cudaEventDestroy(stop_event_);
#endif
    }

    void start()
    {
#if PROJECT_HAS_CUDA
        cudaEventRecord(start_event_);
#endif
        cpu_start_ = std::chrono::high_resolution_clock::now();
    }

    void stop()
    {
        cpu_end_ = std::chrono::high_resolution_clock::now();
#if PROJECT_HAS_CUDA
        cudaEventRecord(stop_event_);
        cudaEventSynchronize(stop_event_);
#endif
    }

    double elapsed_ms() const
    {
#if PROJECT_HAS_CUDA
        float gpu_time_ms = 0.0f;
        cudaEventElapsedTime(&gpu_time_ms, start_event_, stop_event_);
        return static_cast<double>(gpu_time_ms);
#else
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(cpu_end_ - cpu_start_);
        return duration.count() / 1000000.0;
#endif
    }

    double elapsed_ns() const { return elapsed_ms() * 1000000.0; }

private:
#if PROJECT_HAS_CUDA
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;
#endif
    std::chrono::high_resolution_clock::time_point cpu_start_;
    std::chrono::high_resolution_clock::time_point cpu_end_;
};

/**
 * @brief Thread-safe GPU benchmark statistics collector
 */
class gpu_benchmark_stats
{
public:
    void record_allocation(double time_ns, size_t size, bool success)
    {
        if (success)
        {
            // Use compare_exchange_weak for atomic double operations
            double current_time = total_alloc_time_.load();
            while (!total_alloc_time_.compare_exchange_weak(current_time, current_time + time_ns))
            {
                // Retry until successful
            }
            total_allocations_.fetch_add(1);
            total_bytes_allocated_.fetch_add(size);

            // Update size class performance
            std::lock_guard<std::mutex> lock(size_class_mutex_);
            size_class_times_[size].push_back(time_ns);
        }
        else
        {
            allocation_failures_.fetch_add(1);
        }
    }

    void record_deallocation(double time_ns)
    {
        // Use compare_exchange_weak for atomic double operations
        double current_time = total_dealloc_time_.load();
        while (!total_dealloc_time_.compare_exchange_weak(current_time, current_time + time_ns))
        {
            // Retry until successful
        }
        total_deallocations_.fetch_add(1);
    }

    gpu_benchmark_results get_results(const std::string& allocator_name) const
    {
        gpu_benchmark_results results;
        results.allocator_name      = allocator_name;
        results.total_allocations   = total_allocations_.load();
        results.total_deallocations = total_deallocations_.load();
        results.allocation_failures = allocation_failures_.load();

        if (results.total_allocations > 0)
        {
            results.avg_alloc_time_ns = total_alloc_time_.load() / results.total_allocations;
        }

        if (results.total_deallocations > 0)
        {
            results.avg_dealloc_time_ns = total_dealloc_time_.load() / results.total_deallocations;
        }

        size_t total_bytes = total_bytes_allocated_.load();
        if (total_bytes > 0 && total_time_ms_ > 0)
        {
            results.throughput_mb_per_sec =
                (total_bytes / (1024.0 * 1024.0)) / (total_time_ms_ / 1000.0);
        }

        // Calculate size class performance
        std::lock_guard<std::mutex> lock(size_class_mutex_);
        for (const auto& [size, times] : size_class_times_)
        {
            if (!times.empty())
            {
                double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
                results.size_class_performance[size] = avg_time;
            }
        }

        return results;
    }

    void set_total_time(double time_ms) { total_time_ms_ = time_ms; }

private:
    std::atomic<double> total_alloc_time_{0.0};
    std::atomic<double> total_dealloc_time_{0.0};
    std::atomic<size_t> total_allocations_{0};
    std::atomic<size_t> total_deallocations_{0};
    std::atomic<size_t> total_bytes_allocated_{0};
    std::atomic<size_t> allocation_failures_{0};
    double              total_time_ms_{0.0};

    mutable std::mutex                    size_class_mutex_;
    std::map<size_t, std::vector<double>> size_class_times_;
};

/**
 * @brief GPU allocation pattern generator with realistic workload simulation
 */
class gpu_allocation_pattern_generator
{
public:
    explicit gpu_allocation_pattern_generator(const gpu_benchmark_config& config)
        : config_(config),
          rng_(std::random_device{}()),
          size_dist_(0, config.size_classes.size() - 1),
          action_dist_(0.0, 1.0)
    {
    }

    size_t next_allocation_size()
    {
        // Use size classes for more realistic allocation patterns
        size_t class_idx = size_dist_(rng_);
        return config_.size_classes[class_idx];
    }

    bool should_allocate() { return action_dist_(rng_) < config_.allocation_ratio; }

private:
    gpu_benchmark_config                   config_;
    std::mt19937                           rng_;
    std::uniform_int_distribution<size_t>  size_dist_;
    std::uniform_real_distribution<double> action_dist_;
};

/**
 * @brief Generic GPU allocator interface for benchmarking
 */
class gpu_allocator_interface
{
public:
    virtual ~gpu_allocator_interface()                     = default;
    virtual void*       allocate(size_t size)              = 0;
    virtual void        deallocate(void* ptr, size_t size) = 0;
    virtual std::string name() const                       = 0;
};

/**
 * @brief Wrapper for allocator_gpu
 */
class gpu_allocator_wrapper : public gpu_allocator_interface
{
public:
    explicit gpu_allocator_wrapper(std::unique_ptr<allocator_gpu> allocator)
        : allocator_(std::move(allocator))
    {
    }

    void* allocate(size_t size) override
    {
        return allocator_->allocate_raw(256, size);  // 256-byte alignment for GPU
    }

    void deallocate(void* ptr, size_t size) override
    {
        (void)size;  // Suppress unused parameter warning
        allocator_->deallocate_raw(ptr);
    }

    std::string name() const override { return allocator_->Name(); }

private:
    std::unique_ptr<allocator_gpu> allocator_;
};

/**
 * @brief Wrapper for cuda_caching_allocator
 */
class cuda_caching_allocator_wrapper : public gpu_allocator_interface
{
public:
    explicit cuda_caching_allocator_wrapper(int device_id, size_t cache_size = 256 * 1024 * 1024)
        : allocator_(device_id, cache_size)
    {
    }

    void* allocate(size_t size) override { return allocator_.allocate(size); }

    void deallocate(void* ptr, size_t size) override { allocator_.deallocate(ptr, size); }

    std::string name() const override { return "CUDA-Caching-Allocator"; }

private:
    cuda_caching_allocator allocator_;
};

/**
 * @brief Direct CUDA allocator wrapper for comparison
 */
class direct_cuda_allocator_wrapper : public gpu_allocator_interface
{
public:
    void* allocate(size_t size) override
    {
#if PROJECT_HAS_CUDA
        void*       ptr   = nullptr;
        cudaError_t error = cudaMalloc(&ptr, size);
        return (error == cudaSuccess) ? ptr : nullptr;
#else
        (void)size;
        return nullptr;
#endif
    }

    void deallocate(void* ptr, size_t size) override
    {
        (void)size;
#if PROJECT_HAS_CUDA
        if (ptr != nullptr)
        {
            cudaFree(ptr);
        }
#endif
    }

    std::string name() const override { return "Direct-CUDA-Allocator"; }
};

/**
 * @brief Worker thread for GPU benchmark
 */
void gpu_benchmark_worker_thread(
    gpu_allocator_interface*    allocator,
    const gpu_benchmark_config& config,
    gpu_benchmark_stats&        stats,
    std::atomic<bool>&          start_flag)
{
    // Wait for start signal
    while (!start_flag.load())
    {
        std::this_thread::yield();
    }

    gpu_allocation_pattern_generator      pattern(config);
    std::vector<std::pair<void*, size_t>> allocated_ptrs;
    allocated_ptrs.reserve(config.num_iterations / 2);

    gpu_timer timer;

    for (size_t i = 0; i < config.num_iterations; ++i)
    {
        if (pattern.should_allocate() || allocated_ptrs.empty())
        {
            // Perform allocation
            size_t alloc_size = pattern.next_allocation_size();

            timer.start();
            void* ptr = allocator->allocate(alloc_size);
            timer.stop();

            bool success = (ptr != nullptr);
            stats.record_allocation(timer.elapsed_ns(), alloc_size, success);

            if (success)
            {
                allocated_ptrs.emplace_back(ptr, alloc_size);
            }
        }
        else
        {
            // Perform deallocation
            if (!allocated_ptrs.empty())
            {
                std::mt19937                          rng{std::random_device{}()};
                std::uniform_int_distribution<size_t> idx_dist(0, allocated_ptrs.size() - 1);
                size_t                                idx = idx_dist(rng);
                auto [ptr, size]                          = allocated_ptrs[idx];
                allocated_ptrs.erase(allocated_ptrs.begin() + idx);

                timer.start();
                allocator->deallocate(ptr, size);
                timer.stop();

                stats.record_deallocation(timer.elapsed_ns());
            }
        }
    }

    // Clean up remaining allocations
    for (auto [ptr, size] : allocated_ptrs)
    {
        timer.start();
        allocator->deallocate(ptr, size);
        timer.stop();

        stats.record_deallocation(timer.elapsed_ns());
    }
}

/**
 * @brief Run GPU allocator benchmark
 */
gpu_benchmark_results run_gpu_allocator_benchmark(
    gpu_allocator_interface* allocator, const gpu_benchmark_config& config)
{
    gpu_benchmark_stats      stats;
    std::vector<std::thread> threads;
    std::atomic<bool>        start_flag{false};

    // Create worker threads
    for (size_t i = 0; i < config.num_threads; ++i)
    {
        threads.emplace_back(
            gpu_benchmark_worker_thread, allocator, config, std::ref(stats), std::ref(start_flag));
    }

    // Start benchmark
    gpu_timer total_timer;
    total_timer.start();
    start_flag.store(true);

    // Wait for all threads to complete
    for (auto& thread : threads)
    {
        thread.join();
    }

    total_timer.stop();

    // Collect results
    gpu_benchmark_results results = stats.get_results(allocator->name());
    results.total_time_ms         = total_timer.elapsed_ms();

    return results;
}

/**
 * @brief Print GPU benchmark results in formatted table
 */
void print_gpu_benchmark_results(const std::vector<gpu_benchmark_results>& results)
{
    std::cout << "\n=== GPU Memory Allocator Benchmark Results ===\n\n";
    std::cout << std::left << std::setw(25) << "Allocator" << std::setw(12) << "Alloc (ns)"
              << std::setw(12) << "Dealloc (ns)" << std::setw(15) << "Throughput (MB/s)"
              << std::setw(12) << "Total (ms)" << std::setw(10) << "Allocs" << std::setw(10)
              << "Failures" << "\n";
    std::cout << std::string(110, '-') << "\n";

    for (const auto& result : results)
    {
        std::cout << std::left << std::setw(25) << result.allocator_name << std::setw(12)
                  << std::fixed << std::setprecision(2) << result.avg_alloc_time_ns << std::setw(12)
                  << std::fixed << std::setprecision(2) << result.avg_dealloc_time_ns
                  << std::setw(15) << std::fixed << std::setprecision(2)
                  << result.throughput_mb_per_sec << std::setw(12) << std::fixed
                  << std::setprecision(2) << result.total_time_ms << std::setw(10)
                  << result.total_allocations << std::setw(10) << result.allocation_failures
                  << "\n";
    }

    // Print size class performance breakdown
    std::cout << "\n=== Size Class Performance (ns per allocation) ===\n";
    std::cout << std::left << std::setw(25) << "Allocator";
    std::vector<size_t> size_classes = {16, 256, 4096, 65536, 1048576, 16777216};
    for (size_t size : size_classes)
    {
        std::cout << std::setw(12) << (std::to_string(size / 1024) + "KB");
    }
    std::cout << "\n" << std::string(25 + 12 * size_classes.size(), '-') << "\n";

    for (const auto& result : results)
    {
        std::cout << std::left << std::setw(25) << result.allocator_name;
        for (size_t size : size_classes)
        {
            auto it = result.size_class_performance.find(size);
            if (it != result.size_class_performance.end())
            {
                std::cout << std::setw(12) << std::fixed << std::setprecision(1) << it->second;
            }
            else
            {
                std::cout << std::setw(12) << "N/A";
            }
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int get_cuda_device_count()
{
#if PROJECT_HAS_CUDA
    int         device_count = 0;
    cudaError_t error        = cudaGetDeviceCount(&device_count);
    return (error == cudaSuccess) ? device_count : 0;
#else
    return 0;
#endif
}

/**
 * @brief Generate comprehensive benchmark report with analysis
 */
/**
 * @brief Generate comprehensive benchmark report with analysis
 */
void generate_benchmark_report(const std::vector<gpu_benchmark_results>& results)
{
    std::ofstream report("gpu_allocator_benchmark_report.txt");
    if (!report.is_open())
    {
        std::cout << "Warning: Could not create benchmark report file\n";
        return;
    }

    report << "=== Quarisma GPU Memory Allocator Benchmark Report ===\n";
    report << "Generated: " << std::chrono::system_clock::now().time_since_epoch().count()
           << "\n\n";

    // Summary statistics
    report << "=== Summary Statistics ===\n";
    if (!results.empty())
    {
        double      best_alloc_time   = std::numeric_limits<double>::max();
        double      best_dealloc_time = std::numeric_limits<double>::max();
        double      best_throughput   = 0.0;
        std::string best_alloc_name, best_dealloc_name, best_throughput_name;

        for (const auto& result : results)
        {
            if (result.avg_alloc_time_ns < best_alloc_time)
            {
                best_alloc_time = result.avg_alloc_time_ns;
                best_alloc_name = result.allocator_name;
            }
            if (result.avg_dealloc_time_ns < best_dealloc_time)
            {
                best_dealloc_time = result.avg_dealloc_time_ns;
                best_dealloc_name = result.allocator_name;
            }
            if (result.throughput_mb_per_sec > best_throughput)
            {
                best_throughput      = result.throughput_mb_per_sec;
                best_throughput_name = result.allocator_name;
            }
        }

        report << "Best Allocation Performance: " << best_alloc_name << " (" << std::fixed
               << std::setprecision(2) << best_alloc_time << " ns)\n";
        report << "Best Deallocation Performance: " << best_dealloc_name << " (" << std::fixed
               << std::setprecision(2) << best_dealloc_time << " ns)\n";
        report << "Best Throughput: " << best_throughput_name << " (" << std::fixed
               << std::setprecision(2) << best_throughput << " MB/s)\n\n";
    }

    // Detailed results
    report << "=== Detailed Results ===\n";
    for (const auto& result : results)
    {
        report << "Allocator: " << result.allocator_name << "\n";
        report << "  Total Time: " << std::fixed << std::setprecision(2) << result.total_time_ms
               << " ms\n";
        report << "  Avg Allocation Time: " << std::fixed << std::setprecision(2)
               << result.avg_alloc_time_ns << " ns\n";
        report << "  Avg Deallocation Time: " << std::fixed << std::setprecision(2)
               << result.avg_dealloc_time_ns << " ns\n";
        report << "  Throughput: " << std::fixed << std::setprecision(2)
               << result.throughput_mb_per_sec << " MB/s\n";
        report << "  Total Allocations: " << result.total_allocations << "\n";
        report << "  Total Deallocations: " << result.total_deallocations << "\n";
        report << "  Allocation Failures: " << result.allocation_failures << "\n";
        report << "  Peak Memory Usage: " << result.peak_memory_usage << " bytes\n";
        report << "  Memory Fragmentation Ratio: " << std::fixed << std::setprecision(4)
               << result.memory_fragmentation_ratio << "\n\n";
    }

    report << "=== Recommendations ===\n";
    report << "1. For high-frequency small allocations: Use CUDA caching allocator with "
              "appropriate cache size\n";
    report << "2. For large allocations: Direct GPU allocator may be more efficient\n";
    report << "3. For multi-threaded applications: Monitor contention and consider per-thread "
              "allocators\n";
    report
        << "4. For memory-constrained environments: Use smaller cache sizes or direct allocation\n";
    report << "5. Monitor allocation failure rates and adjust strategies accordingly\n\n";

    report.close();
    std::cout << "Benchmark report saved to: gpu_allocator_benchmark_report.txt\n";
}

}  // anonymous namespace

/**
 * @brief Comprehensive GPU allocator benchmark comparing all strategies
 */
MEMORYTEST(GpuAllocatorBenchmark, ComprehensiveBenchmark)
{
    int device_count = get_cuda_device_count();
    if (device_count == 0)
    {
        GTEST_SKIP() << "No CUDA devices available";
    }

    std::vector<gpu_benchmark_results> all_results;

    // Test configurations for comprehensive benchmarking
    std::vector<gpu_benchmark_config> configs = {
        {1, 1000, 16, 1024 * 1024, 0.7, "SmallAllocations"},  // Small allocations (16B-1MB)
        {1,
         1000,
         1024 * 1024,
         16 * 1024 * 1024,
         0.6,
         "LargeAllocations"},                                 // Large allocations (1MB-16MB)
        {4, 1000, 16, 16 * 1024 * 1024, 0.6, "MultiThread"},  // Multi-threaded stress test
        {1, 2000, 256, 256 * 1024, 0.8, "HighFrequency"},     // High frequency small allocations
    };

    for (const auto& config : configs)
    {
        std::cout << "\nRunning " << config.test_name << " benchmark...\n";

        // Test GPU allocator with default options (compile-time strategy)
        {
            auto gpu_allocator = create_gpu_allocator(0, "GPU-Direct-" + config.test_name);
            gpu_allocator_wrapper wrapper(std::move(gpu_allocator));
            auto                  results = run_gpu_allocator_benchmark(&wrapper, config);
            all_results.push_back(results);
        }

        // Test GPU allocator with statistics disabled for performance
        {
            allocator_gpu::Options options;
            options.enable_statistics = false;
            auto perf_allocator =
                create_gpu_allocator(0, options, "GPU-NoStats-" + config.test_name);
            gpu_allocator_wrapper wrapper(std::move(perf_allocator));
            auto                  results = run_gpu_allocator_benchmark(&wrapper, config);
            all_results.push_back(results);
        }

        // Test CUDA caching allocator with different cache sizes
        {
            cuda_caching_allocator_wrapper wrapper(0, 64 * 1024 * 1024);  // 64MB cache
            auto                           results = run_gpu_allocator_benchmark(&wrapper, config);
            results.allocator_name                 = "CUDA-Cache-64MB-" + config.test_name;
            all_results.push_back(results);
        }

        {
            cuda_caching_allocator_wrapper wrapper(0, 256 * 1024 * 1024);  // 256MB cache
            auto                           results = run_gpu_allocator_benchmark(&wrapper, config);
            results.allocator_name                 = "CUDA-Cache-256MB-" + config.test_name;
            all_results.push_back(results);
        }

        // Test direct CUDA allocator for baseline comparison
        {
            direct_cuda_allocator_wrapper wrapper;
            auto                          results = run_gpu_allocator_benchmark(&wrapper, config);
            results.allocator_name                = "Direct-CUDA-" + config.test_name;
            all_results.push_back(results);
        }
    }

    // Print comprehensive results
    print_gpu_benchmark_results(all_results);

    // Basic validation
    EXPECT_FALSE(all_results.empty());
    for (const auto& result : all_results)
    {
        EXPECT_GT(result.total_allocations, 0);
        EXPECT_GE(result.avg_alloc_time_ns, 0.0);
    }

    // Generate comprehensive benchmark report
    generate_benchmark_report(all_results);
}

/**
 * @brief Allocation method comparison test (requires different builds)
 */
MEMORYTEST(GpuAllocatorBenchmark, AllocationMethodComparison)
{
    int device_count = get_cuda_device_count();
    if (device_count == 0)
    {
        GTEST_SKIP() << "No CUDA devices available";
    }

    std::cout << "\n=== GPU Allocation Method Analysis ===\n";

    // Test current allocation method
    auto gpu_allocator = create_gpu_allocator(0, "GPU-Current-Method");

    // Display current allocation method
    std::cout << "Current allocation method: ";
#if defined(QUARISMA_CUDA_ALLOC_SYNC) || defined(QUARISMA_HIP_ALLOC_SYNC)
    std::cout << "SYNC (cuMemAlloc/cuMemFree or hipMalloc/hipFree)\n";
#elif defined(QUARISMA_CUDA_ALLOC_ASYNC) || defined(QUARISMA_HIP_ALLOC_ASYNC)
    std::cout << "ASYNC (cuMemAllocAsync/cuMemFreeAsync or hipMallocAsync/hipFreeAsync)\n";
#elif defined(QUARISMA_CUDA_ALLOC_POOL_ASYNC) || defined(QUARISMA_HIP_ALLOC_POOL_ASYNC)
    std::cout << "POOL_ASYNC (cuMemAllocFromPoolAsync or hipMallocFromPoolAsync)\n";
#else
    std::cout << "DEFAULT (SYNC fallback)\n";
#endif

    // Run focused benchmark on current method
    gpu_benchmark_config  config{1, 1000, 1024, 1024 * 1024, 0.7, "MethodTest"};
    gpu_allocator_wrapper wrapper(std::move(gpu_allocator));
    auto                  results = run_gpu_allocator_benchmark(&wrapper, config);

    std::cout << "Performance Results:\n";
    std::cout << "  Avg Allocation Time: " << std::fixed << std::setprecision(2)
              << results.avg_alloc_time_ns << " ns\n";
    std::cout << "  Avg Deallocation Time: " << std::fixed << std::setprecision(2)
              << results.avg_dealloc_time_ns << " ns\n";
    std::cout << "  Throughput: " << std::fixed << std::setprecision(2)
              << results.throughput_mb_per_sec << " MB/s\n";
    std::cout << "  Total Allocations: " << results.total_allocations << "\n";
    std::cout << "  Allocation Failures: " << results.allocation_failures << "\n\n";

    std::cout << "Note: To test different allocation methods, rebuild with:\n";
    std::cout << "  -DQUARISMA_CUDA_ALLOC=SYNC     (for synchronous allocation)\n";
    std::cout << "  -DQUARISMA_CUDA_ALLOC=ASYNC    (for asynchronous allocation)\n";
    std::cout << "  -DQUARISMA_CUDA_ALLOC=POOL_ASYNC (for pool-based async allocation)\n\n";

    // Validation
    EXPECT_GT(results.total_allocations, 0);
    EXPECT_EQ(results.allocation_failures, 0);
}

#endif
