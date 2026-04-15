/*
 * Quarisma: High-Performance Computational Library
 * Copyright 2025 Quarisma Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include "MemoryTest.h"
#include "backend/allocator_bfc.h"
#include "backend/allocator_pool.h"
#include "backend/allocator_tracking.h"
#include "common/pointer.h"
#include "helper/memory_allocator.h"

using namespace memory;
using namespace memory;

namespace
{

/**
 * @brief Benchmark configuration and results structure
 */
struct benchmark_config
{
    size_t      num_threads      = 1;
    size_t      num_iterations   = 10000;
    size_t      min_alloc_size   = 64;
    size_t      max_alloc_size   = 4096;
    double      allocation_ratio = 0.5;  // Ratio of allocations vs deallocations
    std::string test_name;
};

struct benchmark_results
{
    std::string test_name;
    double      total_time_ms         = 0.0;
    double      avg_alloc_time_ns     = 0.0;
    double      avg_dealloc_time_ns   = 0.0;
    double      throughput_mb_per_sec = 0.0;
    size_t      total_allocations     = 0;
    size_t      total_deallocations   = 0;
    size_t      peak_memory_usage     = 0;
    double      lock_contention_ratio = 0.0;
};

/**
 * @brief High-precision timer for benchmarking
 */
class high_precision_timer
{
public:
    void start() { start_time_ = std::chrono::high_resolution_clock::now(); }

    void stop() { end_time_ = std::chrono::high_resolution_clock::now(); }

    double elapsed_ms() const
    {
        auto duration =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end_time_ - start_time_);
        return duration.count() / 1000000.0;
    }

    double elapsed_ns() const
    {
        auto duration =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end_time_ - start_time_);
        return static_cast<double>(duration.count());
    }

private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
};

/**
 * @brief Thread-safe statistics collector
 */
class benchmark_stats
{
public:
    void record_allocation(double time_ns, size_t size)
    {
        total_alloc_time_.fetch_add(static_cast<int64_t>(time_ns), std::memory_order_relaxed);
        total_allocations_.fetch_add(1, std::memory_order_relaxed);
        total_bytes_allocated_.fetch_add(size, std::memory_order_relaxed);
    }

    void record_deallocation(double time_ns)
    {
        total_dealloc_time_.fetch_add(static_cast<int64_t>(time_ns), std::memory_order_relaxed);
        total_deallocations_.fetch_add(1, std::memory_order_relaxed);
    }

    benchmark_results get_results() const
    {
        benchmark_results results;
        results.total_allocations   = total_allocations_.load();
        results.total_deallocations = total_deallocations_.load();

        if (results.total_allocations > 0)
        {
            results.avg_alloc_time_ns =
                static_cast<double>(total_alloc_time_.load()) / results.total_allocations;
        }

        if (results.total_deallocations > 0)
        {
            results.avg_dealloc_time_ns =
                static_cast<double>(total_dealloc_time_.load()) / results.total_deallocations;
        }

        size_t total_bytes = total_bytes_allocated_.load();
        if (total_bytes > 0 && results.total_time_ms > 0)
        {
            results.throughput_mb_per_sec =
                (total_bytes / (1024.0 * 1024.0)) / (results.total_time_ms / 1000.0);
        }

        return results;
    }

    void set_total_time(double time_ms) { total_time_ms_ = time_ms; }

private:
    std::atomic<int64_t> total_alloc_time_{0};
    std::atomic<int64_t> total_dealloc_time_{0};
    std::atomic<size_t>  total_allocations_{0};
    std::atomic<size_t>  total_deallocations_{0};
    std::atomic<size_t>  total_bytes_allocated_{0};
    double               total_time_ms_{0.0};
};

/**
 * @brief Allocation pattern generator
 */
class allocation_pattern_generator
{
public:
    explicit allocation_pattern_generator(const benchmark_config& config)
        : config_(config),
          rng_(std::random_device{}()),
          size_dist_(config.min_alloc_size, config.max_alloc_size),
          action_dist_(0.0, 1.0)
    {
    }

    size_t next_allocation_size() { return size_dist_(rng_); }

    bool should_allocate() { return action_dist_(rng_) < config_.allocation_ratio; }

private:
    benchmark_config                       config_;
    std::mt19937                           rng_;
    std::uniform_int_distribution<size_t>  size_dist_;
    std::uniform_real_distribution<double> action_dist_;
};

/**
 * @brief Worker thread function for multi-threaded benchmarks
 */
void benchmark_worker_thread(
    Allocator*              allocator,
    const benchmark_config& config,
    benchmark_stats&        stats,
    std::atomic<bool>&      start_flag)
{
    // Wait for start signal
    while (!start_flag.load())
    {
        std::this_thread::yield();
    }

    allocation_pattern_generator pattern(config);
    std::vector<void*>           allocated_ptrs;
    allocated_ptrs.reserve(config.num_iterations / 2);

    high_precision_timer timer;

    for (size_t i = 0; i < config.num_iterations; ++i)
    {
        if (pattern.should_allocate() || allocated_ptrs.empty())
        {
            // Perform allocation
            size_t alloc_size = pattern.next_allocation_size();

            timer.start();
            void* ptr = allocator->allocate_raw(64, alloc_size);  // 64-byte alignment
            timer.stop();

            if (ptr != nullptr)
            {
                allocated_ptrs.push_back(ptr);
                stats.record_allocation(timer.elapsed_ns(), alloc_size);
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
                void*                                 ptr = allocated_ptrs[idx];
                allocated_ptrs.erase(allocated_ptrs.begin() + idx);

                timer.start();
                allocator->deallocate_raw(ptr);
                timer.stop();

                stats.record_deallocation(timer.elapsed_ns());
            }
        }
    }

    // Clean up remaining allocations
    for (void* ptr : allocated_ptrs)
    {
        timer.start();
        allocator->deallocate_raw(ptr);
        timer.stop();

        stats.record_deallocation(timer.elapsed_ns());
    }
}

/**
 * @brief Run benchmark for a specific allocator
 */
benchmark_results run_allocator_benchmark(Allocator* allocator, const benchmark_config& config)
{
    benchmark_stats          stats;
    std::vector<std::thread> threads;
    std::atomic<bool>        start_flag{false};

    // Create worker threads
    for (size_t i = 0; i < config.num_threads; ++i)
    {
        threads.emplace_back(
            benchmark_worker_thread, allocator, config, std::ref(stats), std::ref(start_flag));
    }

    // Start benchmark
    high_precision_timer total_timer;
    total_timer.start();
    start_flag.store(true);

    // Wait for all threads to complete
    for (auto& thread : threads)
    {
        thread.join();
    }

    total_timer.stop();

    // Collect results
    benchmark_results results = stats.get_results();
    results.total_time_ms     = total_timer.elapsed_ms();
    results.test_name         = config.test_name;

    return results;
}

/**
 * @brief Create BFC allocator for testing
 */
std::unique_ptr<allocator_bfc> create_bfc_allocator(const std::string& name)
{
    auto sub_allocator = std::make_unique<basic_cpu_allocator>(
        0,
        std::vector<memory::sub_allocator::Visitor>{},
        std::vector<memory::sub_allocator::Visitor>{});

    allocator_bfc::Options opts;
    opts.allow_growth           = true;
    opts.garbage_collection     = false;
    opts.fragmentation_fraction = 0.0;

    return std::make_unique<allocator_bfc>(
        std::move(sub_allocator), 1024ULL * 1024ULL * 1024ULL, name, opts);
}

/**
 * @brief Create Pool allocator for testing
 */
std::unique_ptr<allocator_pool> create_pool_allocator(const std::string& name)
{
    auto sub_allocator = std::make_unique<basic_cpu_allocator>(
        0,
        std::vector<memory::sub_allocator::Visitor>{},
        std::vector<memory::sub_allocator::Visitor>{});

    // Simple no-op rounder for testing
    class NoopRounder : public round_up_interface
    {
    public:
        size_t RoundUp(size_t num_bytes) override { return num_bytes; }
    };

    return std::make_unique<allocator_pool>(
        1000,   // pool_size_limit
        false,  // auto_resize
        std::move(sub_allocator),
        std::make_unique<NoopRounder>(),
        name);
}

/**
 * @brief Print benchmark results in formatted table
 */
void print_benchmark_results(const std::vector<benchmark_results>& results)
{
    std::cout << "\n=== Memory Allocator Benchmark Results ===\n\n";
    std::cout << std::left << std::setw(25) << "Test Name" << std::setw(12) << "Alloc (ns)"
              << std::setw(12) << "Dealloc (ns)" << std::setw(15) << "Throughput (MB/s)"
              << std::setw(12) << "Total (ms)" << std::setw(10) << "Allocs" << std::setw(10)
              << "Deallocs" << "\n";
    std::cout << std::string(100, '-') << "\n";

    for (const auto& result : results)
    {
        std::cout << std::left << std::setw(25) << result.test_name << std::setw(12) << std::fixed
                  << std::setprecision(2) << result.avg_alloc_time_ns << std::setw(12) << std::fixed
                  << std::setprecision(2) << result.avg_dealloc_time_ns << std::setw(15)
                  << std::fixed << std::setprecision(2) << result.throughput_mb_per_sec
                  << std::setw(12) << std::fixed << std::setprecision(2) << result.total_time_ms
                  << std::setw(10) << result.total_allocations << std::setw(10)
                  << result.total_deallocations << "\n";
    }
    std::cout << "\n";
}

}  // anonymous namespace

/**
 * @brief Comprehensive allocator benchmark test
 */
MEMORYTEST(AllocatorBenchmark, ComprehensiveBenchmark)
{
    std::vector<benchmark_results> all_results;

    // Test configurations
    std::vector<benchmark_config> configs = {
        {1, 10000, 64, 4096, 0.5, "BFC-SingleThread"},
        {4, 10000, 64, 4096, 0.5, "BFC-MultiThread"},
        {1, 10000, 64, 4096, 0.5, "Pool-SingleThread"},
        {4, 10000, 64, 4096, 0.5, "Pool-MultiThread"},
    };

    // Run BFC benchmarks
    auto bfc_allocator = create_bfc_allocator("BFC-Benchmark");
    for (auto& config : configs)
    {
        if (config.test_name.find("BFC") != std::string::npos)
        {
            auto results = run_allocator_benchmark(bfc_allocator.get(), config);
            all_results.push_back(results);
        }
    }

    // Run Pool benchmarks
    auto pool_allocator = create_pool_allocator("Pool-Benchmark");
    for (auto& config : configs)
    {
        if (config.test_name.find("Pool") != std::string::npos)
        {
            auto results = run_allocator_benchmark(pool_allocator.get(), config);
            all_results.push_back(results);
        }
    }

    // Print results
    print_benchmark_results(all_results);

    // Basic validation - ensure we got results
    EXPECT_FALSE(all_results.empty());
    for (const auto& result : all_results)
    {
        EXPECT_GT(result.total_allocations, 0);
        EXPECT_GT(result.avg_alloc_time_ns, 0.0);
    }
}
