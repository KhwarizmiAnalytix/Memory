#include "gpu/gpu_resource_tracker.h"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <mutex>
#include <numeric>
#include <sstream>
#include <thread>

#include "common/memory_containers.h"
#include "common/memory_macros.h"
#include "util/exception.h"

// Hash specialization for std::pair<device_enum, int>
namespace std
{
template <>
struct hash<std::pair<memory::device_enum, int>>
{
    size_t operator()(const std::pair<memory::device_enum, int>& p) const
    {
        return std::hash<int>()(static_cast<int>(p.first)) ^ (std::hash<int>()(p.second) << 1);
    }
};
}  // namespace std

namespace memory
{
namespace gpu
{

namespace
{

/**
 * @brief Custom hash function for void pointers
 *
 * Provides a simple hash implementation for void* that doesn't rely on
 * std::__hash_memory which may not be available in all libc++ versions.
 */
struct void_ptr_hash
{
    std::size_t operator()(void* ptr) const noexcept
    {
        return static_cast<std::size_t>(reinterpret_cast<std::uintptr_t>(ptr));
    }
};

/**
 * @brief Internal implementation of GPU resource tracker
 */
class gpu_resource_tracker_impl : public gpu_resource_tracker
{
private:
    /** @brief Mutex for thread-safe operations */
    mutable std::mutex mutex_;

    /** @brief Whether tracking is enabled */
    std::atomic<bool> tracking_enabled_{true};

    /** @brief Next allocation ID */
    std::atomic<size_t> next_allocation_id_{1};

    /** @brief Map of active allocations (using custom hash for void*) */
    memory_map<void*, std::shared_ptr<gpu_allocation_info>, void_ptr_hash> active_allocations_;

    /** @brief Map of all allocations (including deallocated) */
    memory_map<size_t, std::shared_ptr<gpu_allocation_info>> all_allocations_;

    /** @brief Leak detection configuration */
    leak_detection_config leak_config_;

    /** @brief Statistics */
    mutable unified_resource_stats statistics_;

    /** @brief Background thread for periodic leak scanning */
    std::unique_ptr<std::thread> leak_scan_thread_;

    /** @brief Flag to stop background thread */
    std::atomic<bool> stop_background_thread_{false};

    /**
     * @brief Start background leak scanning thread
     */
    void start_leak_scan_thread()
    {
        if (leak_config_.enable_periodic_scan && !leak_scan_thread_)
        {
            stop_background_thread_.store(false);
            leak_scan_thread_ = std::make_unique<std::thread>(
                [this]()
                {
                    while (!stop_background_thread_.load())
                    {
                        std::this_thread::sleep_for(std::chrono::milliseconds(
                            static_cast<int>(leak_config_.scan_interval_ms)));

                        if (stop_background_thread_.load())
                        {
                            break;
                        }

                        // Perform leak detection
                        auto leaks = detect_leaks();
                        if (!leaks.empty() && leak_config_.enable_auto_reporting)
                        {
                            MEMORY_LOG_WARNING(
                                "GPU memory leak detection found {} potential leaks", leaks.size());
                            // Log details of first few leaks
                            for (size_t i = 0; i < std::min(leaks.size(), size_t(5)); ++i)
                            {
                                const auto& leak = leaks[i];
                                MEMORY_LOG_WARNING(
                                    "Leak {}: {} bytes at {} allocated in {} ({})",
                                    i + 1,
                                    leak->size,
                                    leak->ptr,
                                    leak->function_name,
                                    leak->source_file,
                                    leak->source_line);
                            }
                        }
                    }
                });
        }
    }

    /**
     * @brief Stop background leak scanning thread
     */
    void stop_leak_scan_thread()
    {
        if (leak_scan_thread_)
        {
            stop_background_thread_.store(true);
            if (leak_scan_thread_->joinable())
            {
                leak_scan_thread_->join();
            }
            leak_scan_thread_.reset();
        }
    }

    /**
     * @brief Update statistics after allocation
     */
    void update_statistics_on_allocation(size_t size)
    {
        statistics_.num_allocs.fetch_add(1, std::memory_order_relaxed);
        statistics_.active_allocations.fetch_add(1, std::memory_order_relaxed);
        statistics_.total_bytes_allocated.fetch_add(size, std::memory_order_relaxed);
        statistics_.bytes_in_use.fetch_add(size, std::memory_order_relaxed);

        // Update peak bytes in use with compare-and-swap
        int64_t const current_bytes = statistics_.bytes_in_use.load(std::memory_order_relaxed);
        int64_t       peak_bytes    = statistics_.peak_bytes_in_use.load(std::memory_order_relaxed);
        while (current_bytes > peak_bytes &&
               !statistics_.peak_bytes_in_use.compare_exchange_weak(
                   peak_bytes, current_bytes, std::memory_order_relaxed))
        {
            // Retry if another thread updated peak_bytes_in_use
        }

        // Update largest allocation size
        int64_t current_largest = statistics_.largest_alloc_size.load(std::memory_order_relaxed);
        while (static_cast<int64_t>(size) > current_largest &&
               !statistics_.largest_alloc_size.compare_exchange_weak(
                   current_largest, size, std::memory_order_relaxed))
        {
            // Retry if another thread updated largest_alloc_size
        }
    }

    /**
     * @brief Update statistics after deallocation
     */
    void update_statistics_on_deallocation(size_t size)
    {
        statistics_.num_deallocs.fetch_add(1, std::memory_order_relaxed);
        statistics_.active_allocations.fetch_sub(1, std::memory_order_relaxed);
        statistics_.total_bytes_deallocated.fetch_add(size, std::memory_order_relaxed);
        statistics_.bytes_in_use.fetch_sub(size, std::memory_order_relaxed);

        // Note: average_allocation_lifetime_ms is not available in unified_resource_stats
        // This metric would need to be tracked separately if needed
    }

public:
    gpu_resource_tracker_impl()
    {
        // Initialize leak detection with default configuration
        leak_config_.enabled               = true;
        leak_config_.leak_threshold_ms     = 60000.0;  // 1 minute
        leak_config_.max_call_stack_depth  = 10;
        leak_config_.enable_periodic_scan  = true;
        leak_config_.scan_interval_ms      = 30000.0;  // 30 seconds
        leak_config_.enable_auto_reporting = true;

        start_leak_scan_thread();
    }

    ~gpu_resource_tracker_impl() override
    {
        stop_leak_scan_thread();

        // Check for remaining allocations
        std::scoped_lock const lock(mutex_);
        if (!active_allocations_.empty())
        {
            MEMORY_LOG_WARNING(
                "GPU resource tracker destroyed with {} active allocations (potential memory "
                "leaks)",
                active_allocations_.size());
        }
    }

    void configure_leak_detection(const leak_detection_config& config) override
    {
        std::scoped_lock const lock(mutex_);

        bool const restart_thread =
            (config.enable_periodic_scan != leak_config_.enable_periodic_scan) ||
            (config.scan_interval_ms != leak_config_.scan_interval_ms);

        leak_config_ = config;

        if (restart_thread)
        {
            stop_leak_scan_thread();
            start_leak_scan_thread();
        }
    }

    size_t track_allocation(
        void*              ptr,
        size_t             size,
        device_enum        device_type,
        int                device_index,
        const std::string& tag,
        const char*        source_file,
        int                source_line,
        const char*        function_name) override
    {
        if (!tracking_enabled_.load() || (ptr == nullptr))
        {
            return 0;
        }

        std::scoped_lock const lock(mutex_);

        size_t const allocation_id = next_allocation_id_.fetch_add(1);

        auto info              = std::make_shared<gpu_allocation_info>();
        info->allocation_id    = allocation_id;
        info->ptr              = ptr;
        info->size             = size;
        info->device           = device_option(device_type, device_index);
        info->allocation_time  = std::chrono::high_resolution_clock::now();
        info->last_access_time = info->allocation_time;
        info->is_active        = true;
        info->source_file      = (source_file != nullptr) ? source_file : "";
        info->source_line      = source_line;
        info->function_name    = (function_name != nullptr) ? function_name : "";
        info->tag              = tag;

        // TODO: Capture call stack if enabled
        // This would require platform-specific code or a third-party library

        active_allocations_[ptr]        = info;
        all_allocations_[allocation_id] = info;

        update_statistics_on_allocation(size);

        return allocation_id;
    }

    bool track_deallocation(void* ptr) override
    {
        if (!tracking_enabled_.load() || (ptr == nullptr))
        {
            return false;
        }

        std::scoped_lock const lock(mutex_);

        auto it = active_allocations_.find(ptr);
        if (it == active_allocations_.end())
        {
            return false;  // Allocation not found
        }

        auto info               = it->second;
        info->deallocation_time = std::chrono::high_resolution_clock::now();
        info->is_active         = false;

        update_statistics_on_deallocation(info->size);

        active_allocations_.erase(it);

        return true;
    }

    void record_access(void* ptr) override
    {
        if (!tracking_enabled_.load() || (ptr == nullptr))
        {
            return;
        }

        std::scoped_lock const lock(mutex_);

        auto it = active_allocations_.find(ptr);
        if (it != active_allocations_.end())
        {
            const auto& info = it->second;
            info->access_count.fetch_add(1);
            info->last_access_time = std::chrono::high_resolution_clock::now();
        }
    }

    std::shared_ptr<gpu_allocation_info> get_allocation_info(void* ptr) const override
    {
        if (ptr == nullptr)
        {
            return nullptr;
        }

        std::scoped_lock const lock(mutex_);

        auto it = active_allocations_.find(ptr);
        if (it != active_allocations_.end())
        {
            return it->second;
        }

        return nullptr;
    }

    unified_resource_stats get_statistics() const override
    {
        std::scoped_lock const lock(mutex_);

        // Update potential leaks count
        statistics_.potential_leaks.store(0, std::memory_order_relaxed);

        if (leak_config_.enabled)
        {
            int64_t leak_count = 0;
            auto    now        = std::chrono::high_resolution_clock::now();
            for (const auto& [ptr, info] : active_allocations_)
            {
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - info->allocation_time);
                if (duration.count() > leak_config_.leak_threshold_ms)
                {
                    leak_count++;
                }
            }
            statistics_.potential_leaks.store(leak_count, std::memory_order_relaxed);
        }

        // Return a copy of the statistics
        unified_resource_stats const stats_copy(statistics_);
        return stats_copy;
    }

    std::vector<std::shared_ptr<gpu_allocation_info>> get_active_allocations() const override
    {
        std::scoped_lock const lock(mutex_);

        std::vector<std::shared_ptr<gpu_allocation_info>> result;
        result.reserve(active_allocations_.size());

        for (const auto& [ptr, info] : active_allocations_)
        {
            result.push_back(info);
        }

        return result;
    }

private:
    /**
     * @brief Internal leak detection without mutex lock (assumes lock is already held)
     */
    std::vector<std::shared_ptr<gpu_allocation_info>> detect_leaks_unsafe() const
    {
        if (!leak_config_.enabled)
        {
            return {};
        }

        std::vector<std::shared_ptr<gpu_allocation_info>> leaks;
        auto now = std::chrono::high_resolution_clock::now();

        for (const auto& [ptr, info] : active_allocations_)
        {
            auto duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(now - info->allocation_time);

            if (duration.count() > leak_config_.leak_threshold_ms)
            {
                leaks.push_back(info);
            }
        }

        // Sort by allocation time (oldest first)
        std::sort(
            leaks.begin(),
            leaks.end(),
            [](const auto& a, const auto& b) { return a->allocation_time < b->allocation_time; });

        return leaks;
    }

public:
    std::vector<std::shared_ptr<gpu_allocation_info>> detect_leaks() const override
    {
        std::scoped_lock const lock(mutex_);
        return detect_leaks_unsafe();
    }

    std::vector<std::shared_ptr<gpu_allocation_info>> get_allocations_by_tag(
        const std::string& tag) const override
    {
        std::scoped_lock const lock(mutex_);

        std::vector<std::shared_ptr<gpu_allocation_info>> result;

        for (const auto& [ptr, info] : active_allocations_)
        {
            if (info->tag == tag)
            {
                result.push_back(info);
            }
        }

        return result;
    }

    std::vector<std::shared_ptr<gpu_allocation_info>> get_allocations_by_device(
        device_enum device_type, int device_index) const override
    {
        std::scoped_lock const lock(mutex_);

        std::vector<std::shared_ptr<gpu_allocation_info>> result;

        for (const auto& [ptr, info] : active_allocations_)
        {
            if (info->device.type() == device_type && info->device.index() == device_index)
            {
                result.push_back(info);
            }
        }

        return result;
    }

    void clear_all_data() override
    {
        std::scoped_lock const lock(mutex_);

        active_allocations_.clear();
        all_allocations_.clear();

        // Reset all statistics to zero
        statistics_.num_allocs.store(0, std::memory_order_relaxed);
        statistics_.num_deallocs.store(0, std::memory_order_relaxed);
        statistics_.bytes_in_use.store(0, std::memory_order_relaxed);
        statistics_.peak_bytes_in_use.store(0, std::memory_order_relaxed);
        statistics_.largest_alloc_size.store(0, std::memory_order_relaxed);
        statistics_.active_allocations.store(0, std::memory_order_relaxed);
        statistics_.total_bytes_allocated.store(0, std::memory_order_relaxed);
        statistics_.total_bytes_deallocated.store(0, std::memory_order_relaxed);
        statistics_.failed_allocations.store(0, std::memory_order_relaxed);
        statistics_.potential_leaks.store(0, std::memory_order_relaxed);
        statistics_.bytes_reserved.store(0, std::memory_order_relaxed);
        statistics_.peak_bytes_reserved.store(0, std::memory_order_relaxed);
        statistics_.bytes_reservable_limit.store(0, std::memory_order_relaxed);
        statistics_.largest_free_block_bytes.store(0, std::memory_order_relaxed);
        statistics_.pool_bytes.store(0, std::memory_order_relaxed);
        statistics_.peak_pool_bytes.store(0, std::memory_order_relaxed);
        statistics_.bytes_limit.store(0, std::memory_order_relaxed);
    }

    std::string generate_report(bool include_call_stacks) const override
    {
        std::scoped_lock const lock(mutex_);

        std::ostringstream oss;
        oss << "GPU Resource Tracker Report:\n";
        oss << "============================\n\n";

        // Statistics
        oss << "Statistics:\n";
        oss << "  Total allocations: " << statistics_.num_allocs.load(std::memory_order_relaxed)
            << "\n";
        oss << "  Total deallocations: " << statistics_.num_deallocs.load(std::memory_order_relaxed)
            << "\n";
        oss << "  Active allocations: "
            << statistics_.active_allocations.load(std::memory_order_relaxed) << "\n";
        oss << "  Total bytes allocated: " << std::fixed << std::setprecision(2)
            << (statistics_.total_bytes_allocated.load(std::memory_order_relaxed) / 1024.0 / 1024.0)
            << " MB\n";
        oss << "  Current bytes in use: " << std::fixed << std::setprecision(2)
            << (statistics_.bytes_in_use.load(std::memory_order_relaxed) / 1024.0 / 1024.0)
            << " MB\n";
        oss << "  Peak bytes in use: " << std::fixed << std::setprecision(2)
            << (statistics_.peak_bytes_in_use.load(std::memory_order_relaxed) / 1024.0 / 1024.0)
            << " MB\n";
        oss << "  Average allocation size: " << std::fixed << std::setprecision(2)
            << (statistics_.average_allocation_size() / 1024.0) << " KB\n";
        oss << "  Largest allocation size: " << std::fixed << std::setprecision(2)
            << (statistics_.largest_alloc_size.load(std::memory_order_relaxed) / 1024.0) << " KB\n";

        // Potential leaks
        auto leaks = detect_leaks_unsafe();
        oss << "  Potential leaks: " << leaks.size() << "\n\n";

        // Active allocations by device
        memory_map<
            std::pair<device_enum, int>,
            std::vector<std::shared_ptr<gpu_allocation_info>>,
            std::hash<std::pair<device_enum, int>>>
            by_device;

        for (const auto& [ptr, info] : active_allocations_)
        {
            auto key = std::make_pair(info->device.type(), info->device.index());
            by_device[key].push_back(info);
        }

        oss << "Active Allocations by Device:\n";
        for (const auto& [device_key, allocations] : by_device)
        {
            size_t const total_bytes = std::accumulate(
                allocations.begin(),
                allocations.end(),
                size_t{0},
                [](size_t sum, const auto& alloc) { return sum + alloc->size; });

            oss << "  Device " << static_cast<int>(device_key.first) << ":" << device_key.second
                << " - " << allocations.size() << " allocations, " << std::fixed
                << std::setprecision(2) << (total_bytes / 1024.0 / 1024.0) << " MB\n";
        }

        // Top allocations by size
        std::vector<std::shared_ptr<gpu_allocation_info>> sorted_allocations;
        for (const auto& [ptr, info] : active_allocations_)
        {
            sorted_allocations.push_back(info);
        }

        std::sort(
            sorted_allocations.begin(),
            sorted_allocations.end(),
            [](const auto& a, const auto& b) { return a->size > b->size; });

        oss << "\nTop 10 Largest Active Allocations:\n";
        for (size_t i = 0; i < std::min(sorted_allocations.size(), size_t(10)); ++i)
        {
            const auto& info = sorted_allocations[i];
            oss << "  " << (i + 1) << ". " << std::fixed << std::setprecision(2)
                << (info->size / 1024.0 / 1024.0) << " MB at " << info->ptr << " (allocated in "
                << info->function_name << ")\n";

            if (include_call_stacks && !info->call_stack.empty())
            {
                oss << "     Call stack:\n";
                for (const auto& frame : info->call_stack)
                {
                    oss << "       " << frame << "\n";
                }
            }
        }

        return oss.str();
    }

    std::string generate_leak_report() const override
    {
        auto leaks = detect_leaks();

        std::ostringstream oss;
        oss << "GPU Memory Leak Report:\n";
        oss << "=======================\n\n";

        if (leaks.empty())
        {
            oss << "No potential memory leaks detected.\n";
        }
        else
        {
            oss << "Found " << leaks.size() << " potential memory leaks:\n\n";

            for (size_t i = 0; i < leaks.size(); ++i)
            {
                const auto& leak     = leaks[i];
                auto        now      = std::chrono::high_resolution_clock::now();
                auto        duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - leak->allocation_time);

                oss << "Leak " << (i + 1) << ":\n";
                oss << "  Pointer: " << leak->ptr << "\n";
                oss << "  Size: " << std::fixed << std::setprecision(2) << (leak->size / 1024.0)
                    << " KB\n";
                oss << "  Age: " << std::fixed << std::setprecision(2)
                    << (duration.count() / 1000.0) << " seconds\n";
                oss << "  Device: " << static_cast<int>(leak->device.type()) << ":"
                    << leak->device.index() << "\n";
                oss << "  Allocated in: " << leak->function_name << " (" << leak->source_file << ":"
                    << leak->source_line << ")\n";
                oss << "  Tag: " << (leak->tag.empty() ? "(none)" : leak->tag) << "\n";
                oss << "  Access count: " << leak->access_count.load() << "\n";

                if (!leak->call_stack.empty())
                {
                    oss << "  Call stack:\n";
                    for (const auto& frame : leak->call_stack)
                    {
                        oss << "    " << frame << "\n";
                    }
                }

                oss << "\n";
            }
        }

        return oss.str();
    }

    void set_tracking_enabled(bool enabled) override { tracking_enabled_.store(enabled); }

    bool is_tracking_enabled() const override { return tracking_enabled_.load(); }
};

}  // anonymous namespace

gpu_resource_tracker& gpu_resource_tracker::instance()
{
    static gpu_resource_tracker_impl instance;
    return instance;
}

}  // namespace gpu
}  // namespace memory
