#include "visualization/web_dashboard.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <ios>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

//#include "logger/logger.h"
#include "cpu/allocator.h"

namespace memory
{

web_dashboard::~web_dashboard()
{
    if (server_running_.load())
    {
        stop_dashboard();
    }
}

bool web_dashboard::start_dashboard(int port)
{
    if (server_running_.load())
    {
        MEMORY_LOG_WARNING("Dashboard is already running");
        return false;
    }

    if (port > 0)
    {
        config_.port = port;
    }

    should_stop_.store(false);
    server_running_.store(true);

    // Start server thread
    server_thread_ = std::thread(&web_dashboard::server_thread_main, this);

    // Start metrics collection thread
    metrics_thread_ = std::thread(&web_dashboard::metrics_thread_main, this);

    MEMORY_LOG_INFO("Web dashboard started on {}:{}", config_.host, config_.port);
    return true;
}

void web_dashboard::stop_dashboard()
{
    if (!server_running_.load())
    {
        return;
    }

    should_stop_.store(true);
    server_running_.store(false);

    // Wait for threads to finish
    if (server_thread_.joinable())
    {
        server_thread_.join();
    }

    if (metrics_thread_.joinable())
    {
        metrics_thread_.join();
    }

    // Clear WebSocket clients
    {
        std::scoped_lock const lock(clients_mutex_);
        websocket_clients_.clear();
    }

    MEMORY_LOG_INFO("Web dashboard stopped");
}

bool web_dashboard::register_allocator(const std::string& name, Allocator* allocator)
{
    if (allocator == nullptr)
    {
        MEMORY_LOG_ERROR("Cannot register null allocator: {}", name);
        return false;
    }

    std::scoped_lock const lock(allocators_mutex_);

    if (registered_allocators_.find(name) != registered_allocators_.end())  //NOLINT
    {
        MEMORY_LOG_WARNING("Allocator already registered: {}", name);
        return false;
    }

    allocator_info info;
    info.name          = name;
    info.type          = get_allocator_type_name(allocator);
    info.allocator_ptr = allocator;
    info.is_active     = true;
    info.last_update   = std::chrono::steady_clock::now();

    registered_allocators_[name] = info;

    MEMORY_LOG_INFO("Registered allocator: {} (type: {})", name, info.type);
    return true;
}

bool web_dashboard::unregister_allocator(const std::string& name)
{
    std::scoped_lock const lock(allocators_mutex_);

    auto it = registered_allocators_.find(name);
    if (it == registered_allocators_.end())
    {
        MEMORY_LOG_WARNING("Allocator not found for unregistration: {}", name);
        return false;
    }

    registered_allocators_.erase(it);

    // Also remove history
    {
        std::scoped_lock const history_lock(history_mutex_);
        metrics_history_.erase(name);
    }

    MEMORY_LOG_INFO("Unregistered allocator: {}", name);
    return true;
}

std::vector<std::string> web_dashboard::get_registered_allocators() const
{
    std::scoped_lock const lock(allocators_mutex_);

    std::vector<std::string> names;
    names.reserve(registered_allocators_.size());

    for (const auto& [name, info] : registered_allocators_)
    {
        names.push_back(name);
    }

    return names;
}

std::string web_dashboard::get_allocator_metrics_json(const std::string& allocator_name) const
{
    std::scoped_lock const lock(allocators_mutex_);

    auto it = registered_allocators_.find(allocator_name);
    if (it == registered_allocators_.end())
    {
        return R"({"error": "Allocator not found"})";
    }

    const auto& info      = it->second;
    auto        stats_opt = info.allocator_ptr->GetStats();

    std::ostringstream json;
    json << "{\n";
    json << R"(  "name": ")" << info.name << R"(",
)";
    json << R"(  "type": ")" << info.type << R"(",
)";
    json << R"(  "is_active": )" << (info.is_active ? "true" : "false") << R"(,
)";
    json << R"(  "timestamp": )"
         << std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count()
         << R"(,
)";

    if (stats_opt.has_value())
    {
        const auto& stats = stats_opt.value();
        json << "  \"stats\": {\n";
        json << "    \"bytes_in_use\": " << stats.bytes_in_use << ",\n";
        json << "    \"peak_bytes_in_use\": " << stats.peak_bytes_in_use << ",\n";
        json << "    \"num_allocs\": " << stats.num_allocs << ",\n";
        json << "    \"bytes_limit\": " << stats.bytes_limit << ",\n";
        json << "    \"largest_alloc_size\": " << stats.largest_alloc_size << ",\n";
        json << "    \"fragmentation_ratio\": " << std::fixed << std::setprecision(4) << 0.0
             << "\n";
        json << "  }\n";
    }
    else
    {
        json << "  \"stats\": null\n";
    }

    json << "}";
    return json.str();
}

std::string web_dashboard::get_allocator_history_json(
    const std::string& allocator_name, size_t max_points) const
{
    std::scoped_lock const lock(history_mutex_);

    auto it = metrics_history_.find(allocator_name);
    if (it == metrics_history_.end())
    {
        return R"({"error": "No history available for allocator"})";
    }

    const auto&        history = it->second;
    std::ostringstream json;

    json << "{\n";
    json << R"(  "allocator_name": ")" << allocator_name << R"(",
)";
    json << "  \"history\": [\n";

    // Convert queue to vector for easier access
    std::vector<metrics_history_point> points;
    auto                               temp_queue = history;  // Copy queue
    while (!temp_queue.empty())
    {
        points.push_back(temp_queue.front());
        temp_queue.pop();
    }

    // Limit to max_points
    if (points.size() > max_points)
    {
        points.erase(points.begin(), points.end() - max_points);
    }

    for (size_t i = 0; i < points.size(); ++i)
    {
        const auto& point = points[i];
        json << "    {\n";
        json << "      \"timestamp\": "
             << std::chrono::duration_cast<std::chrono::milliseconds>(
                    point.timestamp.time_since_epoch())
                    .count()
             << ",\n";
        json << "      \"bytes_in_use\": " << point.bytes_in_use << ",\n";
        json << "      \"peak_bytes_in_use\": " << point.peak_bytes_in_use << ",\n";
        json << "      \"num_allocs\": " << point.num_allocs << ",\n";
        json << "      \"fragmentation_metric\": " << std::fixed << std::setprecision(4)
             << point.fragmentation_metric << ",\n";
        json << "      \"allocation_rate\": " << std::fixed << std::setprecision(2)
             << point.allocation_rate << "\n";
        json << "    }";

        if (i < points.size() - 1)
        {
            json << ",";
        }
        json << "\n";
    }

    json << "  ]\n";
    json << "}";
    return json.str();
}

std::string web_dashboard::get_all_allocators_summary_json() const
{
    std::scoped_lock const lock(allocators_mutex_);

    std::ostringstream json;
    json << "{\n";
    json << "  \"timestamp\": "
         << std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count()
         << ",\n";
    json << "  \"allocators\": [\n";

    size_t count = 0;
    for (const auto& [name, info] : registered_allocators_)
    {
        json << "    {\n";
        json << R"(      "name": ")" << info.name << R"(",
)";
        json << R"(      "type": ")" << info.type << R"(",
)";
        json << "      \"is_active\": " << (info.is_active ? "true" : "false") << "\n";
        json << "    }";

        if (++count < registered_allocators_.size())
        {
            json << ",";
        }
        json << "\n";
    }

    json << "  ]\n";
    json << "}";
    return json.str();
}

std::string web_dashboard::export_prometheus_metrics() const
{
    std::scoped_lock const lock(allocators_mutex_);

    std::ostringstream prometheus;

    // Add help and type information
    prometheus << "# HELP memory_allocations_total Total number of allocations\n";
    prometheus << "# TYPE memory_allocations_total counter\n";

    prometheus << "# HELP memory_bytes_allocated Current bytes allocated\n";
    prometheus << "# TYPE memory_bytes_allocated gauge\n";

    prometheus << "# HELP memory_peak_bytes_allocated Peak bytes allocated\n";
    prometheus << "# TYPE memory_peak_bytes_allocated gauge\n";

    prometheus << "# HELP memory_fragmentation_ratio Memory fragmentation ratio\n";
    prometheus << "# TYPE memory_fragmentation_ratio gauge\n";

    // Export metrics for each allocator
    for (const auto& [name, info] : registered_allocators_)
    {
        auto stats_opt = info.allocator_ptr->GetStats();
        if (!stats_opt.has_value())
        {
            continue;
        }

        const auto& stats = stats_opt.value();

        prometheus << "memory_allocations_total{allocator=\"" << name << "\",type=\"" << info.type
                   << "\"} " << stats.num_allocs << "\n";
        prometheus << "memory_bytes_allocated{allocator=\"" << name << "\",type=\"" << info.type
                   << "\"} " << stats.bytes_in_use << "\n";
        prometheus << "memory_peak_bytes_allocated{allocator=\"" << name << "\",type=\""
                   << info.type << "\"} " << stats.peak_bytes_in_use << "\n";
        prometheus << "memory_fragmentation_ratio{allocator=\"" << name << "\",type=\"" << info.type
                   << "\"} " << 0.0 << "\n";
    }

    return prometheus.str();
}

size_t web_dashboard::get_connected_clients_count() const
{
    std::scoped_lock const lock(clients_mutex_);
    return std::count_if(
        websocket_clients_.begin(),
        websocket_clients_.end(),
        [](const auto& pair) { return pair.second.is_active; });
}

void web_dashboard::update_metrics_now()
{
    collect_metrics();
}

void web_dashboard::server_thread_main()
{
    MEMORY_LOG_INFO("Starting web dashboard server thread on port {}", config_.port);

    // Simple HTTP server implementation
    // In a real implementation, this would use a proper HTTP library like httplib or similar

    while (!should_stop_.load())
    {
        // Simulate server processing
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Handle incoming HTTP requests
        // This is a simplified implementation - in practice would use proper HTTP library
    }

    MEMORY_LOG_INFO("Web dashboard server thread stopped");
}

void web_dashboard::metrics_thread_main()
{
    MEMORY_LOG_INFO("Starting metrics collection thread");

    while (!should_stop_.load())
    {
        collect_metrics();
        std::this_thread::sleep_for(config_.update_interval);
    }

    MEMORY_LOG_INFO("Metrics collection thread stopped");
}

void web_dashboard::collect_metrics()
{
    std::scoped_lock<std::mutex> const allocators_lock(allocators_mutex_);
    std::scoped_lock<std::mutex> const history_lock(history_mutex_);

    auto now = std::chrono::steady_clock::now();

    for (auto& [name, info] : registered_allocators_)
    {
        auto stats_opt = info.allocator_ptr->GetStats();
        if (!stats_opt.has_value())
        {
            continue;
        }

        const auto& stats = stats_opt.value();

        // Create history point
        metrics_history_point point;
        point.timestamp            = now;
        point.bytes_in_use         = stats.bytes_in_use;
        point.peak_bytes_in_use    = stats.peak_bytes_in_use;
        point.num_allocs           = stats.num_allocs;
        point.fragmentation_metric = 0.0;  // Placeholder - fragmentation calculation not available
        point.allocation_rate      = calculate_allocation_rate(name);

        // Add to history
        auto& history = metrics_history_[name];
        history.push(point);

        // Cleanup old history
        cleanup_old_history(name);

        // Update allocator info
        info.last_update = now;
    }

    // Broadcast update to WebSocket clients
    if (config_.enable_websocket && get_connected_clients_count() > 0)
    {
        std::string const metrics_json = get_all_allocators_summary_json();
        broadcast_metrics_update(metrics_json);
    }
}

void web_dashboard::broadcast_metrics_update(const std::string& /*metrics_json*/)
{
    std::scoped_lock const lock(clients_mutex_);

    // In a real implementation, this would send WebSocket messages to all connected clients
    // For now, just log that we would broadcast
    if (!websocket_clients_.empty())
    {
        MEMORY_LOG_INFO_DEBUG(
            "Broadcasting metrics update to {} clients", websocket_clients_.size());
    }
}

std::string web_dashboard::get_allocator_type_name(Allocator* allocator)
{
    if (allocator == nullptr)
    {
        return "Unknown";
    }

    // Use typeid to determine allocator type
    // In a real implementation, might use dynamic_cast or a type registry
    const std::string type_name = typeid(*allocator).name();

    if (type_name.find("allocator_bfc") != std::string::npos)
    {
        return "BFC";
    }
    if (type_name.find("allocator_pool") != std::string::npos)
    {
        return "Pool";
    }
    if (type_name.find("allocator_cpu") != std::string::npos)
    {
        return "CPU";
    }
    if (type_name.find("allocator_tracking") != std::string::npos)
    {
        return "Tracking";
    }
    if (type_name.find("allocator_device") != std::string::npos)
    {
        return "device_option";
    }
    return "Custom";
}

void web_dashboard::cleanup_old_history(const std::string& allocator_name)
{
    auto& history = metrics_history_[allocator_name];

    while (history.size() > config_.max_history_points)
    {
        history.pop();
    }
}

double web_dashboard::calculate_allocation_rate(const std::string& allocator_name) const
{
    auto it = metrics_history_.find(allocator_name);
    if (it == metrics_history_.end() || it->second.size() < 2)
    {
        return 0.0;
    }

    // Calculate rate based on last two data points
    auto                               temp_queue = it->second;  // Copy queue
    std::vector<metrics_history_point> recent_points;

    while (!temp_queue.empty() && recent_points.size() < 2)
    {
        recent_points.insert(recent_points.begin(), temp_queue.front());
        temp_queue.pop();
    }

    if (recent_points.size() < 2)
    {
        return 0.0;
    }

    const auto& latest   = recent_points[1];
    const auto& previous = recent_points[0];

    auto time_diff = std::chrono::duration<double>(latest.timestamp - previous.timestamp).count();
    if (time_diff <= 0.0)
    {
        return 0.0;
    }

    int64_t const alloc_diff =
        static_cast<int64_t>(latest.num_allocs) - static_cast<int64_t>(previous.num_allocs);
    return static_cast<double>(alloc_diff) / time_diff;
}

}  // namespace memory
