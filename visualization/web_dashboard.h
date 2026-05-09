#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cpu/allocator.h"
#include "profiler/unified_memory_stats.h"
#include "common/memory_export.h"

namespace memory
{

/**
 * @brief Web-based dashboard for real-time memory allocation monitoring
 * 
 * Provides HTTP server with REST API endpoints for querying allocator metrics,
 * real-time updates via WebSocket, and JSON export capabilities. Designed for
 * development and debugging of memory allocation patterns.
 * 
 * **Thread Safety**: Thread-safe for concurrent access
 * **Performance**: Minimal overhead when disabled, configurable update intervals
 */
class MEMORY_VISIBILITY web_dashboard
{
public:
    /**
     * @brief Configuration for web dashboard
     */
    struct dashboard_config
    {
        int                       port = 8080;                ///< HTTP server port
        std::string               host = "localhost";         ///< HTTP server host
        std::chrono::milliseconds update_interval{1000};      ///< Metrics update interval
        size_t                    max_history_points = 1000;  ///< Maximum history data points
        bool                      enable_websocket   = true;  ///< Enable WebSocket support
        bool                      enable_cors        = true;  ///< Enable CORS headers
        std::string               static_files_path  = "./dashboard";  ///< Path to static files
    };

    /**
     * @brief Allocator information for dashboard
     */
    struct allocator_info
    {
        std::string                           name;  ///< Allocator name
        std::string                           type;  ///< Allocator type (BFC, Pool, CPU, etc.)
        Allocator*                            allocator_ptr;  ///< Pointer to allocator instance
        bool                                  is_active;  ///< Whether allocator is currently active
        std::chrono::steady_clock::time_point last_update;  ///< Last update timestamp
    };

    /**
     * @brief Historical data point for metrics
     */
    struct metrics_history_point
    {
        std::chrono::steady_clock::time_point timestamp;
        size_t                                bytes_in_use;
        size_t                                peak_bytes_in_use;
        size_t                                num_allocs;
        double                                fragmentation_metric;
        double                                allocation_rate;  // allocations per second
    };

    /**
     * @brief WebSocket client connection
     */
    struct websocket_client
    {
        int                                   connection_id;
        std::string                           client_address;
        std::chrono::steady_clock::time_point connected_at;
        bool                                  is_active;
    };

public:
    /**
     * @brief Construct web dashboard with default configuration
     */
    web_dashboard() = default;

    /**
     * @brief Construct web dashboard with custom configuration
     * @param config Dashboard configuration
     */
    explicit web_dashboard(dashboard_config config) : config_(std::move(config)) {}

    /**
     * @brief Destructor - stops dashboard if running
     */
    ~web_dashboard();

    /**
     * @brief Start the web dashboard server
     * @param port Server port (overrides config if specified)
     * @return True if server started successfully
     * 
     * Starts HTTP server and begins collecting metrics from registered allocators.
     * Non-blocking - server runs in background thread.
     * 
     * **Example**:
     * ```cpp
     * web_dashboard dashboard;
     * dashboard.register_allocator("main_bfc", my_bfc_allocator);
     * dashboard.start_dashboard(8080);
     * ```
     */
    MEMORY_API bool start_dashboard(int port = 0);

    /**
     * @brief Stop the web dashboard server
     * 
     * Gracefully shuts down HTTP server and stops metrics collection.
     * Blocks until server is fully stopped.
     */
    MEMORY_API void stop_dashboard();

    /**
     * @brief Register allocator for monitoring
     * @param name Unique name for the allocator
     * @param allocator Pointer to allocator instance
     * @return True if registration successful
     * 
     * Registers an allocator for monitoring and visualization.
     * Must be called before starting the dashboard.
     * 
     * **Thread Safety**: Safe to call from any thread
     */
    MEMORY_API bool register_allocator(const std::string& name, Allocator* allocator);

    /**
     * @brief Unregister allocator from monitoring
     * @param name Name of allocator to unregister
     * @return True if unregistration successful
     */
    MEMORY_API bool unregister_allocator(const std::string& name);

    /**
     * @brief Get list of registered allocators
     * @return Vector of allocator names
     */
    MEMORY_API std::vector<std::string> get_registered_allocators() const;

    /**
     * @brief Get current metrics for specific allocator
     * @param allocator_name Name of allocator
     * @return JSON string with current metrics, empty if not found
     */
    MEMORY_API std::string get_allocator_metrics_json(const std::string& allocator_name) const;

    /**
     * @brief Get historical metrics for specific allocator
     * @param allocator_name Name of allocator
     * @param max_points Maximum number of historical points to return
     * @return JSON string with historical metrics
     */
    MEMORY_API std::string get_allocator_history_json(
        const std::string& allocator_name, size_t max_points = 100) const;

    /**
     * @brief Get summary of all allocators
     * @return JSON string with summary of all registered allocators
     */
    MEMORY_API std::string get_all_allocators_summary_json() const;

    /**
     * @brief Export metrics in Prometheus format
     * @return Prometheus-formatted metrics string
     * 
     * Exports current metrics in Prometheus format for integration
     * with monitoring systems like Grafana.
     */
    MEMORY_API std::string export_prometheus_metrics() const;

    /**
     * @brief Check if dashboard is currently running
     * @return True if dashboard server is active
     */
    bool is_running() const { return server_running_.load(); }

    /**
     * @brief Get current configuration
     * @return Current dashboard configuration
     */
    const dashboard_config& get_config() const { return config_; }

    /**
     * @brief Set dashboard configuration
     * @param config New configuration (takes effect on next start)
     */
    void set_config(const dashboard_config& config) { config_ = config; }

    /**
     * @brief Get number of connected WebSocket clients
     * @return Number of active WebSocket connections
     */
    size_t get_connected_clients_count() const;

    /**
     * @brief Manually trigger metrics update
     * 
     * Forces immediate collection and broadcast of current metrics.
     * Useful for testing or on-demand updates.
     */
    MEMORY_API void update_metrics_now();

private:
    dashboard_config  config_;
    std::atomic<bool> server_running_{false};
    std::atomic<bool> should_stop_{false};

    // Thread management
    std::thread server_thread_;
    std::thread metrics_thread_;

    // Allocator management
    mutable std::mutex                              allocators_mutex_;
    std::unordered_map<std::string, allocator_info> registered_allocators_;

    // Metrics history
    mutable std::mutex                                                 history_mutex_;
    std::unordered_map<std::string, std::queue<metrics_history_point>> metrics_history_;

    // WebSocket clients
    mutable std::mutex                        clients_mutex_;
    std::unordered_map<int, websocket_client> websocket_clients_;
    std::atomic<int>                          next_client_id_{1};

    /**
     * @brief Main server thread function
     */
    void server_thread_main();

    /**
     * @brief Main metrics collection thread function
     */
    void metrics_thread_main();

    /**
     * @brief Collect current metrics from all registered allocators
     */
    void collect_metrics();

    /**
     * @brief Broadcast metrics update to WebSocket clients
     * @param metrics_json JSON string with current metrics
     */
    void broadcast_metrics_update(const std::string& metrics_json);

    /**
     * @brief Handle HTTP GET request for allocator list
     * @return JSON response with allocator list
     */
    std::string handle_get_allocators() const;

    /**
     * @brief Handle HTTP GET request for specific allocator stats
     * @param allocator_name Name of allocator
     * @return JSON response with allocator statistics
     */
    std::string handle_get_allocator_stats(const std::string& allocator_name) const;

    /**
     * @brief Handle HTTP GET request for allocator history
     * @param allocator_name Name of allocator
     * @param max_points Maximum number of points to return
     * @return JSON response with historical data
     */
    std::string handle_get_allocator_history(
        const std::string& allocator_name, size_t max_points) const;

    /**
     * @brief Handle WebSocket connection
     * @param client_id Unique client identifier
     * @param client_address Client IP address
     */
    void handle_websocket_connection(int client_id, const std::string& client_address);

    /**
     * @brief Handle WebSocket disconnection
     * @param client_id Client identifier
     */
    void handle_websocket_disconnection(int client_id);

    /**
     * @brief Get allocator type name from pointer
     * @param allocator Pointer to allocator
     * @return String representation of allocator type
     */
    static std::string get_allocator_type_name(Allocator* allocator);

    /**
     * @brief Create JSON string from allocator stats
     * @param stats Allocator statistics
     * @return JSON string representation
     */
    std::string stats_to_json(const allocator_stats& stats) const;

    /**
     * @brief Create JSON string from metrics history point
     * @param point Historical metrics point
     * @return JSON string representation
     */
    std::string history_point_to_json(const metrics_history_point& point) const;

    /**
     * @brief Add CORS headers to HTTP response
     * @param headers Map of HTTP headers to modify
     */
    void add_cors_headers(std::unordered_map<std::string, std::string>& headers) const;

    /**
     * @brief Cleanup old history data points
     * @param allocator_name Name of allocator to cleanup
     */
    void cleanup_old_history(const std::string& allocator_name);

    /**
     * @brief Calculate allocation rate from history
     * @param allocator_name Name of allocator
     * @return Allocations per second over last minute
     */
    double calculate_allocation_rate(const std::string& allocator_name) const;
};

}  // namespace memory
