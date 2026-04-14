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
 * Contact: licensing@quarisma.co.uk
 * Website: https://www.quarisma.co.uk
 */

#pragma once

#include <memory>
#include <string>

#include "common/memory_macros.h"
#include "cpu/allocator.h"
#include "common/memory_export.h"

namespace memory
{

/**
 * @brief Allocation context describing workload characteristics
 *
 * Provides information about allocation patterns to help select
 * the optimal allocator for specific use cases.
 */
struct MEMORY_VISIBILITY allocation_context
{
    size_t allocation_size{0};           ///< Typical allocation size in bytes
    size_t min_size{0};                  ///< Minimum allocation size
    size_t max_size{0};                  ///< Maximum allocation size
    size_t estimated_frequency{0};       ///< Estimated allocations per second
    size_t estimated_lifetime_ms{0};     ///< Estimated lifetime in milliseconds
    size_t thread_count{1};              ///< Number of concurrent threads
    bool   size_predictable{false};      ///< Whether sizes are predictable
    bool   lifetime_predictable{false};  ///< Whether lifetimes are predictable
    bool   memory_constrained{false};    ///< Whether running under memory pressure
    bool   require_tracking{false};      ///< Whether detailed tracking is needed

    /**
     * @brief Create context for small, high-frequency allocations
     */
    static allocation_context high_frequency_small()
    {
        allocation_context ctx;
        ctx.allocation_size       = 256;
        ctx.min_size              = 64;
        ctx.max_size              = 1024;
        ctx.estimated_frequency   = 10000;
        ctx.estimated_lifetime_ms = 10;
        ctx.size_predictable      = true;
        ctx.lifetime_predictable  = true;
        return ctx;
    }

    /**
     * @brief Create context for large, long-lived allocations
     */
    static allocation_context large_long_lived()
    {
        allocation_context ctx;
        ctx.allocation_size       = 10ULL * 1024ULL * 1024ULL;   // 10 MB
        ctx.min_size              = 1ULL * 1024ULL * 1024ULL;    // 1 MB
        ctx.max_size              = 100ULL * 1024ULL * 1024ULL;  // 100 MB
        ctx.estimated_frequency   = 100;
        ctx.estimated_lifetime_ms = 60000;  // 1 minute
        ctx.size_predictable      = true;
        ctx.lifetime_predictable  = true;
        return ctx;
    }

    /**
     * @brief Create context for general-purpose allocations
     */
    static allocation_context general_purpose()
    {
        allocation_context ctx;
        ctx.allocation_size       = 4096;
        ctx.min_size              = 16;
        ctx.max_size              = 1ULL * 1024ULL * 1024ULL;
        ctx.estimated_frequency   = 1000;
        ctx.estimated_lifetime_ms = 1000;
        ctx.size_predictable      = false;
        ctx.lifetime_predictable  = false;
        return ctx;
    }
};

/**
 * @brief Allocator selection policy for optimal performance
 *
 * Provides intelligent allocator selection based on workload characteristics,
 * performance requirements, and system constraints. Implements the strategy
 * described in docs/Allocator_Selection_Strategy.md.
 *
 * **Thread Safety**: Thread-safe for read operations, not thread-safe for configuration
 * **Performance**: O(1) selection based on pre-computed decision tree
 */
class MEMORY_VISIBILITY allocator_selector
{
public:
    /**
     * @brief Allocator recommendation with rationale
     */
    struct recommendation
    {
        std::string allocator_type;  ///< Recommended allocator type
        std::string rationale;       ///< Explanation for recommendation
        std::string configuration;   ///< Suggested configuration parameters
        double      confidence;      ///< Confidence score (0.0 to 1.0)
    };

    /**
     * @brief Get allocator recommendation based on context
     *
     * @param ctx Allocation context describing workload
     * @return Recommendation with allocator type and configuration
     *
     * **Algorithm**: Decision tree based on allocation patterns
     * **Performance**: O(1) - simple conditional checks
     * **Thread Safety**: Thread-safe (read-only operation)
     */
    MEMORY_API static recommendation recommend(const allocation_context& ctx);

    /**
     * @brief Get detailed analysis of allocation context
     *
     * @param ctx Allocation context to analyze
     * @return Multi-line string with detailed analysis
     *
     * **Use Cases**: Debugging, performance tuning, documentation
     * **Performance**: O(1) - string formatting only
     */
    MEMORY_API static std::string analyze_context(const allocation_context& ctx);

    /**
     * @brief Compare multiple allocators for given context
     *
     * @param ctx Allocation context
     * @return Vector of recommendations sorted by confidence
     *
     * **Use Cases**: Evaluating alternatives, A/B testing
     * **Performance**: O(n) where n is number of allocator types
     */
    MEMORY_API static std::vector<recommendation> compare_allocators(
        const allocation_context& ctx);

    /**
     * @brief Validate allocator choice against context
     *
     * @param allocator_type Chosen allocator type
     * @param ctx Allocation context
     * @return Validation result with warnings/suggestions
     *
     * **Use Cases**: Configuration validation, code review
     * **Performance**: O(1) - simple checks
     */
    MEMORY_API static std::string validate_choice(
        const std::string& allocator_type, const allocation_context& ctx);

private:
    /**
     * @brief Calculate confidence score for allocator choice
     */
    static double calculate_confidence(
        const std::string& allocator_type, const allocation_context& ctx);

    /**
     * @brief Generate configuration string for allocator
     */
    static std::string generate_configuration(
        const std::string& allocator_type, const allocation_context& ctx);

    /**
     * @brief Check if pool allocator is suitable
     */
    static bool is_pool_suitable(const allocation_context& ctx);

    /**
     * @brief Check if BFC allocator is suitable
     */
    static bool is_bfc_suitable(const allocation_context& ctx);

    /**
     * @brief Check if CPU allocator is suitable
     */
    static bool is_cpu_suitable(const allocation_context& ctx);
};

/**
 * @brief Adaptive allocator manager with runtime selection
 *
 * Manages multiple allocator instances and selects the optimal one
 * based on runtime allocation patterns. Provides automatic adaptation
 * and performance monitoring.
 *
 * **Thread Safety**: Fully thread-safe with internal synchronization
 * **Performance**: Minimal overhead for allocator selection
 *
 * **Example Usage**:
 * ```cpp
 * adaptive_allocator_manager manager;
 * manager.initialize();
 *
 * // Get optimal allocator for context
 * allocation_context ctx = allocation_context::high_frequency_small();
 * Allocator* alloc = manager.get_allocator(ctx);
 *
 * // Use allocator
 * void* ptr = alloc->allocate_raw(64, 256);
 * // ...
 * alloc->deallocate_raw(ptr);
 *
 * // Get performance report
 * std::string report = manager.generate_report();
 * ```
 */
class MEMORY_VISIBILITY adaptive_allocator_manager
{
public:
    /**
     * @brief Construct adaptive allocator manager
     */
    MEMORY_API adaptive_allocator_manager();

    /**
     * @brief Destructor - cleans up managed allocators
     */
    MEMORY_API ~adaptive_allocator_manager();

    /**
     * @brief Initialize allocator instances
     *
     * @param enable_pool Whether to create pool allocator
     * @param enable_bfc Whether to create BFC allocator
     * @param enable_tracking Whether to enable tracking wrappers
     *
     * **Thread Safety**: Not thread-safe - call before concurrent use
     * **Performance**: One-time initialization cost
     */
    MEMORY_API void initialize(
        bool enable_pool = true, bool enable_bfc = true, bool enable_tracking = false);

    /**
     * @brief Get optimal allocator for given context
     *
     * @param ctx Allocation context
     * @return Pointer to optimal allocator (not owned by caller)
     *
     * **Thread Safety**: Thread-safe
     * **Performance**: O(1) - fast lookup
     */
    MEMORY_API Allocator* get_allocator(const allocation_context& ctx);

    /**
     * @brief Generate performance report for all allocators
     *
     * @return Multi-line string with statistics and recommendations
     *
     * **Thread Safety**: Thread-safe
     * **Performance**: O(n) where n is number of allocators
     */
    MEMORY_API std::string generate_report() const;

    /**
     * @brief Get statistics for specific allocator type
     *
     * @param allocator_type Type of allocator
     * @return Statistics if available
     *
     * **Thread Safety**: Thread-safe
     */
    MEMORY_API std::optional<allocator_stats> get_stats(const std::string& allocator_type) const;

    /**
     * @brief Reset all allocator statistics
     *
     * **Thread Safety**: Thread-safe
     */
    MEMORY_API void reset_stats();

private:
    struct impl;
    std::unique_ptr<impl> pimpl_;
};

}  // namespace memory
