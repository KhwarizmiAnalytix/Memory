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

#include "common/device.h"
#include "gpu/cuda_caching_allocator.h"
#include "common/memory_export.h"

namespace memory
{
namespace gpu
{

/**
 * @brief GPU memory allocation strategies for different use cases
 *
 * Each strategy is optimized for specific allocation patterns common
 * in quantitative finance applications:
 *
 * - DIRECT: Direct CUDA malloc/free for simple, infrequent allocations
 * - POOL: Memory pool for frequent allocations of similar sizes
 * - CACHING: Intelligent caching for complex allocation patterns
 */
enum class gpu_allocation_strategy
{
    DIRECT  = 0,  ///< Direct CUDA allocation (cudaMalloc/cudaFree)
    POOL    = 1,  ///< Memory pool-based allocation
    CACHING = 2   ///< CUDA caching allocator with stream awareness
};

/**
 * @brief Configuration for GPU allocator creation
 *
 * Provides comprehensive configuration options for different allocation
 * strategies, allowing fine-tuning for specific quantitative applications.
 */
struct MEMORY_VISIBILITY gpu_allocator_config
{
    gpu_allocation_strategy strategy     = gpu_allocation_strategy::DIRECT;
    device_enum             device_type  = device_enum::CUDA;
    int                     device_index = 0;
    size_t                  alignment    = 256ULL;  ///< Memory alignment in bytes

    // Pool-specific configuration
    size_t pool_min_block_size = 1024;                   ///< Minimum pool block size
    size_t pool_max_block_size = 64ULL * 1024ULL;        ///< Maximum pool block size
    double pool_growth_factor  = 1.5;                    ///< Pool growth factor
    size_t pool_max_size       = 1ULL * 1024ULL * 1024;  ///< Maximum pool size

    // Caching-specific configuration
    size_t cache_max_bytes = std::numeric_limits<size_t>::max();  ///< Maximum cache size

    /**
     * @brief Create default configuration for strategy
     * @param strategy Allocation strategy
     * @param device_index CUDA device index
     * @return Optimized configuration
     */
    MEMORY_API static gpu_allocator_config create_default(
        gpu_allocation_strategy strategy, int device_index = 0);

    /**
     * @brief Create configuration optimized for Monte Carlo simulations
     * @param device_index CUDA device index
     * @return Monte Carlo optimized configuration
     */
    MEMORY_API static gpu_allocator_config create_monte_carlo_optimized(int device_index = 0);

    /**
     * @brief Create configuration optimized for PDE solvers
     * @param device_index CUDA device index
     * @return PDE solver optimized configuration
     */
    MEMORY_API static gpu_allocator_config create_pde_optimized(int device_index = 0);
};

/**
 * @brief Factory for creating GPU allocators with different strategies
 *
 * This factory provides a unified interface for creating GPU allocators
 * optimized for different use cases in quantitative finance. It handles
 * device validation, configuration optimization, and allocator lifecycle.
 *
 * Features:
 * - Strategy-based allocator selection
 * - Device validation and management
 * - Configuration optimization for specific use cases
 * - Thread-safe allocator creation
 * - Comprehensive error handling
 *
 * @note All allocators created by this factory are thread-safe
 */
class MEMORY_VISIBILITY gpu_allocator_factory
{
public:
    /**
     * @brief Create a GPU allocator with specified configuration
     * @tparam T Element type for template allocators
     * @param config Allocator configuration
     * @return Unique pointer to created allocator
     * @throws std::invalid_argument if configuration is invalid
     * @throws std::runtime_error if device is unavailable
     *
     * @deprecated This method is deprecated after gpu_allocator removal.
     *             Use direct CUDA allocation or create_caching_allocator() instead.
     */
    template <typename T>
    static std::unique_ptr<void> create_allocator(const gpu_allocator_config& config);

    /**
     * @brief Create a caching allocator with template interface
     * @tparam T Element type
     * @tparam alignment Memory alignment requirement
     * @param config Allocator configuration
     * @return Unique pointer to caching allocator
     */
    template <typename T, std::size_t alignment = 256ULL>
    static std::unique_ptr<cuda_caching_allocator_template<T, alignment>> create_caching_allocator(
        const gpu_allocator_config& config)
    {
        // Validate device support
        if (!validate_device_support(config.strategy, config.device_type, config.device_index))
        {
            throw std::runtime_error("Device does not support caching allocation strategy");
        }

        // Create and return the caching allocator
        return std::make_unique<cuda_caching_allocator_template<T, alignment>>(
            config.device_index, config.cache_max_bytes);
    }

    /**
     * @brief Get recommended strategy for allocation pattern
     * @param avg_allocation_size Average allocation size in bytes
     * @param allocation_frequency Allocations per second (approximate)
     * @param allocation_lifetime Average lifetime in seconds
     * @return Recommended allocation strategy
     */
    MEMORY_API static gpu_allocation_strategy recommend_strategy(
        size_t avg_allocation_size, double allocation_frequency, double allocation_lifetime);

    /**
     * @brief Validate device availability for strategy
     * @param strategy Allocation strategy
     * @param device_type Device type
     * @param device_index Device index
     * @return true if device supports strategy
     */
    MEMORY_API static bool validate_device_support(
        gpu_allocation_strategy strategy, device_enum device_type, int device_index);

    /**
     * @brief Get human-readable strategy name
     * @param strategy Allocation strategy
     * @return Strategy name string
     */
    MEMORY_API static std::string strategy_name(gpu_allocation_strategy strategy);

private:
    gpu_allocator_factory() = delete;  // Static factory only
};

}  // namespace gpu
}  // namespace memory
