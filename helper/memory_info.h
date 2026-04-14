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
 * - Separated compression functionality to dedicated module
 *
 * Contact: licensing@quarisma.co.uk
 * Website: https://www.quarisma.co.uk
 */

#pragma once

#include <cstdint>
#include <type_traits>

#include "common/memory_macros.h"
#include "common/memory_export.h"

namespace memory
{
/**
 * @brief Platform abstraction layer for memory management and system information.
 *
 * The port namespace provides cross-platform abstractions for memory allocation,
 * system information retrieval, and platform-specific optimizations. It serves
 * as the foundation for higher-level memory management systems in Memory.
 *
 * **Key Features**:
 * - Cross-platform aligned memory allocation
 * - System memory information and monitoring
 * - Memory bandwidth analysis
 * - Platform-specific optimizations
 * - NUMA awareness and affinity control
 *
 * **Design Principles**:
 * - Platform abstraction with consistent interface
 * - High-performance memory operations
 * - Comprehensive system information access
 * - Integration with system memory managers
 * - Support for specialized hardware configurations
 */
namespace port
{

/**
 * @brief Comprehensive system memory information.
 *
 * Contains detailed information about system memory availability
 * and usage for monitoring and resource management decisions.
 */
struct memory_info
{
    /**
     * @brief Total system memory in bytes.
     *
     * **Value**: Total physical RAM installed in the system
     * **Special**: INT64_MAX if information unavailable
     * **Use Cases**: Resource planning, capacity analysis
     */
    int64_t total{0};

    /**
     * @brief Available free memory in bytes.
     *
     * **Value**: Memory currently available for allocation
     * **Special**: INT64_MAX if information unavailable
     * **Dynamic**: Changes frequently based on system activity
     * **Use Cases**: Memory pressure detection, allocation decisions
     */
    int64_t free{0};
};

/**
 * @brief System memory bandwidth utilization information.
 *
 * Provides insights into memory bandwidth usage across the system
 * for performance analysis and optimization decisions.
 */
struct memory_bandwidth_info
{
    /**
     * @brief Current memory bandwidth utilization.
     *
     * **Value**: Memory bandwidth used across all CPUs in MB/second
     * **Special**: INT64_MAX if information unavailable
     * **Measurement**: Aggregate across all CPU cores and memory controllers
     * **Use Cases**: Performance bottleneck analysis, NUMA optimization
     */
    int64_t bw_used{0};
};

/**
 * @brief Retrieves comprehensive host memory information.
 *
 * Queries the operating system for detailed memory statistics including
 * total installed memory and currently available free memory.
 *
 * @return memory_info structure with current memory statistics
 *
 * **Availability**: Fields set to INT64_MAX if information unavailable
 * **Performance**: May involve system calls - consider caching results
 * **Thread Safety**: Thread-safe
 * **Platform**: Cross-platform implementation with OS-specific backends
 *
 * **Use Cases**:
 * - Memory pressure monitoring
 * - Resource allocation decisions
 * - System health monitoring
 * - Capacity planning and analysis
 *
 * **Example**:
 * ```cpp
 * auto mem_info = GetMemoryInfo();
 * if (mem_info.free != INT64_MAX && mem_info.free < threshold) {
 *     // Handle low memory condition
 * }
 * ```
 */
MEMORY_API memory_info GetMemoryInfo();

/**
 * @brief Retrieves host memory bandwidth utilization information.
 *
 * Queries system performance counters to determine current memory
 * bandwidth usage across all CPU cores and memory controllers.
 *
 * @return memory_bandwidth_info structure with bandwidth statistics
 *
 * **Availability**: Fields set to INT64_MAX if information unavailable
 * **Performance**: May involve performance counter queries
 * **Thread Safety**: Thread-safe
 * **Platform**: Requires platform-specific performance monitoring support
 *
 * **Use Cases**:
 * - Performance bottleneck identification
 * - Memory-bound operation optimization
 * - NUMA topology analysis
 * - System performance monitoring
 *
 * **Limitations**: Not all platforms provide bandwidth information
 */
MEMORY_API memory_bandwidth_info GetMemoryBandwidthInfo();

/**
 * @brief Returns amount of available RAM in bytes.
 *
 * Convenience function that extracts the free memory amount from
 * system memory information. Provides quick access to available
 * memory for allocation decisions.
 *
 * @return Available RAM in bytes, or INT64_MAX if unknown
 *
 * **Performance**: Involves system call - consider caching for frequent use
 * **Thread Safety**: Thread-safe
 * **Convenience**: Equivalent to GetMemoryInfo().free
 *
 * **Use Cases**:
 * - Quick memory availability checks
 * - Allocation size validation
 * - Memory pressure detection
 * - Resource planning decisions
 *
 * **Example**:
 * ```cpp
 * int64_t available = available_ram();
 * if (available != INT64_MAX && requested_size > available * 0.8) {
 *     // Allocation might cause memory pressure
 * }
 * ```
 */
MEMORY_NODISCARD inline int64_t available_ram() noexcept
{
    return GetMemoryInfo().free;
}

}  // namespace port
}  // namespace memory
