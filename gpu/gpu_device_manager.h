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
#include <vector>

#include "common/memory_macros.h"
#include "common/device.h"

#if MEMORY_HAS_CUDA
#include <cuda_runtime.h>
#include "common/memory_export.h"
#endif

namespace memory
{
namespace gpu
{

/**
 * @brief GPU device capabilities and properties
 *
 * Contains detailed information about a GPU device including compute
 * capabilities, memory specifications, and performance characteristics
 * relevant for Monte Carlo simulations and PDE solvers.
 */
struct MEMORY_VISIBILITY gpu_device_info
{
    /** @brief Device index */
    int device_index = -1;

    /** @brief Device type (CUDA or HIP) */
    device_enum device_type = device_enum::CPU;

    /** @brief Device name */
    std::string name;

    /** @brief Total global memory in bytes */
    size_t total_memory = 0;

    /** @brief Available memory in bytes */
    size_t available_memory = 0;

    /** @brief Maximum memory allocation size in bytes */
    size_t max_allocation_size = 0;

    /** @brief Number of multiprocessors/compute units */
    int multiprocessor_count = 0;

    /** @brief Maximum threads per block */
    int max_threads_per_block = 0;

    /** @brief Maximum block dimensions [x, y, z] */
    int max_block_dims[3] = {0, 0, 0};

    /** @brief Maximum grid dimensions [x, y, z] */
    int max_grid_dims[3] = {0, 0, 0};

    /** @brief Shared memory per block in bytes */
    size_t shared_memory_per_block = 0;

    /** @brief Memory bus width in bits */
    int memory_bus_width = 0;

    /** @brief Memory clock rate in kHz */
    int memory_clock_rate = 0;

    /** @brief Compute capability major version (CUDA) */
    int compute_capability_major = 0;

    /** @brief Compute capability minor version (CUDA) */
    int compute_capability_minor = 0;

    /** @brief Whether device supports double precision */
    bool supports_double_precision = false;

    /** @brief Whether device supports unified memory */
    bool supports_unified_memory = false;

    /** @brief Whether device supports concurrent kernels */
    bool supports_concurrent_kernels = false;

    /** @brief PCI bus ID */
    std::string pci_bus_id;

    /** @brief Device utilization percentage (0-100) */
    float utilization_percentage = 0.0f;

    /** @brief Device temperature in Celsius */
    float temperature_celsius = 0.0f;

    /** @brief Power consumption in watts */
    float power_consumption_watts = 0.0f;

    /** @brief Memory bandwidth in GB/s */
    double memory_bandwidth_gb_per_sec = 0.0;

    /** @brief Total memory in bytes (alias for total_memory for compatibility) */
    size_t total_memory_bytes = 0;
};

/**
 * @brief GPU runtime environment information
 *
 * Contains information about the available GPU runtime environments
 * and their capabilities for optimal backend selection.
 */
struct MEMORY_VISIBILITY gpu_runtime_info
{
    /** @brief Whether CUDA runtime is available */
    bool cuda_available = false;

    /** @brief CUDA runtime version */
    int cuda_runtime_version = 0;

    /** @brief CUDA driver version */
    int cuda_driver_version = 0;

    /** @brief Number of CUDA devices */
    int cuda_device_count = 0;

    /** @brief Recommended backend for current system */
    device_enum recommended_backend = device_enum::CPU;
};

/**
 * @brief GPU device manager for runtime detection and device selection
 *
 * Provides comprehensive GPU device management capabilities including
 * runtime detection, device enumeration, capability querying, and
 * optimal device selection for Monte Carlo simulations and PDE solvers.
 *
 * Key features:
 * - Automatic CUDA runtime detection
 * - Device capability enumeration and comparison
 * - Optimal device selection based on workload characteristics
 * - Device health monitoring and utilization tracking
 * - Thread-safe device context management
 * - Performance benchmarking for device ranking
 *
 * The manager uses heuristics to select the best available device
 * based on memory capacity, compute capability, and current utilization.
 * For Monte Carlo simulations, it prioritizes devices with high memory
 * bandwidth and many cores. For PDE solvers, it considers shared memory
 * capacity and double precision performance.
 *
 * @example
 * ```cpp
 * auto& manager = gpu_device_manager::instance();
 *
 * // Get runtime information
 * auto runtime_info = manager.get_runtime_info();
 * if (runtime_info.cuda_available) {
 *     std::cout << "CUDA " << runtime_info.cuda_runtime_version << " available\n";
 * }
 *
 * // Find best device for Monte Carlo simulation
 * auto device = manager.select_optimal_device_for_monte_carlo(1024ULL * 1024); // 1GB
 * if (device.device_index >= 0) {
 *     std::cout << "Selected device: " << device.name << "\n";
 * }
 * ```
 */
class MEMORY_VISIBILITY gpu_device_manager
{
public:
    /**
     * @brief Get the singleton instance of the device manager
     * @return Reference to the global device manager instance
     */
    MEMORY_API static gpu_device_manager& instance();

    /**
     * @brief Virtual destructor
     */
    MEMORY_API virtual ~gpu_device_manager() = default;

    /**
     * @brief Initialize the device manager and detect available GPUs
     * @throws std::runtime_error if initialization fails
     */
    MEMORY_API virtual void initialize() = 0;

    /**
     * @brief Get information about available GPU runtimes
     * @return Runtime information structure
     */
    MEMORY_API virtual gpu_runtime_info get_runtime_info() const = 0;

    /**
     * @brief Get list of all available GPU devices
     * @return Vector of device information structures
     */
    MEMORY_API virtual std::vector<gpu_device_info> get_available_devices() const = 0;

    /**
     * @brief Get information about a specific device
     * @param device_type Device type (CUDA or HIP)
     * @param device_index Device index
     * @return Device information structure
     * @throws std::invalid_argument if device is not found
     */
    MEMORY_API virtual gpu_device_info get_device_info(
        device_enum device_type, int device_index) const = 0;

    /**
     * @brief Check if a device is currently available and healthy
     * @param device_type Device type
     * @param device_index Device index
     * @return True if device is available and healthy
     */
    MEMORY_API virtual bool is_device_available(
        device_enum device_type, int device_index) const = 0;

    /**
     * @brief Set the current device context
     * @param device_type Device type
     * @param device_index Device index
     * @throws std::runtime_error if device context cannot be set
     */
    MEMORY_API virtual void set_device_context(device_enum device_type, int device_index) = 0;

    /**
     * @brief Get the currently active device
     * @return Current device information
     */
    MEMORY_API virtual gpu_device_info get_current_device() const = 0;

    /**
     * @brief Refresh device information and update utilization statistics
     */
    MEMORY_API virtual void refresh_device_info() = 0;

    /**
     * @brief Get detailed system report including all devices and capabilities
     * @return Formatted string with comprehensive system information
     */
    MEMORY_API virtual std::string get_system_report() const = 0;

protected:
    gpu_device_manager() = default;
    MEMORY_DELETE_COPY_AND_MOVE(gpu_device_manager);
};

}  // namespace gpu
}  // namespace memory
