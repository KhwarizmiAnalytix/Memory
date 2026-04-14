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

#include <chrono>
#include <functional>
#include <future>
#include <memory>
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
 * @brief Memory transfer direction enumeration
 */
enum class transfer_direction
{
    HOST_TO_DEVICE,    ///< Transfer from host memory to device memory
    DEVICE_TO_HOST,    ///< Transfer from device memory to host memory
    DEVICE_TO_DEVICE,  ///< Transfer between device memories
    HOST_TO_HOST       ///< Transfer between host memories (for completeness)
};

/**
 * @brief Memory transfer operation status
 */
enum class transfer_status
{
    PENDING,    ///< Transfer is queued but not started
    RUNNING,    ///< Transfer is currently in progress
    COMPLETED,  ///< Transfer completed successfully
    FAILED,     ///< Transfer failed with error
    CANCELLED   ///< Transfer was cancelled
};

/**
 * @brief Memory transfer operation metadata
 *
 * Contains information about a memory transfer operation including
 * timing, bandwidth, and status for performance monitoring and debugging.
 */
struct MEMORY_VISIBILITY gpu_transfer_info
{
    /** @brief Unique transfer ID */
    size_t transfer_id = 0;

    /** @brief Transfer direction */
    transfer_direction direction = transfer_direction::HOST_TO_DEVICE;

    /** @brief Source device (if applicable) */
    device_option source_device{device_enum::CPU, 0};

    /** @brief Destination device (if applicable) */
    device_option destination_device{device_enum::CPU, 0};

    /** @brief Number of bytes transferred */
    size_t bytes_transferred = 0;

    /** @brief Transfer status */
    transfer_status status = transfer_status::PENDING;

    /** @brief Start time of transfer */
    std::chrono::high_resolution_clock::time_point start_time;

    /** @brief End time of transfer */
    std::chrono::high_resolution_clock::time_point end_time;

    /** @brief Achieved bandwidth in GB/s */
    double bandwidth_gbps = 0.0;

    /** @brief Transfer duration in microseconds */
    uint64_t duration_us = 0;

    /** @brief Error message (if failed) */
    std::string error_message;

    /**
     * @brief Get transfer duration in milliseconds
     * @return Duration in milliseconds
     */
    double get_duration_ms() const
    {
        if (status == transfer_status::COMPLETED || status == transfer_status::FAILED)
        {
            auto duration =
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            return duration.count() / 1000.0;
        }
        return 0.0;
    }
};

/**
 * @brief Callback function type for transfer completion
 * @param info Transfer information
 */
using transfer_callback = std::function<void(const gpu_transfer_info& info)>;

/**
 * @brief GPU stream handle abstraction
 *
 * Provides a unified interface for GPU streams across different backends
 * (CUDA streams, etc.).
 */
class MEMORY_VISIBILITY gpu_stream
{
public:
    /**
     * @brief Create a new GPU stream
     * @param device_type Device type (CUDA or HIP)
     * @param device_index Device index
     * @param priority Stream priority (0 = default, higher values = higher priority)
     * @return Unique pointer to created stream
     */
    MEMORY_API static std::unique_ptr<gpu_stream> create(
        device_enum device_type, int device_index, int priority = 0);

    /**
     * @brief Virtual destructor
     */
    MEMORY_API virtual ~gpu_stream() = default;

    /**
     * @brief Get the device associated with this stream
     * @return Device option
     */
    MEMORY_API virtual device_option get_device() const = 0;

    /**
     * @brief Synchronize the stream (wait for all operations to complete)
     */
    MEMORY_API virtual void synchronize() = 0;

    /**
     * @brief Check if all operations in the stream have completed
     * @return True if stream is idle
     */
    MEMORY_API virtual bool is_idle() const = 0;

    /**
     * @brief Get native stream handle (CUDA stream)
     * @return Native handle as void pointer
     */
    MEMORY_API virtual void* get_native_handle() const = 0;

protected:
    gpu_stream() = default;
    MEMORY_DELETE_COPY_AND_MOVE(gpu_stream);
};

/**
 * @brief High-performance GPU memory transfer manager
 *
 * Provides efficient asynchronous memory transfer capabilities between
 * host and device memory with stream management, bandwidth optimization,
 * and comprehensive performance monitoring.
 *
 * Key features:
 * - Asynchronous transfers with callback support
 * - Multi-stream management for overlapping transfers
 * - Automatic bandwidth optimization and monitoring
 * - Support for both CUDA and HIP backends
 * - Memory coalescing for optimal transfer patterns
 * - Transfer queue management and prioritization
 * - Comprehensive error handling and recovery
 *
 * Mathematical foundation:
 * Transfer bandwidth is calculated as: BW = Size / Time
 * Optimal transfer size follows: S_opt = min(max_transfer_size, align(requested_size, alignment))
 * where alignment ensures coalesced memory access patterns.
 *
 * @example
 * ```cpp
 * auto& transfer_manager = gpu_memory_transfer::instance();
 *
 * // Create a high-priority stream for critical transfers
 * auto stream = gpu_stream::create(device_enum::CUDA, 0, 1);
 *
 * // Asynchronous transfer with callback
 * auto future = transfer_manager.transfer_async(
 *     host_data, device_data, size,
 *     transfer_direction::HOST_TO_DEVICE,
 *     stream.get(),
 *     [](const gpu_transfer_info& info) {
 *         std::cout << "Transfer completed: " << info.bandwidth_gbps << " GB/s\n";
 *     }
 * );
 *
 * // Continue with other work...
 * future.wait(); // Wait for completion if needed
 * ```
 */
class MEMORY_VISIBILITY gpu_memory_transfer
{
public:
    /**
     * @brief Get the singleton instance of the memory transfer manager
     * @return Reference to the global transfer manager instance
     */
    MEMORY_API static gpu_memory_transfer& instance();

    /**
     * @brief Virtual destructor
     */
    MEMORY_API virtual ~gpu_memory_transfer() = default;

    /**
     * @brief Perform synchronous memory transfer
     * @param src Source memory pointer
     * @param dst Destination memory pointer
     * @param size Number of bytes to transfer
     * @param direction Transfer direction
     * @param stream GPU stream to use (optional, uses default if null)
     * @return Transfer information with timing and bandwidth data
     * @throws std::invalid_argument if parameters are invalid
     * @throws std::runtime_error if transfer fails
     */
    MEMORY_API virtual gpu_transfer_info transfer_sync(
        const void*        src,
        void*              dst,
        size_t             size,
        transfer_direction direction,
        gpu_stream*        stream = nullptr) = 0;

    /**
     * @brief Perform asynchronous memory transfer
     * @param src Source memory pointer
     * @param dst Destination memory pointer
     * @param size Number of bytes to transfer
     * @param direction Transfer direction
     * @param stream GPU stream to use (optional, uses default if null)
     * @param callback Completion callback (optional)
     * @return Future that can be used to wait for completion and get transfer info
     * @throws std::invalid_argument if parameters are invalid
     */
    MEMORY_API virtual std::future<gpu_transfer_info> transfer_async(
        const void*        src,
        void*              dst,
        size_t             size,
        transfer_direction direction,
        gpu_stream*        stream   = nullptr,
        transfer_callback  callback = nullptr) = 0;

    /**
     * @brief Perform batched memory transfers
     * @param transfers Vector of transfer specifications
     * @param stream GPU stream to use (optional, uses default if null)
     * @return Vector of futures for each transfer
     */
    MEMORY_API virtual std::vector<std::future<gpu_transfer_info>> transfer_batch_async(
        const std::vector<std::tuple<const void*, void*, size_t, transfer_direction>>& transfers,
        gpu_stream* stream = nullptr) = 0;

    /**
     * @brief Get optimal transfer chunk size for given parameters
     * @param total_size Total size to transfer
     * @param direction Transfer direction
     * @param device_type Device type
     * @return Optimal chunk size in bytes
     */
    MEMORY_API virtual size_t get_optimal_chunk_size(
        size_t total_size, transfer_direction direction, device_enum device_type) const = 0;

    /**
     * @brief Get transfer statistics and performance metrics
     * @return String containing detailed transfer statistics
     */
    MEMORY_API virtual std::string get_transfer_statistics() const = 0;

    /**
     * @brief Clear transfer statistics
     */
    MEMORY_API virtual void clear_statistics() = 0;

    /**
     * @brief Wait for all pending transfers to complete
     */
    MEMORY_API virtual void wait_for_all_transfers() = 0;

    /**
     * @brief Cancel all pending transfers
     */
    MEMORY_API virtual void cancel_all_transfers() = 0;

protected:
    gpu_memory_transfer() = default;
    MEMORY_DELETE_COPY_AND_MOVE(gpu_memory_transfer);
};

}  // namespace gpu
}  // namespace memory
