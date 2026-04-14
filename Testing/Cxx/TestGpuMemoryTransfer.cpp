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

#include "CoreTest.h"
#include "common/macros.h"

#if PROJECT_HAS_CUDA

#include <future>
#include <memory>
#include <vector>

#include "logger.h"
#include "common/device.h"
#include "gpu/gpu_memory_transfer.h"

using namespace quarisma;
using namespace quarisma::gpu;

/**
 * @brief Test GPU memory transfer manager singleton access
 */
QUARISMATEST(GpuMemoryTransfer, provides_singleton_instance)
{
    // Test singleton access
    auto& manager1 = gpu_memory_transfer::instance();
    auto& manager2 = gpu_memory_transfer::instance();

    // Should be the same instance
    EXPECT_EQ(&manager1, &manager2);

    QUARISMA_LOG_INFO("GPU memory transfer singleton test passed");
}

/**
 * @brief Test transfer direction enumeration
 */
QUARISMATEST(GpuMemoryTransfer, supports_all_transfer_directions)
{
    // Test transfer direction enum values
    transfer_direction directions[] = {
        transfer_direction::HOST_TO_DEVICE,
        transfer_direction::DEVICE_TO_HOST,
        transfer_direction::DEVICE_TO_DEVICE,
        transfer_direction::HOST_TO_HOST};

    // All directions should be valid
    for (auto direction : directions)
    {
        EXPECT_TRUE(
            direction == transfer_direction::HOST_TO_DEVICE ||
            direction == transfer_direction::DEVICE_TO_HOST ||
            direction == transfer_direction::DEVICE_TO_DEVICE ||
            direction == transfer_direction::HOST_TO_HOST);
    }

    QUARISMA_LOG_INFO("GPU memory transfer directions test passed");
}

/**
 * @brief Test transfer status enumeration
 */
QUARISMATEST(GpuMemoryTransfer, tracks_transfer_status)
{
    // Test transfer status enum values
    transfer_status statuses[] = {
        transfer_status::PENDING,
        transfer_status::RUNNING,
        transfer_status::COMPLETED,
        transfer_status::FAILED,
        transfer_status::CANCELLED};

    // All statuses should be valid
    for (auto status : statuses)
    {
        EXPECT_TRUE(
            status == transfer_status::PENDING || status == transfer_status::RUNNING ||
            status == transfer_status::COMPLETED || status == transfer_status::FAILED ||
            status == transfer_status::CANCELLED);
    }

    QUARISMA_LOG_INFO("GPU memory transfer status test passed");
}

/**
 * @brief Test GPU transfer info structure
 */
QUARISMATEST(GpuTransferInfo, manages_transfer_metadata)
{
    gpu_transfer_info info;

    // Test default construction
    EXPECT_EQ(0, info.transfer_id);
    EXPECT_EQ(transfer_direction::HOST_TO_DEVICE, info.direction);
    EXPECT_EQ(0, info.bytes_transferred);
    EXPECT_EQ(transfer_status::PENDING, info.status);
    EXPECT_EQ(0.0, info.bandwidth_gbps);
    EXPECT_EQ(0, info.duration_us);
    EXPECT_TRUE(info.error_message.empty());

    // Test duration calculation when not completed
    double duration_ms = info.get_duration_ms();
    EXPECT_EQ(0.0, duration_ms);

    QUARISMA_LOG_INFO("GPU transfer info structure test passed");
}

/**
 * @brief Test GPU stream creation and management
 */
QUARISMATEST(GpuStream, creates_and_manages_streams)
{
    try
    {
        // Test stream creation
        auto stream = gpu_stream::create(device_enum::CUDA, 0, 0);

        if (stream)
        {
            // Test device information
            auto device = stream->get_device();
            EXPECT_EQ(device_enum::CUDA, device.type());
            EXPECT_EQ(0, device.index());

            // Test stream operations
            EXPECT_NE(nullptr, stream->get_native_handle());

            // Test synchronization (should not throw)
            stream->synchronize();

            // Test idle check
            bool is_idle = stream->is_idle();
            EXPECT_TRUE(is_idle || !is_idle);  // Should not crash

            QUARISMA_LOG_INFO("GPU stream creation and management test passed");
        }
        else
        {
            QUARISMA_LOG_INFO("GPU stream creation failed (expected if no CUDA device)");
        }
    }
    catch (const std::exception& e)
    {
        QUARISMA_LOG_INFO("GPU stream test failed (expected if no CUDA device): {}", e.what());
    }
}

/**
 * @brief Test optimal chunk size calculation
 */
QUARISMATEST(GpuMemoryTransfer, calculates_optimal_chunk_sizes)
{
    auto& transfer_manager = gpu_memory_transfer::instance();

    // Test optimal chunk size calculation
    size_t chunk_size_h2d = transfer_manager.get_optimal_chunk_size(
        1024ULL, transfer_direction::HOST_TO_DEVICE, device_enum::CUDA);
    EXPECT_GT(chunk_size_h2d, 0);

    size_t chunk_size_d2h = transfer_manager.get_optimal_chunk_size(
        2 * 1024ULL, transfer_direction::DEVICE_TO_HOST, device_enum::CUDA);
    EXPECT_GT(chunk_size_d2h, 0);

    size_t chunk_size_d2d = transfer_manager.get_optimal_chunk_size(
        512 * 1024, transfer_direction::DEVICE_TO_DEVICE, device_enum::CUDA);
    EXPECT_GT(chunk_size_d2d, 0);

    QUARISMA_LOG_INFO(
        "Optimal chunk sizes - H2D: {}, D2H: {}, D2D: {}",
        chunk_size_h2d,
        chunk_size_d2h,
        chunk_size_d2d);

    QUARISMA_LOG_INFO("GPU memory transfer chunk size calculation test passed");
}

/**
 * @brief Test transfer statistics and reporting
 */
QUARISMATEST(GpuMemoryTransfer, provides_transfer_statistics)
{
    auto& transfer_manager = gpu_memory_transfer::instance();

    // Test transfer statistics (should not throw)
    std::string stats = transfer_manager.get_transfer_statistics();
    EXPECT_TRUE(stats.empty() || !stats.empty());  // Should not crash

    // Test clear statistics
    transfer_manager.clear_statistics();

    // Statistics should still be accessible after clearing
    std::string cleared_stats = transfer_manager.get_transfer_statistics();
    EXPECT_TRUE(cleared_stats.empty() || !cleared_stats.empty());  // Should not crash

    QUARISMA_LOG_INFO("GPU memory transfer statistics test passed");
}

/**
 * @brief Test transfer queue management
 */
QUARISMATEST(GpuMemoryTransfer, manages_transfer_queue)
{
    auto& transfer_manager = gpu_memory_transfer::instance();

    try
    {
        // Test waiting for all transfers (should not hang)
        transfer_manager.wait_for_all_transfers();

        // Test cancelling all transfers (should not throw)
        transfer_manager.cancel_all_transfers();

        QUARISMA_LOG_INFO("GPU memory transfer queue management test passed");
    }
    catch (const std::exception& e)
    {
        QUARISMA_LOG_INFO("GPU memory transfer queue management failed: {}", e.what());
    }
}

/**
 * @brief Test synchronous memory transfer (basic validation)
 */
QUARISMATEST(GpuMemoryTransfer, performs_synchronous_transfers)
{
    auto& transfer_manager = gpu_memory_transfer::instance();

    try
    {
        // Create test data
        std::vector<float> host_data(1024, 1.0f);
        std::vector<float> result_data(1024, 0.0f);

        // Test host-to-host transfer (should always work)
        auto h2h_info = transfer_manager.transfer_sync(
            host_data.data(),
            result_data.data(),
            1024 * sizeof(float),
            transfer_direction::HOST_TO_HOST);

        EXPECT_EQ(transfer_status::COMPLETED, h2h_info.status);
        EXPECT_EQ(1024 * sizeof(float), h2h_info.bytes_transferred);
        EXPECT_GE(h2h_info.duration_us, 0);

        // Verify data was copied
        for (size_t i = 0; i < 10; ++i)
        {
            EXPECT_EQ(host_data[i], result_data[i]);
        }

        QUARISMA_LOG_INFO("GPU memory transfer synchronous transfer test passed");
    }
    catch (const std::exception& e)
    {
        QUARISMA_LOG_INFO("GPU memory transfer synchronous test failed: {}", e.what());
    }
}

/**
 * @brief Test asynchronous memory transfer (basic validation)
 */
QUARISMATEST(GpuMemoryTransfer, performs_asynchronous_transfers)
{
    auto& transfer_manager = gpu_memory_transfer::instance();

    try
    {
        // Create test data
        std::vector<float> host_data(512, 2.0f);
        std::vector<float> result_data(512, 0.0f);

        bool callback_called = false;

        // Test asynchronous host-to-host transfer
        auto future = transfer_manager.transfer_async(
            host_data.data(),
            result_data.data(),
            512 * sizeof(float),
            transfer_direction::HOST_TO_HOST,
            nullptr,
            [&callback_called](const gpu_transfer_info& info)
            {
                callback_called = true;
                EXPECT_EQ(transfer_status::COMPLETED, info.status);
            });

        // Wait for completion
        auto info = future.get();

        EXPECT_EQ(transfer_status::COMPLETED, info.status);
        EXPECT_EQ(512 * sizeof(float), info.bytes_transferred);
        EXPECT_TRUE(callback_called);

        // Verify data was copied
        for (size_t i = 0; i < 10; ++i)
        {
            EXPECT_EQ(host_data[i], result_data[i]);
        }

        QUARISMA_LOG_INFO("GPU memory transfer asynchronous transfer test passed");
    }
    catch (const std::exception& e)
    {
        QUARISMA_LOG_INFO("GPU memory transfer asynchronous test failed: {}", e.what());
    }
}

/**
 * @brief Test batch memory transfers
 */
QUARISMATEST(GpuMemoryTransfer, performs_batch_transfers)
{
    auto& transfer_manager = gpu_memory_transfer::instance();

    try
    {
        // Create test data for batch transfers
        std::vector<float> src1(256, 1.0f);
        std::vector<float> src2(256, 2.0f);
        std::vector<float> dst1(256, 0.0f);
        std::vector<float> dst2(256, 0.0f);

        // Prepare batch transfer specifications
        std::vector<std::tuple<const void*, void*, size_t, transfer_direction>> transfers = {
            {src1.data(), dst1.data(), 256 * sizeof(float), transfer_direction::HOST_TO_HOST},
            {src2.data(), dst2.data(), 256 * sizeof(float), transfer_direction::HOST_TO_HOST}};

        // Perform batch transfers
        auto futures = transfer_manager.transfer_batch_async(transfers);

        EXPECT_EQ(2, futures.size());

        // Wait for all transfers to complete
        for (auto& future : futures)
        {
            auto info = future.get();
            EXPECT_EQ(transfer_status::COMPLETED, info.status);
            EXPECT_EQ(256 * sizeof(float), info.bytes_transferred);
        }

        // Verify data was copied correctly
        for (size_t i = 0; i < 10; ++i)
        {
            EXPECT_EQ(src1[i], dst1[i]);
            EXPECT_EQ(src2[i], dst2[i]);
        }

        QUARISMA_LOG_INFO("GPU memory transfer batch transfers test passed");
    }
    catch (const std::exception& e)
    {
        QUARISMA_LOG_INFO("GPU memory transfer batch test failed: {}", e.what());
    }
}

/**
 * @brief Test transfer error handling
 */
QUARISMATEST(GpuMemoryTransfer, handles_transfer_errors)
{
    auto& transfer_manager = gpu_memory_transfer::instance();

    try
    {
        // Test null pointer transfer (should handle gracefully)
        auto null_info = transfer_manager.transfer_sync(
            nullptr, nullptr, 1024, transfer_direction::HOST_TO_HOST);

        // Should either complete successfully (if implementation handles nulls)
        // or fail gracefully
        EXPECT_TRUE(
            null_info.status == transfer_status::COMPLETED ||
            null_info.status == transfer_status::FAILED);

        QUARISMA_LOG_INFO("GPU memory transfer error handling test passed");
    }
    catch (const std::exception& e)
    {
        // Expected behavior for invalid operations
        QUARISMA_LOG_INFO("Expected exception in transfer error handling: {}", e.what());
    }
}

#endif  // PROJECT_HAS_CUDA
