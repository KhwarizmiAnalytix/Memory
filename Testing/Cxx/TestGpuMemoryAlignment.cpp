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

#include "MemoryTest.h"
#include "common/memory_macros.h"

#if PROJECT_HAS_CUDA

#include <string>
#include <vector>

//#include "logger/logger.h"
#include "common/device.h"
#include "gpu/gpu_memory_alignment.h"

using namespace memory;
using namespace memory::gpu;

/**
 * @brief Test alignment configuration validation
 */
MEMORYTEST(GpuMemoryAlignment, validates_alignment_configuration)
{
    // Test valid configuration
    alignment_config valid_config;
    valid_config.base_alignment    = 128;
    valid_config.vector_alignment  = 32;
    valid_config.texture_alignment = 128;
    valid_config.work_group_size   = 256;

    EXPECT_TRUE(gpu_memory_alignment::validate_config(valid_config));

    // Test invalid configuration (zero alignment)
    alignment_config invalid_config;
    invalid_config.base_alignment   = 0;
    invalid_config.vector_alignment = 32;

    EXPECT_FALSE(gpu_memory_alignment::validate_config(invalid_config));

    MEMORY_LOG_INFO("GPU memory alignment configuration validation test passed");
}

/**
 * @brief Test basic size alignment functionality
 */
MEMORYTEST(GpuMemoryAlignment, aligns_sizes_correctly)
{
    // Test size alignment
    size_t size         = 100;
    size_t aligned_size = gpu_memory_alignment::align_size(size, 64);

    EXPECT_GE(aligned_size, size);
    EXPECT_EQ(aligned_size % 64, 0);
    EXPECT_EQ(128, aligned_size);  // 100 aligned to 64 should be 128

    // Test already aligned size
    size_t aligned_input = 128;
    size_t result        = gpu_memory_alignment::align_size(aligned_input, 64);
    EXPECT_EQ(aligned_input, result);

    // Test power-of-2 alignments
    EXPECT_EQ(8, gpu_memory_alignment::align_size(5, 8));
    EXPECT_EQ(16, gpu_memory_alignment::align_size(9, 16));
    EXPECT_EQ(32, gpu_memory_alignment::align_size(17, 32));

    MEMORY_LOG_INFO("GPU memory alignment size alignment test passed");
}

/**
 * @brief Test pointer alignment functionality
 */
MEMORYTEST(GpuMemoryAlignment, aligns_pointers_correctly)
{
    // Test pointer alignment
    char  buffer[256];
    char* aligned_ptr = gpu_memory_alignment::align_pointer(buffer, 64);

    EXPECT_TRUE(gpu_memory_alignment::is_aligned(aligned_ptr, 64));
    EXPECT_GE(aligned_ptr, buffer);
    EXPECT_LT(aligned_ptr, buffer + 256);

    // Test alignment checking
    char* unaligned_ptr = buffer + 1;
    EXPECT_FALSE(gpu_memory_alignment::is_aligned(unaligned_ptr, 64));

    // Test different alignment values
    int* int_buffer  = new int[100];
    int* aligned_int = gpu_memory_alignment::align_pointer(int_buffer, 32);
    EXPECT_TRUE(gpu_memory_alignment::is_aligned(aligned_int, 32));

    delete[] int_buffer;

    MEMORY_LOG_INFO("GPU memory alignment pointer alignment test passed");
}

/**
 * @brief Test coalesced memory access alignment
 */
MEMORYTEST(GpuMemoryAlignment, optimizes_for_coalesced_access)
{
    alignment_config config;
    config.base_alignment    = 128;
    config.enable_coalescing = true;

    // Test float array alignment
    size_t float_count = 100;
    size_t aligned_float_size =
        gpu_memory_alignment::align_size_for_coalescing<float>(float_count, config);

    EXPECT_GE(aligned_float_size, float_count * sizeof(float));
    EXPECT_EQ(aligned_float_size % config.base_alignment, 0);

    // Test double array alignment
    size_t double_count = 50;
    size_t aligned_double_size =
        gpu_memory_alignment::align_size_for_coalescing<double>(double_count, config);

    EXPECT_GE(aligned_double_size, double_count * sizeof(double));
    EXPECT_EQ(aligned_double_size % config.base_alignment, 0);

    // Test with coalescing disabled
    config.enable_coalescing = false;
    size_t unaligned_size =
        gpu_memory_alignment::align_size_for_coalescing<float>(float_count, config);
    EXPECT_EQ(unaligned_size, float_count * sizeof(float));

    MEMORY_LOG_INFO("GPU memory alignment coalesced access test passed");
}

/**
 * @brief Test optimal stride calculation for 2D arrays
 */
MEMORYTEST(GpuMemoryAlignment, calculates_optimal_strides)
{
    alignment_config config;
    config.base_alignment       = 128;
    config.avoid_bank_conflicts = true;
    config.bank_conflict_stride = 32;

    // Test stride calculation for different types
    size_t width = 100;

    size_t float_stride = gpu_memory_alignment::calculate_optimal_stride<float>(width, config);
    EXPECT_GE(float_stride, width);
    EXPECT_EQ((float_stride * sizeof(float)) % config.bank_conflict_stride, 0);

    size_t double_stride = gpu_memory_alignment::calculate_optimal_stride<double>(width, config);
    EXPECT_GE(double_stride, width);
    EXPECT_EQ((double_stride * sizeof(double)) % config.bank_conflict_stride, 0);

    // Test with bank conflict avoidance disabled
    config.avoid_bank_conflicts = false;
    size_t simple_stride = gpu_memory_alignment::calculate_optimal_stride<float>(width, config);
    EXPECT_GE(simple_stride, width);

    MEMORY_LOG_INFO("GPU memory alignment stride calculation test passed");
}

/**
 * @brief Test padded width calculation for bank conflict avoidance
 */
MEMORYTEST(GpuMemoryAlignment, calculates_padded_widths)
{
    alignment_config config;
    config.base_alignment       = 128;
    config.avoid_bank_conflicts = true;
    config.bank_conflict_stride = 32;

    size_t original_width = 64;

    // Test padded width calculation
    size_t padded_width =
        gpu_memory_alignment::calculate_padded_width<float>(original_width, config);
    EXPECT_GE(padded_width, original_width);

    // Padded width should avoid bank conflicts
    size_t byte_width = padded_width * sizeof(float);
    if (config.avoid_bank_conflicts && (byte_width % config.bank_conflict_stride == 0))
    {
        // Should have added padding
        EXPECT_GT(padded_width, original_width);
    }

    MEMORY_LOG_INFO("GPU memory alignment padded width test passed");
}

/**
 * @brief Test SIMD alignment requirements
 */
MEMORYTEST(GpuMemoryAlignment, provides_simd_alignment)
{
    // Test SIMD alignment for different vector sizes
    size_t float_simd_8 = gpu_memory_alignment::get_simd_alignment<float>(8);
    EXPECT_GE(float_simd_8, 8 * sizeof(float));
    EXPECT_EQ(float_simd_8 & (float_simd_8 - 1), 0);  // Should be power of 2

    size_t double_simd_4 = gpu_memory_alignment::get_simd_alignment<double>(4);
    EXPECT_GE(double_simd_4, 4 * sizeof(double));
    EXPECT_EQ(double_simd_4 & (double_simd_4 - 1), 0);  // Should be power of 2

    // Test default vector size
    size_t default_simd = gpu_memory_alignment::get_simd_alignment<float>();
    EXPECT_GT(default_simd, 0);
    EXPECT_EQ(default_simd & (default_simd - 1), 0);  // Should be power of 2

    MEMORY_LOG_INFO("GPU memory alignment SIMD alignment test passed");
}

/**
 * @brief Test optimal memory layout calculation for multi-dimensional arrays
 */
MEMORYTEST(GpuMemoryAlignment, calculates_optimal_layouts)
{
    alignment_config config;
    config.base_alignment       = 128;
    config.avoid_bank_conflicts = true;

    // Test 2D layout
    std::vector<size_t> dimensions_2d = {100, 200};
    auto strides_2d = gpu_memory_alignment::calculate_optimal_layout<float>(dimensions_2d, config);

    EXPECT_EQ(strides_2d.size(), 2);
    EXPECT_EQ(strides_2d[0], 1);                 // Innermost dimension stride is 1
    EXPECT_GE(strides_2d[1], dimensions_2d[0]);  // Outer stride >= inner dimension

    // Test 3D layout
    std::vector<size_t> dimensions_3d = {50, 100, 150};
    auto strides_3d = gpu_memory_alignment::calculate_optimal_layout<double>(dimensions_3d, config);

    EXPECT_EQ(strides_3d.size(), 3);
    EXPECT_EQ(strides_3d[0], 1);
    EXPECT_GE(strides_3d[1], dimensions_3d[0]);
    EXPECT_GE(strides_3d[2], strides_3d[1] * dimensions_3d[1]);

    // Test empty dimensions
    std::vector<size_t> empty_dimensions;
    auto                empty_strides =
        gpu_memory_alignment::calculate_optimal_layout<float>(empty_dimensions, config);
    EXPECT_TRUE(empty_strides.empty());

    MEMORY_LOG_INFO("GPU memory alignment layout calculation test passed");
}

/**
 * @brief Test GPU architecture detection
 */
MEMORYTEST(GpuMemoryAlignment, detects_gpu_architecture)
{
    // Test CUDA architecture detection
    auto arch_75 = gpu_memory_alignment::detect_architecture(device_enum::CUDA, 7, 5, "");
    EXPECT_EQ(gpu_architecture::CUDA_COMPUTE_75, arch_75);

    auto arch_80 = gpu_memory_alignment::detect_architecture(device_enum::CUDA, 8, 0, "");
    EXPECT_EQ(gpu_architecture::CUDA_COMPUTE_80, arch_80);

    auto arch_60 = gpu_memory_alignment::detect_architecture(device_enum::CUDA, 6, 0, "");
    EXPECT_EQ(gpu_architecture::CUDA_COMPUTE_60, arch_60);

    // Test older architectures
    auto arch_35 = gpu_memory_alignment::detect_architecture(device_enum::CUDA, 3, 5, "");
    EXPECT_EQ(gpu_architecture::CUDA_COMPUTE_35, arch_35);

    MEMORY_LOG_INFO("GPU memory alignment architecture detection test passed");
}

/**
 * @brief Test optimal configuration for different architectures and access patterns
 */
MEMORYTEST(GpuMemoryAlignment, provides_optimal_configurations)
{
    // Test configuration for different architectures and access patterns
    auto sequential_config = gpu_memory_alignment::get_optimal_config(
        gpu_architecture::CUDA_COMPUTE_80, memory_access_pattern::SEQUENTIAL);

    EXPECT_GT(sequential_config.base_alignment, 0);
    EXPECT_TRUE(sequential_config.enable_coalescing);

    auto random_config = gpu_memory_alignment::get_optimal_config(
        gpu_architecture::CUDA_COMPUTE_75, memory_access_pattern::RANDOM);

    EXPECT_GT(random_config.base_alignment, 0);

    auto stencil_config = gpu_memory_alignment::get_optimal_config(
        gpu_architecture::CUDA_COMPUTE_70, memory_access_pattern::STENCIL);

    EXPECT_GT(stencil_config.base_alignment, 0);
    EXPECT_TRUE(stencil_config.avoid_bank_conflicts);

    MEMORY_LOG_INFO("GPU memory alignment optimal configuration test passed");
}

/**
 * @brief Test alignment report generation
 */
MEMORYTEST(GpuMemoryAlignment, generates_alignment_reports)
{
    alignment_config config;
    config.base_alignment       = 128;
    config.vector_alignment     = 32;
    config.texture_alignment    = 512;
    config.work_group_size      = 256;
    config.enable_coalescing    = true;
    config.avoid_bank_conflicts = true;

    std::string report = gpu_memory_alignment::get_alignment_report(config);

    // Report should not be empty
    EXPECT_FALSE(report.empty());

    // Report should contain configuration values
    EXPECT_TRUE(report.find("128") != std::string::npos);  // base_alignment
    EXPECT_TRUE(report.find("32") != std::string::npos);   // vector_alignment
    EXPECT_TRUE(report.find("512") != std::string::npos);  // texture_alignment

    MEMORY_LOG_INFO("Alignment report length: {} characters", report.length());

    MEMORY_LOG_INFO("GPU memory alignment report generation test passed");
}

#endif  // PROJECT_HAS_CUDA
