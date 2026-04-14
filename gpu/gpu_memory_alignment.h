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

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include "common/memory_macros.h"
#include "common/device.h"
#include "common/memory_export.h"

namespace memory
{
namespace gpu
{

/**
 * @brief GPU memory alignment constants
 *
 * These constants define optimal alignment boundaries for different
 * GPU architectures and memory access patterns, ensuring coalesced
 * memory access for maximum bandwidth utilization.
 */
namespace alignment
{
/** @brief CUDA warp size (32 threads) */
constexpr size_t CUDA_WARP_SIZE = 32;

/** @brief CUDA memory coalescing boundary (128 bytes) */
constexpr size_t CUDA_COALESCING_BOUNDARY = 128;

/** @brief CUDA texture alignment (512 bytes) */
constexpr size_t CUDA_TEXTURE_ALIGNMENT = 512;

/** @brief SIMD vector alignment for AVX-512 (64ULL bytes) */
constexpr size_t SIMD_VECTOR_ALIGNMENT = 64ULL;

/** @brief Cache line size (64ULL bytes on most modern CPUs) */
constexpr size_t CACHE_LINE_SIZE = 64ULL;

/** @brief Page size (4KB on most systems) */
constexpr size_t PAGE_SIZE = 4096;
}  // namespace alignment

/**
 * @brief Memory access pattern enumeration
 *
 * Defines different memory access patterns that affect optimal
 * alignment strategies for GPU kernels and data structures.
 */
enum class memory_access_pattern
{
    SEQUENTIAL,  ///< Sequential access (coalesced)
    STRIDED,     ///< Strided access with regular pattern
    RANDOM,      ///< Random access pattern
    BROADCAST,   ///< Single value broadcast to multiple threads
    REDUCTION,   ///< Reduction operations across threads
    TRANSPOSE,   ///< Matrix transpose operations
    STENCIL      ///< Stencil operations (PDE solvers)
};

/**
 * @brief GPU architecture enumeration
 *
 * Different GPU architectures have different optimal alignment
 * requirements and memory access characteristics.
 */
enum class gpu_architecture
{
    CUDA_COMPUTE_30,  ///< Kepler architecture
    CUDA_COMPUTE_35,  ///< Kepler GK110
    CUDA_COMPUTE_50,  ///< Maxwell architecture
    CUDA_COMPUTE_60,  ///< Pascal architecture
    CUDA_COMPUTE_70,  ///< Volta architecture
    CUDA_COMPUTE_75,  ///< Turing architecture
    CUDA_COMPUTE_80,  ///< Ampere architecture
    CUDA_COMPUTE_90   ///< Hopper architecture
};

/**
 * @brief Memory alignment configuration
 *
 * Contains alignment parameters optimized for specific GPU
 * architectures and memory access patterns.
 */
struct MEMORY_VISIBILITY alignment_config
{
    /** @brief Base alignment boundary in bytes */
    size_t base_alignment = alignment::CUDA_COALESCING_BOUNDARY;

    /** @brief Vector alignment for SIMD operations */
    size_t vector_alignment = alignment::SIMD_VECTOR_ALIGNMENT;

    /** @brief Texture alignment for texture memory */
    size_t texture_alignment = alignment::CUDA_TEXTURE_ALIGNMENT;

    /** @brief Preferred work group/block size */
    size_t work_group_size = alignment::CUDA_WARP_SIZE;

    /** @brief Memory bank conflict avoidance stride */
    size_t bank_conflict_stride = 32;

    /** @brief Enable padding to avoid bank conflicts */
    bool avoid_bank_conflicts = true;

    /** @brief Enable alignment for coalesced access */
    bool enable_coalescing = true;
};

/**
 * @brief GPU memory alignment utilities
 *
 * Provides comprehensive memory alignment functions optimized for
 * GPU architectures, ensuring optimal memory access patterns for
 * Monte Carlo simulations and PDE solvers.
 *
 * Key features:
 * - Architecture-specific alignment optimization
 * - SIMD vector alignment for CPU-GPU interoperability
 * - Bank conflict avoidance for shared memory
 * - Coalesced memory access pattern optimization
 * - Template-based type-safe alignment functions
 * - Compile-time alignment calculations where possible
 *
 * Mathematical foundation:
 * Aligned address calculation: addr_aligned = (addr + alignment - 1) & ~(alignment - 1)
 * Padding calculation: padding = (alignment - (size % alignment)) % alignment
 * Stride calculation for bank conflict avoidance: stride = lcm(bank_size, element_size)
 *
 * @example
 * ```cpp
 * // Align memory for CUDA coalesced access
 * size_t aligned_size = gpu_memory_alignment::align_size_for_coalescing<float>(1000);
 *
 * // Get optimal configuration for Monte Carlo simulation
 * auto config = gpu_memory_alignment::get_optimal_config(
 *     gpu_architecture::CUDA_COMPUTE_80,
 *     memory_access_pattern::SEQUENTIAL
 * );
 *
 * // Calculate padding for 2D array to avoid bank conflicts
 * size_t padded_width = gpu_memory_alignment::calculate_padded_width<double>(
 *     width, config
 * );
 * ```
 */
class MEMORY_VISIBILITY gpu_memory_alignment
{
public:
    /**
     * @brief Get optimal alignment configuration for given architecture and access pattern
     * @param arch GPU architecture
     * @param pattern Memory access pattern
     * @return Optimal alignment configuration
     */
    MEMORY_API static alignment_config get_optimal_config(
        gpu_architecture arch, memory_access_pattern pattern);

    /**
     * @brief Detect GPU architecture from device information
     * @param device_type Device type (CUDA or HIP)
     * @param compute_major Compute capability major version
     * @param compute_minor Compute capability minor version
     * @param vendor_name Vendor name (for HIP)
     * @return Detected GPU architecture
     */
    MEMORY_API static gpu_architecture detect_architecture(
        device_enum        device_type,
        int                compute_major,
        int                compute_minor,
        const std::string& vendor_name = "");

    /**
     * @brief Align size to specified boundary
     * @param size Size to align
     * @param alignment Alignment boundary (must be power of 2)
     * @return Aligned size
     */
    MEMORY_API static constexpr size_t align_size(size_t size, size_t alignment) noexcept
    {
        return (size + alignment - 1) & ~(alignment - 1);
    }

    /**
     * @brief Align pointer to specified boundary
     * @param ptr Pointer to align
     * @param alignment Alignment boundary (must be power of 2)
     * @return Aligned pointer
     */
    template <typename T>
    static constexpr T* align_pointer(T* ptr, size_t alignment) noexcept
    {
        auto addr         = reinterpret_cast<uintptr_t>(ptr);
        auto aligned_addr = align_size(addr, alignment);
        return reinterpret_cast<T*>(aligned_addr);
    }

    /**
     * @brief Check if pointer is aligned to specified boundary
     * @param ptr Pointer to check
     * @param alignment Alignment boundary
     * @return True if pointer is aligned
     */
    template <typename T>
    static constexpr bool is_aligned(const T* ptr, size_t alignment) noexcept
    {
        return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
    }

    /**
     * @brief Align size for optimal coalesced memory access
     * @tparam T Element type
     * @param count Number of elements
     * @param config Alignment configuration
     * @return Aligned size in bytes
     */
    template <typename T>
    static size_t align_size_for_coalescing(
        size_t count, const alignment_config& config = alignment_config{})
    {
        size_t element_size = sizeof(T);
        size_t total_size   = count * element_size;

        if (config.enable_coalescing)
        {
            return align_size(total_size, config.base_alignment);
        }

        return total_size;
    }

    /**
     * @brief Calculate optimal stride for 2D array access
     * @tparam T Element type
     * @param width Array width
     * @param config Alignment configuration
     * @return Optimal stride in elements
     */
    template <typename T>
    static size_t calculate_optimal_stride(
        size_t width, const alignment_config& config = alignment_config{})
    {
        size_t element_size = sizeof(T);
        size_t byte_width   = width * element_size;

        // Align to coalescing boundary
        size_t aligned_byte_width = align_size(byte_width, config.base_alignment);

        // Add padding to avoid bank conflicts if enabled
        if (config.avoid_bank_conflicts)
        {
            size_t stride_elements = aligned_byte_width / element_size;

            // Check if stride causes bank conflicts
            if ((stride_elements * element_size) % config.bank_conflict_stride == 0)
            {
                // Add minimal padding to avoid conflicts
                stride_elements += config.bank_conflict_stride / element_size;
            }

            return stride_elements;
        }

        return aligned_byte_width / element_size;
    }

    /**
     * @brief Calculate padded width for 2D arrays to avoid bank conflicts
     * @tparam T Element type
     * @param width Original width
     * @param config Alignment configuration
     * @return Padded width in elements
     */
    template <typename T>
    static size_t calculate_padded_width(
        size_t width, const alignment_config& config = alignment_config{})
    {
        return calculate_optimal_stride<T>(width, config);
    }

    /**
     * @brief Get alignment requirement for SIMD vector operations
     * @tparam T Element type
     * @param vector_size Number of elements in SIMD vector
     * @return Required alignment in bytes
     */
    template <typename T>
    static constexpr size_t get_simd_alignment(size_t vector_size = 8) noexcept
    {
        size_t vector_bytes = vector_size * sizeof(T);

        // Ensure alignment is at least the vector size and a power of 2
        size_t alignment = 1;
        while (alignment < vector_bytes)
        {
            alignment <<= 1;
        }

        return alignment;
    }

    /**
     * @brief Calculate memory layout for optimal GPU kernel performance
     * @tparam T Element type
     * @param dimensions Array dimensions [width, height, depth, ...]
     * @param config Alignment configuration
     * @return Vector of strides for each dimension
     */
    template <typename T>
    static std::vector<size_t> calculate_optimal_layout(
        const std::vector<size_t>& dimensions, const alignment_config& config = alignment_config{})
    {
        if (dimensions.empty())
            return {};

        std::vector<size_t> strides(dimensions.size());

        // Start with the innermost dimension
        strides[0] = 1;

        // Calculate strides for each dimension
        for (size_t i = 1; i < dimensions.size(); ++i)
        {
            size_t prev_dim_size = dimensions[i - 1];

            // Apply padding to avoid bank conflicts for the previous dimension
            if (i == 1)  // Only pad the innermost dimension
            {
                prev_dim_size = calculate_padded_width<T>(prev_dim_size, config);
            }

            strides[i] = strides[i - 1] * prev_dim_size;
        }

        return strides;
    }

    /**
     * @brief Validate alignment configuration
     * @param config Configuration to validate
     * @return True if configuration is valid
     */
    MEMORY_API static bool validate_config(const alignment_config& config) noexcept;

    /**
     * @brief Get alignment report for debugging
     * @param config Alignment configuration
     * @return Formatted string with alignment information
     */
    MEMORY_API static std::string get_alignment_report(const alignment_config& config);

private:
    gpu_memory_alignment() = default;
    MEMORY_DELETE_COPY_AND_MOVE(gpu_memory_alignment);
};

}  // namespace gpu
}  // namespace memory
