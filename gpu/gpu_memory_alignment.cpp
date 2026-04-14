#include "gpu/gpu_memory_alignment.h"

#include <algorithm>
#include <iomanip>
#include <sstream>

#include "common/memory_macros.h"

namespace memory
{
namespace gpu
{

alignment_config gpu_memory_alignment::get_optimal_config(
    gpu_architecture arch, memory_access_pattern pattern)
{
    alignment_config config;

    // Set base configuration based on architecture
    switch (arch)
    {
    case gpu_architecture::CUDA_COMPUTE_30:
    case gpu_architecture::CUDA_COMPUTE_35:
        config.base_alignment       = alignment::CUDA_COALESCING_BOUNDARY;
        config.vector_alignment     = 32;  // Older architectures prefer smaller alignment
        config.texture_alignment    = alignment::CUDA_TEXTURE_ALIGNMENT;
        config.work_group_size      = alignment::CUDA_WARP_SIZE;
        config.bank_conflict_stride = 32;
        break;

    case gpu_architecture::CUDA_COMPUTE_50:
    case gpu_architecture::CUDA_COMPUTE_60:
    case gpu_architecture::CUDA_COMPUTE_70:
    case gpu_architecture::CUDA_COMPUTE_75:
        config.base_alignment       = alignment::CUDA_COALESCING_BOUNDARY;
        config.vector_alignment     = alignment::SIMD_VECTOR_ALIGNMENT;
        config.texture_alignment    = alignment::CUDA_TEXTURE_ALIGNMENT;
        config.work_group_size      = alignment::CUDA_WARP_SIZE;
        config.bank_conflict_stride = 32;
        break;

    case gpu_architecture::CUDA_COMPUTE_80:
    case gpu_architecture::CUDA_COMPUTE_90:
        // Latest architectures have more flexible memory systems
        config.base_alignment       = alignment::CUDA_COALESCING_BOUNDARY;
        config.vector_alignment     = alignment::SIMD_VECTOR_ALIGNMENT;
        config.texture_alignment    = alignment::CUDA_TEXTURE_ALIGNMENT;
        config.work_group_size      = alignment::CUDA_WARP_SIZE;
        config.bank_conflict_stride = 32;
        config.avoid_bank_conflicts = false;  // Modern architectures handle this better
        break;
    }

    // Adjust configuration based on memory access pattern
    switch (pattern)
    {
    case memory_access_pattern::SEQUENTIAL:
        // Sequential access benefits most from coalescing
        config.enable_coalescing    = true;
        config.avoid_bank_conflicts = true;
        break;

    case memory_access_pattern::STRIDED:
        // Strided access may need larger alignment
        config.base_alignment       = std::max(config.base_alignment, size_t(256ULL));
        config.enable_coalescing    = true;
        config.avoid_bank_conflicts = true;
        break;

    case memory_access_pattern::RANDOM:
        // Random access doesn't benefit much from coalescing
        config.enable_coalescing    = false;
        config.avoid_bank_conflicts = false;
        break;

    case memory_access_pattern::BROADCAST:
        // Broadcast benefits from cache line alignment
        config.base_alignment       = std::max(config.base_alignment, alignment::CACHE_LINE_SIZE);
        config.enable_coalescing    = true;
        config.avoid_bank_conflicts = false;
        break;

    case memory_access_pattern::REDUCTION:
        // Reduction operations benefit from avoiding bank conflicts
        config.avoid_bank_conflicts = true;
        config.enable_coalescing    = true;
        break;

    case memory_access_pattern::TRANSPOSE:
        // Matrix transpose needs careful alignment
        config.base_alignment       = std::max(config.base_alignment, size_t(256ULL));
        config.avoid_bank_conflicts = true;
        config.enable_coalescing    = true;
        break;

    case memory_access_pattern::STENCIL:
        // Stencil operations (PDE solvers) benefit from cache-friendly alignment
        config.base_alignment       = std::max(config.base_alignment, alignment::CACHE_LINE_SIZE);
        config.avoid_bank_conflicts = true;
        config.enable_coalescing    = true;
        break;
    }

    return config;
}

gpu_architecture gpu_memory_alignment::detect_architecture(
    device_enum device_type,
    int         compute_major,
    int         compute_minor,
    const std::string& /*vendor_name*/)
{
    switch (device_type)
    {
    case device_enum::CUDA:
    {
        int const compute_capability = (compute_major * 10) + compute_minor;

        if (compute_capability >= 90)
        {
            return gpu_architecture::CUDA_COMPUTE_90;
        }
        if (compute_capability >= 80)
        {
            return gpu_architecture::CUDA_COMPUTE_80;
        }
        if (compute_capability >= 75)
        {
            return gpu_architecture::CUDA_COMPUTE_75;
        }
        if (compute_capability >= 70)
        {
            return gpu_architecture::CUDA_COMPUTE_70;
        }
        if (compute_capability >= 60)
        {
            return gpu_architecture::CUDA_COMPUTE_60;
        }
        if (compute_capability >= 50)
        {
            return gpu_architecture::CUDA_COMPUTE_50;
        }
        if (compute_capability >= 35)
        {
            return gpu_architecture::CUDA_COMPUTE_35;
        }
        return gpu_architecture::CUDA_COMPUTE_30;
    }

    case device_enum::HIP:
    {
        // HIP architecture detection (placeholder for future HIP support)
        return gpu_architecture::CUDA_COMPUTE_80;  // Default to modern architecture
    }

    default:
        return gpu_architecture::CUDA_COMPUTE_80;  // Default to modern architecture
    }
}

bool gpu_memory_alignment::validate_config(const alignment_config& config) noexcept
{
    // Check that alignments are powers of 2
    auto is_power_of_2 = [](size_t n) { return n > 0 && (n & (n - 1)) == 0; };

    if (!is_power_of_2(config.base_alignment) || !is_power_of_2(config.vector_alignment) ||
        !is_power_of_2(config.texture_alignment))
    {
        return false;
    }

    // Check reasonable bounds
    if (config.base_alignment < 32 || config.base_alignment > 4096)
    {
        return false;
    }

    if (config.vector_alignment < 16 || config.vector_alignment > 512)
    {
        return false;
    }

    if (config.texture_alignment < 64ULL || config.texture_alignment > 4096)
    {
        return false;
    }

    if (config.work_group_size < 16 || config.work_group_size > 1024)
    {
        return false;
    }

    if (config.bank_conflict_stride < 16 || config.bank_conflict_stride > 128)
    {
        return false;
    }

    return true;
}

std::string gpu_memory_alignment::get_alignment_report(const alignment_config& config)
{
    std::ostringstream oss;
    oss << "GPU Memory Alignment Configuration:\n";
    oss << "===================================\n";
    oss << "Base alignment: " << config.base_alignment << " bytes\n";
    oss << "Vector alignment: " << config.vector_alignment << " bytes\n";
    oss << "Texture alignment: " << config.texture_alignment << " bytes\n";
    oss << "Work group size: " << config.work_group_size << " threads\n";
    oss << "Bank conflict stride: " << config.bank_conflict_stride << " bytes\n";
    oss << "Avoid bank conflicts: " << (config.avoid_bank_conflicts ? "Yes" : "No") << "\n";
    oss << "Enable coalescing: " << (config.enable_coalescing ? "Yes" : "No") << "\n";
    oss << "Configuration valid: " << (validate_config(config) ? "Yes" : "No") << "\n";

    // Add some example calculations
    oss << "\nExample Alignments:\n";
    oss << "  float array (1000 elements): " << align_size_for_coalescing<float>(1000, config)
        << " bytes\n";
    oss << "  double array (1000 elements): " << align_size_for_coalescing<double>(1000, config)
        << " bytes\n";
    oss << "  2D float array stride (width=1024): " << calculate_optimal_stride<float>(1024, config)
        << " elements\n";
    oss << "  2D double array stride (width=1024): "
        << calculate_optimal_stride<double>(1024, config) << " elements\n";

    return oss.str();
}

}  // namespace gpu
}  // namespace memory
