#include "gpu/gpu_allocator_factory.h"

#include <sstream>
#include <stdexcept>

#include "common/memory_macros.h"
//#include "logger/logger.h"
#include "gpu/gpu_device_manager.h"

#if MEMORY_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace memory
{
namespace gpu
{

gpu_allocator_config gpu_allocator_config::create_default(
    gpu_allocation_strategy strategy, int device_index)
{
    gpu_allocator_config config;
    config.strategy     = strategy;
    config.device_type  = device_enum::CUDA;
    config.device_index = device_index;
    config.alignment    = 256ULL;  // Optimal for CUDA coalescing

    switch (strategy)
    {
    case gpu_allocation_strategy::DIRECT:
        // Direct allocation needs no special configuration
        break;

    case gpu_allocation_strategy::POOL:
        config.pool_min_block_size = 4096;             // 4KB minimum
        config.pool_max_block_size = 64ULL * 1024ULL;  // 64MB maximum
        config.pool_growth_factor  = 1.5;
        config.pool_max_size       = 512ULL * 1024ULL;  // 512MB pool
        break;

    case gpu_allocation_strategy::CACHING:
        config.cache_max_bytes = 256ULL * 1024ULL;  // 256MB cache
        break;
    }

    return config;
}

gpu_allocator_config gpu_allocator_config::create_monte_carlo_optimized(int device_index)
{
    auto config = create_default(gpu_allocation_strategy::CACHING, device_index);

    // Monte Carlo simulations benefit from caching with larger cache
    config.cache_max_bytes = 1ULL * 1024ULL * 1024;  // 1GB cache
    config.alignment       = 512;                    // Larger alignment for vectorized operations

    // Log info (simplified for build compatibility)
    return config;
}

gpu_allocator_config gpu_allocator_config::create_pde_optimized(int device_index)
{
    auto config = create_default(gpu_allocation_strategy::POOL, device_index);

    // PDE solvers typically use large, long-lived allocations
    config.pool_min_block_size = 64ULL * 1024;           // 64KB minimum
    config.pool_max_block_size = 256ULL * 1024ULL;       // 256MB maximum
    config.pool_max_size       = 2ULL * 1024ULL * 1024;  // 2GB pool
    config.alignment           = 1024;                   // Large alignment for matrix operations

    // Log info (simplified for build compatibility)
    return config;
}

template <typename T>
std::unique_ptr<void> gpu_allocator_factory::create_allocator(
    const gpu_allocator_config& /*config*/)
{
    // Note: gpu_allocator class has been removed. This factory method is deprecated.
    // Use direct CUDA allocation or cuda_caching_allocator instead.

    throw std::runtime_error(
        "gpu_allocator class has been removed. "
        "Use direct CUDA allocation (cudaMalloc/cudaFree) or "
        "create_caching_allocator() for advanced memory management.");
}

gpu_allocation_strategy gpu_allocator_factory::recommend_strategy(
    size_t avg_allocation_size, double allocation_frequency, double allocation_lifetime)
{
    // Simple heuristics for strategy recommendation
    const size_t large_allocation_threshold = 1024ULL;  // 1MB
    const double high_frequency_threshold   = 100.0;    // 100 allocs/sec
    const double short_lifetime_threshold   = 1.0;      // 1 second

    if (allocation_frequency > high_frequency_threshold &&
        allocation_lifetime < short_lifetime_threshold)
    {
        // High frequency, short-lived allocations benefit from caching
        return gpu_allocation_strategy::CACHING;
    }

    if (avg_allocation_size < large_allocation_threshold && allocation_frequency > 10.0)
    {
        // Medium frequency, small allocations benefit from pooling
        return gpu_allocation_strategy::POOL;
    }

    // Default to direct allocation for simple cases
    return gpu_allocation_strategy::DIRECT;
}

bool gpu_allocator_factory::validate_device_support(
    gpu_allocation_strategy /*strategy*/, device_enum device_type, int device_index)
{
    // Only CUDA devices are supported currently
    if (device_type != device_enum::CUDA)
    {
        return false;
    }

#if MEMORY_HAS_CUDA
    // Validate CUDA device exists
    int               device_count = 0;
    cudaError_t const result       = cudaGetDeviceCount(&device_count);

    return (result == cudaSuccess && device_index >= 0 && device_index < device_count);

#else
    return false;
#endif
}

std::string gpu_allocator_factory::strategy_name(gpu_allocation_strategy strategy)
{
    switch (strategy)
    {
    case gpu_allocation_strategy::DIRECT:
        return "Direct";
    case gpu_allocation_strategy::POOL:
        return "Pool";
    case gpu_allocation_strategy::CACHING:
        return "Caching";
    default:
        return "Unknown";
    }
}

// Note: Explicit template instantiations for create_allocator removed
// after gpu_allocator class removal. The create_allocator method is now
// deprecated and throws an exception.

// Caching allocator template instantiations for common alignments
template std::unique_ptr<cuda_caching_allocator_template<float, 64ULL>>
gpu_allocator_factory::create_caching_allocator<float, 64ULL>(const gpu_allocator_config&);

template std::unique_ptr<cuda_caching_allocator_template<float, 128>>
gpu_allocator_factory::create_caching_allocator<float, 128>(const gpu_allocator_config&);

template std::unique_ptr<cuda_caching_allocator_template<float, 256ULL>>
gpu_allocator_factory::create_caching_allocator<float, 256ULL>(const gpu_allocator_config&);

template std::unique_ptr<cuda_caching_allocator_template<float, 512>>
gpu_allocator_factory::create_caching_allocator<float, 512>(const gpu_allocator_config&);

template std::unique_ptr<cuda_caching_allocator_template<double, 64ULL>>
gpu_allocator_factory::create_caching_allocator<double, 64ULL>(const gpu_allocator_config&);

template std::unique_ptr<cuda_caching_allocator_template<double, 128>>
gpu_allocator_factory::create_caching_allocator<double, 128>(const gpu_allocator_config&);

template std::unique_ptr<cuda_caching_allocator_template<double, 256ULL>>
gpu_allocator_factory::create_caching_allocator<double, 256ULL>(const gpu_allocator_config&);

template std::unique_ptr<cuda_caching_allocator_template<double, 512>>
gpu_allocator_factory::create_caching_allocator<double, 512>(const gpu_allocator_config&);

// Additional instantiations for Random library (uint64_t, uint32_t)
template std::unique_ptr<cuda_caching_allocator_template<uint64_t, 64ULL>>
gpu_allocator_factory::create_caching_allocator<uint64_t, 64ULL>(const gpu_allocator_config&);

template std::unique_ptr<cuda_caching_allocator_template<uint64_t, 256ULL>>
gpu_allocator_factory::create_caching_allocator<uint64_t, 256ULL>(const gpu_allocator_config&);

template std::unique_ptr<cuda_caching_allocator_template<uint32_t, 64ULL>>
gpu_allocator_factory::create_caching_allocator<uint32_t, 64ULL>(const gpu_allocator_config&);

template std::unique_ptr<cuda_caching_allocator_template<uint32_t, 256ULL>>
gpu_allocator_factory::create_caching_allocator<uint32_t, 256ULL>(const gpu_allocator_config&);

// Additional instantiations for test coverage (int type)
template std::unique_ptr<cuda_caching_allocator_template<int, 128>>
gpu_allocator_factory::create_caching_allocator<int, 128>(const gpu_allocator_config&);

}  // namespace gpu
}  // namespace memory
