#include "gpu/gpu_device_manager.h"

#include <algorithm>
#include <iomanip>
#include <mutex>
#include <sstream>

#include "common/memory_macros.h"
#include "logger.h"
#include "util/exception.h"

#if MEMORY_HAS_CUDA
#include <cuda_runtime.h>
// #ifdef _WIN32
// // NVML is available on Windows
// #include <nvml.h>
// #define NVML_AVAILABLE
// #endif
#endif

namespace memory
{
namespace gpu
{

namespace
{

/**
 * @brief Internal implementation of GPU device manager
 *
 * Provides concrete implementation of device detection, enumeration,
 * and management for CUDA backend.
 */
class gpu_device_manager_impl : public gpu_device_manager
{
private:
    /** @brief Mutex for thread-safe operations */
    mutable std::mutex mutex_;

    /** @brief Whether the manager has been initialized */
    bool initialized_ = false;

    /** @brief Runtime information */
    gpu_runtime_info runtime_info_;

    /** @brief List of available devices */
    std::vector<gpu_device_info> available_devices_;

    /** @brief Current active device */
    gpu_device_info current_device_;

    /**
     * @brief Initialize CUDA runtime and detect CUDA devices
     */
    void initialize_cuda()
    {
#if MEMORY_HAS_CUDA
        // Check CUDA runtime availability
        cudaError_t result = cudaGetDeviceCount(&runtime_info_.cuda_device_count);
        if (result == cudaSuccess && runtime_info_.cuda_device_count > 0)
        {
            runtime_info_.cuda_available = true;

            // Get CUDA runtime and driver versions
            cudaRuntimeGetVersion(&runtime_info_.cuda_runtime_version);
            cudaDriverGetVersion(&runtime_info_.cuda_driver_version);

#ifdef NVML_AVAILABLE
            // Initialize NVML for device monitoring
            nvmlReturn_t nvml_result = nvmlInit();
            (void)(nvml_result ==
                   NVML_SUCCESS);  // Suppress unused variable warning for nvml_available
#else
            // NVML not available
#endif

            // Enumerate CUDA devices
            for (int i = 0; i < runtime_info_.cuda_device_count; ++i)
            {
                gpu_device_info device_info;
                device_info.device_index = i;
                device_info.device_type  = device_enum::CUDA;

                cudaDeviceProp props;
                result = cudaGetDeviceProperties(&props, i);
                if (result == cudaSuccess)
                {
                    device_info.name                    = props.name;
                    device_info.total_memory            = props.totalGlobalMem;
                    device_info.multiprocessor_count    = props.multiProcessorCount;
                    device_info.max_threads_per_block   = props.maxThreadsPerBlock;
                    device_info.max_block_dims[0]       = props.maxThreadsDim[0];
                    device_info.max_block_dims[1]       = props.maxThreadsDim[1];
                    device_info.max_block_dims[2]       = props.maxThreadsDim[2];
                    device_info.max_grid_dims[0]        = props.maxGridSize[0];
                    device_info.max_grid_dims[1]        = props.maxGridSize[1];
                    device_info.max_grid_dims[2]        = props.maxGridSize[2];
                    device_info.shared_memory_per_block = props.sharedMemPerBlock;
                    device_info.memory_bus_width        = props.memoryBusWidth;
                    device_info.memory_clock_rate =
                        props.memoryBusWidth;  // memoryClockRate deprecated in CUDA 13+
                    device_info.compute_capability_major = props.major;
                    device_info.compute_capability_minor = props.minor;
                    device_info.supports_double_precision =
                        (props.major >= 2) || (props.major == 1 && props.minor >= 3);
                    device_info.supports_unified_memory     = (props.managedMemory != 0);
                    device_info.supports_concurrent_kernels = (props.concurrentKernels != 0);

                    // Calculate memory bandwidth (GB/s) from bus width
                    // Note: memoryClockRate is deprecated in CUDA 13+, use approximation
                    // Formula: (bus_width_bits / 8) * estimated_clock_rate_khz * 2 / 1000000
                    // Use conservative estimate of 7000 MHz for modern GPUs
                    double const estimated_memory_clock_khz = 7000000.0;  // 7 GHz in kHz
                    device_info.memory_bandwidth_gb_per_sec =
                        (static_cast<double>(props.memoryBusWidth) / 8.0) *
                        estimated_memory_clock_khz * 2.0 / 1000000.0;

                    // Set total_memory_bytes as alias for total_memory for compatibility
                    device_info.total_memory_bytes = props.totalGlobalMem;

                    // Get PCI bus ID
                    char pci_bus_id[16];
                    result = cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), i);
                    if (result == cudaSuccess)
                    {
                        device_info.pci_bus_id = pci_bus_id;
                    }

                    // Get available memory
                    size_t free_mem;
                    size_t total_mem;
                    cudaSetDevice(i);
                    result = cudaMemGetInfo(&free_mem, &total_mem);
                    if (result == cudaSuccess)
                    {
                        device_info.available_memory = free_mem;
                        device_info.max_allocation_size =
                            std::min(free_mem, device_info.total_memory / 2);
                    }

#ifdef NVML_AVAILABLE
                    // Get device utilization and temperature using NVML
                    if (nvml_available)
                    {
                        nvmlDevice_t nvml_device;
                        nvml_result = nvmlDeviceGetHandleByPciBusId(
                            device_info.pci_bus_id.c_str(), &nvml_device);
                        if (nvml_result == NVML_SUCCESS)
                        {
                            nvmlUtilization_t utilization;
                            nvml_result = nvmlDeviceGetUtilizationRates(nvml_device, &utilization);
                            if (nvml_result == NVML_SUCCESS)
                            {
                                device_info.utilization_percentage =
                                    static_cast<float>(utilization.gpu);
                            }

                            unsigned int temperature;
                            nvml_result = nvmlDeviceGetTemperature(
                                nvml_device, NVML_TEMPERATURE_GPU, &temperature);
                            if (nvml_result == NVML_SUCCESS)
                            {
                                device_info.temperature_celsius = static_cast<float>(temperature);
                            }

                            unsigned int power;
                            nvml_result = nvmlDeviceGetPowerUsage(nvml_device, &power);
                            if (nvml_result == NVML_SUCCESS)
                            {
                                device_info.power_consumption_watts =
                                    static_cast<float>(power) / 1000.0f;
                            }
                        }
                    }
#endif

                    available_devices_.push_back(device_info);
                }
            }

#ifdef NVML_AVAILABLE
            if (nvml_available)
            {
                nvmlShutdown();
            }
#endif
        }
        else
        {
            runtime_info_.cuda_available    = false;
            runtime_info_.cuda_device_count = 0;
        }
#else
        runtime_info_.cuda_available    = false;
        runtime_info_.cuda_device_count = 0;
#endif
    }

    /**
     * @brief Determine the recommended backend based on available devices
     */
    void determine_recommended_backend()
    {
        if (runtime_info_.cuda_available && runtime_info_.cuda_device_count > 0)
        {
            runtime_info_.recommended_backend = device_enum::CUDA;
        }

        else
        {
            runtime_info_.recommended_backend = device_enum::CPU;
        }
    }

    /**
     * @brief Calculate device score for Monte Carlo simulations
     * @param device Device to score
     * @return Score (higher is better)
     */
    static double calculate_monte_carlo_score(const gpu_device_info& device)
    {
        if (device.device_index < 0)
        {
            return 0.0;
        }

        double score = 0.0;

        // Memory capacity (40% weight)
        score += 0.4 * (static_cast<double>(device.available_memory) / (1024.0 * 1024.0 * 1024.0));

        // Compute units (30% weight)
        score += 0.3 * device.multiprocessor_count;

        // Memory bandwidth (20% weight) - approximated from bus width and clock
        double const memory_bandwidth =
            (device.memory_bus_width / 8.0) * device.memory_clock_rate * 2.0 / 1000.0;  // GB/s
        score += 0.2 * (memory_bandwidth / 100.0);  // Normalize to ~100 GB/s

        // Utilization penalty (10% weight) - prefer less utilized devices
        score += 0.1 * (100.0 - device.utilization_percentage) / 100.0;

        return score;
    }

    /**
     * @brief Calculate device score for PDE solvers
     * @param device Device to score
     * @return Score (higher is better)
     */
    static double calculate_pde_score(const gpu_device_info& device)
    {
        if (device.device_index < 0)
        {
            return 0.0;
        }

        double score = 0.0;

        // Memory capacity (35% weight)
        score += 0.35 * (static_cast<double>(device.available_memory) / (1024.0 * 1024.0 * 1024.0));

        // Shared memory per block (25% weight) - important for PDE stencils
        score += 0.25 * (static_cast<double>(device.shared_memory_per_block) /
                         (64.0 * 1024.0));  // Normalize to 64KB

        // Compute capability (20% weight) - higher is better for PDE
        double const compute_score =
            device.compute_capability_major + (device.compute_capability_minor * 0.1);
        score += 0.2 * (compute_score / 8.0);  // Normalize to compute capability 8.0

        // Double precision support (15% weight)
        if (device.supports_double_precision)
        {
            score += 0.15;
        }

        // Utilization penalty (5% weight)
        score += 0.05 * (100.0 - device.utilization_percentage) / 100.0;

        return score;
    }

public:
    gpu_device_manager_impl() = default;

    ~gpu_device_manager_impl() override = default;

    void initialize() override
    {
        std::scoped_lock const lock(mutex_);

        if (initialized_)
        {
            return;
        }

        // Clear previous state
        available_devices_.clear();
        runtime_info_ = gpu_runtime_info{};

        // Initialize backends
        initialize_cuda();

        // Determine recommended backend
        determine_recommended_backend();

        // Set current device to first available device
        if (!available_devices_.empty())
        {
            current_device_ = available_devices_[0];
        }

        initialized_ = true;

        LOGGING_LOG_INFO(
            "GPU device manager initialized with {} devices", available_devices_.size());
    }

    gpu_runtime_info get_runtime_info() const override
    {
        std::scoped_lock const lock(mutex_);
        return runtime_info_;
    }

    std::vector<gpu_device_info> get_available_devices() const override
    {
        std::scoped_lock const lock(mutex_);
        return available_devices_;
    }

    gpu_device_info get_device_info(device_enum device_type, int device_index) const override
    {
        std::scoped_lock const lock(mutex_);

        auto device_it = std::find_if(
            available_devices_.begin(),
            available_devices_.end(),
            [device_type, device_index](const auto& device)
            { return device.device_type == device_type && device.device_index == device_index; });

        if (device_it != available_devices_.end())
        {
            return *device_it;
        }

        MEMORY_THROW(
            "Device not found: type={}, index={}", static_cast<int>(device_type), device_index);
        return {};  // Added to satisfy compiler C4715
    }

    //gpu_device_info select_optimal_device_for_monte_carlo(
    //    size_t required_memory, bool prefer_double_precision) const override
    //{
    //    std::lock_guard<std::mutex> lock(mutex_);

    //    gpu_device_info best_device;
    //    double          best_score = -1.0;

    //    for (const auto& device : available_devices_)
    //    {
    //        // Check memory requirement
    //        if (device.available_memory < required_memory)
    //            continue;

    //        // Check double precision requirement
    //        if (prefer_double_precision && !device.supports_double_precision)
    //            continue;

    //        double score = calculate_monte_carlo_score(device);
    //        if (score > best_score)
    //        {
    //            best_score  = score;
    //            best_device = device;
    //        }
    //    }

    //    return best_device;
    //}

    //gpu_device_info select_optimal_device_for_pde(
    //    size_t required_memory, bool prefer_double_precision) const override
    //{
    //    std::lock_guard<std::mutex> lock(mutex_);

    //    gpu_device_info best_device;
    //    double          best_score = -1.0;

    //    for (const auto& device : available_devices_)
    //    {
    //        // Check memory requirement
    //        if (device.available_memory < required_memory)
    //            continue;

    //        // Check double precision requirement
    //        if (prefer_double_precision && !device.supports_double_precision)
    //            continue;

    //        double score = calculate_pde_score(device);
    //        if (score > best_score)
    //        {
    //            best_score  = score;
    //            best_device = device;
    //        }
    //    }

    //    return best_device;
    //}

    bool is_device_available(device_enum device_type, int device_index) const override
    {
        std::scoped_lock const lock(mutex_);

        return std::any_of(
            available_devices_.cbegin(),
            available_devices_.cend(),
            [&](const gpu_device_info& device)
            { return device.device_type == device_type && device.device_index == device_index; });
    }

    void set_device_context(device_enum device_type, int device_index) override
    {
        std::scoped_lock const lock(mutex_);

        // Find the device
        auto device_it = std::find_if(
            available_devices_.begin(),
            available_devices_.end(),
            [device_type, device_index](const auto& device)
            { return device.device_type == device_type && device.device_index == device_index; });

        if (device_it == available_devices_.end())
        {
            MEMORY_THROW(
                "Cannot set context for unknown device: type={}, index={}",
                static_cast<int>(device_type),
                device_index);
        }

        // Set device context based on type
        switch (device_type)
        {
#if MEMORY_HAS_CUDA
        case device_enum::CUDA:
        {
            cudaError_t const result = cudaSetDevice(device_index);
            if (result != cudaSuccess)
            {
                MEMORY_THROW(
                    "Failed to set CUDA device context: {}",
                    std::string(cudaGetErrorString(result)));
            }
            break;
        }
#endif

        default:
            MEMORY_THROW("Unsupported device type for context setting");
        }

        current_device_ = *device_it;
    }

    gpu_device_info get_current_device() const override
    {
        std::scoped_lock const lock(mutex_);
        return current_device_;
    }

    void refresh_device_info() override
    {
        std::scoped_lock const lock(mutex_);

        // Re-initialize to refresh device information
        initialized_ = false;
        initialize();
    }

    std::string get_system_report() const override
    {
        std::scoped_lock const lock(mutex_);

        std::ostringstream oss;
        oss << "GPU System Report:\n";
        oss << "==================\n\n";

        // Runtime information
        oss << "Runtime Information:\n";
        oss << "  CUDA Available: " << (runtime_info_.cuda_available ? "Yes" : "No") << "\n";
        if (runtime_info_.cuda_available)
        {
            oss << "  CUDA Runtime Version: " << runtime_info_.cuda_runtime_version << "\n";
            oss << "  CUDA Driver Version: " << runtime_info_.cuda_driver_version << "\n";
            oss << "  CUDA Device Count: " << runtime_info_.cuda_device_count << "\n";
        }

        oss << "  Recommended Backend: ";
        switch (runtime_info_.recommended_backend)
        {
        case device_enum::CUDA:
            oss << "CUDA";
            break;
        case device_enum::HIP:
            oss << "HIP";
            break;
        default:
            oss << "CPU";
            break;
        }
        oss << "\n\n";

        // Device information
        oss << "Available Devices (" << available_devices_.size() << "):\n";
        for (size_t i = 0; i < available_devices_.size(); ++i)
        {
            const auto& device = available_devices_[i];
            oss << "  Device " << i << ":\n";
            oss << "    Name: " << device.name << "\n";
            oss << "    Type: ";
            switch (device.device_type)
            {
            case device_enum::CUDA:
                oss << "CUDA";
                break;
            case device_enum::HIP:
                oss << "HIP";
                break;
            default:
                oss << "Unknown";
                break;
            }
            oss << "\n";
            oss << "    Index: " << device.device_index << "\n";
            oss << "    Total Memory: " << std::fixed << std::setprecision(2)
                << (device.total_memory / 1024.0 / 1024.0 / 1024.0) << " GB\n";
            oss << "    Available Memory: " << std::fixed << std::setprecision(2)
                << (device.available_memory / 1024.0 / 1024.0 / 1024.0) << " GB\n";
            oss << "    Multiprocessors: " << device.multiprocessor_count << "\n";
            oss << "    Max Threads/Block: " << device.max_threads_per_block << "\n";
            oss << "    Shared Memory/Block: " << (device.shared_memory_per_block / 1024)
                << " KB\n";
            oss << "    Compute Capability: " << device.compute_capability_major << "."
                << device.compute_capability_minor << "\n";
            oss << "    Double Precision: " << (device.supports_double_precision ? "Yes" : "No")
                << "\n";
            oss << "    Unified Memory: " << (device.supports_unified_memory ? "Yes" : "No")
                << "\n";
            oss << "    Concurrent Kernels: " << (device.supports_concurrent_kernels ? "Yes" : "No")
                << "\n";
            oss << "    PCI Bus ID: " << device.pci_bus_id << "\n";
            oss << "    Utilization: " << std::fixed << std::setprecision(1)
                << device.utilization_percentage << "%\n";
            oss << "    Temperature: " << std::fixed << std::setprecision(1)
                << device.temperature_celsius << "°C\n";
            oss << "    Power: " << std::fixed << std::setprecision(1)
                << device.power_consumption_watts << "W\n";

            // Calculate scores
            double const mc_score  = calculate_monte_carlo_score(device);
            double const pde_score = calculate_pde_score(device);
            oss << "    Monte Carlo Score: " << std::fixed << std::setprecision(3) << mc_score
                << "\n";
            oss << "    PDE Solver Score: " << std::fixed << std::setprecision(3) << pde_score
                << "\n";

            if (i < available_devices_.size() - 1)
            {
                oss << "\n";
            }
        }

        return oss.str();
    }
};

}  // anonymous namespace

gpu_device_manager& gpu_device_manager::instance()
{
    static gpu_device_manager_impl instance;
    static std::once_flag          initialized;

    std::call_once(initialized, [&]() { instance.initialize(); });

    return instance;
}

}  // namespace gpu
}  // namespace memory
