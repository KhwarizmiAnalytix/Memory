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

#include <functional>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

#include "common/memory_macros.h"
#include "common/device.h"
#include "gpu/gpu_memory_pool.h"
#include "gpu/gpu_resource_tracker.h"

#if MEMORY_HAS_CUDA
#include <cuda_runtime.h>
#include "common/memory_export.h"
#endif

namespace memory
{
namespace gpu
{

/**
 * @brief Allocation strategy enumeration
 */
enum class allocation_strategy
{
    USE_POOL,        ///< Use provided memory pool
    USE_DIRECT,      ///< Use direct GPU allocator
    FALLBACK_DIRECT  ///< Fallback to direct allocation when pool is unsuitable
};

/**
 * @brief Result of allocation strategy determination
 */
struct allocation_result
{
    allocation_strategy strategy;
    std::string         reason;  ///< Reason for the chosen strategy

    // C++17 structured binding support
    template <std::size_t N>
    [[nodiscard]] constexpr auto get() const noexcept
    {
        if constexpr (N == 0)
            return strategy;
        else if constexpr (N == 1)
            return reason;
        else
            static_assert(N < 2, "allocation_result only has 2 members");
    }
};

/**
 * @brief Deallocation strategy enumeration
 */
enum class deallocation_strategy
{
    USE_POOL,       ///< Use memory pool for deallocation
    USE_ALLOCATOR,  ///< Use GPU allocator for deallocation
    USE_DIRECT      ///< Use direct device-specific deallocation
};

/**
 * @brief Pool ownership mode for enhanced RAII semantics
 */
enum class pool_ownership_mode
{
    SHARED,    ///< Shared ownership with reference counting
    WEAK,      ///< Weak reference to prevent circular dependencies
    EXCLUSIVE  ///< Exclusive ownership with automatic cleanup
};

/**
 * @brief RAII GPU memory wrapper with automatic resource management
 *
 * Provides exception-safe GPU memory management using RAII principles.
 * Automatically handles allocation, deallocation, and resource tracking
 * with support for both raw pointers and typed arrays.
 *
 * Key features:
 * - Automatic memory deallocation on destruction
 * - Exception-safe resource management
 * - Integration with GPU memory pool and resource tracker
 * - Type-safe operations with template specialization
 * - Move semantics for efficient transfers
 * - Custom deleter support for specialized cleanup
 * - Alignment-aware allocation for optimal performance
 *
 * The wrapper ensures that GPU memory is properly released even in
 * the presence of exceptions, preventing memory leaks in complex
 * Monte Carlo simulations and PDE solver computations.
 *
 * @tparam T Element type (void for raw memory)
 *
 * @example
 * ```cpp
 * // Allocate typed GPU memory
 * auto gpu_array = gpu_memory_wrapper<float>::allocate(1000, device_enum::CUDA, 0);
 *
 * // Use memory
 * float* ptr = gpu_array.get();
 * // ... perform GPU computations ...
 *
 * // Memory is automatically freed when gpu_array goes out of scope
 *
 * // Transfer ownership
 * auto moved_array = std::move(gpu_array);
 *
 * // Create from existing memory with custom deleter
 * auto custom_wrapper = gpu_memory_wrapper<double>::wrap(
 *     existing_ptr, 500, device_enum::CUDA, 0,
 *     [](double* ptr) { custom_free_function(ptr); }
 * );
 * ```
 */
template <typename T = void>
class MEMORY_VISIBILITY gpu_memory_wrapper
{
public:
    using element_type  = T;
    using pointer       = T*;
    using const_pointer = const T*;
    using size_type     = std::size_t;
    using deleter_type  = std::function<void(pointer)>;

private:
    /** @brief Managed pointer */
    pointer ptr_ = nullptr;

    /** @brief Number of elements (for arrays) */
    size_type count_ = 0;

    /** @brief Device information */
    device_option device_;

    /** @brief Memory pool used for allocation with enhanced ownership semantics */
    std::shared_ptr<gpu_memory_pool> pool_;

    /** @brief Weak reference to pool for circular dependency prevention */
    std::weak_ptr<gpu_memory_pool> weak_pool_;

    /** @brief Pool ownership mode for RAII semantics */
    pool_ownership_mode pool_mode_ = pool_ownership_mode::SHARED;

    /** @brief Memory block information (if allocated through pool) */
    gpu_memory_block block_;

    /** @brief Custom deleter function with C++17 optional support */
    std::optional<deleter_type> deleter_;

    /** @brief Whether this wrapper owns the memory */
    bool owns_memory_ = false;

    /** @brief Resource tracker allocation ID */
    size_t tracker_id_ = 0;

    /**
     * @brief Determine the best allocation strategy
     * @param pool Requested memory pool (can be null)
     * @param device_type Target device type
     * @param device_index Target device index
     * @param bytes Number of bytes to allocate
     * @return Allocation strategy result
     */
    static allocation_result determine_allocation_strategy(
        std::shared_ptr<gpu_memory_pool> pool,
        device_enum                      device_type,
        int                              device_index,
        size_t                           bytes)
    {
        // If no pool is provided, use direct allocation
        if (!pool)
        {
            return {allocation_strategy::USE_DIRECT, "No pool provided"};
        }

        // Check if the pool supports the requested device type
        if (!is_pool_compatible_with_device(pool, device_type, device_index))
        {
            return {allocation_strategy::FALLBACK_DIRECT, "Pool incompatible with device type"};
        }

        // Check if the pool can handle the requested allocation size
        if (!can_pool_handle_allocation(pool, bytes))
        {
            return {allocation_strategy::FALLBACK_DIRECT, "Pool cannot handle allocation size"};
        }

        // Pool is suitable, use it
        return {allocation_strategy::USE_POOL, "Pool is compatible and suitable"};
    }

    /**
     * @brief Check if pool is compatible with device
     * @param pool Memory pool to check
     * @param device_type Target device type
     * @param device_index Target device index
     * @return True if compatible
     */
    static bool is_pool_compatible_with_device(
        std::shared_ptr<gpu_memory_pool> pool,
        MEMORY_UNUSED device_enum      device_type,
        MEMORY_UNUSED int              device_index)
    {
        if (!pool)
            return false;

        // For now, assume pools are device-agnostic or check pool configuration
        // This can be enhanced based on actual pool implementation
        return true;  // Simplified for now
    }

    /**
     * @brief Check if pool can handle the allocation
     * @param pool Memory pool to check
     * @param bytes Number of bytes to allocate
     * @return True if pool can handle the allocation
     */
    static bool can_pool_handle_allocation(std::shared_ptr<gpu_memory_pool> pool, size_t bytes)
    {
        if (!pool)
            return false;

        // Check if the pool has sufficient capacity
        // This is a simplified check - real implementation would query pool statistics
        return bytes > 0;  // Simplified for now
    }

    /**
     * @brief Determine the best deallocation strategy
     * @return Deallocation strategy to use
     */
    deallocation_strategy determine_deallocation_strategy() const
    {
        // If we have a pool and the block matches our pointer, use pool
        if (pool_ && block_.ptr == ptr_)
        {
            return deallocation_strategy::USE_POOL;
        }

        // If we have a typed allocator, use it
        if constexpr (!std::is_void_v<T>)
        {
            return deallocation_strategy::USE_ALLOCATOR;
        }

        // For void type, use direct deallocation
        return deallocation_strategy::USE_DIRECT;
    }

    /**
     * @brief Deallocate void memory using device-specific methods
     */
    void deallocate_void_memory()
    {
        switch (device_.type())
        {
#if MEMORY_HAS_CUDA
        case device_enum::CUDA:
            cudaFree(ptr_);
            break;
#endif
        default:
            // For unsupported device types, we can't safely deallocate
            // Log the issue but don't crash
            break;
        }
    }

    /**
     * @brief Default deleter using memory pool
     */
    void default_delete()
    {
        if (ptr_ && owns_memory_)
        {
            // Track deallocation
            if (tracker_id_ != 0)
            {
                gpu_resource_tracker::instance().track_deallocation(ptr_);
            }

            // Determine deallocation strategy and execute
            deallocation_strategy strategy = determine_deallocation_strategy();

            switch (strategy)
            {
            case deallocation_strategy::USE_POOL:
            {
                pool_->deallocate(block_);
                break;
            }
            case deallocation_strategy::USE_ALLOCATOR:
            {
                // Direct CUDA deallocation (gpu_allocator removed)
#if MEMORY_HAS_CUDA
                if (device_.type() == device_enum::CUDA)
                {
                    cudaError_t result = cudaSetDevice(device_.index());
                    if (result == cudaSuccess)
                    {
                        result = cudaFree(ptr_);
                    }
                }
#endif
                break;
            }
            case deallocation_strategy::USE_DIRECT:
            {
                deallocate_void_memory();
                break;
            }
            }
        }
    }

public:
    /**
     * @brief Default constructor - creates empty wrapper
     */
    gpu_memory_wrapper()
        : ptr_(nullptr),
          count_(0),
          device_(device_enum::CUDA, 0),
          pool_(nullptr),
          block_(),
          deleter_(nullptr),
          owns_memory_(false)
    {
    }

    /**
     * @brief Constructor with explicit parameters
     * @param ptr Pointer to manage
     * @param count Number of elements
     * @param device Device information
     * @param pool Memory pool (optional)
     * @param block Memory block information (optional)
     * @param deleter Custom deleter (optional)
     * @param owns_memory Whether wrapper owns the memory
     */
    gpu_memory_wrapper(
        pointer                          ptr,
        size_type                        count,
        const device_option&             device,
        std::shared_ptr<gpu_memory_pool> pool        = nullptr,
        gpu_memory_block&&               block       = gpu_memory_block{},
        std::optional<deleter_type>      deleter     = std::nullopt,
        bool                             owns_memory = true,
        pool_ownership_mode              pool_mode   = pool_ownership_mode::SHARED)
        : ptr_(ptr),
          count_(count),
          device_(device),
          pool_(pool),
          pool_mode_(pool_mode),
          block_(std::move(block)),
          deleter_(std::move(deleter)),
          owns_memory_(owns_memory),
          tracker_id_(0)
    {
        // C++17 enhanced initialization
        if constexpr (true)
        {
            // Set up weak reference based on ownership mode
            if (pool_ && pool_mode_ == pool_ownership_mode::WEAK)
            {
                weak_pool_ = pool_;
                pool_.reset();
            }

            // Track allocation if we own the memory
            if (owns_memory_ && ptr_)
            {
                tracker_id_ = gpu_resource_tracker::instance().track_allocation(
                    ptr_,
                    count_ * sizeof(T),
                    device_.type(),
                    device_.index(),
                    "gpu_memory_wrapper");
            }
        }
    }

    /**
     * @brief Destructor - automatically releases memory
     */
    ~gpu_memory_wrapper() { reset(); }

    /**
     * @brief Move constructor (C++17 enhanced)
     */
    gpu_memory_wrapper(gpu_memory_wrapper&& other) noexcept
        : ptr_(std::exchange(other.ptr_, nullptr)),
          count_(std::exchange(other.count_, 0)),
          device_(std::move(other.device_)),
          pool_(std::move(other.pool_)),
          weak_pool_(std::move(other.weak_pool_)),
          pool_mode_(std::exchange(other.pool_mode_, pool_ownership_mode::SHARED)),
          block_(std::move(other.block_)),
          deleter_(std::move(other.deleter_)),
          owns_memory_(std::exchange(other.owns_memory_, false)),
          tracker_id_(std::exchange(other.tracker_id_, 0))
    {
        // C++17 std::exchange provides cleaner move semantics
        // All resources are properly transferred
    }

    /**
     * @brief Copy constructor (C++17 enhanced)
     */
    gpu_memory_wrapper(const gpu_memory_wrapper& other)
        : ptr_(nullptr),
          count_(0),
          device_(other.device_),
          pool_(other.pool_),
          weak_pool_(other.weak_pool_),
          pool_mode_(other.pool_mode_),
          block_(),
          deleter_(other.deleter_),
          owns_memory_(false),
          tracker_id_(0)
    {
        // Copy creates non-owning wrapper to avoid double deletion
        if (other.ptr_)
        {
            ptr_   = other.ptr_;
            count_ = other.count_;
            // Deliberately not setting owns_memory_ = true
        }
    }

    /**
     * @brief Move assignment operator
     */
    gpu_memory_wrapper& operator=(gpu_memory_wrapper&& other) noexcept
    {
        if (this != &other)
        {
            reset();

            // C++17 std::exchange for cleaner move assignment
            ptr_         = std::exchange(other.ptr_, nullptr);
            count_       = std::exchange(other.count_, 0);
            device_      = std::move(other.device_);
            pool_        = std::move(other.pool_);
            weak_pool_   = std::move(other.weak_pool_);
            pool_mode_   = std::exchange(other.pool_mode_, pool_ownership_mode::SHARED);
            block_       = std::move(other.block_);
            deleter_     = std::move(other.deleter_);
            owns_memory_ = std::exchange(other.owns_memory_, false);
            tracker_id_  = std::exchange(other.tracker_id_, 0);
        }
        return *this;
    }

    /**
     * @brief Copy assignment operator (C++17 enhanced)
     * Creates non-owning copy to prevent double deletion
     */
    gpu_memory_wrapper& operator=(const gpu_memory_wrapper& other)
    {
        if (this != &other)
        {
            reset();  // Clean up current resources

            // Copy creates non-owning wrapper
            device_      = other.device_;
            pool_        = other.pool_;
            weak_pool_   = other.weak_pool_;
            pool_mode_   = other.pool_mode_;
            deleter_     = other.deleter_;
            owns_memory_ = false;  // Don't transfer ownership in copy
            tracker_id_  = 0;

            if (other.ptr_)
            {
                ptr_   = other.ptr_;
                count_ = other.count_;
            }
        }
        return *this;
    }

    /**
     * @brief Allocate GPU memory using memory pool
     * @param count Number of elements to allocate
     * @param device_type Device type
     * @param device_index Device index
     * @param pool Memory pool to use (optional, uses default if null)
     * @param tag Tag for resource tracking
     * @return GPU memory wrapper managing the allocated memory
     */
    MEMORY_NODISCARD static gpu_memory_wrapper allocate(
        size_type                        count,
        device_enum                      device_type,
        int                              device_index = 0,
        std::shared_ptr<gpu_memory_pool> pool         = nullptr,
        MEMORY_UNUSED const std::string& tag        = "")
    {
        if (count == 0)
        {
            return gpu_memory_wrapper{};
        }

        size_t        bytes = count * sizeof(T);
        device_option device(device_type, device_index);

        // Determine allocation strategy based on pool availability and device support
        allocation_result result =
            determine_allocation_strategy(pool, device_type, device_index, bytes);

        pointer          ptr = nullptr;
        gpu_memory_block block;

        switch (result.strategy)
        {
        case allocation_strategy::USE_POOL:
        {
            block = pool->allocate(bytes, device_type, device_index);
            ptr   = static_cast<pointer>(block.ptr);
            break;
        }
        case allocation_strategy::USE_DIRECT:
        {
            // Direct CUDA allocation (gpu_allocator removed)
#if MEMORY_HAS_CUDA
            if (device_type == device_enum::CUDA)
            {
                cudaError_t cuda_result = cudaSetDevice(device_index);
                if (cuda_result != cudaSuccess)
                {
                    throw std::runtime_error(
                        "Failed to set CUDA device: " +
                        std::string(cudaGetErrorString(cuda_result)));
                }
                cuda_result = cudaMalloc(reinterpret_cast<void**>(&ptr), bytes);
                if (cuda_result != cudaSuccess)
                {
                    throw std::bad_alloc();
                }
            }
#endif
            block = gpu_memory_block(ptr, bytes, device);
            pool  = nullptr;  // Clear pool reference since we used direct allocation
            break;
        }
        case allocation_strategy::FALLBACK_DIRECT:
        {
            // Pool was requested but not suitable, use direct allocation
#if MEMORY_HAS_CUDA
            if (device_type == device_enum::CUDA)
            {
                cudaError_t cuda_result = cudaSetDevice(device_index);
                if (cuda_result != cudaSuccess)
                {
                    throw std::runtime_error(
                        "Failed to set CUDA device: " +
                        std::string(cudaGetErrorString(cuda_result)));
                }
                cuda_result = cudaMalloc(reinterpret_cast<void**>(&ptr), bytes);
                if (cuda_result != cudaSuccess)
                {
                    throw std::bad_alloc();
                }
            }
#endif
            block = gpu_memory_block(ptr, bytes, device);
            pool  = nullptr;  // Clear pool reference
            break;
        }
        }

        return gpu_memory_wrapper(ptr, count, device, pool, std::move(block), nullptr, true);
    }

    /**
     * @brief Wrap existing GPU memory with RAII management
     * @param ptr Existing pointer to wrap
     * @param count Number of elements
     * @param device_type Device type
     * @param device_index Device index
     * @param deleter Custom deleter function
     * @param tag Tag for resource tracking
     * @return GPU memory wrapper managing the existing memory
     */
    MEMORY_NODISCARD static gpu_memory_wrapper wrap(
        pointer               ptr,
        size_type             count,
        device_enum           device_type,
        int                   device_index     = 0,
        deleter_type          deleter          = nullptr,
        MEMORY_UNUSED const std::string& tag = "")
    {
        device_option device(device_type, device_index);
        return gpu_memory_wrapper(ptr, count, device, nullptr, gpu_memory_block{}, deleter, true);
    }

    /**
     * @brief Create non-owning wrapper (does not delete on destruction)
     * @param ptr Pointer to wrap
     * @param count Number of elements
     * @param device_type Device type
     * @param device_index Device index
     * @return Non-owning GPU memory wrapper
     */
    MEMORY_NODISCARD static gpu_memory_wrapper non_owning(
        pointer ptr, size_type count, device_enum device_type, int device_index = 0)
    {
        device_option device(device_type, device_index);
        return gpu_memory_wrapper(ptr, count, device, nullptr, gpu_memory_block{}, nullptr, false);
    }

    /**
     * @brief Get raw pointer
     * @return Raw pointer to GPU memory
     */
    MEMORY_NODISCARD pointer get() const noexcept { return ptr_; }

    /**
     * @brief Get const pointer
     * @return Const pointer to GPU memory
     */
    MEMORY_NODISCARD const_pointer get_const() const noexcept { return ptr_; }

    /**
     * @brief Get number of elements
     * @return Number of elements in the array
     */
    MEMORY_NODISCARD size_type size() const noexcept { return count_; }

    /**
     * @brief Get size in bytes
     * @return Size in bytes
     */
    MEMORY_NODISCARD size_type size_bytes() const noexcept { return count_ * sizeof(T); }

    /**
     * @brief Get device information
     * @return Device where memory is allocated
     */
    MEMORY_NODISCARD const device_option& device() const noexcept { return device_; }

    /**
     * @brief Check if wrapper is empty
     * @return True if wrapper does not manage any memory
     */
    MEMORY_NODISCARD bool empty() const noexcept { return ptr_ == nullptr; }

    /**
     * @brief Check if wrapper owns the memory
     * @return True if wrapper will delete memory on destruction
     */
    MEMORY_NODISCARD bool owns_memory() const noexcept { return owns_memory_; }

    /**
     * @brief Release ownership of memory
     * @return Raw pointer to released memory
     */
    MEMORY_NODISCARD pointer release() noexcept
    {
        pointer result = ptr_;
        ptr_           = nullptr;
        count_         = 0;
        owns_memory_   = false;
        tracker_id_    = 0;
        return result;
    }

    /**
     * @brief Reset wrapper and release memory
     * @param new_ptr New pointer to manage (optional)
     * @param new_count New element count (optional)
     */
    void reset(pointer new_ptr = nullptr, size_type new_count = 0)
    {
        if (ptr_ && owns_memory_)
        {
            // C++17 optional-based deleter handling
            if (deleter_ != nullptr && deleter_.has_value())
            {
                (*deleter_)(ptr_);
            }
            else
            {
                default_delete();
            }
        }

        // Enhanced pool cleanup with C++17 features
        reset_pool_ownership();

        ptr_         = new_ptr;
        count_       = new_count;
        owns_memory_ = (new_ptr != nullptr);
        tracker_id_  = 0;

        if (new_ptr)
        {
            tracker_id_ = gpu_resource_tracker::instance().track_allocation(
                new_ptr,
                new_count * sizeof(T),
                device_.type(),
                device_.index(),
                "gpu_memory_wrapper");
        }
    }

    /**
     * @brief Reset pool ownership with proper RAII semantics
     */
    void reset_pool_ownership() noexcept
    {
        if constexpr (true)  // C++17 if constexpr for compile-time optimization
        {
            // Reset shared pointer with proper reference counting
            if (pool_ && pool_.use_count() == 1 && pool_mode_ == pool_ownership_mode::EXCLUSIVE)
            {
                // Last reference with exclusive ownership - trigger cleanup
                // pool_->clear_pools();  // Would need this method in gpu_memory_pool
            }

            pool_.reset();
            weak_pool_.reset();
            pool_mode_ = pool_ownership_mode::SHARED;
        }
    }

    /**
     * @brief Swap contents with another wrapper (C++17 enhanced)
     * @param other Other wrapper to swap with
     */
    void swap(gpu_memory_wrapper& other) noexcept
    {
        using std::swap;

        // Enhanced swap with all new members
        swap(ptr_, other.ptr_);
        swap(count_, other.count_);
        swap(device_, other.device_);
        swap(pool_, other.pool_);
        swap(weak_pool_, other.weak_pool_);
        swap(pool_mode_, other.pool_mode_);
        swap(block_, other.block_);
        swap(deleter_, other.deleter_);
        swap(owns_memory_, other.owns_memory_);
        swap(tracker_id_, other.tracker_id_);
    }

    // C++17 Pool Management Methods

    /**
     * @brief Get pool reference count
     * @return Number of references to the pool, or 0 if no pool
     */
    [[nodiscard]] auto pool_use_count() const noexcept -> long
    {
        return pool_ ? pool_.use_count() : 0;
    }

    /**
     * @brief Check if pool is unique (only this wrapper references it)
     * @return True if pool has only one reference
     */
    [[nodiscard]] bool pool_unique() const noexcept { return pool_ && pool_.unique(); }

    /**
     * @brief Get pool ownership mode
     * @return Current pool ownership mode
     */
    [[nodiscard]] auto pool_ownership() const noexcept -> pool_ownership_mode { return pool_mode_; }

    /**
     * @brief Set pool ownership mode with validation
     * @param mode New ownership mode
     */
    void set_pool_ownership(pool_ownership_mode mode) noexcept
    {
        if constexpr (true)  // C++17 compile-time branch
        {
            switch (mode)
            {
            case pool_ownership_mode::WEAK:
                if (pool_)
                {
                    weak_pool_ = pool_;
                    pool_.reset();
                }
                break;

            case pool_ownership_mode::SHARED:
                if (auto shared_pool = weak_pool_.lock())
                {
                    pool_ = shared_pool;
                }
                break;

            case pool_ownership_mode::EXCLUSIVE:
                // Keep current pool_ as exclusive reference
                weak_pool_.reset();
                break;
            }
            pool_mode_ = mode;
        }
    }

    /**
     * @brief Get shared pool reference (may be null)
     * @return Shared pointer to pool
     */
    [[nodiscard]] auto get_pool() const noexcept -> std::shared_ptr<gpu_memory_pool>
    {
        return pool_;
    }

    /**
     * @brief Get weak pool reference
     * @return Weak pointer to pool
     */
    [[nodiscard]] auto get_weak_pool() const noexcept -> std::weak_ptr<gpu_memory_pool>
    {
        return weak_pool_;
    }

    /**
     * @brief Try to lock weak pool reference
     * @return Shared pointer if weak reference is still valid, null otherwise
     */
    [[nodiscard]] auto try_lock_pool() const noexcept -> std::shared_ptr<gpu_memory_pool>
    {
        return weak_pool_.lock();
    }

    /**
     * @brief Boolean conversion operator
     * @return True if wrapper manages memory
     */
    MEMORY_NODISCARD explicit operator bool() const noexcept { return ptr_ != nullptr; }

    /**
     * @brief Equality comparison
     * @param other Other wrapper to compare with
     * @return True if both wrappers manage the same memory
     */
    MEMORY_NODISCARD bool operator==(const gpu_memory_wrapper& other) const noexcept
    {
        return ptr_ == other.ptr_;
    }

    /**
     * @brief Inequality comparison
     * @param other Other wrapper to compare with
     * @return True if wrappers manage different memory
     */
    MEMORY_NODISCARD bool operator!=(const gpu_memory_wrapper& other) const noexcept
    {
        return ptr_ != other.ptr_;
    }
};

/**
 * @brief Specialization for void type (raw memory)
 */
template <>
class MEMORY_VISIBILITY gpu_memory_wrapper<void>
{
public:
    using element_type  = void;
    using pointer       = void*;
    using const_pointer = const void*;
    using size_type     = std::size_t;
    using deleter_type  = std::function<void(pointer)>;

private:
    pointer                          ptr_   = nullptr;
    size_type                        bytes_ = 0;
    device_option                    device_;
    std::shared_ptr<gpu_memory_pool> pool_;
    gpu_memory_block                 block_;
    deleter_type                     deleter_;
    bool                             owns_memory_ = false;
    size_t                           tracker_id_  = 0;

    void default_delete()
    {
        if (ptr_ && owns_memory_)
        {
            if (tracker_id_ != 0)
            {
                gpu_resource_tracker::instance().track_deallocation(ptr_);
            }

            if (pool_ && block_.ptr == ptr_)
            {
                pool_->deallocate(block_);
            }
            else
            {
                switch (device_.type())
                {
#if MEMORY_HAS_CUDA
                case device_enum::CUDA:
                    cudaFree(ptr_);
                    break;
#endif
                default:
                    break;
                }
            }
        }
    }

public:
    gpu_memory_wrapper() = default;

    gpu_memory_wrapper(
        pointer                          ptr,
        size_type                        bytes,
        const device_option&             device,
        std::shared_ptr<gpu_memory_pool> pool        = nullptr,
        gpu_memory_block&&               block       = gpu_memory_block{},
        deleter_type                     deleter     = nullptr,
        bool                             owns_memory = true)
        : ptr_(ptr),
          bytes_(bytes),
          device_(device),
          pool_(pool),
          block_(std::move(block)),
          deleter_(deleter),
          owns_memory_(owns_memory),
          tracker_id_(0)
    {
        if (owns_memory_ && ptr_)
        {
            tracker_id_ = gpu_resource_tracker::instance().track_allocation(
                ptr_, bytes_, device_.type(), device_.index(), "gpu_memory_wrapper");
        }
    }

    ~gpu_memory_wrapper() { reset(); }

    gpu_memory_wrapper(gpu_memory_wrapper&& other) noexcept
        : ptr_(other.ptr_),
          bytes_(other.bytes_),
          device_(other.device_),
          pool_(std::move(other.pool_)),
          block_(std::move(other.block_)),
          deleter_(std::move(other.deleter_)),
          owns_memory_(other.owns_memory_),
          tracker_id_(other.tracker_id_)
    {
        other.ptr_         = nullptr;
        other.bytes_       = 0;
        other.owns_memory_ = false;
        other.tracker_id_  = 0;
    }

    gpu_memory_wrapper& operator=(gpu_memory_wrapper&& other) noexcept
    {
        if (this != &other)
        {
            reset();

            ptr_         = other.ptr_;
            bytes_       = other.bytes_;
            device_      = other.device_;
            pool_        = std::move(other.pool_);
            block_       = std::move(other.block_);
            deleter_     = std::move(other.deleter_);
            owns_memory_ = other.owns_memory_;
            tracker_id_  = other.tracker_id_;

            other.ptr_         = nullptr;
            other.bytes_       = 0;
            other.owns_memory_ = false;
            other.tracker_id_  = 0;
        }
        return *this;
    }

    gpu_memory_wrapper(const gpu_memory_wrapper&)            = delete;
    gpu_memory_wrapper& operator=(const gpu_memory_wrapper&) = delete;

    static gpu_memory_wrapper allocate(
        size_type                        bytes,
        device_enum                      device_type,
        int                              device_index = 0,
        std::shared_ptr<gpu_memory_pool> pool         = nullptr)
    {
        if (bytes == 0)
        {
            device_option device(device_type, device_index);
            return gpu_memory_wrapper(
                nullptr, 0, device, nullptr, gpu_memory_block{}, nullptr, false);
        }

        device_option device(device_type, device_index);

        if (!pool)
        {
            gpu_memory_pool_config config;
            pool = gpu_memory_pool::create(config);
        }

        auto block = pool->allocate(bytes, device_type, device_index);
        return gpu_memory_wrapper(block.ptr, bytes, device, pool, std::move(block), nullptr, true);
    }

    static gpu_memory_wrapper non_owning(
        pointer ptr, size_type bytes, device_enum device_type, int device_index = 0)
    {
        device_option device(device_type, device_index);
        return gpu_memory_wrapper(ptr, bytes, device, nullptr, gpu_memory_block{}, nullptr, false);
    }

    pointer              get() const noexcept { return ptr_; }
    const_pointer        get_const() const noexcept { return ptr_; }
    size_type            size_bytes() const noexcept { return bytes_; }
    const device_option& device() const noexcept { return device_; }
    bool                 empty() const noexcept { return ptr_ == nullptr; }
    bool                 owns_memory() const noexcept { return owns_memory_; }

    pointer release() noexcept
    {
        pointer result = ptr_;
        ptr_           = nullptr;
        bytes_         = 0;
        owns_memory_   = false;
        tracker_id_    = 0;
        return result;
    }

    void reset(pointer new_ptr = nullptr, size_type new_bytes = 0)
    {
        if (ptr_ && owns_memory_)
        {
            if (deleter_)
            {
                deleter_(ptr_);
            }
            else
            {
                default_delete();
            }
        }

        ptr_         = new_ptr;
        bytes_       = new_bytes;
        owns_memory_ = (new_ptr != nullptr);
        tracker_id_  = 0;

        if (new_ptr)
        {
            tracker_id_ = gpu_resource_tracker::instance().track_allocation(
                new_ptr, new_bytes, device_.type(), device_.index(), "gpu_memory_wrapper");
        }
    }

    explicit operator bool() const noexcept { return ptr_ != nullptr; }
    bool operator==(const gpu_memory_wrapper& other) const noexcept { return ptr_ == other.ptr_; }
    bool operator!=(const gpu_memory_wrapper& other) const noexcept { return ptr_ != other.ptr_; }
};

/**
 * @brief Convenience function to create GPU memory wrapper
 * @tparam T Element type
 * @param count Number of elements
 * @param device_type Device type
 * @param device_index Device index
 * @return GPU memory wrapper
 */
template <typename T>
gpu_memory_wrapper<T> make_gpu_memory(
    std::size_t count, device_enum device_type, int device_index = 0)
{
    return gpu_memory_wrapper<T>::allocate(count, device_type, device_index);
}

/**
 * @brief Swap function for GPU memory wrappers
 * @tparam T Element type
 * @param a First wrapper
 * @param b Second wrapper
 */
template <typename T>
void swap(gpu_memory_wrapper<T>& a, gpu_memory_wrapper<T>& b) noexcept
{
    a.swap(b);
}

}  // namespace gpu
}  // namespace memory

// C++17 structured binding support for allocation_result
namespace std
{
template <>
struct tuple_size<memory::gpu::allocation_result> : std::integral_constant<std::size_t, 2>
{
};

template <>
struct tuple_element<0, memory::gpu::allocation_result>
{
    using type = memory::gpu::allocation_strategy;
};

template <>
struct tuple_element<1, memory::gpu::allocation_result>
{
    using type = std::string;
};
}  // namespace std
