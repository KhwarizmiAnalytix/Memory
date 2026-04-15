#include "gpu/gpu_memory_transfer.h"

#include <algorithm>
#include <atomic>
#include <iomanip>
#include <mutex>
#include <queue>
#include <sstream>
#include <thread>
#include <utility>

#include "common/memory_containers.h"
#include "common/memory_macros.h"
#include "util/exception.h"

// Hash specialization for std::pair<device_enum, int>
namespace std
{
template <>
struct hash<std::pair<memory::device_enum, int>>
{
    size_t operator()(const std::pair<memory::device_enum, int>& p) const
    {
        return std::hash<int>()(static_cast<int>(p.first)) ^ (std::hash<int>()(p.second) << 1);
    }
};
}  // namespace std

#if MEMORY_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace memory
{
namespace gpu
{

namespace
{

/**
 * @brief CUDA stream implementation
 */
#if MEMORY_HAS_CUDA
class cuda_stream_impl : public gpu_stream
{
private:
    cudaStream_t  stream_;
    device_option device_;

public:
    cuda_stream_impl(device_enum device_type, int device_index, int priority)
        : device_(device_type, device_index)
    {
        cudaSetDevice(device_index);

        if (priority == 0)
        {
            cudaError_t const result = cudaStreamCreate(&stream_);
            if (result != cudaSuccess)
            {
                MEMORY_THROW(
                    "Failed to create CUDA stream: {}", std::string(cudaGetErrorString(result)));
            }
        }
        else
        {
            // Create stream with priority
            int least_priority;
            int greatest_priority;
            cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
            int const actual_priority =
                std::max(greatest_priority, std::min(least_priority, priority));

            cudaError_t const result =
                cudaStreamCreateWithPriority(&stream_, cudaStreamDefault, actual_priority);
            if (result != cudaSuccess)
            {
                MEMORY_THROW(
                    "Failed to create CUDA stream with priority: {}",
                    std::string(cudaGetErrorString(result)));
            }
        }
    }

    ~cuda_stream_impl() override
    {
        if (stream_ != nullptr)
        {
            cudaStreamDestroy(stream_);
        }
    }

    [[nodiscard]] device_option get_device() const override { return device_; }

    void synchronize() override
    {
        cudaError_t const result = cudaStreamSynchronize(stream_);
        if (result != cudaSuccess)
        {
            MEMORY_THROW(
                "CUDA stream synchronization failed: {}", std::string(cudaGetErrorString(result)));
        }
    }

    [[nodiscard]] bool is_idle() const override
    {
        cudaError_t const result = cudaStreamQuery(stream_);
        return (result == cudaSuccess);
    }

    [[nodiscard]] void* get_native_handle() const override { return static_cast<void*>(stream_); }
};
#endif

/**
 * @brief Transfer operation for async processing
 */
struct transfer_operation
{
    size_t                          id;
    const void*                     src;
    void*                           dst;
    size_t                          size;
    transfer_direction              direction;
    gpu_stream*                     stream;
    transfer_callback               callback;
    std::promise<gpu_transfer_info> promise;
    gpu_transfer_info               info;

    transfer_operation(
        size_t             id_,
        const void*        src_,
        void*              dst_,
        size_t             size_,
        transfer_direction dir,
        gpu_stream*        stream_,
        transfer_callback  cb)
        : id(id_),
          src(src_),
          dst(dst_),
          size(size_),
          direction(dir),
          stream(stream_),
          callback(std::move(cb))
    {
        info.transfer_id       = id;
        info.direction         = direction;
        info.bytes_transferred = size;
        info.status            = transfer_status::PENDING;
    }
};

/**
 * @brief Internal implementation of GPU memory transfer manager
 */
class gpu_memory_transfer_impl : public gpu_memory_transfer
{
private:
    /** @brief Mutex for thread-safe operations */
    mutable std::mutex mutex_;

    /** @brief Next transfer ID */
    std::atomic<size_t> next_transfer_id_{1};

    /** @brief Active transfer operations */
    memory_map<size_t, std::unique_ptr<transfer_operation>> active_transfers_;

    /** @brief Transfer statistics */
    std::atomic<size_t> total_transfers_{0};
    std::atomic<size_t> total_bytes_transferred_{0};
    std::atomic<double> total_transfer_time_ms_{0.0};
    std::atomic<size_t> failed_transfers_{0};

    /** @brief Default streams for each device */
    memory_map<
        std::pair<device_enum, int>,
        std::unique_ptr<gpu_stream>,
        std::hash<std::pair<device_enum, int>>>
        default_streams_;

    /**
     * @brief Get or create default stream for device
     */
    gpu_stream* get_default_stream(device_enum device_type, int device_index)
    {
        std::scoped_lock const lock(mutex_);

        auto key = std::make_pair(device_type, device_index);
        auto it  = default_streams_.find(key);
        if (it == default_streams_.end())
        {
            auto        stream     = gpu_stream::create(device_type, device_index, 0);
            gpu_stream* stream_ptr = stream.get();  //NOLINT
            default_streams_[key]  = std::move(stream);
            return stream_ptr;
        }

        return it->second.get();
    }

    /**
     * @brief Perform the actual memory transfer
     */
    void perform_transfer(transfer_operation& op)
    {
        op.info.start_time = std::chrono::high_resolution_clock::now();
        op.info.status     = transfer_status::RUNNING;

        try
        {
            switch (op.direction)
            {
#if MEMORY_HAS_CUDA
            case transfer_direction::HOST_TO_DEVICE:
            case transfer_direction::DEVICE_TO_HOST:
            case transfer_direction::DEVICE_TO_DEVICE:
            {
                cudaMemcpyKind kind;
                switch (op.direction)
                {
                case transfer_direction::HOST_TO_DEVICE:
                    kind = cudaMemcpyHostToDevice;
                    break;
                case transfer_direction::DEVICE_TO_HOST:
                    kind = cudaMemcpyDeviceToHost;
                    break;
                case transfer_direction::DEVICE_TO_DEVICE:
                    kind = cudaMemcpyDeviceToDevice;
                    break;
                default:
                    kind = cudaMemcpyDefault;
                    break;
                }

                cudaStream_t cuda_stream = nullptr;
                if (op.stream != nullptr)
                {
                    cuda_stream = static_cast<cudaStream_t>(op.stream->get_native_handle());
                }

                cudaError_t result = cudaMemcpyAsync(op.dst, op.src, op.size, kind, cuda_stream);
                if (result != cudaSuccess)
                {
                    op.info.error_message =
                        "CUDA memcpy failed: " + std::string(cudaGetErrorString(result));
                    op.info.status = transfer_status::FAILED;
                    failed_transfers_.fetch_add(1);
                    return;
                }

                // Synchronize to ensure transfer completion for timing
                if (cuda_stream != nullptr)
                {
                    result = cudaStreamSynchronize(cuda_stream);
                }
                else
                {
                    result = cudaDeviceSynchronize();
                }

                if (result != cudaSuccess)
                {
                    op.info.error_message =
                        "CUDA synchronization failed: " + std::string(cudaGetErrorString(result));
                    op.info.status = transfer_status::FAILED;
                    failed_transfers_.fetch_add(1);
                    return;
                }

                break;
            }
#endif
            case transfer_direction::HOST_TO_HOST:
            {
                std::memcpy(op.dst, op.src, op.size);
                break;
            }
            default:
                op.info.error_message = "Unsupported transfer direction";
                op.info.status        = transfer_status::FAILED;
                failed_transfers_.fetch_add(1);
                return;
            }

            op.info.end_time = std::chrono::high_resolution_clock::now();
            op.info.status   = transfer_status::COMPLETED;

            // Calculate bandwidth
            double const duration_ms = op.info.get_duration_ms();
            if (duration_ms > 0.0)
            {
                op.info.bandwidth_gbps =
                    (op.size / 1024.0 / 1024.0 / 1024.0) / (duration_ms / 1000.0);
            }

            // Update statistics
            total_transfers_.fetch_add(1);
            total_bytes_transferred_.fetch_add(op.size);
            // Atomic add for double (C++20 feature, use compare_exchange for compatibility)
            double expected = total_transfer_time_ms_.load();
            while (!total_transfer_time_ms_.compare_exchange_weak(expected, expected + duration_ms))
            {
                ;
            }
        }
        catch (const std::exception& e)
        {
            op.info.end_time      = std::chrono::high_resolution_clock::now();
            op.info.error_message = e.what();
            op.info.status        = transfer_status::FAILED;
            failed_transfers_.fetch_add(1);
        }
    }

public:
    gpu_memory_transfer_impl() = default;

    ~gpu_memory_transfer_impl() override { gpu_memory_transfer_impl::wait_for_all_transfers(); }

    gpu_transfer_info transfer_sync(
        const void*        src,
        void*              dst,
        size_t             size,
        transfer_direction direction,
        gpu_stream*        stream) override
    {
        if ((src == nullptr) || (dst == nullptr) || size == 0)
        {
            MEMORY_THROW("Invalid transfer parameters");
        }

        size_t const       transfer_id = next_transfer_id_.fetch_add(1);
        transfer_operation op(transfer_id, src, dst, size, direction, stream, nullptr);

        perform_transfer(op);

        return op.info;
    }

    std::future<gpu_transfer_info> transfer_async(
        const void*        src,
        void*              dst,
        size_t             size,
        transfer_direction direction,
        gpu_stream*        stream,
        transfer_callback  callback) override
    {
        if ((src == nullptr) || (dst == nullptr) || size == 0)
        {
            MEMORY_THROW("Invalid transfer parameters");
        }

        size_t const transfer_id = next_transfer_id_.fetch_add(1);
        auto         op          = std::make_unique<transfer_operation>(
            transfer_id, src, dst, size, direction, stream, callback);

        auto future = op->promise.get_future();

        // Launch transfer in separate thread
        std::thread(
            [this, op_ptr = op.get()]()
            {
                perform_transfer(*op_ptr);

                // Call callback if provided
                if (op_ptr->callback)
                {
                    op_ptr->callback(op_ptr->info);
                }

                // Set promise value
                op_ptr->promise.set_value(op_ptr->info);

                // Remove from active transfers
                std::scoped_lock const lock(mutex_);
                active_transfers_.erase(op_ptr->id);
            })
            .detach();

        // Store operation
        {
            std::scoped_lock const lock(mutex_);
            active_transfers_[transfer_id] = std::move(op);
        }

        return future;
    }

    std::vector<std::future<gpu_transfer_info>> transfer_batch_async(
        const std::vector<std::tuple<const void*, void*, size_t, transfer_direction>>& transfers,
        gpu_stream* stream) override
    {
        std::vector<std::future<gpu_transfer_info>> futures;
        futures.reserve(transfers.size());

        for (const auto& [src, dst, size, direction] : transfers)
        {
            futures.push_back(transfer_async(src, dst, size, direction, stream, nullptr));
        }

        return futures;
    }

    size_t get_optimal_chunk_size(
        size_t total_size, transfer_direction direction, device_enum device_type) const override
    {
        // Base chunk size recommendations
        size_t base_chunk_size = size_t{64ULL} * 1024ULL;  // 64MB default

        switch (device_type)
        {
        case device_enum::CUDA:
            // CUDA typically performs well with larger chunks
            base_chunk_size = size_t{128} * 1024ULL;  // 128MB
            break;
        case device_enum::HIP:  // HIP (placeholder)
            // HIP may prefer smaller chunks
            base_chunk_size = size_t{32} * 1024ULL;  // 32MB
            break;
        default:
            break;
        }

        // Adjust based on transfer direction
        switch (direction)
        {
        case transfer_direction::HOST_TO_DEVICE:
            // Host to device transfers can use larger chunks
            break;
        case transfer_direction::DEVICE_TO_HOST:
            // Device to host may benefit from smaller chunks
            base_chunk_size = std::min(base_chunk_size, static_cast<size_t>(64ULL * 1024ULL));
            break;
        case transfer_direction::DEVICE_TO_DEVICE:
            // Device to device can use very large chunks
            base_chunk_size = std::max(base_chunk_size, static_cast<size_t>(256ULL * 1024ULL));
            break;
        default:
            break;
        }

        // Don't exceed total size
        return std::min(base_chunk_size, total_size);
    }

    std::string get_transfer_statistics() const override
    {
        std::ostringstream oss;
        oss << "GPU Memory Transfer Statistics:\n";
        oss << "==============================\n";

        size_t const total   = total_transfers_.load();
        size_t const failed  = failed_transfers_.load();
        size_t const bytes   = total_bytes_transferred_.load();
        double const time_ms = total_transfer_time_ms_.load();

        oss << "Total transfers: " << total << "\n";
        oss << "Successful transfers: " << (total - failed) << "\n";
        oss << "Failed transfers: " << failed << "\n";
        oss << "Success rate: " << std::fixed << std::setprecision(2)
            << (total > 0 ? 100.0 * (total - failed) / total : 0.0) << "%\n";

        oss << "Total bytes transferred: " << std::fixed << std::setprecision(2)
            << (bytes / 1024.0 / 1024.0 / 1024.0) << " GB\n";

        oss << "Total transfer time: " << std::fixed << std::setprecision(2) << (time_ms / 1000.0)
            << " seconds\n";

        if (time_ms > 0.0)
        {
            double const avg_bandwidth = (bytes / 1024.0 / 1024.0 / 1024.0) / (time_ms / 1000.0);
            oss << "Average bandwidth: " << std::fixed << std::setprecision(2) << avg_bandwidth
                << " GB/s\n";
        }

        {
            std::scoped_lock const lock(mutex_);
            oss << "Active transfers: " << active_transfers_.size() << "\n";
        }

        return oss.str();
    }

    void clear_statistics() override
    {
        total_transfers_.store(0);
        total_bytes_transferred_.store(0);
        total_transfer_time_ms_.store(0.0);
        failed_transfers_.store(0);
    }

    void wait_for_all_transfers() override
    {
        std::vector<std::unique_ptr<transfer_operation>> operations;

        {
            std::scoped_lock const lock(mutex_);
            operations.reserve(active_transfers_.size());
            for (auto& [id, op] : active_transfers_)
            {
                operations.push_back(std::move(op));
            }
            active_transfers_.clear();
        }

        // Wait for all operations to complete
        for (const auto& op : operations)
        {
            op->promise.get_future().wait();
        }
    }

    void cancel_all_transfers() override
    {
        std::scoped_lock const lock(mutex_);

        for (auto& [id, op] : active_transfers_)
        {
            op->info.status        = transfer_status::CANCELLED;
            op->info.error_message = "Transfer cancelled";

            op->promise.set_value(op->info);
        }

        active_transfers_.clear();
    }
};

}  // anonymous namespace

std::unique_ptr<gpu_stream> gpu_stream::create(
    device_enum device_type, int device_index, int priority)
{
#if !MEMORY_HAS_CUDA
    (void)device_index;
    (void)priority;
#endif
    switch (device_type)
    {
#if MEMORY_HAS_CUDA
    case device_enum::CUDA:
        return std::make_unique<cuda_stream_impl>(device_type, device_index, priority);
#endif
    default:
        MEMORY_THROW("Unsupported device type for stream creation");
    }
}

gpu_memory_transfer& gpu_memory_transfer::instance()
{
    static gpu_memory_transfer_impl instance;
    return static_cast<gpu_memory_transfer&>(instance);
}

}  // namespace gpu
}  // namespace memory
