/*
 * Quarisma: High-Performance Computational Library
 *
 * Original work Copyright 2015 The TensorFlow Authors
 * Modified work Copyright 2025 Quarisma Contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * This file contains code modified from TensorFlow (Apache 2.0 licensed)
 * and is part of Quarisma, licensed under a dual-license model:
 *
 *   - Open-source License (GPLv3):
 *       Free for personal, academic, and research use under the terms of
 *       the GNU General Public License v3.0 or later.
 *
 *   - Commercial License:
 *       A commercial license is required for proprietary, closed-source,
 *       or SaaS usage. Contact us to obtain a commercial agreement.
 *
 * MODIFICATIONS FROM ORIGINAL:
 * - Adapted for Quarisma quantitative computing requirements
 * - Added high-performance memory allocation optimizations
 * - Integrated NUMA-aware allocation strategies
 *
 * Contact: licensing@quarisma.co.uk
 * Website: https://www.quarisma.co.uk
 */

#include "backend/allocator_bfc.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iomanip>
#include <ios>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "common/memory_macros.h"
#include "logger.h"
#include "cpu/allocator.h"
#if MEMORY_HAS_NATIVE_PROFILER && 0
#include "native/memory/scoped_memory_debug_annotation.h"
#include "native/tracing/traceme.h"
#include "native/tracing/traceme_encode.h"
#endif
#include "util/exception.h"
#include "util/flat_hash.h"
#include "util/string_util.h"

#ifdef MEMORY_MEM_DEBUG
#define LOGGING_LOG_INFO_DEBUG_BFC(...) LOGGING_LOG_INFO_DEBUG("[mem-verbose] " __VA_ARGS__)
#else
#define LOGGING_LOG_INFO_DEBUG_BFC(...)
#endif

namespace memory
{

// Helper function to format bytes in human-readable format (IEC 60027-2 binary prefixes)
// Inline implementation to avoid adding to string_util.h
static std::string format_human_readable_bytes(int64_t bytes)
{
    if (bytes < 0)
    {
        return "-" + format_human_readable_bytes(-bytes);
    }
    if (bytes < 1024)
    {
        return std::to_string(bytes) + "B";
    }
    const char*  units[]    = {"KiB", "MiB", "GiB", "TiB", "PiB"};  //NOLINT
    const double divisors[] = {
        1024.0, 1048576.0, 1073741824.0, 1099511627776.0, 1125899906842624.0};
    for (int i = 4; i >= 0; --i)
    {
        if (bytes >= divisors[i])
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << (bytes / divisors[i]) << units[i];
            return oss.str();
        }
    }
    return std::to_string(bytes) + "B";
}

// NOLINTNEXTLINE(misc-use-internal-linkage)
class MemAllocatorStats
{
public:
    // Getters
    MEMORY_NODISCARD int64_t num_allocs() const { return num_allocs_; }
    MEMORY_NODISCARD int64_t bytes_in_use() const { return bytes_in_use_; }
    MEMORY_NODISCARD int64_t peak_bytes_in_use() const { return peak_bytes_in_use_; }
    MEMORY_NODISCARD int64_t largest_alloc_size() const { return largest_alloc_size_; }
    MEMORY_NODISCARD float   fragmentation_metric() const { return fragmentation_metric_; }

    // Setters
    void set_num_allocs(int64_t value) { num_allocs_ = value; }
    void set_bytes_in_use(int64_t value) { bytes_in_use_ = value; }
    void set_peak_bytes_in_use(int64_t value) { peak_bytes_in_use_ = value; }
    void set_largest_alloc_size(int64_t value) { largest_alloc_size_ = value; }
    void set_fragmentation_metric(float value) { fragmentation_metric_ = value; }

private:
    int64_t num_allocs_{0};
    int64_t bytes_in_use_{0};
    int64_t peak_bytes_in_use_{0};
    int64_t largest_alloc_size_{0};
    float   fragmentation_metric_{0.0F};
};

// NOLINTNEXTLINE(misc-use-internal-linkage)
class MemChunk
{
public:
    // Getters
    MEMORY_NODISCARD uint64_t address() const { return address_; }
    MEMORY_NODISCARD int64_t  size() const { return size_; }
    MEMORY_NODISCARD int64_t  requested_size() const { return requested_size_; }
    MEMORY_NODISCARD int32_t  bin() const { return bin_; }
    MEMORY_NODISCARD const std::string& op_name() const { return op_name_; }
    MEMORY_NODISCARD uint64_t           freed_at_count() const { return freed_at_count_; }
    MEMORY_NODISCARD uint64_t           action_count() const { return action_count_; }
    MEMORY_NODISCARD bool               in_use() const { return in_use_; }
    MEMORY_NODISCARD uint64_t           step_id() const { return step_id_; }

    // Setters
    void set_address(uint64_t value) { address_ = value; }
    void set_size(int64_t value) { size_ = value; }
    void set_requested_size(int64_t value) { requested_size_ = value; }
    void set_bin(int32_t value) { bin_ = value; }
    void set_op_name(const std::string& value) { op_name_ = value; }
    void set_op_name(std::string&& value) { op_name_ = std::move(value); }
    void set_freed_at_count(uint64_t value) { freed_at_count_ = value; }
    void set_action_count(uint64_t value) { action_count_ = value; }
    void set_in_use(bool value) { in_use_ = value; }
    void set_step_id(uint64_t value) { step_id_ = value; }

private:
    uint64_t    address_{0};
    int64_t     size_{0};
    int64_t     requested_size_{0};
    int32_t     bin_{0};
    std::string op_name_;
    uint64_t    freed_at_count_{0};
    uint64_t    action_count_{0};
    bool        in_use_{false};
    uint64_t    step_id_{0};
};

// NOLINTNEXTLINE(misc-use-internal-linkage)
class BinSummary
{
public:
    // Getters
    MEMORY_NODISCARD int32_t bin() const { return bin_; }
    MEMORY_NODISCARD int64_t total_bytes_in_use() const { return total_bytes_in_use_; }
    MEMORY_NODISCARD int64_t total_bytes_in_bin() const { return total_bytes_in_bin_; }
    MEMORY_NODISCARD int64_t total_chunks_in_use() const { return total_chunks_in_use_; }
    MEMORY_NODISCARD int64_t total_chunks_in_bin() const { return total_chunks_in_bin_; }

    // Setters
    void set_bin(int32_t value) { bin_ = value; }
    void set_total_bytes_in_use(int64_t value) { total_bytes_in_use_ = value; }
    void set_total_bytes_in_bin(int64_t value) { total_bytes_in_bin_ = value; }
    void set_total_chunks_in_use(int64_t value) { total_chunks_in_use_ = value; }
    void set_total_chunks_in_bin(int64_t value) { total_chunks_in_bin_ = value; }

private:
    int32_t bin_{0};
    int64_t total_bytes_in_use_{0};
    int64_t total_bytes_in_bin_{0};
    int64_t total_chunks_in_use_{0};
    int64_t total_chunks_in_bin_{0};
};

// NOLINTNEXTLINE(misc-use-internal-linkage)
class SnapShot
{
public:
    // Getters
    MEMORY_NODISCARD uint64_t action_count() const { return action_count_; }
    MEMORY_NODISCARD int64_t  size() const { return size_; }

    // Setters
    void set_action_count(uint64_t value) { action_count_ = value; }
    void set_size(int64_t value) { size_ = value; }

private:
    uint64_t action_count_{0};
    int64_t  size_{0};
};

class memory_dump
{
public:
    // Getters
    MEMORY_NODISCARD const std::string& allocator_name() const { return allocator_name_; }
    MEMORY_NODISCARD const std::vector<BinSummary>& bin_summary() const { return bin_summary_; }
    MEMORY_NODISCARD const std::vector<MemChunk>& chunk() const { return chunk_; }
    MEMORY_NODISCARD const std::vector<SnapShot>& snap_shot() const { return snap_shot_; }
    MEMORY_NODISCARD const MemAllocatorStats&     stats() const { return stats_; }

    // Mutable accessors
    MemAllocatorStats* stats() { return &stats_; }

    // Setters
    void set_allocator_name(const std::string& name) { allocator_name_ = name; }
    void set_allocator_name(std::string&& name) { allocator_name_ = std::move(name); }

    // Add methods
    BinSummary* add_bin_summary()
    {
        bin_summary_.emplace_back();
        return &bin_summary_.back();
    }

    MemChunk* add_chunk()
    {
        chunk_.emplace_back();
        return &chunk_.back();
    }

    SnapShot* add_snap_shot()
    {
        snap_shot_.emplace_back();
        return &snap_shot_.back();
    }

    void Clear()
    {
        allocator_name_.clear();
        bin_summary_.clear();
        chunk_.clear();
        snap_shot_.clear();
        stats_ = MemAllocatorStats{};
    }

private:
    std::string             allocator_name_;
    std::vector<BinSummary> bin_summary_;
    std::vector<MemChunk>   chunk_;
    std::vector<SnapShot>   snap_shot_;
    MemAllocatorStats       stats_;
};

//constexpr allocator_bfc::ChunkHandle allocator_bfc::kInvalidChunkHandle;

allocator_bfc::allocator_bfc(
    std::unique_ptr<memory::sub_allocator> sub_allocator,
    size_t                                   total_memory,
    std::string                              name,
    const Options&                           opts)
    : opts_(opts),
      coalesce_regions_(sub_allocator->SupportsCoalescing()),
      sub_allocator_(std::move(sub_allocator)),
      name_(std::move(name))

{
    if (opts.allow_growth)
    {
        // 2MiB smallest initial allocation, unless total memory available
        // is less.
        curr_region_allocation_bytes_ = RoundedBytes(std::min(total_memory, size_t{2 << 20}));
    }
    else
    {
        curr_region_allocation_bytes_ = RoundedBytes(total_memory);
    }

    // Initially, we have not allocated any memory from the sub-allocator; our
    // pool of memory is empty.
    stats_.pool_bytes.store(0, std::memory_order_relaxed);
    stats_.peak_pool_bytes.store(0, std::memory_order_relaxed);

    // Allocate the requested amount of memory.
    memory_limit_ = total_memory;
    stats_.bytes_limit.store(static_cast<int64_t>(total_memory), std::memory_order_relaxed);

    // Create a bunch of bins of various good sizes.

    // We create bins to fit all possible ranges that cover the
    // memory_limit_ starting from allocations up to 256 bytes to
    // allocations up to (and including) the memory limit.
    LOGGING_LOG_INFO_DEBUG_BFC("Creating new allocator_bfc named: {}", name);
    for (BinNum b = 0; b < kNumBins; b++)
    {
        size_t const bin_size = BinNumToSize(b);
        LOGGING_LOG_INFO_DEBUG_BFC(
            "Creating bin of max chunk size {}", format_human_readable_bytes(bin_size));
        new (BinFromIndex(b)) Bin(this, bin_size);
        LOGGING_CHECK(BinForSize(bin_size) == BinFromIndex(b));
        LOGGING_CHECK(BinForSize(bin_size + 255) == BinFromIndex(b));
        LOGGING_CHECK(BinForSize(bin_size * 2 - 1) == BinFromIndex(b));  //NOLINT
        if (b + 1 < kNumBins)
        {
            LOGGING_CHECK(BinForSize(bin_size * 2) != BinFromIndex(b));
        }
    }
}

allocator_bfc::~allocator_bfc()
{
    // Lock the mutex to make sure that all memory effects are safely published
    // and available to a thread running the destructor (i.e., deallocations
    // happened on a different thread right before the destructor).
    std::scoped_lock const l(mutex_);

    // Return memory back.
    LOGGING_LOG_INFO_DEBUG_BFC(
        "Number of regions allocated: {}", region_manager_.regions().size());
    for (const auto& region : region_manager_.regions())
    {
        sub_allocator_->Free(region.ptr(), region.memory_size());
    }

    for (BinNum b = 0; b < kNumBins; b++)
    {
        BinFromIndex(b)->~Bin();
    }
}

allocator_bfc::Chunk* allocator_bfc::ChunkFromHandle(ChunkHandle h)
{
    LOGGING_CHECK_DEBUG(h >= 0);
    LOGGING_CHECK_DEBUG(h < static_cast<int>(chunks_.size()));
    return &(chunks_[h]);
}

const allocator_bfc::Chunk* allocator_bfc::ChunkFromHandle(ChunkHandle h) const
{
    LOGGING_CHECK_DEBUG(h >= 0);
    LOGGING_CHECK_DEBUG(h < static_cast<int>(chunks_.size()));
    return &(chunks_[h]);
}

bool allocator_bfc::Extend(size_t alignment, size_t rounded_bytes)
{
    size_t available_bytes = memory_limit_ - stats_.pool_bytes.load(std::memory_order_relaxed);
    // Rounds available_bytes down to the nearest multiple of kMinAllocationSize.
    available_bytes = (available_bytes / kMinAllocationSize) * kMinAllocationSize;

    // Do we have enough space to handle the client's request?
    // If not, fail immediately.
    if (rounded_bytes > available_bytes)
    {
        return false;
    }

    // If curr_region_allocation_bytes_ is not enough to satisfy the
    // allocation, keep multiplying by a power of two until that is
    // sufficient.
    bool increased_allocation = false;
    while (rounded_bytes > curr_region_allocation_bytes_)
    {
        curr_region_allocation_bytes_ *= 2;
        increased_allocation = true;
    }

    // Try allocating.
    size_t bytes = std::min(curr_region_allocation_bytes_, available_bytes);
    size_t bytes_received;
    void*  mem_addr = sub_allocator_->Alloc(alignment, bytes, &bytes_received);
    if (mem_addr == nullptr)
    {
        static constexpr float kBackpedalFactor = 0.9F;

        // Try allocating less memory.
        while (mem_addr == nullptr)
        {
            bytes = RoundedBytes(bytes * kBackpedalFactor);
            if (bytes < rounded_bytes)
            {
                return false;
            }
            mem_addr = sub_allocator_->Alloc(alignment, bytes, &bytes_received);
        }
    }

    if (!increased_allocation)
    {
        // Increase the region size of the next required allocation.
        curr_region_allocation_bytes_ *= 2;
    }

    LOGGING_LOG_INFO_DEBUG_BFC(
        "Extending allocation by {} bytes for {}",
        format_human_readable_bytes(bytes_received),
        Name());

    stats_.pool_bytes.fetch_add(bytes_received, std::memory_order_relaxed);

    // Update peak pool bytes atomically
    int64_t const current_pool = stats_.pool_bytes.load(std::memory_order_relaxed);
    int64_t       peak_pool    = stats_.peak_pool_bytes.load(std::memory_order_relaxed);
    while (current_pool > peak_pool && !stats_.peak_pool_bytes.compare_exchange_weak(
                                           peak_pool, current_pool, std::memory_order_relaxed))
    {
        // Retry if another thread updated peak_pool
    }

    LOGGING_LOG_INFO_DEBUG_BFC(
        "Total allocated bytes: {}",
        format_human_readable_bytes(stats_.pool_bytes.load(std::memory_order_relaxed)));

    LOGGING_LOG_INFO_DEBUG_BFC(
        "Allocated memory at {} to {}",
        mem_addr,
        static_cast<void*>(static_cast<char*>(mem_addr) + bytes_received));

    AllocationRegion const* maybe_extended_region = nullptr;
    if (coalesce_regions_)
    {
        maybe_extended_region =
            region_manager_.AddOrExtendAllocationRegion(mem_addr, bytes_received);
    }
    else
    {
        region_manager_.AddAllocationRegion(mem_addr, bytes_received);
    }

    // Create one large chunk for the whole memory space that will
    // be chunked later.
    ChunkHandle const     h = AllocateChunk();
    allocator_bfc::Chunk* c = ChunkFromHandle(h);
    c->ptr                  = mem_addr;
    c->size                 = bytes_received;
    c->allocation_id        = -1;
    c->prev                 = kInvalidChunkHandle;
    c->next                 = kInvalidChunkHandle;
    c->freed_at_count       = 0;

    region_manager_.set_handle(c->ptr, h);

    // If the region was extended, then there exists a previous chunk that should
    // be linked to the new chunk.
    if (maybe_extended_region != nullptr)
    {
        ChunkHandle prev = maybe_extended_region->get_handle(maybe_extended_region->ptr());
        allocator_bfc::Chunk* prev_chunk = ChunkFromHandle(prev);
        // Find the last recorded chunk in the extended region.
        while (prev_chunk->next != kInvalidChunkHandle)
        {
            prev       = prev_chunk->next;
            prev_chunk = ChunkFromHandle(prev);
        }
        c->prev          = prev;
        prev_chunk->next = h;
    }

    // Maybe merge adjacent chunks and insert the chunk into the right bin.
    InsertFreeChunkIntoBin(TryToCoalesce(h, /*ignore_freed_at=*/false));

    return true;
}

allocator_bfc::ChunkHandle allocator_bfc::AllocateChunk()
{
    if (free_chunks_list_ != kInvalidChunkHandle)
    {
        ChunkHandle const h = free_chunks_list_;
        const Chunk*      c = ChunkFromHandle(h);
        free_chunks_list_   = c->next;
        return h;
    }

    ChunkHandle const h = chunks_.size();
    chunks_.resize(h + 1);
    return h;
}

void allocator_bfc::DeallocateChunk(ChunkHandle h)
{
    Chunk* c          = ChunkFromHandle(h);
    c->allocation_id  = -1;
    c->bin_num        = kInvalidBinNum;
    c->next           = free_chunks_list_;
    free_chunks_list_ = h;
}

void* allocator_bfc::AllocateRawInternalWithRetry(
    size_t unused_alignment, size_t num_bytes, const allocation_attributes& allocation_attr)
{
    // Fast path: Try once to allocate without getting the retry_helper_ involved
    uint64_t freed_by_count = 0;
    if (allocation_attr.freed_by_func != nullptr)
    {
        freed_by_count = (*allocation_attr.freed_by_func)();
    }
    void* r = AllocateRawInternal(unused_alignment, num_bytes, false, freed_by_count);  //NOLINT

    if (r != nullptr)
    {
        return r;
    }
    static const int64_t kMaxMillisToWait = 10000;  // 10 seconds

    r = retry_helper_.allocate_raw(
        [this, &allocation_attr](size_t a, size_t nb, bool v)
        {
            uint64_t freed_by_count = 0;
            if (allocation_attr.freed_by_func != nullptr)
            {
                freed_by_count = (*allocation_attr.freed_by_func)();
            }
            return AllocateRawInternal(a, nb, v, freed_by_count);
        },
        kMaxMillisToWait,
        unused_alignment,
        num_bytes);

    return r;
}

void* allocator_bfc::allocate_raw(
    size_t unused_alignment, size_t num_bytes, const allocation_attributes& allocation_attr)
{
    //LOGGING_LOG_INFO_DEBUG_BFC("allocate_raw {}  {}", Name(), num_bytes);
    void* result = [&]  //NOLINT
    {
        if (!opts_.allow_retry_on_failure || !allocation_attr.retry_on_failure)
        {
            // If we have globally disabled retry-on-failure and fail to allocate an
            // "important" alloc, we want to print a log, because the program may be
            // about to fail due to OOM.
            //
            // Bit of a hack: We deem "important" allocs as those which are retryable.
            // In TF, *non*-retryable allocations are usually those which we can
            // tolerate failing.  For example, we allocate convolution scratch memory
            // as non-retryable; if it fails, we'll just use a fallback algorithm that
            // uses no scratch.
            static std::atomic<int32_t> log_counter{0};
            constexpr int               kMaxFailureLogs = 10;
            bool const                  dump_log_on_failure =
                (/*retry is globally disabled*/ !opts_.allow_retry_on_failure &&
                 /*alloc is "important"*/ allocation_attr.retry_on_failure &&
                 log_counter.load(std::memory_order_relaxed) < kMaxFailureLogs);

            uint64_t freed_by_count = 0;
            if (allocation_attr.freed_by_func != nullptr)
            {
                freed_by_count = (*allocation_attr.freed_by_func)();
            }
            void* res = AllocateRawInternal(  //NOLINT
                unused_alignment,
                num_bytes,
                dump_log_on_failure,
                freed_by_count);
            if (res == nullptr)
            {
                int32_t const counter_value = log_counter.load(std::memory_order_relaxed);
                if (counter_value < kMaxFailureLogs)
                {
                    log_counter.store(counter_value + 1, std::memory_order_relaxed);
                    LOGGING_LOG_WARNING(
                        "Allocator ({}) ran out of memory trying to allocate {} with "
                        "freed_by_count={}.{}",
                        Name(),
                        format_human_readable_bytes(num_bytes),
                        freed_by_count,
                        (!allocation_attr.retry_on_failure
                             ? " The caller indicates that this is not a failure, but"
                               " this may mean that there could be performance gains "
                               "if more memory were available."
                             : ""));
                }
            }
            return res;
        }

        return AllocateRawInternalWithRetry(unused_alignment, num_bytes, allocation_attr);
    }();
    //LOGGING_LOG_INFO_DEBUG_BFC("allocate_raw {}  {} {}", Name(), num_bytes, result);
    //LOGGING_LOG_INFO_DEBUG_BFC(
    //   "[mem-verbose] allocate_raw,{},{},{},{},{}", Name(), num_bytes, result, "", "");
    return result;
}

// static
size_t allocator_bfc::RoundedBytes(size_t bytes) noexcept
{
    size_t const rounded_bytes =
        (kMinAllocationSize * ((bytes + kMinAllocationSize - 1) / kMinAllocationSize));
    return rounded_bytes;
}

bool allocator_bfc::DeallocateFreeRegions(size_t rounded_bytes)
    MEMORY_EXCLUSIVE_LOCKS_REQUIRED(mutex_)
{
    // Do nothing if garbage collection is off.
    if (!opts_.garbage_collection)
    {
        return false;
    }

    // Searching for free regions.
    logging::flat_hash_set<void*> free_region_ptrs;
    size_t               total_free_bytes = 0;
    for (const AllocationRegion& region : region_manager_.regions())
    {
        ChunkHandle h       = region_manager_.get_handle(region.ptr());
        bool        any_use = false;
        while (h != kInvalidChunkHandle)
        {
            const Chunk* c = ChunkFromHandle(h);
            if (c->in_use())
            {
                any_use = true;
                break;
            }
            h = c->next;
        }

        if (!any_use)
        {
            LOGGING_LOG_INFO("Found free region with ptr = {}", region.ptr());
            free_region_ptrs.insert(region.ptr());
            total_free_bytes += region.memory_size();
        }
    }

    if (total_free_bytes == 0)
    {
        return false;
    }

    // Rough estimation to check whether deallocation can help.
    size_t const available_bytes =
        memory_limit_ - stats_.pool_bytes.load(std::memory_order_relaxed) + total_free_bytes;
    if (rounded_bytes > available_bytes)
    {
        return false;
    }

    LOGGING_LOG_WARNING(
        "Garbage collection: deallocate free memory regions"
        " (i.e., allocations) so that we can re-allocate a larger"
        " region to avoid OOM due to memory fragmentation. If you"
        " see this message frequently, you are running near the"
        " threshold of the available device memory and re-allocation"
        " may incur great performance overhead. You may try smaller"
        " batch sizes to observe the performance impact."
        " Set ENABLE_GPU_GARBAGE_COLLECTION=false if you'd like to"
        " disable this feature.");

    // Deallocate free regions.
    DeallocateRegions(free_region_ptrs);

    return true;
}

void allocator_bfc::DeallocateRegions(const logging::flat_hash_set<void*>& region_ptrs)
    MEMORY_EXCLUSIVE_LOCKS_REQUIRED(mutex_)
{
    // Explicitly remove the const qualifier as some compilers disallow passing
    // const_iterator to std::vector::erase(), which is used in
    // RemoveAllocationRegion().
    auto* regions = const_cast<std::vector<AllocationRegion>*>(&region_manager_.regions());
    auto  it      = regions->begin();
    while (it != regions->end())
    {
        if (region_ptrs.find(it->ptr()) == region_ptrs.end())
        {
            ++it;
            continue;
        }

        LOGGING_LOG_WARNING("Deallocate region with ptr = {}", it->ptr());
        // Remove all chunk registrations from Bins.
        ChunkHandle h = region_manager_.get_handle(it->ptr());
        while (h != kInvalidChunkHandle)
        {
            const Chunk* c = ChunkFromHandle(h);
            if (c->bin_num != kInvalidBinNum)
            {
                RemoveFreeChunkFromBin(h);
            }
            auto h_to_delete = h;
            h                = c->next;
            DeleteChunk(h_to_delete);
        }

        // Deallocate the memory.
        sub_allocator_->Free(it->ptr(), it->memory_size());
        stats_.pool_bytes.fetch_sub(it->memory_size(), std::memory_order_relaxed);
        it = region_manager_.RemoveAllocationRegion(it);
    }
}

void* allocator_bfc::AllocateRawInternal(
    size_t unused_alignment, size_t num_bytes, bool dump_log_on_failure, uint64_t freed_before)
{
    if (num_bytes == 0)
    {
        LOGGING_LOG_WARNING("tried to allocate 0 bytes");
        return nullptr;
    }
    // First, always allocate memory of at least kMinAllocationSize
    // bytes, and always allocate multiples of kMinAllocationSize bytes
    // so all memory addresses are nicely byte aligned.
    size_t rounded_bytes = RoundedBytes(num_bytes);

    // The BFC allocator tries to find the best fit first.
    BinNum const bin_num = BinNumForSize(rounded_bytes);

    std::scoped_lock const lock(mutex_);
    if (!timestamped_chunks_.empty())
    {
        // Merge timestamped chunks whose counts have become safe for general use.
        MergeTimestampedChunks(0);
    }
    void* ptr = FindChunkPtr(bin_num, rounded_bytes, num_bytes, freed_before);  //NOLINT
    if (ptr != nullptr)
    {
#if MEMORY_HAS_NATIVE_PROFILER && 0
        AddTraceMe("MemoryAllocation", ptr);
#endif
        return ptr;
    }

    // Try to extend
    if (Extend(unused_alignment, rounded_bytes))
    {
        ptr = FindChunkPtr(bin_num, rounded_bytes, num_bytes, freed_before);
        if (ptr != nullptr)
        {
#if MEMORY_HAS_NATIVE_PROFILER && 0
            AddTraceMe("MemoryAllocation", ptr);
#endif
            return ptr;
        }
    }

    if ((freed_before == 0) && (!timestamped_chunks_.empty()))
    {
        // We're unable to satisfy an allocation request without a specific
        // timestamp requirement.  Rather than fail, try merging any held-out
        // timestamped chunks more aggressively until a free chunk of the necessary
        // size is formed.
        if (MergeTimestampedChunks(rounded_bytes))
        {
            ptr = FindChunkPtr(bin_num, rounded_bytes, num_bytes, freed_before);
            if (ptr != nullptr)
            {
#if MEMORY_HAS_NATIVE_PROFILER && 0
                AddTraceMe("MemoryAllocation", ptr);
#endif
                return ptr;
            }
        }
    }

    // Reaching this point means that no chunks can satisfy the request. Also,
    // the unallocated bytes cannot satisfy the request. Before giving up, let's
    // try deallocating free regions so that suballocator can combine them with
    // the unallocated bytes and form a larger region.
    if (DeallocateFreeRegions(rounded_bytes) && Extend(unused_alignment, rounded_bytes))
    {
        ptr = FindChunkPtr(bin_num, rounded_bytes, num_bytes, freed_before);
        if (ptr != nullptr)
        {
#if MEMORY_HAS_NATIVE_PROFILER && 0
            AddTraceMe("MemoryAllocation", ptr);
#endif
            return ptr;
        }
    }

    // We searched all bins for an existing free chunk to use and
    // couldn't find one.  This means we must have run out of memory,
    // Dump the memory log for analysis.
    MaybeWriteMemoryMap();
    if (dump_log_on_failure)
    {
        LOGGING_LOG_WARNING(
            "Allocator ({}) ran out of memory trying to allocate {} (rounded to {})requested by op "
            "\nIf the cause is memory fragmentation maybe the environment "
            "variable 'MEMORY_GPU_ALLOCATOR=cuda_malloc_async' will "
            "improve the situation. \nCurrent allocation summary follows."
            "\nCurrent allocation summary follows.",
            Name(),
            format_human_readable_bytes(num_bytes),
            rounded_bytes);

        DumpMemoryLog(rounded_bytes);
        LOGGING_LOG_WARNING("RenderOccupancy: {}", RenderOccupancy());
    }
    return nullptr;
}

int64_t allocator_bfc::LargestFreeChunk()
{
    for (int i = kNumBins - 1; i >= 0; i--)
    {
        if (!BinFromIndex(i)->free_chunks.empty())
        {
            return ChunkFromHandle(*BinFromIndex(i)->free_chunks.rbegin())->size;
        }
    }
    return 0;
}

double allocator_bfc::GetFragmentation()
{
    int64_t const bytes_available = stats_.pool_bytes.load(std::memory_order_relaxed) -
                                    stats_.bytes_in_use.load(std::memory_order_relaxed);
    LOGGING_CHECK_DEBUG(bytes_available >= 0);
    return static_cast<double>(bytes_available - LargestFreeChunk()) / bytes_available;
}

#if MEMORY_HAS_NATIVE_PROFILER && 0
void allocator_bfc::AddTraceMe(std::string_view traceme_name, const void* ptr)
{
    allocator_bfc::Chunk const* chunk = ChunkFromHandle(region_manager_.get_handle(ptr));
    AddTraceMe(traceme_name, chunk->ptr, chunk->requested_size, chunk->size);
}

void allocator_bfc::AddTraceMe(
    MEMORY_UNUSED std::string_view traceme_name,
    MEMORY_UNUSED const void*      chunk_ptr,
    MEMORY_UNUSED int64_t          req_bytes,
    MEMORY_UNUSED int64_t          alloc_bytes)
{
    memory::traceme::instant_activity(
        [this, traceme_name, chunk_ptr, req_bytes, alloc_bytes]() MEMORY_NO_THREAD_SAFETY_ANALYSIS
        {
            int64_t bytes_available = memory_limit_ -
                                      stats_.bytes_reserved.load(std::memory_order_relaxed) -
                                      stats_.bytes_in_use.load(std::memory_order_relaxed);
            const auto& annotation = memory::scoped_memory_debug_annotation::current_annotation();
            const auto* const op_name =
                (annotation.pending_op_name != nullptr) ? annotation.pending_op_name : "(null)";
            const auto* const region_type = (annotation.pending_region_type != nullptr)
                                                ? annotation.pending_region_type
                                                : "(null)";
            return memory::traceme_encode(
                std::string(traceme_name),
                {{"allocator_name", name_},
                 {"bytes_reserved", stats_.bytes_reserved.load(std::memory_order_relaxed)},
                 {"bytes_allocated", stats_.bytes_in_use.load(std::memory_order_relaxed)},
                 {"bytes_available", bytes_available},
                 {"fragmentation", GetFragmentation()},
                 {"peak_bytes_in_use", stats_.peak_bytes_in_use.load(std::memory_order_relaxed)},
                 {"requested_bytes", req_bytes},
                 {"allocation_bytes", alloc_bytes},
                 {"addr", reinterpret_cast<uint64_t>(chunk_ptr)},
                 {"tf_op", op_name},
                 {"id", annotation.pending_step_id},
                 {"region_type", region_type},
                 {"data_type", annotation.pending_data_type},
                 {"shape", annotation.pending_shape_func()}});
        },
        /*level=*/static_cast<int>(memory::traceme_level_enum::INFO));
}
#endif

void* allocator_bfc::FindChunkPtr(
    BinNum bin_num, size_t rounded_bytes, size_t num_bytes, uint64_t freed_before)
{
    // First identify the first bin that could satisfy rounded_bytes.
    for (; bin_num < kNumBins; bin_num++)
    {
        // Start searching from the first bin for the smallest chunk that fits
        // rounded_bytes.
        Bin* b = BinFromIndex(bin_num);
        for (auto citer = b->free_chunks.begin(); citer != b->free_chunks.end(); ++citer)
        {
            const allocator_bfc::ChunkHandle h     = (*citer);
            allocator_bfc::Chunk*            chunk = ChunkFromHandle(h);
            LOGGING_CHECK_DEBUG(!chunk->in_use());
            if (freed_before > 0 && freed_before < chunk->freed_at_count)
            {
                continue;
            }
            if (chunk->size >= rounded_bytes)
            {
                // We found an existing chunk that fits us that wasn't in use, so remove
                // it from the free bin structure prior to using.
                RemoveFreeChunkIterFromBin(&b->free_chunks, citer);

                // If we can break the size of the chunk into two reasonably large
                // pieces, do don't waste more than max_internal_fragmentation_bytes on
                // padding. If this threshold is not set by the user, then use 128MB as
                // the default.
                const int64_t max_internal_fragmentation_bytes =
                    (opts_.fragmentation_fraction > 0.0)
                        ? opts_.fragmentation_fraction * memory_limit_
                        : 128 << 20;

                if (chunk->size >= rounded_bytes * 2 ||
                    static_cast<int64_t>(chunk->size) >=
                        max_internal_fragmentation_bytes + static_cast<int64_t>(rounded_bytes))
                {
                    SplitChunk(h, rounded_bytes);
                    chunk = ChunkFromHandle(h);  // Update chunk pointer in case it moved
                }

                // The requested size of the returned chunk is what the user
                // has allocated.
                chunk->requested_size = num_bytes;
                // Assign a unique id and increment the id counter, marking the
                // chunk as being in use.
                chunk->allocation_id = next_allocation_id_++;

                // Update stats.
                stats_.num_allocs.fetch_add(1, std::memory_order_relaxed);
                stats_.bytes_in_use.fetch_add(chunk->size, std::memory_order_relaxed);

                int64_t const current_bytes = stats_.bytes_in_use.load(std::memory_order_relaxed);
                int64_t       peak_bytes = stats_.peak_bytes_in_use.load(std::memory_order_relaxed);
                // if (current_bytes > peak_bytes)
                // {
                //     LOGGING_LOG_INFO_DEBUG_BFC(
                //         "New Peak memory usage of {} bytes for {}", current_bytes, Name());
                // }

                // Update peak bytes atomically
                while (current_bytes > peak_bytes &&
                       !stats_.peak_bytes_in_use.compare_exchange_weak(
                           peak_bytes, current_bytes, std::memory_order_relaxed))
                {
                    // Retry if another thread updated peak_bytes
                }

                // Update largest allocation size atomically
                auto const chunk_size_int64 = static_cast<int64_t>(chunk->size);
                int64_t    largest_size = stats_.largest_alloc_size.load(std::memory_order_relaxed);
                while (chunk_size_int64 > largest_size &&
                       !stats_.largest_alloc_size.compare_exchange_weak(
                           largest_size, chunk_size_int64, std::memory_order_relaxed))
                {
                    // Retry if another thread updated largest_size
                }

#ifdef MEMORY_MEM_DEBUG
                if (ShouldRecordOpName())
                {
                    const auto& annotation = ScopedMemoryDebugAnnotation::CurrentAnnotation();
                    if (annotation.pending_op_name != nullptr)
                    {
                        chunk->op_name = annotation.pending_op_name;
                    }
                    else
                    {
                        LOGGING_LOG_INFO(
                            "missing pending_op_name for {} reading addr {}\n{}",
                            Name(),
                            static_cast<const void*>(&annotation.pending_op_name),
                            CurrentStackTrace());
                        chunk->op_name = nullptr;
                    }
                    chunk->action_count = ++action_counter_;
                    chunk->step_id      = annotation.pending_step_id;
                    int slot            = chunk->action_count % MEM_DEBUG_SIZE_HISTORY_SIZE;
                    size_history_[slot] = stats_.bytes_in_use;
                }
#endif

                //LOGGING_LOG_INFO_DEBUG_BFC("Returning: {}\nA: {}", chunk->ptr, RenderOccupancy());
                return chunk->ptr;
            }
        }
    }

    return nullptr;
}

void allocator_bfc::SplitChunk(allocator_bfc::ChunkHandle h, size_t num_bytes)
{
    // Allocate the new chunk before we do any ChunkFromHandle
    ChunkHandle const h_new_chunk = AllocateChunk();

    Chunk* c = ChunkFromHandle(h);
    LOGGING_CHECK(!c->in_use() && (c->bin_num == kInvalidBinNum));  //NOLINT

    // Create a new chunk starting num_bytes after c
    allocator_bfc::Chunk* new_chunk = ChunkFromHandle(h_new_chunk);
    new_chunk->ptr                  = static_cast<void*>(static_cast<char*>(c->ptr) + num_bytes);
    region_manager_.set_handle(new_chunk->ptr, h_new_chunk);

    // Set the new sizes of the chunks.
    new_chunk->size = c->size - num_bytes;
    c->size         = num_bytes;

    // The new chunk is not in use.
    new_chunk->allocation_id = -1;

    // It inherits the freed time.
    new_chunk->freed_at_count = c->freed_at_count;

    // Maintain the pointers.
    // c <-> c_neighbor becomes
    // c <-> new_chunk <-> c_neighbor
    allocator_bfc::ChunkHandle const h_neighbor = c->next;
    new_chunk->prev                             = h;
    new_chunk->next                             = h_neighbor;
    c->next                                     = h_new_chunk;
    if (h_neighbor != kInvalidChunkHandle)
    {
        Chunk* c_neighbor = ChunkFromHandle(h_neighbor);
        c_neighbor->prev  = h_new_chunk;
    }

    // Add the newly free chunk to the free bin.
    InsertFreeChunkIntoBin(h_new_chunk);
}

void allocator_bfc::deallocate_raw(void* ptr)
{
    LOGGING_LOG_INFO_DEBUG_BFC(
        "deallocate_raw {} {} [mem-verbose] deallocate_raw, {},{},{}",
        Name(),
        (ptr ? RequestedSize(ptr) : 0),
        Name(),
        (ptr ? RequestedSize(ptr) : 0),
        ptr);

    DeallocateRawInternal(ptr);
    retry_helper_.NotifyDealloc();
}

void allocator_bfc::DeallocateRawInternal(void* ptr)
{
    if (ptr == nullptr)
    {
        LOGGING_LOG(INFO, "tried to deallocate nullptr");
        return;
    }
    std::scoped_lock const lock(mutex_);

    // Find the chunk from the ptr.
    allocator_bfc::ChunkHandle const h = region_manager_.get_handle(ptr);
    LOGGING_CHECK(h != kInvalidChunkHandle);

#if MEMORY_HAS_NATIVE_PROFILER && 0
    // Record chunk information before it's freed (only needed for profiling).
    const Chunk* const chunk       = ChunkFromHandle(h);
    void const* const  chunk_ptr   = chunk->ptr;
    int64_t const      req_bytes   = chunk->requested_size;
    int64_t const      alloc_bytes = chunk->size;
#endif

    MarkFree(h);

    // Consider coalescing it.
    if (timing_counter_ != nullptr)
    {
        InsertFreeChunkIntoBin(h);
        timestamped_chunks_.push_back(h);
    }
    else
    {
        InsertFreeChunkIntoBin(TryToCoalesce(h, false));
    }

#if MEMORY_HAS_NATIVE_PROFILER && 0
    // TraceMe needs to be added after MarkFree and InsertFreeChunkIntoBin for
    // correct aggregation stats (bytes_in_use, fragmentation).
    AddTraceMe("MemoryDeallocation", chunk_ptr, req_bytes, alloc_bytes);
#endif

    LOGGING_LOG_INFO_DEBUG_BFC("F: {}", RenderOccupancy());
}

// Merges h1 and h2 when Chunk(h1)->next is h2 and Chunk(h2)->prev is c1.
// We merge Chunk(h2) into Chunk(h1).
void allocator_bfc::Merge(allocator_bfc::ChunkHandle h1, allocator_bfc::ChunkHandle h2)
{
    Chunk*       c1 = ChunkFromHandle(h1);
    Chunk const* c2 = ChunkFromHandle(h2);
    // We can only merge chunks that are not in use.
    LOGGING_CHECK(!c1->in_use() && !c2->in_use());  //NOLINT

    // c1's prev doesn't change, still points to the same ptr, and is
    // still not in use.

    // Fix up neighbor pointers
    //
    // c1 <-> c2 <-> c3 should become
    // c1 <-> c3

    allocator_bfc::ChunkHandle const h3 = c2->next;
    c1->next                            = h3;
    LOGGING_CHECK(c2->prev == h1);
    if (h3 != kInvalidChunkHandle)
    {
        allocator_bfc::Chunk* c3 = ChunkFromHandle(h3);
        c3->prev                 = h1;
    }

    // Set the new size
    c1->size += c2->size;

    // Pick latest free time.
    c1->freed_at_count = std::max(c1->freed_at_count, c2->freed_at_count);

    DeleteChunk(h2);
}

void allocator_bfc::DeleteChunk(ChunkHandle h)
{
    // Delete h and cleanup all state
    Chunk const* c = ChunkFromHandle(h);
    //  VLOG(4) << "Removing: " << c->ptr;
    region_manager_.erase(c->ptr);
    DeallocateChunk(h);
}

void allocator_bfc::InsertFreeChunkIntoBin(allocator_bfc::ChunkHandle h)
{
    Chunk* c = ChunkFromHandle(h);
    LOGGING_CHECK(!c->in_use() && (c->bin_num == kInvalidBinNum));  //NOLINT
    BinNum const bin_num = BinNumForSize(c->size);
    Bin*         new_bin = BinFromIndex(bin_num);
    c->bin_num           = bin_num;
    new_bin->free_chunks.insert(h);
}

void allocator_bfc::RemoveFreeChunkIterFromBin(
    allocator_bfc::Bin::FreeChunkSet*                 free_chunks,
    const allocator_bfc::Bin::FreeChunkSet::iterator& citer)
{
    ChunkHandle const h = *citer;
    Chunk*            c = ChunkFromHandle(h);
    LOGGING_CHECK(!c->in_use() && (c->bin_num != kInvalidBinNum));  //NOLINT
    free_chunks->erase(citer);
    c->bin_num = kInvalidBinNum;
}

void allocator_bfc::RemoveFreeChunkFromBin(allocator_bfc::ChunkHandle h)
{
    Chunk* c = ChunkFromHandle(h);
    LOGGING_CHECK(!c->in_use() && (c->bin_num != kInvalidBinNum));  //NOLINT
    LOGGING_CHECK(
        BinFromIndex(c->bin_num)->free_chunks.erase(h) > 0, "Could not find chunk in bin");
    c->bin_num = kInvalidBinNum;
}

void allocator_bfc::MarkFree(allocator_bfc::ChunkHandle h)
{
    Chunk* c = ChunkFromHandle(h);
    LOGGING_CHECK(c->in_use() && (c->bin_num == kInvalidBinNum));  //NOLINT

    // Mark the chunk as no longer in use.
    c->allocation_id = -1;

    // Optionally record the free time.
    if (timing_counter_ != nullptr)
    {
        c->freed_at_count = timing_counter_->next();
    }

    // Updates the stats.
    stats_.bytes_in_use -= c->size;

#ifdef MEMORY_MEM_DEBUG
    if (ShouldRecordOpName())
    {
        c->action_count     = ++action_counter_;
        int slot            = c->action_count % MEM_DEBUG_SIZE_HISTORY_SIZE;
        size_history_[slot] = stats_.bytes_in_use;
    }
#endif
}

allocator_bfc::ChunkHandle allocator_bfc::TryToCoalesce(ChunkHandle h, bool ignore_freed_at)
{
    Chunk const* c = ChunkFromHandle(h);
    if ((!ignore_freed_at) && c->freed_at_count > 0)
    {
        return h;
    }
    ChunkHandle coalesced_chunk = h;

    // If the next chunk is free, merge it into c and delete it.
    if (c->next != kInvalidChunkHandle && !ChunkFromHandle(c->next)->in_use())
    {
        Chunk const* n = ChunkFromHandle(c->next);
        if ((n->freed_at_count == 0) || ignore_freed_at)
        {
            LOGGING_LOG_INFO_DEBUG_BFC("Merging c->next {} with c {}", n->ptr, c->ptr);
            RemoveFreeChunkFromBin(c->next);
            Merge(h, c->next);
        }
    }

    // If the previous chunk is free, merge c into it and delete c.
    if (c->prev != kInvalidChunkHandle && !ChunkFromHandle(c->prev)->in_use())
    {
        Chunk const* n = ChunkFromHandle(c->prev);
        if ((n->freed_at_count == 0) || ignore_freed_at)
        {
            LOGGING_LOG_INFO_DEBUG_BFC("Merging c {} into c->prev {}", c->ptr, n->ptr);
            coalesced_chunk = c->prev;
            RemoveFreeChunkFromBin(c->prev);
            Merge(c->prev, h);
        }
    }

    return coalesced_chunk;
}

void allocator_bfc::SetSafeFrontier(uint64_t count) noexcept
{
    uint64_t current = safe_frontier_.load(std::memory_order_relaxed);
    while (count > current)
    {
        if (safe_frontier_.compare_exchange_strong(current, count))
        {
            retry_helper_.NotifyDealloc();
            return;
        }

        current = safe_frontier_.load(std::memory_order_relaxed);
    }
}

bool allocator_bfc::MergeTimestampedChunks(size_t required_bytes)
{
    LOGGING_LOG_INFO_DEBUG_BFC(
        "MergeTimestampedChunks queue_len={} required_bytes={}",
        timestamped_chunks_.size(),
        required_bytes);

    bool satisfied = (required_bytes == 0);

    std::vector<void*>      to_merge;
    std::deque<ChunkHandle> new_ts_queue;
    while (!timestamped_chunks_.empty())
    {
        ChunkHandle h = timestamped_chunks_.front();
        timestamped_chunks_.pop_front();
        LOGGING_CHECK_DEBUG(h != kInvalidChunkHandle);
        Chunk* c = ChunkFromHandle(h);
        // It's possible this chunk has already been merged so refetch and retest
        // the handle.
        h = region_manager_.get_handle(c->ptr);
        if (h == kInvalidChunkHandle)
        {
            continue;
        }
        if (c->in_use() || (c->bin_num == kInvalidBinNum))
        {
            // This chunk has already been reallocated.
            continue;
        }
        if (c->freed_at_count == 0)
        {
            to_merge.push_back(c->ptr);
            continue;
        }
        // Chunk should be free and assigned to a bin.
        LOGGING_CHECK_DEBUG(c->bin_num != kInvalidBinNum);
        if (c->freed_at_count < safe_frontier_)
        {
            c->freed_at_count = 0;
            to_merge.push_back(c->ptr);
        }
        else if (required_bytes > 0)
        {
            to_merge.push_back(c->ptr);
        }
        else
        {
            new_ts_queue.push_back(h);
        }
    }
    LOGGING_CHECK_DEBUG(timestamped_chunks_.empty());
    std::swap(timestamped_chunks_, new_ts_queue);

    // At this point all candidate chunks have been moved from timestamped_chunks_
    // to to_merge.  If this is a standard merge (required_bytes == 0) then
    // merge them all, otherwise merge just until a Chunk of the required size
    // is produced.
    for (auto* ptr : to_merge)
    {
        // It's possible that the Chunk associated with this memory location got
        // merged and deallocated in a prior iteration so refetch the handle and
        // retest.
        ChunkHandle const h = region_manager_.get_handle(ptr);
        if (h == kInvalidChunkHandle)
        {
            continue;
        }
        if (required_bytes == 0 || !satisfied)
        {
            Chunk const* c = ChunkFromHandle(h);
            LOGGING_CHECK_DEBUG(c->bin_num != kInvalidBinNum);
            LOGGING_CHECK_DEBUG(!c->in_use());
            RemoveFreeChunkFromBin(h);
            ChunkHandle const new_h = TryToCoalesce(h, (required_bytes > 0));
            InsertFreeChunkIntoBin(new_h);
            if (required_bytes > 0)
            {
                c = ChunkFromHandle(new_h);
                if (new_h != h && c->freed_at_count > 0)
                {
                    timestamped_chunks_.push_back(new_h);
                }
                if (c->size >= required_bytes)
                {
                    satisfied = true;
                }
            }
        }
        else
        {
            // We were force merging Chunks with unsafe timestamps, but managed
            // to create a satisfying Chunk so just requeue the rest.
            timestamped_chunks_.push_back(h);
        }
    }
    return satisfied;
}

bool allocator_bfc::tracks_allocation_sizes() const noexcept
{
    return true;
}

size_t allocator_bfc::RequestedSize(const void* ptr) const
{
    LOGGING_CHECK(ptr != nullptr);
    std::scoped_lock const           lock(mutex_);
    allocator_bfc::ChunkHandle const h = region_manager_.get_handle(ptr);
    LOGGING_CHECK(
        h != kInvalidChunkHandle, "Asked for requested size of pointer we never allocated: ", ptr);

    const allocator_bfc::Chunk* c = ChunkFromHandle(h);
    return c->requested_size;
}

size_t allocator_bfc::AllocatedSize(const void* ptr) const
{
    std::scoped_lock const           lock(mutex_);
    allocator_bfc::ChunkHandle const h = region_manager_.get_handle(ptr);

    LOGGING_CHECK(
        h != kInvalidChunkHandle, "Asked for allocated size of pointer we never allocated: ", ptr);

    const allocator_bfc::Chunk* c = ChunkFromHandle(h);
    return c->size;
}

int64_t allocator_bfc::AllocationId(const void* ptr) const
{
    std::scoped_lock const           lock(mutex_);
    allocator_bfc::ChunkHandle const h = region_manager_.get_handle(ptr);

    LOGGING_CHECK(
        h != kInvalidChunkHandle, "Asked for allocation id of pointer we never allocated: ", ptr);

    const allocator_bfc::Chunk* c = ChunkFromHandle(h);
    return c->allocation_id;
}

namespace
{

void RenderRegion(
    char*        rendered,
    const size_t resolution,
    const size_t total_render_size,
    const size_t offset,
    const void*  base_ptr,
    const void*  ptr,
    const size_t size,
    const char   c)
{
    const char* base_ptr_c = static_cast<const char*>(base_ptr);
    const char* ptr_c      = static_cast<const char*>(ptr);

    size_t const start_location = ((ptr_c - base_ptr_c + offset) * resolution) / total_render_size;
    LOGGING_CHECK_DEBUG(start_location < resolution);
    size_t const end_location =
        ((ptr_c + size - 1 - base_ptr_c + offset) * resolution) / total_render_size;
    LOGGING_CHECK_DEBUG(end_location < resolution);

    // Ensure we don't exceed array bounds (clang-analyzer check)
    size_t const safe_start = std::min(start_location, resolution - 1);
    size_t const safe_end   = std::min(end_location, resolution - 1);
    for (size_t i = safe_start; i <= safe_end; ++i)
    {
        rendered[i] = c;
    }
}

}  // namespace

std::string allocator_bfc::RenderOccupancy()
{
    // Make a buffer for the ASCII-art representation.
    const size_t resolution = 100;
    char         rendered[resolution];

    // Compute the total region size to render over
    size_t const total_region_size = std::accumulate(
        region_manager_.regions().begin(),
        region_manager_.regions().end(),
        size_t{0},
        [](size_t sum, const auto& region) { return sum + region.memory_size(); });

    if (total_region_size == 0)
    {
        return "<allocator contains no memory>";
    }

    // Start out with everything empty
    RenderRegion(
        rendered, resolution, total_region_size, 0, nullptr, nullptr, total_region_size, '_');

    size_t region_offset = 0;
    for (const auto& region : region_manager_.regions())
    {
        ChunkHandle h = region_manager_.get_handle(region.ptr());
        // Then render each chunk left to right.
        while (h != kInvalidChunkHandle)
        {
            Chunk const* c = ChunkFromHandle(h);
            if (c->in_use())
            {
                // Render the wasted space
                size_t const wasted = c->size - c->requested_size;
                if (wasted > 0)
                {
                    RenderRegion(
                        rendered,
                        resolution,
                        total_region_size,
                        region_offset + c->requested_size,
                        region.ptr(),
                        c->ptr,
                        wasted,
                        'x');
                }
                // Then the occupied space
                RenderRegion(
                    rendered,
                    resolution,
                    total_region_size,
                    region_offset,
                    region.ptr(),
                    c->ptr,
                    c->requested_size,
                    '*');
            }
            h = c->next;
        }
        region_offset += region.memory_size();
    }

    return std::string(rendered, resolution);  //NOLINT
}

void allocator_bfc::DumpMemoryLog(size_t num_bytes)
{
    const std::array<BinDebugInfo, kNumBins> bin_infos = get_bin_debug_info();

    LOGGING_LOG_INFO("allocator_bfc dump for {}", Name());

    for (BinNum bin_num = 0; bin_num < kNumBins; bin_num++)
    {
        Bin*                b        = BinFromIndex(bin_num);
        const BinDebugInfo& bin_info = bin_infos[bin_num];
        LOGGING_CHECK_DEBUG(
            b->free_chunks.size() == bin_info.total_chunks_in_bin - bin_info.total_chunks_in_use);

        LOGGING_LOG_INFO(
            "Bin ({}): \tTotal Chunks: {}, Chunks in use: {}. {} allocated for chunks. {} in use "
            "in bin. {} client-requested in use in bin.",
            b->bin_size,
            bin_info.total_chunks_in_bin,
            bin_info.total_chunks_in_use,
            format_human_readable_bytes(bin_info.total_bytes_in_bin),
            format_human_readable_bytes(bin_info.total_bytes_in_use),
            format_human_readable_bytes(bin_info.total_requested_bytes_in_use));
    }

    // Find the bin that we would have liked to allocate in, so we
    // can get some further analysis about fragmentation.
    Bin const* b = BinForSize(num_bytes);

    LOGGING_LOG_INFO(
        "Bin for {} was {}, Chunk State: ",
        format_human_readable_bytes(num_bytes),
        format_human_readable_bytes(b->bin_size));

    for (ChunkHandle const h : b->free_chunks)
    {
        Chunk const* c = ChunkFromHandle(h);
        LOGGING_LOG_INFO("{}", c->debug_string(this, true));
    }

    // Next show the chunks that are in use, and also summarize their
    // number by size.
    logging::logging_map<size_t, int> in_use_by_size;
    for (const auto& region : region_manager_.regions())
    {
        LOGGING_LOG_INFO("Next region of size {}", region.memory_size());
        ChunkHandle h = region_manager_.get_handle(region.ptr());
        while (h != kInvalidChunkHandle)
        {
            const Chunk* c = ChunkFromHandle(h);
            if (c->in_use())
            {
                in_use_by_size[c->size]++;
            }
            std::string buf = logging::strings::str_cat(
                (c->in_use() ? "InUse" : "Free "),
                " at 0x",
                logging::strings::format_hex(reinterpret_cast<uint64_t>(c->ptr)),
                " of size ",
                c->size);
#ifdef MEMORY_MEM_DEBUG
            if (ShouldRecordOpName())
            {
                logging::strings::str_append(
                    &buf,
                    " by op ",
                    c->op_name,
                    " action_count ",
                    c->action_count,
                    " step ",
                    c->step_id);
            }
#endif
            logging::strings::str_append(&buf, " next ", c->next);
            if (timing_counter_ != nullptr)
            {
                logging::strings::str_append(&buf, " freed_at_count ", c->freed_at_count);
            }
            LOGGING_LOG_INFO("{}", buf);
            h = c->next;
        }
    }

    LOGGING_LOG_INFO("     Summary of in-use Chunks by size: ");
    size_t total_bytes = 0;
    for (auto& it : in_use_by_size)
    {
        LOGGING_LOG_INFO(
            "{} Chunks of size {} totalling {}",
            it.second,
            it.first,
            format_human_readable_bytes(it.first * it.second));
        total_bytes += (it.first * it.second);
    }
    LOGGING_LOG_INFO("Sum Total of in-use chunks: {}", format_human_readable_bytes(total_bytes));
    LOGGING_LOG_INFO(
        "Total bytes in pool: {} memory_limit_: {} available bytes: {} "
        "curr_region_allocation_bytes_: {}",
        stats_.pool_bytes.load(std::memory_order_relaxed),
        memory_limit_,
        (memory_limit_ - stats_.pool_bytes.load(std::memory_order_relaxed)),
        curr_region_allocation_bytes_);
    LOGGING_LOG_INFO("Stats: \n{}", stats_.debug_string());
}

void allocator_bfc::MaybeWriteMemoryMap()
{
#if 0
    const char* gpu_memory_map_file = std::getenv("TF_BFC_MEMORY_DUMP");
    if (gpu_memory_map_file != nullptr)
    {
        std::unique_ptr<WritableFile> dump_file;
        std::string                   file_name = logging::strings::str_cat(
            gpu_memory_map_file, "_", Name(), ".", Env::Default()->NowMicros());
        std::Status status = Env::Default()->NewWritableFile(file_name, &dump_file);
        if (!status.ok())
        {
            LOGGING_LOG_ERROR("Failed to open file {}", file_name);
            return;
        }
        memory_dump md = RecordMemoryMapInternal();
        status        = dump_file->Append(md.SerializeAsstd::string());
        if (!status.ok())
        {
            LOGGING_LOG_ERROR("Error on writing to file {}: {}", gpu_memory_map_file, status);
        }
    }
#endif  // 0
}

memory_dump allocator_bfc::RecordMemoryMap()
{
    std::scoped_lock const lock(mutex_);
    return RecordMemoryMapInternal();
}

memory_dump allocator_bfc::RecordMemoryMapInternal()
{
    memory_dump md;

    md.set_allocator_name(Name());

    // Record the general stats
    MemAllocatorStats* mas = md.stats();
    mas->set_num_allocs(stats_.num_allocs);
    mas->set_bytes_in_use(stats_.bytes_in_use);
    mas->set_peak_bytes_in_use(stats_.peak_bytes_in_use);
    mas->set_largest_alloc_size(stats_.largest_alloc_size);

    // Record summary data for every bin.
    const std::array<BinDebugInfo, kNumBins> bin_infos = get_bin_debug_info();
    for (BinNum bin_num = 0; bin_num < kNumBins; bin_num++)
    {
        const BinDebugInfo& bin_info = bin_infos[bin_num];
#ifndef NDEBUG
        Bin const* b = BinFromIndex(bin_num);
        LOGGING_CHECK_DEBUG(
            b->free_chunks.size() == bin_info.total_chunks_in_bin - bin_info.total_chunks_in_use);
#endif
        BinSummary* bs = md.add_bin_summary();
        bs->set_bin(bin_num);
        bs->set_total_bytes_in_use(bin_info.total_bytes_in_use);
        bs->set_total_bytes_in_bin(bin_info.total_bytes_in_bin);
        bs->set_total_chunks_in_use(bin_info.total_chunks_in_use);
        bs->set_total_chunks_in_bin(bin_info.total_chunks_in_bin);
    }

    // Record state of every defined Chunk.
    for (const auto& region : region_manager_.regions())
    {
        ChunkHandle h = region_manager_.get_handle(region.ptr());
        while (h != kInvalidChunkHandle)
        {
            const Chunk* c  = ChunkFromHandle(h);
            MemChunk*    mc = md.add_chunk();
            mc->set_in_use(c->in_use());
            mc->set_address(reinterpret_cast<uint64_t>(c->ptr));
            mc->set_size(c->size);
            mc->set_requested_size(c->requested_size);
            mc->set_bin(c->bin_num);
#ifdef MEMORY_MEM_DEBUG
            mc->set_op_name(c->op_name ? std::string(c->op_name) : "UNKNOWN");
            mc->set_step_id(c->step_id);
            mc->set_action_count(c->action_count);
#endif
            if (timing_counter_ != nullptr)
            {
                mc->set_freed_at_count(c->in_use() ? 0 : c->freed_at_count);
            }
            h = c->next;
        }
    }

    mas->set_fragmentation_metric(GetFragmentation());

#ifdef MEMORY_MEM_DEBUG
    // Record the recent size history
    int history_len = static_cast<int>(
        std::min(action_counter_, static_cast<int64>(MEM_DEBUG_SIZE_HISTORY_SIZE)));
    for (int i = action_counter_ - history_len; i < action_counter_; ++i)
    {
        SnapShot* ss = md.add_snap_shot();
        ss->set_action_count(i);
        int slot = i % MEM_DEBUG_SIZE_HISTORY_SIZE;
        ss->set_size(size_history_[slot]);
    }
#endif

    return md;
}

std::optional<allocator_stats> allocator_bfc::GetStats() const
{
    std::scoped_lock const lock(mutex_);
    // Create a copy of the atomic stats structure
    allocator_stats stats_copy(stats_);
    return stats_copy;
}

bool allocator_bfc::ClearStats()
{
    std::scoped_lock const lock(mutex_);
    stats_.num_allocs.store(0, std::memory_order_relaxed);
    stats_.peak_bytes_in_use.store(
        stats_.bytes_in_use.load(std::memory_order_relaxed), std::memory_order_relaxed);
    stats_.largest_alloc_size.store(0, std::memory_order_relaxed);
    return true;
}

std::array<allocator_bfc::BinDebugInfo, allocator_bfc::kNumBins> allocator_bfc::get_bin_debug_info()
{
    std::array<BinDebugInfo, kNumBins> bin_infos;
    for (const auto& region : region_manager_.regions())
    {
        ChunkHandle h = region_manager_.get_handle(region.ptr());
        while (h != kInvalidChunkHandle)
        {
            const Chunk*  c        = ChunkFromHandle(h);
            BinNum const  bin_num  = BinNumForSize(c->size);
            BinDebugInfo& bin_info = bin_infos[bin_num];
            bin_info.total_bytes_in_bin += c->size;
            bin_info.total_chunks_in_bin++;
            if (c->in_use())
            {
                bin_info.total_bytes_in_use += c->size;
                bin_info.total_requested_bytes_in_use += c->requested_size;
                bin_info.total_chunks_in_use++;
            }
            else
            {
                Bin const* bin = BinFromIndex(bin_num);
                LOGGING_CHECK(bin->free_chunks.count(h) == 1);
                LOGGING_CHECK(c->bin_num == bin_num);
            }
            h = c->next;
        }
    }
    return bin_infos;
}

allocator_memory_enum allocator_bfc::GetMemoryType() const noexcept
{
    return sub_allocator_->GetMemoryType();
}
}  // namespace memory
