/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * This file is part of XSigma and is licensed under a dual-license model:
 *
 *   - Open-source License (GPLv3):
 *       Free for personal, academic, and research use under the terms of
 *       the GNU General Public License v3.0 or later.
 *
 *   - Commercial License:
 *       A commercial license is required for proprietary, closed-source,
 *       or SaaS usage. Contact us to obtain a commercial agreement.
 *
 * Contact: licensing@xsigma.co.uk
 * Website: https://www.xsigma.co.uk
 */

// Fault-injection regression coverage for gpu/cuda_caching_allocator.cpp,
// compiled against Testing/CudaCachingAllocator/fake_runtime.h instead of a
// real CUDA/HIP driver. See Docs/memory_runtime_implementation_plan.md Appendix A and
// this suite's CMakeLists.txt for why this builds under the HIP labels.

#include <atomic>
#include <exception>
#include <cstdio>
#include <map>
#include <gtest/gtest.h>

#include <stdexcept>
#include <thread>
#include <vector>

#include "fake_runtime.h"
#include "common/storage_handle.h"
#include "common/cleanup_diagnostic.h"
#include "gpu/cuda_caching_allocator.h"
#include "include/util/exception.h"

using memory::gpu::cuda_caching_allocator;
namespace rt = fake_runtime;

namespace
{
constexpr size_t kSegmentSize = 20 * 1024 * 1024;  // exact kLargeBuffer multiple
}

class CudaCachingAllocatorRuntime : public ::testing::Test
{
    void SetUp() override { rt::reset(); }
    void TearDown() override
    {
        EXPECT_EQ(rt::event_creates, rt::event_destroys);
        rt::reset();
    }
};

// Regression for the P0 fix landed 2026-09-28 (commit e34ea89): a partial
// event-insertion failure (first stream's cudaEventRecord succeeds, second
// stream's fails) must quarantine the block, never free it back to the pool
// -- the first stream's real in-flight work was never proven complete.
TEST_F(CudaCachingAllocatorRuntime, PartialEventFailureOnSecondStreamQuarantinesBlock)
{
    cuda_caching_allocator allocator(0);
    void*                  ptr = allocator.allocate(4096);
    ASSERT_NE(nullptr, ptr);
    allocator.record_stream(ptr, rt::stream(1));
    allocator.record_stream(ptr, rt::stream(2));

    rt::fail_event_record_at_call = 2;  // stream(1) succeeds, stream(2) fails
    EXPECT_THROW(allocator.deallocate(ptr, 4096), std::runtime_error);

    // The quarantined block must never be handed back out: a same-size
    // allocate() must return a different pointer.
    void* other = allocator.allocate(4096);
    ASSERT_NE(nullptr, other);
    EXPECT_NE(ptr, other);
}

// Regression for the same fix's other branch: a single stream whose only
// event fails (event_count==0, streams non-empty in the catch block) must
// also quarantine, not take the "no uses were ever registered" free path.
TEST_F(CudaCachingAllocatorRuntime, SingleStreamEventFailureQuarantinesNotFrees)
{
    cuda_caching_allocator allocator(0);
    void*                  ptr = allocator.allocate(4096);
    ASSERT_NE(nullptr, ptr);
    allocator.record_stream(ptr, rt::stream(1));

    rt::fail_event_record_at_call = 1;  // the only cudaEventRecord call fails
    EXPECT_THROW(allocator.deallocate(ptr, 4096), std::runtime_error);

    void* other = allocator.allocate(4096);
    ASSERT_NE(nullptr, other);
    EXPECT_NE(ptr, other);
}

// Order 2, item 1: the OOM-chain retry (after release_cached_blocks_locked())
// used to call alloc_segment_unlocked() again without rechecking the budget.
// alloc_segment_unlocked drops the lock around its own cudaMalloc call, so a
// concurrent set_memory_fraction() landing during that window can shrink
// headroom between this function's earlier checks and the retry. Pause the
// first cudaMalloc call mid-flight, shrink the budget from another thread,
// then release: the fixed code must recheck and reject rather than
// committing a second, now-over-budget driver allocation.
TEST_F(CudaCachingAllocatorRuntime, RetryAfterDriverFailureRechecksShrunkenBudget)
{
    rt::device_total_bytes = 100 * kSegmentSize;
    cuda_caching_allocator allocator(0);
    allocator.set_memory_fraction(2.0 * static_cast<double>(kSegmentSize) / rt::device_total_bytes);

    rt::malloc_pause_on_call = 1;  // pause the 1st cudaMalloc call
    rt::fail_malloc_calls    = 1;  // ...which then fails once released

    std::thread worker(
        [&] { EXPECT_THROW(allocator.allocate(kSegmentSize), std::bad_alloc); });
    rt::wait_for_malloc_pause();

    // Shrink the budget while the worker is paused mid-driver-call for its
    // first attempt -- the only real concurrent-mutation window given the
    // allocator's lock discipline (release_cached_blocks_locked holds the
    // lock throughout; only alloc_segment_unlocked's own cudaMalloc drops it).
    allocator.set_memory_fraction(
        0.5 * static_cast<double>(kSegmentSize) / rt::device_total_bytes);
    rt::release_malloc_pause();
    worker.join();

    EXPECT_EQ(1, rt::malloc_calls) << "the recheck must reject before spending a 2nd cudaMalloc "
                                       "once it sees the shrunken budget";
}

// Order 2, item 2: if malloc_segment's device_guard throws (its cudaGetDevice
// call fails) while the allocator lock is dropped for the driver call, the
// pending-budget reservation for that request must still be rolled back.
// Leaking it would make every subsequent allocate() see stale phantom
// headroom consumption and wrongly reject requests that should fit.
TEST_F(CudaCachingAllocatorRuntime, SegmentAllocDeviceGuardThrowRollsBackPendingBudget)
{
    rt::device_total_bytes = 100 * kSegmentSize;
    cuda_caching_allocator allocator(0);
    // Room for exactly one segment: set_memory_fraction's own device_guard
    // (inside query_device_total_memory) consumes one cudaGetDevice call here,
    // before the injected failure below is armed.
    allocator.set_memory_fraction(1.0 * static_cast<double>(kSegmentSize) / rt::device_total_bytes);

    rt::fail_get_device_calls = 1;  // fails the next cudaGetDevice call only:
                                     // malloc_segment's device_guard, inside
                                     // the retry-free first allocate() attempt.
    EXPECT_THROW(allocator.allocate(kSegmentSize), std::runtime_error);

    // If pending_reserved_bytes_ leaked from the failed attempt above, this
    // retry's budget check sees (stale pending + fresh alloc_size) > budget
    // and wrongly throws bad_alloc; the fix must have rolled it back to 0.
    void* ptr = nullptr;
    EXPECT_NO_THROW(ptr = allocator.allocate(kSegmentSize));
    ASSERT_NE(nullptr, ptr);
    allocator.deallocate(ptr, kSegmentSize);
}

// ---------------------------------------------------------------------------
// Task 2.10 / R1 gate tests (plan §2.10 exit criterion)
// ---------------------------------------------------------------------------

// deallocate_with_stream_lookup locates the block's allocation stream from the
// cache's own bookkeeping and returns it to the free pool.  A second allocation
// of the same size must reuse the recycled block (same pointer).
TEST_F(CudaCachingAllocatorRuntime, DeallocateWithStreamLookupReturnsBlockToPool)
{
    cuda_caching_allocator allocator(0);
    void* ptr = allocator.allocate(4096);
    ASSERT_NE(nullptr, ptr);
    EXPECT_EQ(4096u, allocator.bytes_allocated_now());

    // Free via the stream-lookup path (no stream hint required by the caller).
    allocator.deallocate_with_stream_lookup(ptr, 4096);
    EXPECT_EQ(0u, allocator.bytes_allocated_now());

    // The cache should hand the recycled block back on the next same-size request.
    void* ptr2 = allocator.allocate(4096);
    ASSERT_NE(nullptr, ptr2);
    EXPECT_EQ(ptr, ptr2);
    allocator.deallocate(ptr2, 4096);
}

// A bare storage_handle wired with a gpu_free_fn-style deleter (plan §2.10,
// R1) must return its block to the cache when it is destroyed, without the
// caller supplying the allocation stream.  This exercises the path taken by
// allocate_bytes() GPU handles in production.
TEST_F(CudaCachingAllocatorRuntime, BareStorageHandleFreesViaDeleter)
{
    using memory::storage_handle;
    using memory::device;
    using memory::next_allocation_id;

    // Local mirror of src/storage.cpp::gpu_free_fn (plan §2.10, R1).
    auto gpu_free_fn = [](void* cache_ctx, void* p, std::size_t nb) noexcept {
        static_cast<cuda_caching_allocator*>(cache_ctx)->deallocate_with_stream_lookup(p, nb);
    };

    cuda_caching_allocator allocator(0);
    void* raw = allocator.allocate(8192);
    ASSERT_NE(nullptr, raw);
    EXPECT_EQ(8192u, allocator.bytes_allocated_now());

    {
        // Build a handle that owns this block.
        storage_handle h(raw, 8192,
                         static_cast<memory::deleter_fn>(gpu_free_fn),
                         static_cast<void*>(&allocator),
                         device::cuda(0),
                         next_allocation_id());
        // Handle goes out of scope here → destructor → gpu_free_fn → deallocate_with_stream_lookup.
    }
    EXPECT_EQ(0u, allocator.bytes_allocated_now());

    // Cache recycled the block: second same-size alloc returns same pointer.
    void* ptr2 = allocator.allocate(8192);
    ASSERT_NE(nullptr, ptr2);
    EXPECT_EQ(raw, ptr2);
    allocator.deallocate(ptr2, 8192);
}

// ---------------------------------------------------------------------------
// Phase 0.3 hot-path probes (plan §6.1): counting operator new + fake-runtime
// driver counters. Heap counts include everything the allocator does.
// ---------------------------------------------------------------------------
namespace
{
std::atomic<bool>   g_count_new{false};
std::atomic<size_t> g_new_calls{0};
// Failure injection (task 1.7): while g_inject_new is set, the g_inject_new_at-th
// allocation (1-based) throws std::bad_alloc.
std::atomic<bool> g_inject_new{false};
std::atomic<long> g_inject_new_at{0};
std::atomic<long> g_inject_new_seen{0};
std::atomic<bool> g_inject_new_fired{false};
}  // namespace

void* operator new(std::size_t n)
{
    if (g_count_new.load(std::memory_order_relaxed))
    {
        g_new_calls.fetch_add(1, std::memory_order_relaxed);
    }
    if (g_inject_new.load(std::memory_order_relaxed) &&
        g_inject_new_seen.fetch_add(1, std::memory_order_relaxed) + 1 ==
            g_inject_new_at.load(std::memory_order_relaxed))
    {
        g_inject_new_fired.store(true, std::memory_order_relaxed);
        throw std::bad_alloc();
    }
    if (void* p = std::malloc(n ? n : 1))
    {
        return p;
    }
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

namespace
{
struct new_probe
{
    new_probe()
    {
        g_new_calls = 0;
        g_count_new = true;
    }
    ~new_probe() { g_count_new = false; }
    size_t count() const { return g_new_calls.load(); }
};
}  // namespace

// The zero-allocation probes count operator new. With debug iterators (the MSVC STL
// in a Debug build) every container carries an allocated proxy, so the counts
// are not those of the shipped code: the probes run in Release only.
#if defined(_ITERATOR_DEBUG_LEVEL) && _ITERATOR_DEBUG_LEVEL != 0
#define SKIP_HEAP_PROBE_UNDER_DEBUG_ITERATORS()     GTEST_SKIP() << "heap-allocation probes need a build without debug iterators"
#else
#define SKIP_HEAP_PROBE_UNDER_DEBUG_ITERATORS() (void)0
#endif

TEST_F(CudaCachingAllocatorRuntime, ProbeGpuWarmAllocFreeHeapAndDriverCalls)
{
    SKIP_HEAP_PROBE_UNDER_DEBUG_ITERATORS();
    cuda_caching_allocator allocator(0);
    allocator.deallocate(allocator.allocate(4096), 4096);  // warm the pool
    int const malloc_before = rt::malloc_calls;
    size_t    heap          = 0;
    {
        new_probe probe;
        for (int i = 0; i < 100; ++i)
        {
            void* p = allocator.allocate(4096);
            allocator.deallocate(p, 4096);
        }
        heap = probe.count();
    }
    RecordProperty("warm_alloc_free_heap_allocations_per_100", static_cast<int>(heap));
    EXPECT_EQ(malloc_before, rt::malloc_calls) << "warm alloc/free must make 0 driver calls";
    // Plan 3.8: live-map and free-pool nodes are recycled (was 3 per pair).
    EXPECT_EQ(0u, heap) << "warm alloc/free must not allocate";
}

TEST_F(CudaCachingAllocatorRuntime, ProbeGpuSplitHeapAllocations)
{
    SKIP_HEAP_PROBE_UNDER_DEBUG_ITERATORS();
    cuda_caching_allocator allocator(0);
    auto const             split_and_merge = [&]
    {
        void* a = allocator.allocate(512);
        void* b = allocator.allocate(512);
        allocator.deallocate(a, 512);
        allocator.deallocate(b, 512);
    };
    allocator.deallocate(allocator.allocate(4096), 4096);
    split_and_merge();  // first use grows block metadata and nodes once
    size_t heap = 0;
    {
        new_probe probe;
        split_and_merge();
        heap = probe.count();
    }
    RecordProperty("split_heap_allocations", static_cast<int>(heap));
    EXPECT_EQ(0u, heap) << "steady-state split/merge must not allocate";
}

TEST_F(CudaCachingAllocatorRuntime, ProbeRecordStreamUpToFourStreamsNoHeap)
{
    cuda_caching_allocator allocator(0);
    allocator.deallocate(allocator.allocate(4096), 4096);
    void*  p    = allocator.allocate(4096);
    size_t heap = 0;
    {
        new_probe probe;
        for (size_t s = 1; s <= 4; ++s)
        {
            allocator.record_stream(p, rt::stream(s));
        }
        heap = probe.count();
    }
    allocator.deallocate(p, 4096);
    RecordProperty("record_stream_4_heap_allocations", static_cast<int>(heap));
    EXPECT_EQ(0u, heap) << "record_stream up to 4 streams must not allocate (inline_stream_set)";
}

// Task 1.1 / R7: ownership violations throw logging::exception in every build
// type (this target compiles with NDEBUG), and never corrupt the cache.
TEST_F(CudaCachingAllocatorRuntime, ForeignPointerDeallocateAndRecordStreamThrowLoggingException)
{
    cuda_caching_allocator allocator(0);
    int                    foreign = 0;
    EXPECT_THROW(allocator.deallocate(&foreign, 16), logging::exception);
    EXPECT_THROW(allocator.record_stream(&foreign, rt::stream(1)), logging::exception);
    // Destructor path is noexcept: a foreign pointer is counted, never thrown.
    auto const before = memory::cleanup_diagnostic::failure_count();
    auto const before_cache =
        memory::cleanup_diagnostic::failure_count(memory::cleanup_source::gpu_cache);
    EXPECT_NO_THROW(allocator.deallocate_with_stream_lookup(&foreign, 16));
    EXPECT_EQ(before + 1, memory::cleanup_diagnostic::failure_count());
    EXPECT_EQ(
        before_cache + 1,
        memory::cleanup_diagnostic::failure_count(memory::cleanup_source::gpu_cache));
}

namespace
{
std::atomic<int>                 g_handler_calls{0};
std::atomic<int>                 g_handler_last_source{-1};
std::atomic<cuda_caching_allocator*> g_probe_allocator{nullptr};
std::atomic<bool>                g_probe_reentered{false};

void counting_handler(memory::cleanup_source source) noexcept
{
    g_handler_last_source.store(static_cast<int>(source));
    g_handler_calls.fetch_add(1);
}

// Deliberately violates the handler contract (calls into the allocator) from a
// second thread to prove the cache mutex is not held when the handler runs:
// if it were held, the join below would deadlock.
void reentrant_handler(memory::cleanup_source) noexcept
{
    auto* allocator = g_probe_allocator.load();
    std::thread([allocator] {
        (void)allocator->stats();
        g_probe_reentered.store(true);
    }).join();
}
}  // namespace

// Task 1.3: the handler hook observes the failure, with its source, and is
// never invoked under the cache lock.
TEST_F(CudaCachingAllocatorRuntime, CleanupHandlerSeesFailureOutsideCacheLock)
{
    cuda_caching_allocator allocator(0);
    int                    foreign = 0;

    g_handler_calls.store(0);
    auto const previous = memory::cleanup_diagnostic::set_handler(&counting_handler);
    allocator.deallocate_with_stream_lookup(&foreign, 16);
    EXPECT_EQ(1, g_handler_calls.load());
    EXPECT_EQ(
        static_cast<int>(memory::cleanup_source::gpu_cache), g_handler_last_source.load());

    g_probe_allocator.store(&allocator);
    g_probe_reentered.store(false);
    memory::cleanup_diagnostic::set_handler(&reentrant_handler);
    allocator.deallocate_with_stream_lookup(&foreign, 16);
    EXPECT_TRUE(g_probe_reentered.load());

    memory::cleanup_diagnostic::set_handler(previous);
    g_probe_allocator.store(nullptr);
    g_handler_calls.store(0);
    allocator.deallocate_with_stream_lookup(&foreign, 16);
    EXPECT_EQ(0, g_handler_calls.load());  // handler cleared
}

TEST_F(CudaCachingAllocatorRuntime, DoubleFreeThrowsLoggingException)
{
    cuda_caching_allocator allocator(0);
    void*                  p = allocator.allocate(4096);
    allocator.deallocate(p, 4096);
    EXPECT_THROW(allocator.deallocate(p, 4096), logging::exception);
}

// §6.1 "GPU free": event record only for cross-stream uses, 0 heap allocations.
// Plan 3.3: a steady-state cross-stream free takes no device guard, creates no
// event and allocates nothing.
TEST_F(CudaCachingAllocatorRuntime, ProbeGpuCrossStreamFreeHeapAllocations)
{
    SKIP_HEAP_PROBE_UNDER_DEBUG_ITERATORS();
    cuda_caching_allocator allocator(0);
    auto const             cross_stream_pair = [&](size_t streams)
    {
        void* p = allocator.allocate(4096);
        for (size_t s = 1; s <= streams; ++s)
        {
            allocator.record_stream(p, rt::stream(s));
        }
        allocator.deallocate(p, 4096);
    };
    allocator.deallocate(allocator.allocate(4096), 4096);
    cross_stream_pair(4);  // warm: creates the pooled events, grows the queue
    cross_stream_pair(4);
    int const devices_before = rt::get_device_calls;
    int const creates_before = rt::event_create_calls;
    int const records_before = rt::event_record_calls;
    size_t    heap           = 0;
    {
        new_probe probe;
        for (int i = 0; i < 10; ++i)
        {
            cross_stream_pair(4);
        }
        heap = probe.count();
    }
    RecordProperty("cross_stream_free_heap_allocations_per_10", static_cast<int>(heap));
    EXPECT_EQ(0u, heap);
    EXPECT_EQ(devices_before, rt::get_device_calls) << "idle/pooled path must not query the device";
    EXPECT_EQ(creates_before, rt::event_create_calls) << "steady state must reuse pooled events";
    EXPECT_EQ(records_before + 40, rt::event_record_calls) << "one record per cross-stream use";
}

// Plan 3.5: cached bytes are an O(1) atomic read that tracks the free pools
// through allocate, split, free+merge and release.
TEST_F(CudaCachingAllocatorRuntime, CachedBytesTrackFreePoolsWithoutLocking)
{
    cuda_caching_allocator allocator(0);
    EXPECT_EQ(0u, allocator.bytes_cached_now());
    void* p = allocator.allocate(4096);
    EXPECT_EQ(allocator.bytes_reserved_now() - allocator.bytes_allocated_now(), allocator.bytes_cached_now())
        << "the rest of the segment is cached after a split";
    allocator.deallocate(p, 4096);
    EXPECT_EQ(allocator.bytes_reserved_now(), allocator.bytes_cached_now());
    EXPECT_EQ(allocator.stats().bytes_cached.load(), allocator.bytes_cached_now());
    size_t const peak = allocator.peak_bytes_cached_now();
    EXPECT_GE(peak, allocator.bytes_cached_now());
    allocator.empty_cache();
    EXPECT_EQ(0u, allocator.bytes_cached_now());
    EXPECT_EQ(peak, allocator.peak_bytes_cached_now());
    allocator.reset_peak_stats();
    EXPECT_EQ(0u, allocator.peak_bytes_cached_now());
}

// Plan 3.3: polling is bounded per call, but a cache miss forces progress so
// completed events never starve a request that the cache could serve.
TEST_F(CudaCachingAllocatorRuntime, BoundedPollingStillReclaimsUnderPressure)
{
    cuda_caching_allocator allocator(0);
    constexpr int          kBlocks = 40;  // more pending events than one poll examines
    std::vector<void*>     ptrs;
    for (int i = 0; i < kBlocks; ++i)
    {
        ptrs.push_back(allocator.allocate(4096));
    }
    rt::event_ready[1] = false;  // stream(0)'s events are not ready: they stay queued
    for (void* p : ptrs)
    {
        allocator.record_stream(p, rt::stream(0));
        allocator.deallocate(p, 4096);
    }
    int const mallocs = rt::malloc_calls;
    rt::event_ready[1] = true;  // all work completes
    // The first call polls a bounded number of events; the cache still holds
    // enough completed blocks to serve every request without a driver call.
    std::vector<void*> again;
    for (int i = 0; i < kBlocks; ++i)
    {
        again.push_back(allocator.allocate(4096));
    }
    EXPECT_EQ(mallocs, rt::malloc_calls) << "pressure poll must reap completed events before the driver";
    for (void* p : again)
    {
        allocator.deallocate(p, 4096);
    }
}

// Plan 3.3: not-ready events keep their block withheld and later events of the
// same stream are not queried past the first not-ready one.
TEST_F(CudaCachingAllocatorRuntime, NotReadyEventKeepsBlockWithheldUntilComplete)
{
    cuda_caching_allocator allocator(0);
    void*                  p = allocator.allocate(4096);
    allocator.record_stream(p, rt::stream(0));
    rt::event_ready[1] = false;
    allocator.deallocate(p, 4096);
    void* q = allocator.allocate(4096);
    EXPECT_NE(p, q) << "a block with an unfinished cross-stream use must not be reused";
    rt::event_ready[1] = true;
    void* r            = allocator.allocate(4096);
    EXPECT_EQ(p, r) << "the block returns once its event completes";
    allocator.deallocate(q, 4096);
    allocator.deallocate(r, 4096);
}

// ---------------------------------------------------------------------------
// Task 1.7: native-cache rollback is exact at every boundary. Each scenario is
// re-run with a failure injected at the 1st, 2nd, ... call of one boundary
// (operator new, cudaGetDevice, cudaMalloc, event create/record) until the call
// no longer reaches that many. After every injected failure:
//   * accounting matches the driver (bytes_reserved == live driver bytes),
//   * the whole budget is available again (two full segments fit under a cap of
//     exactly two: a leaked pending reservation or an orphaned segment would
//     make the second fail),
//   * releasing the cache returns every byte to the driver (no leaked segment).
// ---------------------------------------------------------------------------
// Arms a failure of the n-th operator new while in scope; fired() reports whether
// the allocation count reached n.
namespace
{
struct new_failure_injection
{
    explicit new_failure_injection(long n)
    {
        g_inject_new_seen  = 0;
        g_inject_new_fired = false;
        g_inject_new_at    = n;
        g_inject_new       = true;
    }
    ~new_failure_injection() { g_inject_new = false; }
    static bool fired() { return g_inject_new_fired.load(); }
};

constexpr double kTwoSegmentFraction(double total)
{
    return 2.0 * static_cast<double>(kSegmentSize) / total;
}

// Budget and driver accounting must be exact after the scenario, whatever happened.
void expect_exact_after_failure(cuda_caching_allocator& allocator, long n)
{
    EXPECT_EQ(allocator.bytes_reserved_now(), rt::device_backing_bytes) << "injection #" << n;
    void* first  = nullptr;
    void* second = nullptr;
    EXPECT_NO_THROW(first = allocator.allocate(kSegmentSize)) << "injection #" << n;
    EXPECT_NO_THROW(second = allocator.allocate(kSegmentSize))
        << "budget not fully restored after injection #" << n;
    if (first != nullptr)
    {
        allocator.deallocate(first, kSegmentSize);
    }
    if (second != nullptr)
    {
        allocator.deallocate(second, kSegmentSize);
    }
    allocator.empty_cache();
    EXPECT_EQ(0u, allocator.bytes_allocated_now()) << "injection #" << n;
    EXPECT_EQ(0u, allocator.bytes_reserved_now()) << "injection #" << n;
    EXPECT_EQ(0u, rt::device_backing_bytes) << "segment leaked after injection #" << n;
}
}  // namespace

TEST_F(CudaCachingAllocatorRuntime, RollbackIsExactAtEveryAllocationBoundaryOfASegmentAlloc)
{
    long fired = 0;
    for (long n = 1; n < 256; ++n)
    {
        rt::reset();
        rt::device_total_bytes = 100 * kSegmentSize;
        cuda_caching_allocator allocator(0);
        allocator.set_memory_fraction(kTwoSegmentFraction(static_cast<double>(rt::device_total_bytes)));

        void* ptr   = nullptr;
        bool  threw = false;
        bool  hit   = false;
        {
            new_failure_injection inject(n);
            try
            {
                ptr = allocator.allocate(kSegmentSize);
            }
            catch (...)
            {
                threw = true;
            }
            hit = inject.fired();
        }
        if (hit)
        {
            ++fired;
            EXPECT_TRUE(threw) << "an injected allocation failure was swallowed (#" << n << ")";
            EXPECT_EQ(nullptr, ptr);
            EXPECT_EQ(0u, allocator.bytes_allocated_now()) << "injection #" << n;
        }
        else
        {
            EXPECT_NE(nullptr, ptr);
            if (ptr != nullptr)
            {
                allocator.deallocate(ptr, kSegmentSize);
            }
        }
        expect_exact_after_failure(allocator, n);
        if (!hit)
        {
            break;
        }
    }
    EXPECT_GE(fired, 2) << "the scenario must reach at least the metadata and registry inserts";
}

TEST_F(CudaCachingAllocatorRuntime, RollbackIsExactAtEveryAllocationBoundaryOfASplit)
{
    long fired = 0;
    for (long n = 1; n < 256; ++n)
    {
        rt::reset();
        rt::device_total_bytes = 100 * kSegmentSize;
        cuda_caching_allocator allocator(0);
        allocator.set_memory_fraction(kTwoSegmentFraction(static_cast<double>(rt::device_total_bytes)));
        allocator.deallocate(allocator.allocate(kSegmentSize), kSegmentSize);  // one cached segment
        int const mallocs = rt::malloc_calls;

        void* ptr   = nullptr;
        bool  threw = false;
        bool  hit   = false;
        {
            new_failure_injection inject(n);
            try
            {
                ptr = allocator.allocate(2 * 1024 * 1024);  // splits the cached 20 MiB block
            }
            catch (...)
            {
                threw = true;
            }
            hit = inject.fired();
        }
        EXPECT_EQ(mallocs, rt::malloc_calls) << "a cached block must serve the split";
        if (hit)
        {
            ++fired;
            EXPECT_TRUE(threw) << "#" << n;
            EXPECT_EQ(0u, allocator.bytes_allocated_now()) << "injection #" << n;
            // The block was put back whole: the full segment is still servable
            // from the cache with no new driver allocation.
            void* whole = allocator.allocate(kSegmentSize);
            EXPECT_EQ(mallocs, rt::malloc_calls) << "block lost or fragmented by injection #" << n;
            allocator.deallocate(whole, kSegmentSize);
        }
        else if (ptr != nullptr)
        {
            allocator.deallocate(ptr, 2 * 1024 * 1024);
        }
        expect_exact_after_failure(allocator, n);
        if (!hit)
        {
            break;
        }
    }
    // Plan 3.8: container nodes are recycled, so on a warmed cache the split's
    // only allocation boundary left is the block metadata. The node-allocation
    // boundaries (live map, free pool) are covered by the cold segment case.
    EXPECT_GE(fired, 1);
}

TEST_F(CudaCachingAllocatorRuntime, FreeFailingBeforeCommitLeavesTheBlockLiveAndRetryable)
{
    long fired = 0;
    for (long n = 1; n < 256; ++n)
    {
        rt::reset();
        cuda_caching_allocator allocator(0);
        void* ptr = allocator.allocate(4096);
        for (size_t s = 1; s <= 4; ++s)  // fill the inline stream set
        {
            allocator.record_stream(ptr, rt::stream(s));
        }

        bool threw = false;
        bool hit   = false;
        {
            new_failure_injection inject(n);
            try
            {
                // A fifth distinct stream hint overflows the inline set (heap).
                allocator.deallocate(ptr, 4096, rt::stream(5));
            }
            catch (...)
            {
                threw = true;
            }
            hit = inject.fired();
        }
        if (hit)
        {
            ++fired;
            EXPECT_TRUE(threw) << "#" << n;
            if (allocator.bytes_allocated_now() == 4096u)
            {
                // Failed before any state change: the block is still live and the
                // free can simply be retried.
                EXPECT_NO_THROW(allocator.deallocate(ptr, 4096, rt::stream(5))) << "#" << n;
            }
            else
            {
                // Failed after commit: the block was quarantined, never recycled.
                EXPECT_EQ(0u, allocator.bytes_allocated_now()) << "#" << n;
                void* other = allocator.allocate(4096);
                EXPECT_NE(ptr, other) << "quarantined block was recycled (#" << n << ")";
                allocator.deallocate(other, 4096);
            }
        }
        if (!hit)
        {
            break;
        }
    }
    EXPECT_GE(fired, 1);
    EXPECT_EQ(0u, rt::device_backing_bytes) << "teardown must return every segment";
}

TEST_F(CudaCachingAllocatorRuntime, RollbackIsExactWhenTheDriverRetriesAfterOom)
{
    rt::device_total_bytes = 100 * kSegmentSize;
    {
        // First cudaMalloc is out of memory, the flush-and-retry succeeds.
        cuda_caching_allocator allocator(0);
        allocator.set_memory_fraction(kTwoSegmentFraction(static_cast<double>(rt::device_total_bytes)));
        rt::fail_malloc_calls = 1;
        void* ptr             = allocator.allocate(kSegmentSize);
        ASSERT_NE(nullptr, ptr);
        EXPECT_EQ(2, rt::malloc_calls);
        EXPECT_EQ(kSegmentSize, allocator.bytes_reserved_now());
        allocator.deallocate(ptr, kSegmentSize);
        expect_exact_after_failure(allocator, 0);
    }
    rt::reset();
    rt::device_total_bytes = 100 * kSegmentSize;
    {
        // Both attempts fail: bad_alloc, and the budget reservation is returned
        // once per attempt (not leaked, not double-released).
        cuda_caching_allocator allocator(0);
        allocator.set_memory_fraction(kTwoSegmentFraction(static_cast<double>(rt::device_total_bytes)));
        rt::fail_malloc_calls = 2;
        EXPECT_THROW(allocator.allocate(kSegmentSize), std::bad_alloc);
        EXPECT_EQ(2, rt::malloc_calls);
        expect_exact_after_failure(allocator, 0);
    }
}

TEST_F(CudaCachingAllocatorRuntime, RollbackIsExactForANonOomDriverError)
{
    rt::device_total_bytes = 100 * kSegmentSize;
    cuda_caching_allocator allocator(0);
    allocator.set_memory_fraction(kTwoSegmentFraction(static_cast<double>(rt::device_total_bytes)));
    rt::malloc_error      = cudaErrorUnknown;
    rt::fail_malloc_calls = 1;
    EXPECT_THROW(allocator.allocate(kSegmentSize), std::runtime_error);
    EXPECT_EQ(1, rt::malloc_calls) << "a non-OOM error is not retried";
    rt::malloc_error = cudaErrorMemoryAllocation;
    expect_exact_after_failure(allocator, 0);
}

TEST_F(CudaCachingAllocatorRuntime, RollbackIsExactAtEveryDeviceActivationOfARetryingAlloc)
{
    long fired = 0;
    for (long n = 1; n < 64; ++n)
    {
        rt::reset();
        rt::device_total_bytes = 100 * kSegmentSize;
        cuda_caching_allocator allocator(0);
        allocator.set_memory_fraction(kTwoSegmentFraction(static_cast<double>(rt::device_total_bytes)));

        rt::get_device_calls       = 0;
        rt::fail_get_device_at_call = static_cast<int>(n);
        rt::fail_malloc_calls      = 1;  // force the flush-and-retry chain
        void* ptr                  = nullptr;
        bool  threw                = false;
        try
        {
            ptr = allocator.allocate(kSegmentSize);
        }
        catch (...)
        {
            threw = true;
        }
        bool const hit             = rt::get_device_calls >= n;
        rt::fail_get_device_at_call = 0;
        rt::fail_malloc_calls      = 0;
        if (hit)
        {
            ++fired;
            EXPECT_TRUE(threw || ptr != nullptr);
            if (ptr != nullptr)
            {
                allocator.deallocate(ptr, kSegmentSize);
            }
        }
        else if (ptr != nullptr)
        {
            allocator.deallocate(ptr, kSegmentSize);
        }
        expect_exact_after_failure(allocator, n);
        if (!hit)
        {
            break;
        }
    }
    EXPECT_GE(fired, 3) << "first attempt, flush and retry each activate the device";
}

TEST_F(CudaCachingAllocatorRuntime, EventAllocationFailureOnCrossStreamFreeQuarantinesWithoutLeak)
{
    for (int fail_create = 0; fail_create <= 1; ++fail_create)
    {
        for (int at = 1; at <= 2; ++at)
        {
            rt::reset();
            {
                cuda_caching_allocator allocator(0);
                void*                  ptr = allocator.allocate(4096);
                allocator.record_stream(ptr, rt::stream(1));
                allocator.record_stream(ptr, rt::stream(2));
                if (fail_create != 0)
                {
                    rt::fail_event_create_at_call = at;
                }
                else
                {
                    rt::fail_event_record_at_call = at;
                }
                EXPECT_THROW(allocator.deallocate(ptr, 4096), std::runtime_error);
                EXPECT_EQ(0u, allocator.bytes_allocated_now());
                void* other = allocator.allocate(4096);
                EXPECT_NE(ptr, other) << "an unproven block must never be recycled";
                allocator.deallocate(other, 4096);
            }
            EXPECT_EQ(0u, rt::device_backing_bytes) << "teardown must free the segment";
            EXPECT_EQ(rt::event_creates, rt::event_destroys) << "event leaked";
        }
    }
}

// Task 7.2: a trace replay sees one allocation id from alloc through free (also
// through a merge) and a different id when the same address is handed out again;
// the requested size is the caller's, not the rounded capacity.
TEST_F(CudaCachingAllocatorRuntime, TraceKeepsOneAllocationIdUntilTheBlockIsReused)
{
    using memory::gpu::gpu_memory_trace_action;
    cuda_caching_allocator allocator(0);
    allocator.record_memory_history(true, 64);

    void* const low  = allocator.allocate(1000);
    void* const high = allocator.allocate(3000);
    allocator.deallocate(low, 1000);
    allocator.deallocate(high, 3000);  // merges with the free neighbour
    void* const again = allocator.allocate(1000);
    allocator.deallocate(again, 1000);

    auto const trace = allocator.snapshot().device_trace;
    std::map<void*, std::vector<std::uint64_t>> allocs;
    std::map<void*, std::vector<std::uint64_t>> frees;
    for (auto const& e : trace)
    {
        if (e.action == gpu_memory_trace_action::alloc)
        {
            EXPECT_NE(0u, e.alloc_id);
            allocs[e.address].push_back(e.alloc_id);
            if (e.address == low)
            {
                EXPECT_EQ(1000u, e.requested_size);
            }
            if (e.address == high)
            {
                EXPECT_EQ(3000u, e.requested_size);
            }
        }
        else if (
            e.action == gpu_memory_trace_action::free_requested ||
            e.action == gpu_memory_trace_action::free_completed)
        {
            frees[e.address].push_back(e.alloc_id);
        }
        else if (e.action == gpu_memory_trace_action::segment_alloc)
        {
            EXPECT_EQ(0u, e.alloc_id) << "a segment is not an allocation";
        }
    }
    ASSERT_EQ(2u, allocs[low].size()) << "low's address was handed out twice";
    EXPECT_NE(allocs[low][0], allocs[low][1]) << "reuse must mint a new id";
    EXPECT_NE(allocs[low][0], allocs[high][0]);
    ASSERT_EQ(4u, frees[low].size());  // requested + completed, twice
    EXPECT_EQ(allocs[low][0], frees[low][0]);
    EXPECT_EQ(allocs[low][0], frees[low][1]);
    EXPECT_EQ(allocs[low][1], frees[low][2]);
    EXPECT_EQ(allocs[low][1], frees[low][3]);
    ASSERT_EQ(2u, frees[high].size());
    EXPECT_EQ(allocs[high][0], frees[high][0]);
    EXPECT_EQ(allocs[high][0], frees[high][1]) << "the merge must not change the freed id";
    allocator.record_memory_history(false);
}

// Tasks 7.1/7.4: reserved == live + pending + reusable + quarantined at every
// step of a replay that splits, defers a cross-stream free and quarantines a
// block, and the waste fields match their definitions.
namespace
{
void expect_backing_equation(cuda_caching_allocator const& a)
{
    auto const   s   = a.stats();
    size_t const sum = s.bytes_allocated + s.bytes_pending + s.bytes_cached + s.bytes_quarantined;
    EXPECT_EQ(s.bytes_reserved, sum + s.bytes_unaccounted);
    EXPECT_EQ(0u, s.bytes_unaccounted);
}
}  // namespace

TEST_F(CudaCachingAllocatorRuntime, BackingEquationReconcilesThroughSplitPendingAndQuarantine)
{
    cuda_caching_allocator allocator(0);
    expect_backing_equation(allocator);

    void* a = allocator.allocate(1000);  // splits a fresh segment
    void* b = allocator.allocate(5000);
    expect_backing_equation(allocator);
    {
        auto const s = allocator.stats();
        EXPECT_EQ(1000u + 5000u, s.bytes_requested);
        EXPECT_EQ(s.bytes_allocated - 6000u, s.internal_waste_bytes());
        EXPECT_GT(s.inactive_split_bytes, 0u);
        EXPECT_EQ(s.bytes_cached, s.largest_cached_block)
            << "one free tail: the largest reusable block is all of the cache";
    }

    // Pending: a cross-stream use whose event has not completed.
    allocator.record_stream(a, rt::stream(0));
    rt::event_ready[1] = false;
    allocator.deallocate(a, 1000);
    expect_backing_equation(allocator);
    EXPECT_GT(allocator.stats().bytes_pending, 0u);

    // Quarantine: event creation fails for a second cross-stream block.
    allocator.record_stream(b, rt::stream(1));
    allocator.record_stream(b, rt::stream(2));
    rt::fail_event_record_at_call = 2;
    EXPECT_THROW(allocator.deallocate(b, 5000), std::runtime_error);
    expect_backing_equation(allocator);
    EXPECT_GT(allocator.stats().bytes_quarantined, 0u);

    rt::event_ready[1] = true;
    void* c = allocator.allocate(1000);
    expect_backing_equation(allocator);
    allocator.deallocate(c, 1000);
    expect_backing_equation(allocator);
    EXPECT_EQ(0u, allocator.stats().bytes_pending);
}
