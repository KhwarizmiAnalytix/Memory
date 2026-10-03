/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

// Task 1.10: host-side churn replay of the Appendix B reproduction against the
// deterministic fake runtime. The documented crash is a real-hardware, real-driver
// observation; this harness removes the driver and GPU from the picture so the
// cache's own bookkeeping (block lists, pools, free list, event queues) can be run
// under AddressSanitizer / UBSan at volumes the hardware benchmark uses.
//
// Scenarios (predeclared, fixed here; no scenario is chosen after seeing results):
//   warm   : the warm_stress path -- allocate(size)/deallocate on one stream,
//            sizes 4096, 32768, 262144, 2097152, 4194304, N iterations each
//   cold   : empty_cache() + allocate + deallocate, the cold_alloc_free path
//   cycle  : create/destroy many allocator instances with a little activity each
//   mixed  : seeded random sizes, up to 64 live blocks, cross-stream uses with
//            random event completion, periodic empty_cache()
//
// Usage: MemoryGpuShimChurn [--iterations N] [--seed S] [--scenarios warm,cold,cycle,mixed]
// Exit code 0 only if every invariant held; the final line prints a summary.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "fake_runtime.h"
#include "gpu/cuda_caching_allocator.h"

using memory::gpu::cuda_caching_allocator;
namespace rt = fake_runtime;

namespace
{
int g_failures = 0;

void check(bool ok, char const* what, long long a = 0, long long b = 0)
{
    if (!ok)
    {
        ++g_failures;
        std::fprintf(stderr, "INVARIANT VIOLATED: %s (%lld vs %lld)\n", what, a, b);
    }
}

struct xorshift
{
    explicit xorshift(std::uint64_t seed) : s(seed != 0 ? seed : 0x9E3779B97F4A7C15ULL) {}
    std::uint64_t next()
    {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
    std::uint64_t below(std::uint64_t n) { return next() % n; }
    std::uint64_t s;
};

constexpr std::size_t kSizes[] = {4096, 32768, 262144, 2097152, 4194304};

void expect_clean_teardown(cuda_caching_allocator& allocator, char const* scenario)
{
    allocator.empty_cache();
    check(allocator.bytes_allocated_now() == 0, scenario, static_cast<long long>(allocator.bytes_allocated_now()), 0);
    check(allocator.bytes_reserved_now() == 0, scenario, static_cast<long long>(allocator.bytes_reserved_now()), 0);
    check(rt::device_backing_bytes == 0, "driver bytes leaked", static_cast<long long>(rt::device_backing_bytes), 0);
}

void run_warm(long long iterations)
{
    for (std::size_t size : kSizes)
    {
        cuda_caching_allocator allocator(0);
        for (long long i = 0; i < iterations; ++i)
        {
            void* p = allocator.allocate(size);
            allocator.deallocate(p, size);
        }
        expect_clean_teardown(allocator, "warm");
    }
}

void run_cold(long long iterations)
{
    for (std::size_t size : kSizes)
    {
        cuda_caching_allocator allocator(0);
        for (long long i = 0; i < iterations; ++i)
        {
            allocator.empty_cache();
            void* p = allocator.allocate(size);
            allocator.deallocate(p, size);
        }
        expect_clean_teardown(allocator, "cold");
    }
}

void run_cycle(long long iterations)
{
    for (long long i = 0; i < iterations; ++i)
    {
        cuda_caching_allocator allocator(0);
        std::size_t const      size = kSizes[static_cast<std::size_t>(i) % 5];
        void*                  a    = allocator.allocate(size);
        void*                  b    = allocator.allocate(size / 2 + 512);
        allocator.deallocate(a, size);
        allocator.deallocate(b, size / 2 + 512);
        // Destructor runs with a populated cache and no explicit empty_cache().
    }
    check(rt::device_backing_bytes == 0, "cycle: driver bytes leaked", static_cast<long long>(rt::device_backing_bytes), 0);
}

void run_mixed(long long iterations, std::uint64_t seed)
{
    xorshift                                   rng(seed);
    cuda_caching_allocator                     allocator(0);
    struct live_block
    {
        void*       ptr;
        std::size_t size;
    };
    std::vector<live_block> live;
    for (long long i = 0; i < iterations; ++i)
    {
        std::uint64_t const action = rng.below(100);
        if (live.size() < 64 && (action < 55 || live.empty()))
        {
            std::size_t const size = 512 + static_cast<std::size_t>(rng.below(8u * 1024 * 1024));
            void*             p    = allocator.allocate(size);
            if (rng.below(4) == 0)
            {
                std::uint64_t const streams = 1 + rng.below(3);
                for (std::uint64_t s = 0; s < streams; ++s)
                {
                    allocator.record_stream(p, rt::stream(rng.below(7)));
                }
            }
            live.push_back({p, size});
        }
        else if (action < 97)
        {
            std::size_t const at = static_cast<std::size_t>(rng.below(live.size()));
            allocator.deallocate(live[at].ptr, live[at].size);
            live[at] = live.back();
            live.pop_back();
        }
        else
        {
            for (bool& ready : rt::event_ready)
            {
                ready = rng.below(2) == 0;
            }
            if (rng.below(8) == 0)
            {
                allocator.empty_cache();
            }
        }
    }
    for (live_block const& b : live)
    {
        allocator.deallocate(b.ptr, b.size);
    }
    rt::event_ready.fill(true);
    expect_clean_teardown(allocator, "mixed");
}

bool wants(std::string const& list, char const* name)
{
    return list.find(name) != std::string::npos;
}
}  // namespace

int main(int argc, char** argv)
{
    long long     iterations = 200000;
    std::uint64_t seed       = 1;
    std::string   scenarios  = "warm,cold,cycle,mixed";
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--iterations") == 0 && i + 1 < argc)
        {
            iterations = std::atoll(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
        {
            seed = static_cast<std::uint64_t>(std::atoll(argv[++i]));
        }
        else if (std::strcmp(argv[i], "--scenarios") == 0 && i + 1 < argc)
        {
            scenarios = argv[++i];
        }
    }
    rt::reset();
    rt::device_total_bytes = 1ULL << 40;  // no budget pressure: this is a bookkeeping test
    try
    {
        if (wants(scenarios, "warm"))
        {
            run_warm(iterations);
        }
        if (wants(scenarios, "cold"))
        {
            run_cold(iterations / 40 + 1);
        }
        if (wants(scenarios, "cycle"))
        {
            run_cycle(iterations / 100 + 1);
        }
        if (wants(scenarios, "mixed"))
        {
            run_mixed(iterations, seed);
        }
    }
    catch (std::exception const& e)
    {
        std::fprintf(stderr, "unexpected exception: %s\n", e.what());
        ++g_failures;
    }
    check(rt::event_creates == rt::event_destroys, "event leaked", rt::event_creates, rt::event_destroys);
    std::printf(
        "churn shim: iterations=%lld seed=%llu scenarios=%s failures=%d\n",
        iterations,
        static_cast<unsigned long long>(seed),
        scenarios.c_str(),
        g_failures);
    return g_failures == 0 ? 0 : 1;
}
