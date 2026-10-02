// Tasks 0.2/0.3: GPU caching-allocator host overhead under the deterministic fake
// runtime (no GPU). Latency percentiles plus per-op counters (heap allocations,
// driver calls) for plan §6.1.
//
//   Phase0GpuShimBench --out gpu_shim.json --repo <source dir> [--quick]
//
// Evidence level: deterministic shim. Host-side cost only; says nothing about
// device behaviour, and the fake driver calls are free so driver cost is excluded.

#include <atomic>
#include <cstdlib>
#include <new>
#include <string>
#include <vector>

#include "../Cxx/phase0_harness.h"
#include "fake_runtime.h"
#include "gpu/cuda_caching_allocator.h"

namespace
{
std::atomic<bool>   g_count{false};
std::atomic<size_t> g_news{0};
}  // namespace

void* operator new(std::size_t n)
{
    if (g_count.load(std::memory_order_relaxed))
    {
        g_news.fetch_add(1, std::memory_order_relaxed);
    }
    if (void* p = std::malloc(n ? n : 1))
    {
        return p;
    }
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

namespace rt = fake_runtime;
using memory::gpu::cuda_caching_allocator;

namespace
{
struct snap
{
    double heap, malloc_calls, get_device, ev_create, ev_record;
};
snap take()
{
    return {static_cast<double>(g_news.load()),
            static_cast<double>(rt::malloc_calls),
            static_cast<double>(rt::get_device_calls),
            static_cast<double>(rt::event_creates),
            static_cast<double>(rt::event_record_calls)};
}

phase0::result run(
    std::string                                        name,
    std::vector<std::pair<std::string, std::string>>   params,
    std::function<void()> const&                       op,
    bool                                               quick,
    double                                             res)
{
    phase0::result r;
    r.name   = std::move(name);
    r.params = std::move(params);
    size_t const warmup  = 200;
    size_t const samples = quick ? 40 : 400;
    size_t const batch   = phase0::pick_batch(res, op);
    r.batch              = batch;
    g_news               = 0;
    g_count              = true;
    snap const a         = take();
    r.samples_ns_per_op  = phase0::measure(warmup, samples, batch, op);
    snap const b         = take();
    g_count              = false;
    double const ops     = static_cast<double>(warmup + samples * batch);
    r.counters           = {{"heap_allocs_per_op", (b.heap - a.heap) / ops},
                            {"driver_malloc_per_op", (b.malloc_calls - a.malloc_calls) / ops},
                            {"cudaGetDevice_per_op", (b.get_device - a.get_device) / ops},
                            {"event_create_per_op", (b.ev_create - a.ev_create) / ops},
                            {"event_record_per_op", (b.ev_record - a.ev_record) / ops}};
    return r;
}
}  // namespace

static int run_main(int argc, char** argv)
{
    bool const  quick = phase0::has_flag(argc, argv, "--quick");
    std::string out   = phase0::arg_value(argc, argv, "--out", "gpu_shim.json");
    phase0::report rep("gpu_cache_shim_host_overhead",
                       "deterministic runtime shim (HIP labels); no GPU; host cost only",
                       phase0::arg_value(argc, argv, "--repo", "."));
    rep.set_build(phase0::compiler_string(), phase0::kNdebug);
    rep.set_evidence("hip", "shim");
    rep.note("fake_driver", "cudaMalloc/cudaFree are malloc/free; events never block; driver latency excluded");
    rep.note("quick", quick ? "yes (reduced samples; not baseline-grade)" : "no");
    rep.note("single_thread", "lock contention is not measured here (task 8.1)");

    double const res = phase0::timer_resolution_ns();

    for (size_t size : {size_t{512}, size_t{4096}, size_t{1} << 20})
    {
        rt::reset();
        cuda_caching_allocator a(0);
        a.deallocate(a.allocate(size), size);  // warm: segment cached
        rep.add(run(
            "warm_alloc_free",
            {{"size_bytes", std::to_string(size)}, {"workload", "allocate+deallocate, block cached"}},
            [&] {
                void* p = a.allocate(size);
                a.deallocate(p, size);
            },
            quick, res));
    }

    {
        rt::reset();
        cuda_caching_allocator a(0);
        a.deallocate(a.allocate(4096), 4096);
        rep.add(run(
            "split_merge",
            {{"size_bytes", "512"}, {"workload", "two 512 B allocs split from a cached segment, freed (merge)"}},
            [&] {
                void* x = a.allocate(512);
                void* y = a.allocate(512);
                a.deallocate(x, 512);
                a.deallocate(y, 512);
            },
            quick, res));
    }

    for (size_t streams : {size_t{1}, size_t{4}, size_t{7}})  // fake_runtime has 8 event slots
    {
        rt::reset();
        cuda_caching_allocator a(0);
        a.deallocate(a.allocate(4096), 4096);
        rep.add(run(
            "alloc_record_streams_free",
            {{"size_bytes", "4096"},
             {"cross_streams", std::to_string(streams)},
             {"workload", "allocate, record_stream on N other streams, deallocate (events fake-ready)"}},
            [&] {
                void* p = a.allocate(4096);
                for (size_t s = 1; s <= streams; ++s)
                {
                    a.record_stream(p, rt::stream(s - 1));
                }
                a.deallocate(p, 4096);
            },
            quick, res));
    }

    {
        rt::reset();
        cuda_caching_allocator a(0);
        void* p = a.allocate(4096);
        size_t sink = 0;
        rep.add(run(
            "bytes_allocated_now",
            {{"workload", "memory_allocated() basic stat read"}},
            [&] { sink += a.bytes_allocated_now(); },
            quick, res));
        a.deallocate(p, 4096);
        if (sink == 0)
        {
            std::fprintf(stderr, "unexpected\n");
        }
    }

    return rep.write(out) ? 0 : 1;
}

int main(int argc, char** argv)
{
    try
    {
        return run_main(argc, argv);
    }
    catch (std::exception const& e)
    {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        return 2;
    }
}
