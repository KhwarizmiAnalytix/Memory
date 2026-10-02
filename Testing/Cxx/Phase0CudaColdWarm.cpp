// Task 0.4: one cold/warm series on real CUDA/HIP hardware with a full manifest.
// Replaces ad-hoc reading of the Google Benchmark JSON. Timing boundaries are
// explicit: cold = empty_cache() + allocate + deallocate, each timed separately so
// driver cost is attributed to the call that incurs it. Raw samples are recorded.
//
//   Phase0CudaColdWarm --out cuda_cold_warm.json --repo <source dir> [--quick]
//
// Evidence level: real hardware, single host thread, one stream. Not a contention,
// multi-stream or delayed-consumer measurement (tasks 8.1/8.2).

#include "phase0_harness.h"

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP

#include <cstddef>
#include <cstdio>
#include <string>

#include "gpu/cuda_caching_allocator.h"
#include "gpu/device_guard.h"
#include "gpu/gpu_runtime.h"

namespace
{
using memory::gpu::cuda_caching_allocator;

void sync_device() { (void)cudaDeviceSynchronize(); }

phase0::result series(
    std::string                                       name,
    std::size_t                                       size,
    std::string                                       workload,
    std::function<void()> const&                      op,
    size_t                                            warmup,
    size_t                                            samples,
    double                                            res_ns,
    bool                                              allow_batch)
{
    phase0::result r;
    r.name   = std::move(name);
    r.params = {{"size_bytes", std::to_string(size)}, {"workload", std::move(workload)}};
    r.batch  = allow_batch ? phase0::pick_batch(res_ns, op) : 1;
    r.samples_ns_per_op = phase0::measure(warmup, samples, r.batch, op);
    return r;
}
}  // namespace

int main(int argc, char** argv)
{
    bool const  quick = phase0::has_flag(argc, argv, "--quick");
    std::string out   = phase0::arg_value(argc, argv, "--out", "cuda_cold_warm.json");

    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
    {
        std::fprintf(stderr, "no GPU device: this baseline needs hardware\n");
        return 3;
    }
    cudaDeviceProp prop{};
    (void)cudaGetDeviceProperties(&prop, 0);
    int rt_ver = 0, drv_ver = 0;
    (void)cudaRuntimeGetVersion(&rt_ver);
    (void)cudaDriverGetVersion(&drv_ver);

    phase0::report rep("cuda_cold_warm", "real GPU hardware; host timing of allocator calls",
                       phase0::arg_value(argc, argv, "--repo", "."));
    rep.set_build(phase0::compiler_string(), phase0::kNdebug);
    rep.set_evidence("cuda", "hardware");
    rep.note("gpu", prop.name);
    rep.note("cuda_runtime_version", std::to_string(rt_ver));
    rep.note("cuda_driver_version", std::to_string(drv_ver));
    rep.note("quick", quick ? "yes (reduced samples; not baseline-grade)" : "no");
    rep.note("scope", "single host thread, null stream; cold = empty_cache+alloc+free timed separately");
    rep.note("clocks", "GPU/CPU clocks and power state not controlled; repeat before drawing conclusions");

    double const              res   = phase0::timer_resolution_ns();
    size_t const              warm_samples = quick ? 40 : 500;
    size_t const              cold_samples = quick ? 10 : 100;
    std::size_t const sizes[]              = {4096, 65536, 1u << 20, 4u << 20, 32u << 20};

    for (std::size_t size : sizes)
    {
        // Direct driver pair (the "raw" comparator: same size, same thread).
        rep.add(series(
            "raw_driver_malloc_free", size, "cudaMalloc + cudaFree pair",
            [&] {
                void* p = nullptr;
                memory::gpu::throw_on_cuda_error(cudaMalloc(&p, size), "cudaMalloc");
                memory::gpu::throw_on_cuda_error(cudaFree(p), "cudaFree");
            },
            5, cold_samples, res, false));

        // Warm: same-size cache hit after the first iteration.
        {
            cuda_caching_allocator a(0);
            a.deallocate(a.allocate(size), size);
            rep.add(series(
                "cache_warm_alloc_free", size, "allocate + deallocate, block cached",
                [&] {
                    void* p = a.allocate(size);
                    a.deallocate(p, size);
                },
                500, warm_samples, res, true));
        }

        // Cold, attributed per call.
        {
            cuda_caching_allocator a(0);
            std::vector<double> empty_s, alloc_s, free_s;
            for (size_t i = 0; i < 5 + cold_samples; ++i)
            {
                double const t0 = phase0::now_ns();
                a.empty_cache();
                double const t1 = phase0::now_ns();
                void*        p  = a.allocate(size);
                double const t2 = phase0::now_ns();
                a.deallocate(p, size);
                double const t3 = phase0::now_ns();
                if (i >= 5)
                {
                    empty_s.push_back(t1 - t0);
                    alloc_s.push_back(t2 - t1);
                    free_s.push_back(t3 - t2);
                }
            }
            for (auto const& [nm, v] : {std::pair<char const*, std::vector<double>*>{"cache_cold_empty_cache", &empty_s},
                                        {"cache_cold_allocate", &alloc_s},
                                        {"cache_cold_deallocate", &free_s}})
            {
                phase0::result r;
                r.name              = nm;
                r.params            = {{"size_bytes", std::to_string(size)},
                                       {"workload", "cold: empty_cache, allocate, deallocate timed separately"}};
                r.samples_ns_per_op = *v;
                rep.add(std::move(r));
            }
        }
        sync_device();
        std::fprintf(stderr, "size %zu done\n", size);
    }
    return rep.write(out) ? 0 : 1;
}

#else
int main()
{
    return 3;
}
#endif
