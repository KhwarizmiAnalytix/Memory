// Task 0.1: CPU allocation microbenchmarks with raw samples and p50/p95/p99.
// Facade vs raw backend, sizes 16 B - 64 MiB, 1/2/8/32 threads, cross-thread free.
// Not a Google Benchmark target: percentiles need raw samples (plan §6.7).
//
//   Phase0CpuBaseline --out cpu_baseline.json --repo <source dir> [--quick]

#include <atomic>
#include <cstdlib>
#include <new>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)
#include <malloc.h>
#endif

#include "common/data_ptr.h"
#include "helper/memory_allocator.h"
#include "phase0_harness.h"


namespace
{
std::atomic<bool>   g_count_new{false};
std::atomic<size_t> g_new_calls{0};
}  // namespace

void* operator new(std::size_t n)
{
    if (g_count_new.load(std::memory_order_relaxed))
    {
        g_new_calls.fetch_add(1, std::memory_order_relaxed);
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
namespace memory_allocator = memory::cpu::memory_allocator;
constexpr std::size_t kAlign = 64;

struct backend
{
    std::string                              name;
    std::function<void*(std::size_t)>        alloc;
    std::function<void(void*, std::size_t)>  free;
};

std::vector<backend> make_backends()
{
    std::vector<backend> b;
    b.push_back({"platform_aligned",
                 [](std::size_t n) {
#if defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)
                     return _aligned_malloc(n, kAlign);
#else
                     void* p = nullptr;
                     return posix_memalign(&p, kAlign, n) == 0 ? p : nullptr;
#endif
                 },
                 [](void* p, std::size_t) {
#if defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)
                     _aligned_free(p);
#else
                     std::free(p);
#endif
                 }});
#if MEMORY_HAS_MIMALLOC
    b.push_back({"mimalloc_raw",
                 [](std::size_t n) { return memory_allocator::allocate_mi(n, kAlign); },
                 [](void* p, std::size_t n) { memory_allocator::free_mi(p, n); }});
#endif
#if MEMORY_HAS_TBB
    b.push_back({"tbb_raw",
                 [](std::size_t n) { return memory_allocator::allocate_tbb(n, kAlign); },
                 [](void* p, std::size_t n) { memory_allocator::free_tbb(p, n); }});
#endif
    b.push_back({"memory_allocator_facade",
                 [](std::size_t n) { return memory_allocator::allocate(n, kAlign); },
                 [](void* p, std::size_t) { memory_allocator::free(p); }});
    return b;
}

// Touch one byte per page so the cost includes first-touch, as real users pay it.
inline void touch(void* p, std::size_t n)
{
    auto* c = static_cast<volatile char*>(p);
    for (std::size_t i = 0; i < n; i += 4096)
    {
        c[i] = 1;
    }
    c[n - 1] = 1;
}

struct plan_t
{
    size_t warmup, samples;
};
plan_t plan_for(std::size_t size, bool quick)
{
    plan_t p{};
    if (size <= 4096)        p = {2000, 400};
    else if (size <= 262144) p = {200, 200};
    else if (size <= (1u << 20) * 4) p = {10, 60};
    else                     p = {2, 20};
    if (quick)
    {
        p.warmup  = std::max<size_t>(p.warmup / 10, 1);
        p.samples = std::max<size_t>(p.samples / 10, 5);
    }
    return p;
}

// N threads each time their own alloc/free pairs; samples are pooled.
phase0::result run_pairs(
    backend const& be, std::size_t size, unsigned threads, bool quick, double res_ns, bool as_data_ptr)
{
    phase0::result r;
    r.name = as_data_ptr ? "data_ptr_facade" : be.name;
    r.params = {{"size_bytes", std::to_string(size)},
                {"threads", std::to_string(threads)},
                {"workload", "alloc+touch+free pair"},
                {"alignment", std::to_string(kAlign)}};
    auto const pl = plan_for(size, quick);

    auto op_for = [&](void) -> std::function<void()> {
        if (as_data_ptr)
        {
            return [size] {
                memory::data_ptr<std::uint8_t> d(size, memory::device_enum::CPU, 0);
                touch(d.data(), size);
            };
        }
        return [&be, size] {
            void* p = be.alloc(size);
            touch(p, size);
            be.free(p, size);
        };
    };

    size_t const batch = size <= 4096 ? phase0::pick_batch(res_ns, op_for()) : 1;
    r.batch            = batch;

    std::vector<std::vector<double>> per(threads);
    std::atomic<unsigned>            ready{0};
    std::atomic<bool>                go{false};
    std::vector<std::thread>         ts;
    for (unsigned t = 0; t < threads; ++t)
    {
        ts.emplace_back([&, t] {
            auto op = op_for();
            ready.fetch_add(1);
            while (!go.load())
            {
                std::this_thread::yield();
            }
            per[t] = phase0::measure(pl.warmup, pl.samples, batch, op);
        });
    }
    while (ready.load() < threads)
    {
        std::this_thread::yield();
    }
    go.store(true);
    for (auto& t : ts)
    {
        t.join();
    }
    for (auto& v : per)
    {
        r.samples_ns_per_op.insert(r.samples_ns_per_op.end(), v.begin(), v.end());
    }
    return r;
}

// Producer threads allocate; a distinct consumer thread frees. Measures the free.
phase0::result run_cross_thread(backend const& be, std::size_t size, unsigned pairs, bool quick)
{
    phase0::result r;
    r.name   = be.name;
    r.params = {{"size_bytes", std::to_string(size)},
                {"threads", std::to_string(pairs * 2)},
                {"workload", "cross-thread free (producer allocs, consumer frees)"},
                {"alignment", std::to_string(kAlign)}};
    auto const   pl     = plan_for(size, quick);
    size_t const count  = pl.samples * 16;  // pointers per producer
    size_t const batch  = 16;
    r.batch             = batch;

    std::vector<std::vector<double>> per(pairs);
    std::vector<std::thread>         ts;
    for (unsigned k = 0; k < pairs; ++k)
    {
        ts.emplace_back([&, k] {
            std::vector<void*> ptrs(count);
            std::thread        producer([&] {
                for (auto& p : ptrs)
                {
                    p = be.alloc(size);
                    touch(p, size);
                }
            });
            producer.join();  // allocation thread has exited: free happens elsewhere
            for (size_t s = 0; s < pl.samples; ++s)
            {
                double const t0 = phase0::now_ns();
                for (size_t i = 0; i < batch; ++i)
                {
                    be.free(ptrs[s * batch + i], size);
                }
                per[k].push_back((phase0::now_ns() - t0) / static_cast<double>(batch));
            }
        });
    }
    for (auto& t : ts)
    {
        t.join();
    }
    for (auto& v : per)
    {
        r.samples_ns_per_op.insert(r.samples_ns_per_op.end(), v.begin(), v.end());
    }
    return r;
}

// §6.1 CPU row: heap allocations per allocate/free pair (0 expected). Syscalls and
// Memory-lock counts are not observable without OS/production instrumentation.
void add_heap_probe(phase0::report& rep)
{
    constexpr size_t kOps = 1000;
    auto probe = [&](std::string name, std::function<void()> const& op) {
        for (int i = 0; i < 50; ++i) op();
        g_new_calls = 0;
        g_count_new = true;
        for (size_t i = 0; i < kOps; ++i) op();
        g_count_new = false;
        phase0::result r;
        r.name   = std::move(name);
        r.params = {{"workload", "heap allocations per alloc/free pair (probe, not timed)"},
                    {"size_bytes", "256"}};
        r.counters = {{"heap_allocs_per_op", static_cast<double>(g_new_calls.load()) / kOps}};
        rep.add(std::move(r));
    };
    probe("probe_memory_allocator_facade", [] {
        void* p = memory_allocator::allocate(256, kAlign);
        memory_allocator::free(p);
    });
    probe("probe_data_ptr_facade", [] {
        memory::data_ptr<std::uint8_t> d(256, memory::device_enum::CPU, 0);
        (void)d.data();
    });
}
}  // namespace

int main(int argc, char** argv)
{
    bool const  quick = phase0::has_flag(argc, argv, "--quick");
    std::string out   = phase0::arg_value(argc, argv, "--out", "cpu_baseline.json");

    phase0::report rep(
        "cpu_baseline", "real execution (CPU), no GPU involved", phase0::arg_value(argc, argv, "--repo", "."));
    rep.set_build(phase0::compiler_string(), phase0::kNdebug);
    rep.set_evidence("cpu", "hardware");
    rep.note("mimalloc", MEMORY_HAS_MIMALLOC ? "built" : "not built");
    rep.note("tbb", MEMORY_HAS_TBB ? "built" : "not built: tbb_raw cases absent (held)");
    rep.note("numa", "off in this build; H6 syscall cost not measured");
    rep.note("quick", quick ? "yes (reduced samples; not baseline-grade)" : "no");
    rep.note("memory_cap", "total live bytes per case capped at 1 GiB; larger thread/size combos are skipped");

    auto const backends = make_backends();
    double const res    = [] { return phase0::timer_resolution_ns(); }();
    std::vector<std::size_t> const sizes = {
        16, 64, 256, 1u << 10, 4u << 10, 16u << 10, 64u << 10, 256u << 10,
        1u << 20, 4u << 20, 16u << 20, 64u << 20};
    std::vector<unsigned> const thread_counts = {1, 2, 8, 32};
    constexpr std::size_t       kCap          = 1ull << 30;

    for (auto size : sizes)
    {
        for (auto threads : thread_counts)
        {
            if (static_cast<std::size_t>(threads) * size > kCap)
            {
                phase0::result r;
                r.name   = "all_backends";
                r.params = {{"size_bytes", std::to_string(size)}, {"threads", std::to_string(threads)}};
                r.status = "skipped: threads*size exceeds 1 GiB live-bytes cap";
                rep.add(std::move(r));
                continue;
            }
            for (auto const& be : backends)
            {
                rep.add(run_pairs(be, size, threads, quick, res, false));
            }
            rep.add(run_pairs(backends.back(), size, threads, quick, res, true));
        }
        for (unsigned pairs : {1u, 4u})
        {
            if (static_cast<std::size_t>(pairs) * size * (plan_for(size, quick).samples * 16) > kCap)
            {
                continue;
            }
            for (auto const& be : backends)
            {
                rep.add(run_cross_thread(be, size, pairs, quick));
            }
        }
        std::fprintf(stderr, "size %zu done\n", size);
    }
    add_heap_probe(rep);
    return rep.write(out) ? 0 : 1;
}
