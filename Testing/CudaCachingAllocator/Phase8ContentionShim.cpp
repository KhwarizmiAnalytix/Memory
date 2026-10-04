// Task 8.1: GPU caching-allocator host contention (1-32 threads) and a seeded
// mixed-lifetime replay, under a thread-safe fake driver (no GPU).
//
//   Phase8ContentionShim --out contention.json --repo <source dir> [--quick] [--reps N]
//
// Two workloads at 1,2,4,8,16,32 threads, one fresh allocator per run:
//   warm_alloc_free  every thread allocates+frees 512 B on the default stream (block cached)
//   mixed_replay     per-thread seeded xorshift: 32 live slots, log-uniform sizes
//                    512 B..1 MiB, replace a random slot (free + allocate), 8 % of frees are
//                    preceded by record_stream on a neighbour thread's stream
// Per-thread op sequences are deterministic (seed = 1000 + thread id); the interleaving
// across threads is not.
//
// When built with lock_wait_mutex.h force-included (target ...Lockwait) the allocator's
// std::recursive_mutex records acquisitions, contended acquisitions and wait time.
//
// Evidence level: deterministic shim, host cost only. The fake driver is free, so this
// bounds the allocator's own serialization, not driver or device behaviour. Latency here
// is not evidence about fragmentation (plan 8.1 exit).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "../Cxx/phase0_harness.h"
#include "mt/fake_runtime.h"  // explicit: a bare name would resolve to the single-threaded fake next to this file
#include "gpu/cuda_caching_allocator.h"

#if defined(BENCH_LOCK_WAIT)
#    define LOCK_STATS_ENABLED 1
#else
#    define LOCK_STATS_ENABLED 0
#endif

namespace rt = fake_runtime;
using memory::gpu::cuda_caching_allocator;

namespace
{
struct xorshift
{
    std::uint64_t s;
    std::uint64_t next()
    {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
};

struct lock_counts
{
    std::uint64_t acquisitions{0}, contended{0}, wait_ns{0}, outer{0}, hold_cycles{0};
};

#if LOCK_STATS_ENABLED
double g_cycles_per_ns = 1.0;  // TSC rate, calibrated in run_main
double g_hold_floor_ns = 0.0;  // hold time of an empty lock/unlock (the two TSC reads)
#endif

struct worker_out
{
    std::vector<double> samples;
    double              end_ns{0};
    lock_counts         locks;
};

constexpr size_t kBatch = 256;

lock_counts thread_locks()
{
#if LOCK_STATS_ENABLED
    auto const& s = std::g_bench_lock_stats;
    return {s.acquisitions, s.contended, s.wait_ns, s.outer_acquisitions, s.hold_cycles};
#else
    return {};
#endif
}

// One operation = one allocate + one deallocate pair (mixed: free old slot, allocate new).
template <class Body>
void run_worker(std::atomic<bool> const& go, size_t ops, Body&& body, worker_out& out)
{
    while (!go.load(std::memory_order_acquire))
    {
    }
    lock_counts const l0 = thread_locks();
    size_t            done = 0;
    while (done < ops)
    {
        size_t const n  = std::min(kBatch, ops - done);
        double const t0 = phase0::now_ns();
        for (size_t i = 0; i < n; ++i)
        {
            body();
        }
        out.samples.push_back((phase0::now_ns() - t0) / static_cast<double>(n));
        done += n;
    }
    out.end_ns = phase0::now_ns();
    lock_counts const l1 = thread_locks();
    out.locks = {l1.acquisitions - l0.acquisitions,
                 l1.contended - l0.contended,
                 l1.wait_ns - l0.wait_ns,
                 l1.outer - l0.outer,
                 l1.hold_cycles - l0.hold_cycles};
}

enum class workload
{
    warm,
    mixed
};

// `arenas` independent allocator instances; thread t uses instance t % arenas. With 1 this is
// the shared allocator measured since 8.1. With more it is an upper bound for an arena design
// (task 8.7-C): every free returns to the instance that allocated it, so there is no cross-arena
// free, no shared budget and no shared stats here, which a real design would have to pay for.
phase0::result run_config(workload w, size_t threads, size_t ops_per_thread, size_t rep, size_t arenas = 1)
{
    rt::reset();
    std::vector<std::unique_ptr<cuda_caching_allocator>> instances;
    for (size_t i = 0; i < arenas; ++i)
    {
        instances.push_back(std::make_unique<cuda_caching_allocator>(0));
    }

    std::atomic<bool>        go{false};
    std::vector<worker_out>  outs(threads);
    std::vector<std::thread> pool;
    std::atomic<size_t>      ready{0};
    // Counters are snapshotted after workers finish their private warm-up (below).
    for (size_t t = 0; t < threads; ++t)
    {
        pool.emplace_back([&, t] {
            cuda_caching_allocator& a = *instances[t % instances.size()];
            if (w == workload::warm)
            {
                for (int i = 0; i < 2000; ++i)
                {
                    a.deallocate(a.allocate(512), 512);
                }
                ready.fetch_add(1);
                run_worker(go, ops_per_thread, [&] { a.deallocate(a.allocate(512), 512); }, outs[t]);
            }
            else
            {
                struct slot
                {
                    void*  p{nullptr};
                    size_t n{0};
                };
                std::vector<slot> slots(32);
                xorshift          rng{1000 + t};
                auto const        mine      = rt::stream(t);
                auto const        neighbour = rt::stream((t + 1) % threads);
                auto              size_for  = [&] {
                    // log-uniform 512 B .. 1 MiB: 2^9 .. 2^20
                    size_t const shift = 9 + rng.next() % 12;
                    return (size_t{1} << shift) + rng.next() % (size_t{1} << shift);
                };
                auto step = [&] {
                    slot& s = slots[rng.next() % slots.size()];
                    if (s.p != nullptr)
                    {
                        if (rng.next() % 100 < 8)
                        {
                            a.record_stream(s.p, neighbour);
                        }
                        a.deallocate(s.p, s.n, mine);
                    }
                    s.n = size_for();
                    s.p = a.allocate(s.n, mine);
                };
                for (int i = 0; i < 2000; ++i)
                {
                    step();
                }
                ready.fetch_add(1);
                run_worker(go, ops_per_thread, step, outs[t]);
                for (slot& s : slots)
                {
                    if (s.p != nullptr)
                    {
                        a.deallocate(s.p, s.n, mine);
                    }
                }
            }
        });
    }
    while (ready.load() != threads)
    {
        std::this_thread::yield();
    }
    size_t const m0 = rt::malloc_calls, f0 = rt::free_calls, ec0 = rt::event_creates,
                 er0 = rt::event_record_calls;
    double const start = phase0::now_ns();
    go.store(true, std::memory_order_release);
    for (auto& th : pool)
    {
        th.join();
    }
    size_t const m1 = rt::malloc_calls, f1 = rt::free_calls, ec1 = rt::event_creates,
                 er1 = rt::event_record_calls;
    double end = start;
    phase0::result r;
    lock_counts    lk;
    for (auto& o : outs)
    {
        end = std::max(end, o.end_ns);
        r.samples_ns_per_op.insert(r.samples_ns_per_op.end(), o.samples.begin(), o.samples.end());
        lk.acquisitions += o.locks.acquisitions;
        lk.contended += o.locks.contended;
        lk.wait_ns += o.locks.wait_ns;
        lk.outer += o.locks.outer;
        lk.hold_cycles += o.locks.hold_cycles;
    }
    double const wall_s    = (end - start) / 1e9;
    double const total_ops = static_cast<double>(ops_per_thread * threads);

    // Single-threaded now; the backing equation must hold in every instance.
    double unaccounted = 0, reserved = 0, inactive = 0, hits = 0, misses = 0;
    for (auto const& inst : instances)
    {
        auto const st = inst->stats();
        unaccounted += static_cast<double>(st.bytes_unaccounted.load());
        reserved += static_cast<double>(st.bytes_reserved.load());
        inactive += static_cast<double>(st.inactive_split_bytes.load());
        hits += static_cast<double>(st.cache_hits.load());
        misses += static_cast<double>(st.cache_misses.load());
    }

    r.name   = w == workload::warm ? "warm_alloc_free" : "mixed_replay";
    r.params = {{"threads", std::to_string(threads)},
                {"rep", std::to_string(rep)},
                {"ops_per_thread", std::to_string(ops_per_thread)},
                {"arenas", std::to_string(arenas)},
                {"lock_instrumented", LOCK_STATS_ENABLED ? "yes" : "no"}};
    r.batch  = kBatch;
    r.counters = {{"wall_s", wall_s},
                  {"ops_per_sec", total_ops / wall_s},
                  {"ns_per_op_per_thread", wall_s * 1e9 * static_cast<double>(threads) / total_ops},
                  {"driver_malloc_total", static_cast<double>(m1 - m0)},
                  {"driver_free_total", static_cast<double>(f1 - f0)},
                  {"event_create_total", static_cast<double>(ec1 - ec0)},
                  {"event_record_total", static_cast<double>(er1 - er0)},
                  {"cache_hits_total", hits},
                  {"cache_misses_total", misses},
                  {"bytes_reserved_end", reserved},
                  {"inactive_split_bytes_end", inactive},
                  {"bytes_unaccounted_end", unaccounted}};
#if LOCK_STATS_ENABLED
    r.counters.push_back({"lock_acquisitions_per_op", static_cast<double>(lk.acquisitions) / total_ops});
    r.counters.push_back({"lock_contended_fraction",
                          lk.acquisitions ? static_cast<double>(lk.contended) / static_cast<double>(lk.acquisitions)
                                          : 0.0});
    r.counters.push_back({"lock_wait_ns_per_op", static_cast<double>(lk.wait_ns) / total_ops});
    r.counters.push_back({"lock_wait_share_of_time",
                          static_cast<double>(lk.wait_ns) / (wall_s * 1e9 * static_cast<double>(threads))});
    // Critical-section length: mean outermost hold per acquisition, raw and with the empty-lock
    // floor (two TSC reads) subtracted; hold_share = locked time / wall time (1.0 = lock never idle).
    double const hold_ns =
        lk.outer ? static_cast<double>(lk.hold_cycles) / g_cycles_per_ns / static_cast<double>(lk.outer) : 0.0;
    r.counters.push_back({"lock_hold_ns_per_acq_raw", hold_ns});
    r.counters.push_back({"lock_hold_ns_per_acq_net", std::max(0.0, hold_ns - g_hold_floor_ns)});
    r.counters.push_back({"lock_hold_share_of_wall",
                          static_cast<double>(lk.hold_cycles) / g_cycles_per_ns / (wall_s * 1e9)});
#else
    (void)lk;
#endif
    if (unaccounted != 0.0)
    {
        r.status = "FAILED: bytes_unaccounted != 0";
    }
    return r;
}
}  // namespace

static int run_main(int argc, char** argv)
{
    bool const  quick = phase0::has_flag(argc, argv, "--quick");
    std::string out   = phase0::arg_value(argc, argv, "--out", "contention.json");
    size_t const reps = std::stoul(phase0::arg_value(argc, argv, "--reps", quick ? "1" : "3"));
    phase0::report rep(
        LOCK_STATS_ENABLED ? "gpu_cache_shim_contention_lockwait"
#if defined(BENCH_LOCK_KIND)
                           : "gpu_cache_shim_contention_spin",
#else
                           : "gpu_cache_shim_contention",
#endif
        "deterministic runtime shim (HIP labels); thread-safe fake driver; no GPU; host cost only",
        phase0::arg_value(argc, argv, "--repo", "."));
    rep.set_build(phase0::compiler_string(), phase0::kNdebug);
    rep.set_evidence("hip", "shim");
    rep.note("fake_driver", "cudaMalloc/cudaFree are malloc/free, events never block; driver latency excluded");
    rep.note("quick", quick ? "yes (reduced ops; not baseline-grade)" : "no");
    rep.note("replay", "per-thread seeded xorshift (seed 1000+tid); sequence deterministic, interleaving is not");
    rep.note("lock_wait",
             LOCK_STATS_ENABLED ? "measured: std::recursive_mutex replaced by a timing drop-in (lock_wait_mutex.h); "
                                  "try_lock first, wait timed only when the lock was held"
                                : "not instrumented in this build; see the _lockwait suite");
#if defined(BENCH_LOCK_KIND)
    rep.note("lock_kind", BENCH_LOCK_KIND " (experiment: std::recursive_mutex replaced, task 8.7)");
#else
    rep.note("lock_kind", "std::recursive_mutex (library as shipped)");
#endif
    rep.note("scope", "latency is not evidence about fragmentation; no fragmentation claim is made");

#if LOCK_STATS_ENABLED
    {
        // TSC rate against the steady clock, then the floor of an empty lock/unlock hold.
        auto const    t0 = std::chrono::steady_clock::now();
        std::uint64_t const c0 = BENCH_TSC();
        while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(200))
        {
        }
        std::uint64_t const c1 = BENCH_TSC();
        double const ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count();
        g_cycles_per_ns = static_cast<double>(c1 - c0) / ns;
        std::bench_timed_recursive_mutex m;
        lock_counts const                 a = thread_locks();
        for (int i = 0; i < 1000000; ++i)
        {
            m.lock();
            m.unlock();
        }
        lock_counts const b = thread_locks();
        g_hold_floor_ns = static_cast<double>(b.hold_cycles - a.hold_cycles) / g_cycles_per_ns /
                          static_cast<double>(b.outer - a.outer);
        rep.note("tsc_ghz", std::to_string(g_cycles_per_ns));
        rep.note("hold_floor_ns", std::to_string(g_hold_floor_ns));
    }
#endif
    size_t const warm_ops  = quick ? 20000 : 300000;
    size_t const mixed_ops = quick ? 10000 : 100000;
    // --arena-sweep: threads 4..32 against 1..threads independent allocator instances (8.7-C upper
    // bound). Default: 1..32 threads on one shared allocator (the 8.1 baseline).
    bool const sweep = phase0::has_flag(argc, argv, "--arena-sweep");
    if (sweep)
    {
        rep.note("arena_sweep", "K independent cuda_caching_allocator instances, thread t uses t % K; upper bound for "
                                "an arena design: no cross-arena free, shared budget or shared stats");
    }
    for (size_t r = 1; r <= reps; ++r)
    {
        for (size_t threads : {1, 2, 4, 8, 16, 32})
        {
            if (sweep && threads < 4)
            {
                continue;
            }
            for (size_t arenas : {1, 2, 4, 8, 16, 32})
            {
                if ((!sweep && arenas != 1) || arenas > threads)
                {
                    continue;
                }
                for (workload w : {workload::warm, workload::mixed})
                {
                    phase0::result res =
                        run_config(w, threads, w == workload::warm ? warm_ops : mixed_ops, r, arenas);
                    std::fprintf(stderr, "%-16s T=%-2zu K=%-2zu rep=%zu  %.3g ops/s  %s\n", res.name.c_str(),
                                 threads, arenas, r, res.counters[1].second, res.status.c_str());
                    rep.add(std::move(res));
                }
            }
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
