// Tasks 0.2/0.3: retained async copy + token host overhead under the deterministic
// copy runtime (no GPU). Records latency percentiles and per-op counters
// (heap allocations, event create/destroy/record/query, copies) for plan §6.1.
//
//   Phase0CopyShimBench --out copy_shim.json --repo <source dir> [--quick]
//
// Evidence level: deterministic shim. Says nothing about real device behaviour.

#include <atomic>
#include <cstdlib>
#include <new>
#include <string>

#include "fake_runtime.h"
#include "common/copy_token.h"
#include "common/execution_context.h"
#include "common/retained_operation_service.h"
#include "allocator.h"
#include "../Cxx/phase0_harness.h"

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

using namespace memory;

namespace
{
struct counts
{
    double heap, copies, ev_create, ev_destroy, ev_record, ev_query, stream_query;
};

counts snapshot()
{
    return {static_cast<double>(g_news.load()),
            static_cast<double>(fake_runtime::copies),
            static_cast<double>(fake_runtime::event_creates),
            static_cast<double>(fake_runtime::event_destroys),
            static_cast<double>(fake_runtime::event_records),
            static_cast<double>(fake_runtime::event_queries),
            static_cast<double>(fake_runtime::stream_queries)};
}

void add_counters(phase0::result& r, counts a, counts b, double ops)
{
    r.counters = {{"heap_allocs_per_op", (b.heap - a.heap) / ops},
                  {"memcpy_per_op", (b.copies - a.copies) / ops},
                  {"event_create_per_op", (b.ev_create - a.ev_create) / ops},
                  {"event_destroy_per_op", (b.ev_destroy - a.ev_destroy) / ops},
                  {"event_record_per_op", (b.ev_record - a.ev_record) / ops},
                  {"event_query_per_op", (b.ev_query - a.ev_query) / ops},
                  {"stream_query_per_op", (b.stream_query - a.stream_query) / ops}};
}
}  // namespace

int main(int argc, char** argv)
{
    bool const  quick = phase0::has_flag(argc, argv, "--quick");
    std::string out   = phase0::arg_value(argc, argv, "--out", "copy_shim.json");
    phase0::report rep("copy_shim_host_overhead", "deterministic runtime shim; no GPU; host cost only",
                       phase0::arg_value(argc, argv, "--repo", "."));
    rep.set_build(phase0::compiler_string(), phase0::kNdebug);
    rep.set_evidence("cuda", "shim");
    rep.note("scope",
             "retained copy + token only: plain GPU copy_async needs the real caching allocator "
             "(record_stream), measured on hardware under task 0.4");
    rep.note("quick", quick ? "yes (reduced samples; not baseline-grade)" : "no");

    using T = float;
    fake_runtime::reset();
    auto& service = retained_operation_service::instance();
    service.reset();
    auto stream = reinterpret_cast<void*>(3);
    fake_runtime::set_stream_ready(stream, true);
    execution_context ctx;
    ctx.dev.type = device_enum::CUDA;
    ctx.stream      = stream;

    double const res = phase0::timer_resolution_ns();
    for (size_t n : {size_t{16}, size_t{4096}, size_t{1} << 20})
    {
        auto del = [](T* p, size_t, execution_context const&) { delete[] p; };
        auto src = allocator<T>::allocate_adopted(new T[n]{}, n, ctx, del);
        auto dst = allocator<T>::allocate_adopted(new T[n]{}, n, ctx, del);

        auto copy_and_retire = [&] {
            auto token = allocator<T>::copy_async_retained(src, dst, stream);
            (void)token;
            (void)service.poll();
        };
        size_t const samples = quick ? 40 : 400;
        size_t const batch   = n <= 4096 ? phase0::pick_batch(res, copy_and_retire) : 1;

        phase0::result r;
        r.name   = "retained_copy_plus_poll";
        r.params = {{"elements_float", std::to_string(n)}, {"workload", "copy_async_retained + service.poll, stream ready"}};
        r.batch  = batch;
        g_news = 0;
        g_count = true;
        auto c0 = snapshot();
        r.samples_ns_per_op = phase0::measure(20, samples, batch, copy_and_retire);  // counters span warmup too
        auto c1 = snapshot();
        g_count = false;
        add_counters(r, c0, c1, static_cast<double>(20 + samples * batch));
        rep.add(std::move(r));
    }

    // token.ready() after terminal: §6.1 target is 0 driver calls per call.
    {
        auto del = [](T* p, size_t, execution_context const&) { delete[] p; };
        auto src = allocator<T>::allocate_adopted(new T[64]{}, 64, ctx, del);
        auto dst = allocator<T>::allocate_adopted(new T[64]{}, 64, ctx, del);
        auto token = allocator<T>::copy_async_retained(src, dst, stream);
        (void)token.ready();  // reach terminal state
        auto ready_call = [&] { (void)token.ready(); };
        size_t const batch   = phase0::pick_batch(res, ready_call);
        size_t const samples = quick ? 40 : 400;
        phase0::result r;
        r.name   = "token_ready_after_terminal";
        r.params = {{"workload", "token.ready() on an already-complete token"}};
        r.batch  = batch;
        g_news = 0;
        g_count = true;
        auto c0 = snapshot();
        r.samples_ns_per_op = phase0::measure(100, samples, batch, ready_call);
        auto c1 = snapshot();
        g_count = false;
        add_counters(r, c0, c1, static_cast<double>(100 + samples * batch));
        rep.add(std::move(r));
    }

    (void)service.shutdown(std::chrono::milliseconds(100));
    return rep.write(out) ? 0 : 1;
}
