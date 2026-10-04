// Build: clang++ -O2 -std=c++20 live_map_bench.cpp -o live_map_bench && ./live_map_bench
// Result recorded in docs/memory_runtime_implementation_plan.md task 3.8.
// Live-block map candidates for the GPU cache (plan 3.8): per-op cost of the
// pooled std::unordered_map now in use vs an open-addressing table, for the
// warm alloc/free pattern (insert ptr, find ptr, erase ptr) at several live sizes.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

struct node_pool {
    struct entry { entry* next; };
    entry* head = nullptr; size_t sz = 0;
    void* alloc(size_t b) { if (!sz && b >= sizeof(entry)) sz = b; if (b == sz && head) { auto* e = head; head = e->next; return e; } return ::operator new(b); }
    void free(void* p, size_t b) { if (b == sz) { head = new (p) entry{head}; return; } ::operator delete(p); }
};
template <class T> struct pa {
    using value_type = T; node_pool* p;
    template <class U> struct rebind { using other = pa<U>; };
    explicit pa(node_pool* q) : p(q) {}
    template <class U> pa(pa<U> const& o) : p(o.p) {}
    T* allocate(size_t n) { return n == 1 ? (T*)p->alloc(sizeof(T)) : (T*)::operator new(n * sizeof(T)); }
    void deallocate(T* q, size_t n) { n == 1 ? p->free(q, sizeof(T)) : ::operator delete(q); }
    template <class U> bool operator==(pa<U> const& o) const { return p == o.p; }
    template <class U> bool operator!=(pa<U> const& o) const { return p != o.p; }
};
using pooled_map = std::unordered_map<void*, void*, std::hash<void*>, std::equal_to<void*>, pa<std::pair<void* const, void*>>>;

struct open_map {  // linear probing, tombstones, power-of-two capacity
    struct slot { void* k = nullptr; void* v = nullptr; };
    std::vector<slot> t; size_t used = 0, live = 0;
    open_map() : t(64) {}
    static size_t h(void* p) { uint64_t x = (uint64_t)p >> 4; x *= 0x9E3779B97F4A7C15ull; return x >> 20; }
    static void* tomb() { return (void*)1; }
    void grow() { std::vector<slot> old; old.swap(t); t.assign(old.size() * (live * 2 > old.size() / 2 ? 2 : 1), slot{}); used = live = 0; for (auto& s : old) if (s.k && s.k != tomb()) put(s.k, s.v); }
    void put(void* k, void* v) { if ((used + 1) * 4 > t.size() * 3) grow(); size_t m = t.size() - 1, i = h(k) & m; while (t[i].k && t[i].k != tomb()) i = (i + 1) & m; if (!t[i].k) ++used; t[i] = {k, v}; ++live; }
    void* get(void* k) const { size_t m = t.size() - 1, i = h(k) & m; while (t[i].k) { if (t[i].k == k) return t[i].v; i = (i + 1) & m; } return nullptr; }
    void erase(void* k) { size_t m = t.size() - 1, i = h(k) & m; while (t[i].k) { if (t[i].k == k) { t[i].k = tomb(); --live; return; } i = (i + 1) & m; } }
};

template <class F> double ns_per_op(F&& f, size_t ops) {
    double best = 1e18;
    for (int r = 0; r < 7; ++r) { auto a = std::chrono::steady_clock::now(); f(); auto b = std::chrono::steady_clock::now(); best = std::min(best, std::chrono::duration<double, std::nano>(b - a).count() / ops); }
    return best;
}
int main() {
    for (size_t live : {16u, 256u, 4096u, 65536u}) {
        std::vector<void*> keys(live + 1024);
        std::mt19937_64 rng(1); for (auto& k : keys) k = (void*)((rng() & 0xFFFFFFFFF0ull) | 0x10);
        const size_t iters = 2000000;
        node_pool np; pooled_map pm(0, std::hash<void*>{}, std::equal_to<void*>{}, pa<std::pair<void* const, void*>>(&np));
        open_map om;
        for (size_t i = 0; i < live; ++i) { pm.emplace(keys[i], keys[i]); om.put(keys[i], keys[i]); }
        size_t cursor = 0; volatile void* sink = nullptr;
        double a = ns_per_op([&] { for (size_t i = 0; i < iters; ++i) { void* k = keys[live + (i & 1023)]; pm.emplace(k, k); sink = pm.find(k)->second; pm.erase(k); } }, iters);
        double b = ns_per_op([&] { for (size_t i = 0; i < iters; ++i) { void* k = keys[live + (i & 1023)]; om.put(k, k); sink = om.get(k); om.erase(k); } }, iters);
        std::printf("live=%6zu  pooled unordered_map insert+find+erase: %6.1f ns   open-addressing: %6.1f ns   (x%.2f)\n", live, a, b, a / b);
        (void)cursor; (void)sink;
    }
}
