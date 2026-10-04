/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

// flat_ptr_map backs the GPU cache's live-block table (plan task 8.7). The cases check it
// against std::map as a reference (including erase patterns that exercise backward-shift
// deletion), that churn at a steady size never allocates, and the failure contracts.

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <new>
#include <random>
#include <stdexcept>
#include <vector>

#include "MemoryTest.h"
#include "common/flat_ptr_map.h"

namespace
{
std::atomic<bool>   g_count_news{false};
std::atomic<size_t> g_news{0};
std::atomic<bool>   g_fail_next_new{false};

void* key(std::uintptr_t i)
{
    return reinterpret_cast<void*>((i + 1) * 16);
}
}  // namespace

// Counting / failing global operator new, active only while a test arms it.
void* operator new(std::size_t n)
{
    if (g_fail_next_new.exchange(false))
    {
        throw std::bad_alloc();
    }
    if (g_count_news.load(std::memory_order_relaxed))
    {
        g_news.fetch_add(1, std::memory_order_relaxed);
    }
    if (void* p = std::malloc(n != 0 ? n : 1))
    {
        return p;
    }
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

extern "C" const char* __asan_default_options()
{
    return "alloc_dealloc_mismatch=0";
}

MEMORYTEST(FlatPtrMap, starts_empty_and_find_misses)
{
    memory::flat_ptr_map<int> m;
    EXPECT_TRUE(m.empty());
    EXPECT_EQ(m.size(), 0U);
    EXPECT_TRUE(m.find(key(1)) == m.end());
    EXPECT_TRUE(m.find(nullptr) == m.end());
    EXPECT_FALSE(m.erase(key(1)));
    END_TEST();
}

MEMORYTEST(FlatPtrMap, emplace_find_erase_roundtrip)
{
    memory::flat_ptr_map<int> m;
    auto [it, inserted] = m.emplace(key(7), 70);
    EXPECT_TRUE(inserted);
    EXPECT_EQ(it->second, 70);
    auto [again, inserted_again] = m.emplace(key(7), 99);
    EXPECT_FALSE(inserted_again);
    EXPECT_EQ(again->second, 70);  // the existing entry is kept
    EXPECT_EQ(m.size(), 1U);
    EXPECT_EQ(m.find(key(7))->second, 70);
    EXPECT_TRUE(m.erase(key(7)));
    EXPECT_TRUE(m.find(key(7)) == m.end());
    EXPECT_TRUE(m.empty());
    END_TEST();
}

MEMORYTEST(FlatPtrMap, null_key_is_rejected)
{
    memory::flat_ptr_map<int> m;
    EXPECT_THROW(m.emplace(nullptr, 1), std::invalid_argument);
    EXPECT_TRUE(m.empty());
    END_TEST();
}

MEMORYTEST(FlatPtrMap, matches_a_reference_map_under_random_churn)
{
    // Many inserts and erases at several live sizes force collisions, growth and backward shifts.
    for (std::uint64_t seed : {1ULL, 2ULL, 3ULL})
    {
        memory::flat_ptr_map<std::uintptr_t> m;
        std::map<void*, std::uintptr_t>      ref;
        std::mt19937_64                      rng(seed);
        for (int step = 0; step < 60000; ++step)
        {
            void* const  k    = key(rng() % 3000);
            unsigned     what = static_cast<unsigned>(rng() % 100);
            if (what < 55)
            {
                bool const inserted = m.emplace(k, step).second;
                bool const expected = ref.emplace(k, step).second;
                ASSERT_EQ(inserted, expected);
            }
            else if (what < 95)
            {
                ASSERT_EQ(m.erase(k), ref.erase(k) == 1U);
            }
            else
            {
                auto it = m.find(k);
                auto rt = ref.find(k);
                ASSERT_EQ(it == m.end(), rt == ref.end());
                if (rt != ref.end())
                {
                    ASSERT_EQ(it->second, rt->second);
                }
            }
            ASSERT_EQ(m.size(), ref.size());
        }
        // Iteration visits exactly the reference's entries.
        size_t seen = 0;
        for (auto const& e : m)
        {
            auto rt = ref.find(e.first);
            ASSERT_TRUE(rt != ref.end());
            ASSERT_EQ(e.second, rt->second);
            ++seen;
        }
        EXPECT_EQ(seen, ref.size());
    }
    END_TEST();
}

MEMORYTEST(FlatPtrMap, erase_by_iterator_keeps_the_rest_reachable)
{
    memory::flat_ptr_map<int> m;
    for (int i = 0; i < 40; ++i)
    {
        m.emplace(key(static_cast<std::uintptr_t>(i)), i);
    }
    for (int i = 0; i < 40; i += 2)
    {
        auto it = m.find(key(static_cast<std::uintptr_t>(i)));
        ASSERT_TRUE(it != m.end());
        m.erase(it);
    }
    EXPECT_EQ(m.size(), 20U);
    for (int i = 0; i < 40; ++i)
    {
        auto it = m.find(key(static_cast<std::uintptr_t>(i)));
        EXPECT_EQ(it != m.end(), i % 2 == 1);
    }
    END_TEST();
}

MEMORYTEST(FlatPtrMap, steady_state_churn_does_not_allocate)
{
    memory::flat_ptr_map<int> m;
    for (int i = 0; i < 500; ++i)
    {
        m.emplace(key(static_cast<std::uintptr_t>(i)), i);
    }
    // Same live size, a different key each round: the warm alloc/free pattern.
    g_news = 0;
    g_count_news = true;
    for (int round = 0; round < 200000; ++round)
    {
        void* const k = key(static_cast<std::uintptr_t>(100000 + (round % 4096)));
        m.emplace(k, round);
        m.erase(k);
    }
    g_count_news = false;
    EXPECT_EQ(g_news.load(), 0U);
    EXPECT_EQ(m.size(), 500U);
    END_TEST();
}

MEMORYTEST(FlatPtrMap, failed_growth_leaves_the_map_unchanged)
{
    memory::flat_ptr_map<int> m;
    // 64 slots at load 1/2: the 33rd insert grows.
    for (int i = 0; i < 32; ++i)
    {
        m.emplace(key(static_cast<std::uintptr_t>(i)), i);
    }
    size_t const capacity_before = m.capacity();
    g_fail_next_new = true;
    EXPECT_THROW(m.emplace(key(1000), 1000), std::bad_alloc);
    g_fail_next_new = false;
    EXPECT_EQ(m.size(), 32U);
    EXPECT_EQ(m.capacity(), capacity_before);
    for (int i = 0; i < 32; ++i)
    {
        auto it = m.find(key(static_cast<std::uintptr_t>(i)));
        ASSERT_TRUE(it != m.end());
        EXPECT_EQ(it->second, i);
    }
    EXPECT_TRUE(m.emplace(key(1000), 1000).second);  // and the retry succeeds
    END_TEST();
}

MEMORYTEST(FlatPtrMap, clear_keeps_the_table_for_reuse)
{
    memory::flat_ptr_map<int> m;
    for (int i = 0; i < 100; ++i)
    {
        m.emplace(key(static_cast<std::uintptr_t>(i)), i);
    }
    size_t const capacity = m.capacity();
    m.clear();
    EXPECT_TRUE(m.empty());
    EXPECT_EQ(m.capacity(), capacity);
    g_news = 0;
    g_count_news = true;
    for (int i = 0; i < 100; ++i)
    {
        m.emplace(key(static_cast<std::uintptr_t>(i)), i);
    }
    g_count_news = false;
    EXPECT_EQ(g_news.load(), 0U);
    END_TEST();
}
