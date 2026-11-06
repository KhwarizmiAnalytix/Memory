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

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace memory
{
// Open-addressing map from a non-null pointer to a value, the shape of the
// ska::flat_hash_map/flat_hash_set tables PyTorch's CUDA caching allocator uses
// for its live-block bookkeeping (plan task 8.7, Appendix D).
//
//  - Linear probing, power-of-two capacity, load factor at most 1/2, Fibonacci hash.
//  - Backward-shift deletion (no tombstones), so an insert/erase churn at a steady
//    live size never rehashes and never allocates: only growth allocates.
//  - nullptr is the empty-slot key and cannot be stored (a live allocation is never
//    null): emplace(nullptr, ...) throws std::invalid_argument, find(nullptr) is end().
//  - emplace gives the strong guarantee: a failed growth leaves the map unchanged.
//  - Not thread-safe; the caller serializes (the allocator mutex).
//  - Iterators are invalidated by any emplace or erase.
template <typename V>
class flat_ptr_map
{
public:
    struct entry
    {
        void* first{nullptr};
        V     second{};
    };

    template <bool Const>
    class basic_iterator
    {
    public:
        using value_type      = entry;
        using reference       = std::conditional_t<Const, entry const&, entry&>;
        using pointer         = std::conditional_t<Const, entry const*, entry*>;
        using difference_type = std::ptrdiff_t;

        basic_iterator() = default;
        basic_iterator(pointer cur, pointer end) noexcept : cur_(cur), end_(end) { skip(); }
        reference       operator*() const noexcept { return *cur_; }
        pointer         operator->() const noexcept { return cur_; }
        basic_iterator& operator++() noexcept
        {
            ++cur_;
            skip();
            return *this;
        }
        friend bool operator==(basic_iterator const& a, basic_iterator const& b) noexcept
        {
            return a.cur_ == b.cur_;
        }
        friend bool operator!=(basic_iterator const& a, basic_iterator const& b) noexcept
        {
            return a.cur_ != b.cur_;
        }

    private:
        friend class flat_ptr_map;
        void skip() noexcept
        {
            while (cur_ != end_ && cur_->first == nullptr)
            {
                ++cur_;
            }
        }
        pointer cur_{nullptr};
        pointer end_{nullptr};
    };
    using iterator       = basic_iterator<false>;
    using const_iterator = basic_iterator<true>;

    flat_ptr_map() noexcept                      = default;
    flat_ptr_map(flat_ptr_map const&)            = delete;
    flat_ptr_map& operator=(flat_ptr_map const&) = delete;

    size_t size() const noexcept { return size_; }
    bool   empty() const noexcept { return size_ == 0; }
    size_t capacity() const noexcept { return capacity_; }

    iterator begin() noexcept { return iterator(table_.get(), table_.get() + capacity_); }
    iterator end() noexcept { return iterator(table_.get() + capacity_, table_.get() + capacity_); }
    const_iterator begin() const noexcept
    {
        return const_iterator(table_.get(), table_.get() + capacity_);
    }
    const_iterator end() const noexcept
    {
        return const_iterator(table_.get() + capacity_, table_.get() + capacity_);
    }

    iterator find(const void* key) noexcept
    {
        size_t const i = locate(key);
        return i == npos ? end() : iterator(table_.get() + i, table_.get() + capacity_);
    }
    const_iterator find(const void* key) const noexcept
    {
        size_t const i = locate(key);
        return i == npos ? end() : const_iterator(table_.get() + i, table_.get() + capacity_);
    }

    // Inserts when absent. Returns the entry and whether it was inserted. May throw
    // std::bad_alloc while growing, in which case the map is unchanged.
    std::pair<iterator, bool> emplace(void* key, V value)
    {
        if (key == nullptr)
        {
            throw std::invalid_argument("flat_ptr_map: null key");
        }
        if (size_t const found = locate(key); found != npos)
        {
            return {iterator(table_.get() + found, table_.get() + capacity_), false};
        }
        if ((size_ + 1) * 2 > capacity_)
        {
            grow();
        }
        size_t const mask = capacity_ - 1;
        size_t       i    = ideal(key);
        while (table_[i].first != nullptr)
        {
            i = (i + 1) & mask;
        }
        table_[i].first  = key;
        table_[i].second = std::move(value);
        ++size_;
        return {iterator(table_.get() + i, table_.get() + capacity_), true};
    }

    // Removes the entry; returns whether one was present. Never throws or allocates.
    bool erase(const void* key) noexcept
    {
        size_t const i = locate(key);
        if (i == npos)
        {
            return false;
        }
        erase_at(i);
        return true;
    }
    void erase(iterator it) noexcept { erase_at(static_cast<size_t>(it.cur_ - table_.get())); }

    // Removes every entry and keeps the table, so refilling does not allocate.
    void clear() noexcept
    {
        for (size_t i = 0; i < capacity_; ++i)
        {
            table_[i] = entry{};
        }
        size_ = 0;
    }

private:
    static constexpr size_t npos          = static_cast<size_t>(-1);
    static constexpr size_t kInitialSlots = 64;

    size_t ideal(void const* key) const noexcept
    {
        // Pointers are at least 16-byte aligned in this allocator; drop the low bits and
        // take the top bits of the product (Fibonacci hashing).
        std::uint64_t const x =
            static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(key) >> 4U) *
            0x9E3779B97F4A7C15ULL;
        return static_cast<size_t>(x >> shift_);
    }

    size_t locate(void const* key) const noexcept
    {
        if (key == nullptr || size_ == 0)
        {
            return npos;
        }
        size_t const mask = capacity_ - 1;
        for (size_t i = ideal(key);; i = (i + 1) & mask)
        {
            void const* const k = table_[i].first;
            if (k == key)
            {
                return i;
            }
            if (k == nullptr)
            {
                return npos;
            }
        }
    }

    void erase_at(size_t i) noexcept
    {
        size_t const mask = capacity_ - 1;
        --size_;
        // Backward-shift: pull later members of the probe run into the hole when their ideal
        // slot is not between the hole and their current slot.
        for (size_t j = (i + 1) & mask; table_[j].first != nullptr; j = (j + 1) & mask)
        {
            size_t const home = ideal(table_[j].first);
            if (((j - home) & mask) >= ((j - i) & mask))
            {
                table_[i] = std::move(table_[j]);
                i         = j;
            }
        }
        table_[i] = entry{};
    }

    void grow()
    {
        size_t const new_capacity = capacity_ == 0 ? kInitialSlots : capacity_ * 2;
        // Allocate and fill the new table first: the old table is untouched if this throws.
        std::unique_ptr<entry[]> fresh(new entry[new_capacity]());
        unsigned                 new_shift = 64;
        for (size_t c = new_capacity; c > 1; c >>= 1U)
        {
            --new_shift;
        }
        std::unique_ptr<entry[]> old_table = std::move(table_);
        size_t const             old_cap   = capacity_;
        table_                             = std::move(fresh);
        capacity_                          = new_capacity;
        shift_                             = new_shift;
        size_t const mask                  = capacity_ - 1;
        for (size_t k = 0; k < old_cap; ++k)
        {
            if (old_table[k].first != nullptr)
            {
                size_t i = ideal(old_table[k].first);
                while (table_[i].first != nullptr)
                {
                    i = (i + 1) & mask;
                }
                table_[i] = std::move(old_table[k]);
            }
        }
    }

    std::unique_ptr<entry[]> table_;
    size_t                   capacity_{0};
    size_t                   size_{0};
    unsigned                 shift_{63};
};
}  // namespace memory
