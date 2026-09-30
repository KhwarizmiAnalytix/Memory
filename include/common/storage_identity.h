/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace memory
{

// Unique identifier for each allocation lifetime.
// Assigned once at allocate time; survives address reuse and slicing.
// Zero is reserved as invalid/sentinel; real IDs start at 1.
// Thread-safe; monotonically increasing; never wraps or resets.
struct allocation_id
{
    uint64_t value = 0;

    allocation_id() noexcept = default;
    explicit constexpr allocation_id(uint64_t v) noexcept : value(v) {}

    constexpr bool operator==(allocation_id const& other) const noexcept
    {
        return value == other.value;
    }
    constexpr bool operator!=(allocation_id const& other) const noexcept
    {
        return value != other.value;
    }
    constexpr bool operator<(allocation_id const& other) const noexcept
    {
        return value < other.value;
    }
    constexpr bool operator<=(allocation_id const& other) const noexcept
    {
        return value <= other.value;
    }
    constexpr bool operator>(allocation_id const& other) const noexcept
    {
        return value > other.value;
    }
    constexpr bool operator>=(allocation_id const& other) const noexcept
    {
        return value >= other.value;
    }

    // Check if valid (non-zero)
    constexpr bool valid() const noexcept { return value != 0; }
};

// Global allocation ID generator: thread-safe, starts at 1, never wraps.
class allocation_id_generator
{
public:
    static allocation_id_generator& instance() noexcept
    {
        static allocation_id_generator gen;
        return gen;
    }

    allocation_id next() noexcept
    {
        uint64_t val = counter_.fetch_add(1, std::memory_order_relaxed);
        return allocation_id(val);
    }

    // Reset for testing only
    void reset() noexcept
    {
        counter_.store(1, std::memory_order_relaxed);
    }

private:
    allocation_id_generator() noexcept : counter_(1) {}
    std::atomic<uint64_t> counter_;
};

// Global function wrapper
inline allocation_id next_allocation_id() noexcept
{
    return allocation_id_generator::instance().next();
}

// A stable allocation identity (legacy name, maps to allocation_id + metadata).
// Assigned once at allocate time; survives all derived views and address reuse
// so telemetry can pair alloc/free events even when the block address is
// recycled between trace entries.
struct storage_identity
{
    uint64_t alloc_id = 0;  // 0 = invalid/unset; assigned by next_id()
    void*    base = nullptr;     // allocation base pointer (not a view's start)
    size_t   capacity = 0;       // allocation byte capacity

    bool valid() const noexcept { return alloc_id != 0; }

    // Thread-safe monotonic ID source (starts at 1; 0 is always invalid).
    static uint64_t next_id() noexcept
    {
        return next_allocation_id().value;
    }

    bool operator==(storage_identity const& o) const noexcept
    {
        return alloc_id == o.alloc_id && base == o.base;
    }
    bool operator!=(storage_identity const& o) const noexcept { return !(*this == o); }
};

}  // namespace memory

// Hash specialization for allocation_id
namespace std
{
template <>
struct hash<memory::allocation_id>
{
    size_t operator()(memory::allocation_id id) const noexcept
    {
        return static_cast<size_t>(id.value);
    }
};
}  // namespace std
