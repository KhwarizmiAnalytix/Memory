/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

#include "common/memory_export.h"

namespace memory
{

// Unique identifier for each allocation lifetime.
// Assigned once at allocate time; survives slicing, never reused for a new
// allocation at a recycled address. Zero is reserved as invalid/sentinel; real
// IDs start at 1. Empty handles (default-constructed, moved-from, zero-size) have
// the invalid ID.
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

// Process-wide allocation ID generator: thread-safe, starts at 1, never wraps and
// is never reset. Defined once in the Memory library (src/common/allocation_id.cpp)
// and exported, so header-inline callers in client binaries share the library's
// sequence instead of owning a private counter per binary (plan §5.1, A6).
MEMORY_API allocation_id next_allocation_id() noexcept;

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
