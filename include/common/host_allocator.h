/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#pragma once

#include <cstddef>
#include <limits>
#include <memory_resource>
#include <new>
#include <type_traits>

#include "common/cpu_arena.h"
#include "common/memory_macros.h"
#include "helper/memory_allocator.h"

namespace memory
{

/// STL allocator over the CPU backend (mimalloc / TBB / platform), aligned to
/// at least `alignment` bytes. Host containers only: the GPU caching allocators
/// are stream-ordered and cannot back std containers (plan 4.3, task 2.8).
///
/// Stateless: every instance compares equal, so containers may swap and move
/// freely. Failure is `std::bad_alloc`, as the standard requires.
template <typename T, std::size_t alignment = MEMORY_ALIGNMENT>
class host_allocator
{
    static_assert(
        alignment >= sizeof(void*) && (alignment & (alignment - 1)) == 0,
        "host_allocator alignment must be a power of two >= sizeof(void*)");

public:
    using value_type                             = T;
    using size_type                              = std::size_t;
    using difference_type                        = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal                        = std::true_type;

    template <typename U>
    struct rebind
    {
        using other = host_allocator<U, alignment>;
    };

    constexpr host_allocator() noexcept = default;
    template <typename U>
    constexpr host_allocator(host_allocator<U, alignment> const&) noexcept
    {
    }

    [[nodiscard]] T* allocate(size_type n)
    {
        if (n == 0)
        {
            return nullptr;
        }
        if (n > std::numeric_limits<size_type>::max() / sizeof(T))
        {
            throw std::bad_array_new_length();
        }
        constexpr size_type align = alignof(T) > alignment ? alignof(T) : alignment;
        void* const         p     = cpu::memory_allocator::allocate(n * sizeof(T), align);
        if (p == nullptr)
        {
            throw std::bad_alloc();
        }
        return static_cast<T*>(p);
    }

    void deallocate(T* p, size_type n) noexcept
    {
        if (p != nullptr)
        {
            cpu::memory_allocator::free(p, n * sizeof(T));
        }
    }

    template <typename U>
    constexpr bool operator==(host_allocator<U, alignment> const&) const noexcept
    {
        return true;
    }
    template <typename U>
    constexpr bool operator!=(host_allocator<U, alignment> const&) const noexcept
    {
        return false;
    }
};

/// `std::pmr::memory_resource` over the CPU backend. Equal to every other
/// instance. Use `host_memory_resource::instance()` for a shared one.
class host_memory_resource final : public std::pmr::memory_resource
{
public:
    static host_memory_resource& instance() noexcept
    {
        static host_memory_resource resource;
        return resource;
    }

private:
    void* do_allocate(std::size_t bytes, std::size_t align) override
    {
        if (bytes == 0)
        {
            bytes = 1;  // pmr requires a distinct non-null pointer
        }
        if (align < sizeof(void*))
        {
            align = sizeof(void*);
        }
        void* const p = cpu::memory_allocator::allocate(bytes, align);
        if (p == nullptr)
        {
            throw std::bad_alloc();
        }
        return p;
    }

    void do_deallocate(void* p, std::size_t bytes, std::size_t) override
    {
        cpu::memory_allocator::free(p, bytes == 0 ? 1 : bytes);
    }

    bool do_is_equal(std::pmr::memory_resource const& other) const noexcept override
    {
        return dynamic_cast<host_memory_resource const*>(&other) != nullptr;
    }
};

/// `std::pmr::memory_resource` that serves requests from a `cpu_arena`.
/// Deallocation is a no-op; memory returns to the arena on `cpu_arena::reset()`
/// or destruction, so the arena must outlive every container using the resource
/// and containers must not be used after a reset. Not thread-safe, like the arena.
class arena_memory_resource final : public std::pmr::memory_resource
{
public:
    explicit arena_memory_resource(cpu_arena& arena) noexcept : arena_(&arena) {}

    cpu_arena& arena() const noexcept { return *arena_; }

private:
    void* do_allocate(std::size_t bytes, std::size_t align) override
    {
        void* const p = arena_->alloc_bytes(bytes == 0 ? 1 : bytes, align);
        if (p == nullptr)
        {
            throw std::bad_alloc();
        }
        return p;
    }

    void do_deallocate(void*, std::size_t, std::size_t) noexcept override {}

    bool do_is_equal(std::pmr::memory_resource const& other) const noexcept override
    {
        auto const* o = dynamic_cast<arena_memory_resource const*>(&other);
        return o != nullptr && o->arena_ == arena_;
    }

    cpu_arena* arena_;
};

}  // namespace memory
