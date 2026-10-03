#pragma once

#include <cstddef>
#include <type_traits>

namespace memory
{

/// Largest element alignment typed owners accept when the owner does not carry its own
/// alignment parameter (retained_ptr). Matches the default allocation alignment.
inline constexpr std::size_t storage_max_alignment = 64;

/// Element types that typed owners (`data_ptr`, `retained_ptr`) can hold.
///
/// Storage is uninitialized bytes: no constructor runs on allocation and no
/// destructor on release, and copies are byte copies. That is only sound for
/// trivially copyable, trivially destructible, non-cv-qualified types whose
/// alignment the allocation honors (plan §4.3, task 2.9).
template <typename T, std::size_t alignment>
inline constexpr bool is_storage_element_v =
    std::is_object_v<T> && !std::is_const_v<T> && !std::is_volatile_v<T> &&
    std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T> &&
    alignof(T) <= alignment;

}  // namespace memory

#define MEMORY_STATIC_ASSERT_STORAGE_ELEMENT(T, alignment)                                   \
    static_assert(                                                                           \
        ::memory::is_storage_element_v<T, alignment>,                                        \
        "typed storage requires a trivially copyable, trivially destructible, "              \
        "non-cv-qualified element type with alignof(T) <= the allocation alignment")
