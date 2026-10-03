#pragma once

#include <cstdint>
#include <iosfwd>

#include "common/memory_macros.h"
#include "common/memory_export.h"

namespace memory
{
enum class device_enum : std::uint8_t
{
    CPU         = 0,
    CUDA        = 1,
    HIP         = 2,
    PrivateUse1 = 3,
    METAL       = 4
};

MEMORY_API std::ostream& operator<<(std::ostream& str, device_enum const& s);

// Compact device descriptor (c10::Device analogue): type (1 B) + padding (1 B) +
// index (2 B) = 4 bytes. Replaces the removed device_option and scattered
// (device_enum, int device_index) pairs.
struct device
{
    device_enum  type{device_enum::CPU};
    std::int16_t index{0};

    constexpr bool is_cpu() const noexcept { return type == device_enum::CPU; }
    constexpr bool is_gpu() const noexcept
    {
        return type == device_enum::CUDA || type == device_enum::HIP
            || type == device_enum::METAL;
    }
    constexpr bool operator==(device const& o) const noexcept
    {
        return type == o.type && index == o.index;
    }
    constexpr bool operator!=(device const& o) const noexcept { return !(*this == o); }

    static constexpr device cpu()               noexcept { return {device_enum::CPU,   0}; }
    static constexpr device cuda(int16_t i = 0) noexcept { return {device_enum::CUDA,  i}; }
    static constexpr device hip(int16_t i = 0)  noexcept { return {device_enum::HIP,   i}; }
    static constexpr device metal(int16_t i = 0) noexcept{ return {device_enum::METAL, i}; }
};
static_assert(sizeof(device) == 4, "device must be 4 bytes");

MEMORY_API std::ostream& operator<<(std::ostream& str, device const& d);

}  // namespace memory
