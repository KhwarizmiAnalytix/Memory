#pragma once

#include <cstdint>
#include <iosfwd>

#include "common/memory_macros.h"
#include "common/memory_export.h"

namespace memory
{
enum class device_enum : int16_t
{
    CPU         = 0,
    CUDA        = 1,
    HIP         = 2,
    PrivateUse1 = 3
};

MEMORY_API std::ostream& operator<<(std::ostream& str, device_enum const& s);

class MEMORY_VISIBILITY device_option
{
public:
    using int_t = int16_t;

    MEMORY_API device_option(device_enum type, device_option::int_t index);

    MEMORY_API device_option(device_enum type, int index);

    MEMORY_API bool operator==(const memory::device_option& rhs) const noexcept;

    MEMORY_API int_t index() const noexcept;

    MEMORY_API device_enum type() const noexcept;

private:
    int_t       index_ = -1;
    device_enum type_{};
};

MEMORY_API std::ostream& operator<<(std::ostream& str, memory::device_option const& s);
}  // namespace memory
