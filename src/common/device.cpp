#include "common/device.h"

#include <ostream>

namespace memory
{
device_option::device_option(device_enum type, device_option::int_t index)
    : index_(index), type_(type){};

device_option::device_option(device_enum type, int index)
    : index_(static_cast<device_option::int_t>(index)), type_(type){};

bool device_option::operator==(const memory::device_option& rhs) const noexcept
{
    return (index_ == rhs.index_ && type_ == rhs.type_);
}

device_option::int_t device_option::index() const noexcept
{
    return index_;
}

memory::device_enum device_option::type() const noexcept
{
    return type_;
}

std::ostream& operator<<(std::ostream& str, memory::device_enum const& s)
{
    str << "device_option type: " << static_cast<int>(s);
    return str;
}

std::ostream& operator<<(std::ostream& str, memory::device_option const& s)
{
    str << s.type() << ", index " << s.index();
    return str;
}
}  // namespace memory