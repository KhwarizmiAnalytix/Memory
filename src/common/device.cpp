#include "common/device.h"

#include <ostream>

namespace memory
{
std::ostream& operator<<(std::ostream& str, memory::device_enum const& s)
{
    str << "device type: " << static_cast<int>(s);
    return str;
}

std::ostream& operator<<(std::ostream& str, device const& d)
{
    str << d.type << ", index " << d.index;
    return str;
}
}  // namespace memory
