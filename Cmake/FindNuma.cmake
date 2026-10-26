# Finds libnuma (numa.h + libnuma) for MEMORY_ENABLE_NUMA.
# Sets Numa_INCLUDE_DIR, Numa_LIBRARIES, and Numa_FOUND.

include(FindPackageHandleStandardArgs)

find_path(Numa_INCLUDE_DIR NAMES numa.h)
find_library(Numa_LIBRARIES NAMES numa)

find_package_handle_standard_args(Numa DEFAULT_MSG Numa_INCLUDE_DIR Numa_LIBRARIES)

mark_as_advanced(Numa_INCLUDE_DIR Numa_LIBRARIES)
