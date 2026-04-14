#pragma once

#include <cstddef>

// IWYU pragma: keep
#include "common/memory_export.h"

namespace memory
{
/**
 * Check whether NUMA is enabled
 */
MEMORY_API bool IsNUMAEnabled();

/**
 * Bind to a given NUMA node
 */
MEMORY_API void NUMABind(int numa_node_id);

/**
 * Get the NUMA id for a given pointer `ptr`
 */
MEMORY_API int GetNUMANode(const void* ptr);

/**
 * Get number of NUMA nodes
 */
MEMORY_API int GetNumNUMANodes();

/**
 * Move the memory pointed to by `ptr` of a given size to another NUMA node
 */
MEMORY_API void NUMAMove(void* ptr, size_t size, int numa_node_id);

/**
 * Get the current NUMA node id
 */
MEMORY_API int GetCurrentNUMANode();

}  // namespace memory
