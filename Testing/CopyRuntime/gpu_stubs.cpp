// CopyRuntime shim: free_gpu_with_stream stub.
//
// retained_ptr.h forward-declares free_gpu_with_stream and calls it in the
// GPU promotion free path (fn_del_ctx != nullptr && ctx.is_gpu()).
// CopyRuntime tests only use retained_ptr::adopt() — they never promote a
// data_ptr, so this path is never taken.  The linker still needs the symbol,
// so we provide a no-op here rather than pulling in storage.cpp and all its
// transitive dependencies (caching_allocator, memory_allocator, …).
#include <cstddef>

#include "common/execution_context.h"

namespace memory
{

void free_gpu_with_stream(void* /*cache_ctx*/,
                          void* /*ptr*/,
                          std::size_t /*nbytes*/,
                          stream_handle_t /*stream*/) noexcept
{
}

}  // namespace memory
