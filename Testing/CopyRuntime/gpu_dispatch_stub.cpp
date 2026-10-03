/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

// Test-only definition of the one gpu_dispatch.h symbol the byte copy router
// (src/transfer.cpp) calls. These shim targets never link Memory or a real
// caching allocator, so stream-use registration is a counted no-op here.

#include <atomic>

#include "gpu/gpu_dispatch.h"

namespace memory::gpu
{
namespace test
{
std::atomic<int> record_stream_use_calls{0};
}  // namespace test

void record_stream_use(void* /*ptr*/, int /*device_index*/, stream_handle_t /*stream*/)
{
    ++test::record_stream_use_calls;
}
}  // namespace memory::gpu
