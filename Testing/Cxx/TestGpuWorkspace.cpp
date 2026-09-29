/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#include "MemoryTest.h"
#include "common/execution_context.h"
#include "gpu/gpu_workspace.h"

using namespace memory;
using namespace memory::gpu;

MEMORYTEST(GpuWorkspace, default_is_empty)
{
    gpu_workspace ws;
    EXPECT_TRUE(ws.empty());
    EXPECT_EQ(ws.used(), 0U);
    EXPECT_EQ(ws.capacity(), 0U);
    END_TEST();
}

MEMORYTEST(GpuWorkspace, release_resets_cursor_without_freeing)
{
    // Test the release() path without a real GPU allocation.
    // Build a CPU-context workspace; acquire() will throw on a non-GPU context,
    // so we only test construction and release state here.
    gpu_workspace ws{1024, execution_context::cpu()};
    EXPECT_EQ(ws.capacity(), 1024U);
    ws.release();
    EXPECT_TRUE(ws.empty());
    END_TEST();
}

MEMORYTEST(GpuWorkspace, rebind_changes_context)
{
    gpu_workspace ws;
    auto ctx = execution_context::cuda(1);
    ws.rebind(ctx);
    EXPECT_EQ(ws.ctx().device_index, 1);
    END_TEST();
}

MEMORYTEST(GpuWorkspace, rebind_while_acquired_throws)
{
    // rebind() must CHECK that cursor_ == 0 (release() was called).
    // Simulate a non-zero cursor by acquiring on a GPU device.  If no GPU is
    // available the test is skipped; the CHECK still fires in GPU builds.
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP || MEMORY_HAS_METAL
    int device_count = 0;
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
    {
        END_TEST();
        return;
    }
#endif
    gpu_workspace ws{4096, execution_context::cuda(0)};
    (void)ws.acquire(256);
    // cursor_ > 0: rebind must throw.
    ASSERT_ANY_THROW(ws.rebind(execution_context::cuda(0)));
    // release() before rebind: no throw.
    ws.release();
    ws.rebind(execution_context::cuda(0));
#endif
    END_TEST();
}

MEMORYTEST(GpuWorkspace, move_transfers_ownership)
{
    gpu_workspace a{512, execution_context::cpu()};
    gpu_workspace b = std::move(a);
    EXPECT_EQ(b.capacity(), 512U);
    EXPECT_EQ(a.capacity(), 0U);
    END_TEST();
}

// GPU-only tests are skipped on CPU-only builds.
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP || MEMORY_HAS_METAL

MEMORYTEST(GpuWorkspace, gpu_acquire_returns_non_null)
{
    // Only run if a device is available.
    int device_count = 0;
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
    {
        END_TEST();
        return;
    }
#endif

    gpu_workspace ws{4096, execution_context::cuda(0)};
    void* p = ws.acquire(256);
    EXPECT_NE(p, nullptr);
    EXPECT_GE(ws.used(), 256U);
    ws.release();
    EXPECT_TRUE(ws.empty());
    END_TEST();
}

#endif  // MEMORY_HAS_CUDA || MEMORY_HAS_HIP || MEMORY_HAS_METAL
