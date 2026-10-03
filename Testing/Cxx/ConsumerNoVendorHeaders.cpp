/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

// Task 2.6 exit: the public L2/L3 headers compile with a GPU backend selected and
// no vendor runtime (CUDA/HIP) header on the include path. Compile-only; see
// MemoryConsumerNoVendorHeaders in Testing/Cxx/CMakeLists.txt.

#include "memory.h"
#include "allocator.h"
#include "common/cpu_arena.h"
#include "common/copy_token.h"
#include "common/data_ptr.h"
#include "common/data_view.h"
#include "common/host_allocator.h"
#include "common/pinned_buffer.h"
#include "common/retained_operation_service.h"
#include "common/retained_ptr.h"
#include "common/shared_storage.h"
#include "common/storage_handle.h"
#include "gpu/gpu_workspace.h"

int main()
{
    auto p = memory::allocate<float>(4, memory::execution_context::cuda(0, nullptr));
    return p.size() == 4 ? 0 : 1;
}
