/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#pragma once

// Fake GPU runtime portability shim for copy_token and allocator.h testing.
// Redirects to our fake CUDA/HIP runtime instead of the real headers.

#include "common/memory_macros.h"

#if MEMORY_HAS_CUDA
#include "cuda_runtime.h"
#elif MEMORY_HAS_HIP
#include "hip/hip_runtime.h"
#endif
