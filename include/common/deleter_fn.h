/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#pragma once

#include <cstddef>

#include "common/memory_export.h"

namespace memory
{

// Free callback: (context_ptr, data_ptr, nbytes) — must never throw.
// ctx  is the opaque context passed at construction (nullptr for CPU allocations).
// ptr  is the allocation pointer.
// size is the byte count.
using deleter_fn = void (*)(void* ctx, void* ptr, std::size_t nbytes) noexcept;

}  // namespace memory
