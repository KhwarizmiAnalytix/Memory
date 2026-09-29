/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#pragma once

#include <cstddef>

#include "common/data_ptr.h"
#include "common/execution_context.h"

namespace memory
{

/**
 * Allocate an owning typed buffer on the requested device.
 *
 * This is the public allocation entry point: callers receive a data_ptr that
 * owns the allocation and exposes its data, size, views, device, and stream.
 * Backend selection and matching deallocation remain internal to data_ptr.
 * The allocated storage is uninitialized, matching allocator<T>::allocate().
 */
template <typename T>
MEMORY_FORCE_INLINE data_ptr<T> allocate(
    std::size_t count, execution_context ctx = execution_context::cpu())
{
    return data_ptr<T>(count, ctx);
}

/** Allocate an owning byte buffer. */
MEMORY_FORCE_INLINE data_ptr<std::byte> allocate_bytes(
    std::size_t byte_count, execution_context ctx = execution_context::cpu())
{
    return allocate<std::byte>(byte_count, ctx);
}

}  // namespace memory
