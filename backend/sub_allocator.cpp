/*
 * Quarisma: High-Performance Computational Library
 * Copyright 2025 Quarisma Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#include "backend/sub_allocator.h"

namespace memory
{

sub_allocator::sub_allocator(
    const std::vector<Visitor>& alloc_visitors, const std::vector<Visitor>& free_visitors)
    : alloc_visitors_(alloc_visitors), free_visitors_(free_visitors)
{
}

void sub_allocator::VisitAlloc(void* ptr, int index, size_t num_bytes)
{
    for (const auto& v : alloc_visitors_)
    {
        v(ptr, index, num_bytes);
    }
}

void sub_allocator::VisitFree(void* ptr, int index, size_t num_bytes)
{
    // Although we don't guarantee any order of visitor application, strive
    // to apply free visitors in reverse order of alloc visitors.
    for (int i = static_cast<int>(free_visitors_.size()) - 1; i >= 0; --i)
    {
        free_visitors_[i](ptr, index, num_bytes);
    }
}

}  // namespace memory
