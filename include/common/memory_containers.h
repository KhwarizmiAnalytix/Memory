/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * This file is part of XSigma and is licensed under a dual-license model:
 *
 *   - Open-source License (GPLv3):
 *       Free for personal, academic, and research use under the terms of
 *       the GNU General Public License v3.0 or later.
 *
 *   - Commercial License:
 *       A commercial license is required for proprietary, closed-source,
 *       or SaaS usage. Contact us to obtain a commercial agreement.
 *
 * Contact: licensing@xsigma.co.uk
 * Website: https://www.xsigma.co.uk
 */

// Portable container aliases used throughout the Memory module.
//
// memory_set / memory_map are the standard-library containers. A flat-hash
// switch (MEMORY_USE_FLAT_HASH, "util/flat_hash.h") was removed in plan task 3.8:
// the header never existed in this repository, and the hot GPU live-block map
// uses a node-recycling std::unordered_map instead (measured 8-10 ns per
// insert+find+erase against 3-5 ns for open addressing, at most 7 ns of a 77 ns
// warm alloc/free pair; Testing/tools/live_map_bench.cpp).
//
// Usage:
//   memory_set<void*>                        free_ptrs;
//   memory_map<size_t, Block>                blocks_by_size;
//   memory_map<Key, Val, MyHash>             blocks_custom_hash;

#pragma once

#include <unordered_map>
#include <unordered_set>

template <typename T>
using memory_set = std::unordered_set<T>;
template <typename K, typename V, typename H = std::hash<K>>
using memory_map = std::unordered_map<K, V, H>;
