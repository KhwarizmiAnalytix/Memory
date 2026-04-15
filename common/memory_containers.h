/*
 * Quarisma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * This file is part of Quarisma and is licensed under a dual-license model:
 *
 *   - Open-source License (GPLv3):
 *       Free for personal, academic, and research use under the terms of
 *       the GNU General Public License v3.0 or later.
 *
 *   - Commercial License:
 *       A commercial license is required for proprietary, closed-source,
 *       or SaaS usage. Contact us to obtain a commercial agreement.
 *
 * Contact: licensing@quarisma.co.uk
 * Website: https://www.quarisma.co.uk
 */

// Portable container aliases used throughout the Memory module.
//
// When MEMORY_USE_FLAT_HASH is defined (and "util/flat_hash.h" is available),
// memory_set / memory_map resolve to the faster open-addressing flat-hash
// implementations.  Otherwise they fall back to the standard-library
// equivalents so the module remains self-contained.
//
// Usage:
//   memory_set<void*>                        free_ptrs;
//   memory_map<size_t, Block>                blocks_by_size;
//   memory_map<Key, Val, MyHash>             blocks_custom_hash;

#pragma once

#ifdef MEMORY_USE_FLAT_HASH
#include "util/flat_hash.h"

template <typename T>
using memory_set = flat_hash_set<T>;
template <typename K, typename V, typename H = std::hash<K>>
using memory_map = flat_hash_map<K, V, H>;

#else
#include <unordered_map>
#include <unordered_set>

template <typename T>
using memory_set = std::unordered_set<T>;
template <typename K, typename V, typename H = std::hash<K>>
using memory_map = std::unordered_map<K, V, H>;

#endif
