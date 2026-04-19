/*
 * Quarisma: High-Performance Computational Library
 *
 * Original work Copyright 2015 The TensorFlow Authors
 * Modified work Copyright 2025 Quarisma Contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * This file contains code modified from TensorFlow (Apache 2.0 licensed)
 * and is part of Quarisma, licensed under a dual-license model:
 *
 *   - Open-source License (GPLv3):
 *       Free for personal, academic, and research use under the terms of
 *       the GNU General Public License v3.0 or later.
 *
 *   - Commercial License:
 *       A commercial license is required for proprietary, closed-source,
 *       or SaaS usage. Contact us to obtain a commercial agreement.
 *
 * MODIFICATIONS FROM ORIGINAL:
 * - Adapted for Quarisma quantitative computing requirements
 * - Added high-performance memory allocation optimizations
 * - Integrated NUMA-aware allocation strategies
 * - Unified platform-specific implementations using conditional compilation
 * - Separated compression functionality to dedicated module
 *
 * Contact: licensing@quarisma.co.uk
 * Website: https://www.quarisma.co.uk
 */

#include "helper/memory_info.h"

//#include <cpuinfo.h>

#include <cstdint>
#include <limits>

#include "common/memory_macros.h"

#ifdef _WIN32
// Windows-specific includes
#include <Windows.h>
#include <processthreadsapi.h>
#include <shlwapi.h>

#include <cstdio>
#include <cstdlib>
#else
// Unix/Linux-specific includes
#ifdef __linux__
#include <sched.h>
#include <sys/sysinfo.h>
#else
#include <sys/syscall.h>
#endif

#if (__x86_64__ || __i386__)
#include <cpuid.h>
#endif

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#if (defined(__APPLE__) && defined(__MACH__)) || defined(__FreeBSD__) || defined(__HAIKU__)
#include <thread>
#endif

#if MEMORY_HAS_NUMA
#include "hwloc.h"  // from @hwloc
#endif

#if MEMORY_HAS_CXA_DEMANGLE
#include <cxxabi.h>
#endif
#endif  // _WIN32

namespace memory
{
namespace port
{

memory_info GetMemoryInfo()
{
    memory_info mem_info = {INT64_MAX, INT64_MAX};
#ifdef _WIN32
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof(statex);
    if (GlobalMemoryStatusEx(&statex) != 0)
    {
        mem_info.free  = statex.ullAvailPhys;
        mem_info.total = statex.ullTotalPhys;
    }
#else
#ifdef __linux__
    struct sysinfo info;
    int const      err = sysinfo(&info);
    if (err == 0)
    {
        mem_info.free  = info.freeram;
        mem_info.total = info.totalram;
    }
#endif
#endif
    return mem_info;
}

memory_bandwidth_info GetMemoryBandwidthInfo()
{
    memory_bandwidth_info membw_info = {INT64_MAX};
    return membw_info;
}

}  // namespace port
}  // namespace memory
