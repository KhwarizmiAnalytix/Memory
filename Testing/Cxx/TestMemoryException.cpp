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

// Exercises util/memory_exception.h directly: the two format_check_msg
// overloads and the MEMORY_THROW/MEMORY_CHECK macros. Production code only
// ever calls MEMORY_CHECK with a format string (never zero varargs), so the
// no-format-args format_check_msg overload is otherwise unreachable from
// any other test in this library.

#include <stdexcept>
#include <string>

#include "MemoryTest.h"
#include "util/memory_exception.h"

using namespace memory;

MEMORYTEST(MemoryException, FormatCheckMsgWithArgs)
{
    const std::string msg = details::format_check_msg("x > 0", "value was {}", 5);
    EXPECT_NE(msg.find("Check failed: x > 0"), std::string::npos);
    EXPECT_NE(msg.find("value was 5"), std::string::npos);
    END_TEST();
}

MEMORYTEST(MemoryException, FormatCheckMsgNoArgs)
{
    const std::string msg = details::format_check_msg("ptr != nullptr");
    EXPECT_EQ(msg, "Check failed: ptr != nullptr");
    END_TEST();
}

MEMORYTEST(MemoryException, MemoryThrowMacro)
{
    EXPECT_THROW({ MEMORY_THROW("boom {}", 42); }, std::runtime_error);
    END_TEST();
}

MEMORYTEST(MemoryException, MemoryCheckPassesSilently)
{
    EXPECT_NO_THROW({ MEMORY_CHECK(true, "should not fire"); });
    END_TEST();
}

MEMORYTEST(MemoryException, MemoryCheckFailureThrows)
{
    EXPECT_THROW({ MEMORY_CHECK(1 == 2, "one is not two: {}", 1); }, std::runtime_error);
    END_TEST();
}

MEMORYTEST(MemoryException, LogMacrosDoNotThrow)
{
    EXPECT_NO_THROW({
        MEMORY_LOG_WARNING("test warning: {}", 1);
        MEMORY_LOG_ERROR("test error: {}", 2);
    });
    END_TEST();
}
