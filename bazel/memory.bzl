# C++ standard for Memory — mirrors CMake MEMORY_CXX_STANDARD (default: 20)
MEMORY_CXX_STD = "c++20"

def memory_copts():
    # cstdlib_include is left out: CMake skips -include cstdlib for Memory specifically
    # under Clang (compiler-instability workaround, see CMakeLists.txt), so Bazel omits
    # it unconditionally here instead of selecting it per toolchain.
    return select({
        "@platforms//os:windows": [
            "/std:" + MEMORY_CXX_STD,
            "/Zc:__cplusplus",  # expose correct __cplusplus value (mirrors CMake /Zc:__cplusplus)
            "/EHsc",            # structured exception handling
            "/bigobj",          # large object files (mirrors CMake /bigobj)
            "/utf-8",           # UTF-8 source/output encoding (mirrors CMake /utf-8)
            "/wd4244",          # narrowing conversion (mirrors CMake /wd4244)
            "/wd4267",          # size_t → int conversion (mirrors CMake /wd4267)
            "/wd4715",          # not all control paths return (mirrors CMake /wd4715)
            "/wd4018",          # signed/unsigned comparison (mirrors CMake /wd4018)
            "/WX",              # warnings as errors (mirrors CMake /WX)
        ],
        "//conditions:default": [
            "-std=" + MEMORY_CXX_STD,
            "-Wall",
            "-Wextra",
            "-Wpedantic",
        ],
    })

def memory_defines():
    """Returns compile definitions for the Memory library.

    Mirrors this repo's CMakeLists.txt MEMORY_HAS_* flags.
    """
    # Windows CRT safety defines (mirrors CMake compiler_checks.cmake).
    defines = select({
        "@platforms//os:windows": [
            "_CRT_SECURE_NO_DEPRECATE",
            "_CRT_NONSTDC_NO_DEPRECATE",
            "_CRT_SECURE_NO_WARNINGS",
            "_SCL_SECURE_NO_DEPRECATE",
            "_SCL_SECURE_NO_WARNINGS",
        ],
        "//conditions:default": [],
    })

    # CUDA — MEMORY_HAS_CUDA
    defines += select({
        "//bazel:enable_cuda": ["MEMORY_HAS_CUDA=1"],
        "//conditions:default": ["MEMORY_HAS_CUDA=0"],
    })

    # HIP — MEMORY_HAS_HIP
    defines += select({
        "//bazel:enable_hip": ["MEMORY_HAS_HIP=1"],
        "//conditions:default": ["MEMORY_HAS_HIP=0"],
    })

    # Metal — MEMORY_HAS_METAL (Apple only)
    defines += select({
        "//bazel:enable_metal": ["MEMORY_HAS_METAL=1"],
        "//conditions:default": ["MEMORY_HAS_METAL=0"],
    })

    # TBB scalable allocator — MEMORY_HAS_TBB
    defines += select({
        "//bazel:memory_enable_tbb": ["MEMORY_HAS_TBB=1"],
        "//conditions:default": ["MEMORY_HAS_TBB=0"],
    })

    # mimalloc — MEMORY_HAS_MIMALLOC (default ON; matches CMake MEMORY_ENABLE_MIMALLOC)
    defines += select({
        "//bazel:disable_mimalloc": ["MEMORY_HAS_MIMALLOC=0"],
        "//conditions:default": ["MEMORY_HAS_MIMALLOC=1"],
    })

    # NUMA — MEMORY_HAS_NUMA (Unix only)
    defines += select({
        "//bazel:enable_numa": ["MEMORY_HAS_NUMA=1"],
        "//conditions:default": ["MEMORY_HAS_NUMA=0"],
    })

    # memkind — MEMORY_HAS_MEMKIND (Linux only)
    defines += select({
        "//bazel:enable_memkind": ["MEMORY_HAS_MEMKIND=1"],
        "//conditions:default": ["MEMORY_HAS_MEMKIND=0"],
    })

    # Profiler is optional in the standalone build; the root BUILD.bazel only links
    # @profiler//:Profiler when it's available, so this mirrors CMake's
    # MEMORY_ENABLE_PROFILER default (ON) rather than hardcoding 1.
    defines += select({
        "//bazel:disable_profiler": ["MEMORY_HAS_PROFILER=0"],
        "//conditions:default": ["MEMORY_HAS_PROFILER=1"],
    })

    return defines

def memory_linkopts():
    return select({
        "@platforms//os:windows": [],
        "@platforms//os:macos": [
            "-undefined",
            "dynamic_lookup",
        ],
        "//conditions:default": [
            "-lpthread",
            "-ldl",
        ],
    })
