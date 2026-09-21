load("//bazel:xsigma.bzl", "xsigma_copts", "xsigma_defines", "xsigma_linkopts")

# C++ standard for Memory — mirrors CMake MEMORY_CXX_STANDARD (default: 20)
MEMORY_CXX_STD = "c++20"

def memory_copts():
    # cstdlib_include=False: CMake skips -include cstdlib for Memory specifically under
    # Clang (compiler-instability workaround, see CMakeLists.txt); see xsigma_copts()'s
    # docstring for why Bazel omits it unconditionally here instead.
    return xsigma_copts(cxx_std = MEMORY_CXX_STD, cstdlib_include = False)

def memory_defines():
    """Returns compile definitions for the Memory library.

    Mirrors this repo's CMakeLists.txt MEMORY_HAS_* flags.
    """
    defines = xsigma_defines()

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
    return xsigma_linkopts()
