load("//bazel:xsigma.bzl", "xsigma_copts", "xsigma_defines", "xsigma_linkopts")

# C++ standard for Profiler — mirrors CMake PROFILER_CXX_STANDARD (default: 20)
PROFILER_CXX_STD = "c++20"

def profiler_copts():
    return xsigma_copts(cxx_std = PROFILER_CXX_STD)

def profiler_defines():
    """Returns compile definitions for the ThirdParty Profiler package.

    Native product pipeline only. Project-wide PROJECT_HAS_* flags come from
    xsigma_defines(). Product sources require the HAS_* macros below.
    """
    defines = xsigma_defines()

    defines += [
        "PROFILER_HAS_KINETO=0",
        "PROFILER_HAS_ITT=0",
    ]

    # PROFILER_HAS_CUDA / PROFILER_HAS_HIP — independent of each other's build
    # (MEMORY_GPU_BACKEND only ever selects one GPU vendor). Mirrors
    # CMakeLists.txt's find_package(CUDAToolkit) / find_package(hip) gates.
    defines += select({
        # Match CMake: prefer the NVTX C API (nvToolsExt.h / CUDA::nvToolsExt).
        # CUDA 12's cuda-nvtx package often has no nvtx3.hpp, so forcing
        # PROFILER_CUDA_USE_NVTX3=1 fails the Bazel CUDA compile.
        "//bazel:enable_cuda": [
            "PROFILER_HAS_CUDA=1",
            "PROFILER_HAS_NVTX=1",
        ],
        "//conditions:default": ["PROFILER_HAS_CUDA=0", "PROFILER_HAS_NVTX=0"],
    })
    defines += select({
        # Roctracer/ROCTX is optional; CI's hiplibsdk often has HIP runtime
        # without roctx.h. CMake sets PROFILER_HAS_ROCTX only when find_library
        # succeeds — keep Bazel at 0 so cuda.cpp uses the no-op markers.
        "//bazel:enable_hip": ["PROFILER_HAS_HIP=1", "PROFILER_HAS_ROCTX=0"],
        "//conditions:default": ["PROFILER_HAS_HIP=0", "PROFILER_HAS_ROCTX=0"],
    })
    defines += select({
        "//bazel:enable_metal": ["PROFILER_HAS_METAL=1"],
        "//conditions:default": ["PROFILER_HAS_METAL=0"],
    })

    return defines

def profiler_linkopts():
    return xsigma_linkopts() + select({
        "//bazel:enable_metal": ["-framework", "Metal", "-framework", "Foundation"],
        "//conditions:default": [],
    }) + select({
        # nvtx3.hpp dlopens the injector; FindCUDAToolkit's CUDA::nvtx3
        # target adds CMAKE_DL_LIBS for the same reason. Linux-only: -ldl
        # is invalid on Windows and redundant on Apple.
        "@platforms//os:linux": ["-ldl"],
        "//conditions:default": [],
    })
