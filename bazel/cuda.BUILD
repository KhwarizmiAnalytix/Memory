# @local_config_cuda overlay for NVIDIA CUDA Toolkit (Windows, x86-64).
# Provides the :cuda cc_library that BUILD.bazel selects when
# --define=memory_gpu_backend=cuda is passed.  Path is set by the
# new_local_repository in WORKSPACE.bazel.
package(default_visibility = ["//visibility:public"])

# cuda.lib — driver API stubs (cuInit, cuMemAddressReserve, cuMemCreate, …).
# The actual implementation is in the installed NVIDIA display driver; the .lib
# is just an import stub that resolves the symbols at load time.
cc_import(
    name = "cuda_driver_import",
    hdrs = [],
    interface_library = "lib/x64/cuda.lib",
    system_provided = True,
)

# cudart.lib — CUDA runtime API (cudaMalloc, cudaFree, cudaMemcpy, …).
cc_import(
    name = "cudart_import",
    hdrs = [],
    interface_library = "lib/x64/cudart.lib",
    system_provided = True,  # cudart64_*.dll ships with the CUDA toolkit installer
)

cc_library(
    name = "cuda",
    hdrs = glob([
        "include/*.h",
        "include/**/*.h",
        "include/**/*.hpp",
        "include/**/*.cuh",
        "include/**/*.inl",
    ]),
    includes = ["include"],
    deps = [
        ":cuda_driver_import",
        ":cudart_import",
    ],
)
