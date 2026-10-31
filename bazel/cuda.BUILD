# @local_config_cuda overlay for NVIDIA CUDA Toolkit (Linux/Windows, x86-64).
# Provides the :cuda cc_library that BUILD.bazel selects when
# --define=memory_gpu_backend=cuda is passed. See cuda_repository.bzl for
# toolkit discovery. Linux links the driver stub and runs against the
# installed NVIDIA driver; the stub is not included in runtime runfiles.
package(default_visibility = ["//visibility:public"])

# cuda.lib — driver API stubs (cuInit, cuMemAddressReserve, cuMemCreate, …).
# The actual implementation is in the installed NVIDIA display driver; the .lib
# is just an import stub that resolves the symbols at load time.
cc_import(
    name = "cuda_driver_import",
    hdrs = [],
    interface_library = select({
        "@platforms//os:windows": "lib/x64/cuda.lib",
        "//conditions:default": "lib64/stubs/libcuda.so",
    }),
    system_provided = True,
)

# cudart.lib — CUDA runtime API (cudaMalloc, cudaFree, cudaMemcpy, …).
cc_import(
    name = "cudart_import",
    hdrs = [],
    interface_library = select({
        "@platforms//os:windows": "lib/x64/cudart.lib",
        "//conditions:default": None,
    }),
    shared_library = select({
        "@platforms//os:windows": None,
        "//conditions:default": "lib64/libcudart.so",
    }),
    system_provided = select({
        "@platforms//os:windows": True,  # cudart64_*.dll is on PATH
        "//conditions:default": False,
    }),
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
