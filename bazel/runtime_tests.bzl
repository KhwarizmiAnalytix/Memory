"""Compile the real implementation against test-only GPU runtime headers."""

load("@rules_cc//cc:defs.bzl", "cc_test")
load(":memory.bzl", "memory_copts")

def runtime_test(name, srcs, backend, logging = False):
    cc_test(
        name = name,
        srcs = srcs + native.glob(["**/*.h"]),
        copts = memory_copts() + select({
            "@platforms//os:windows": ["/I" + native.package_name()],
            "//conditions:default": ["-I" + native.package_name()],
        }),
        local_defines = [
            "MEMORY_STATIC_DEFINE",
            "MEMORY_HAS_CUDA=" + ("1" if backend == "cuda" else "0"),
            "MEMORY_HAS_HIP=" + ("1" if backend == "hip" else "0"),
            "MEMORY_HAS_METAL=0",
            "MEMORY_HAS_PROFILER=0",
        ],
        deps = ["//:memory_headers_only", "@com_google_googletest//:gtest_main"] + (["@logging//:Logging"] if logging else []),
        size = "small",
        timeout = "short",
    )
