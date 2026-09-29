# Overlay BUILD for @com_google_benchmark — mirrors ThirdParty/benchmark/BUILD.bazel
# but upgrades MSVC /std:c++17 → /std:c++20 to silence D9025 against our global
# .bazelrc /std:c++20, and replaces module_version() (bzlmod-only) with the
# vendored version string.
#
# Switch from local_repository to new_local_repository in WORKSPACE.bazel to
# activate this overlay instead of the submodule's own BUILD.bazel.

load("@rules_cc//cc:defs.bzl", "cc_library")

licenses(["notice"])

COPTS = [
    "-pedantic",
    "-pedantic-errors",
    "-std=c++20",
    "-Wall",
    "-Wconversion",
    "-Wextra",
    "-Wshadow",
    "-Wfloat-equal",
    "-Wformat=2",
    "-fstrict-aliasing",
    "-Wno-unused-variable",
    "-Werror=old-style-cast",
]

MSVC_COPTS = [
    "/std:c++20",
]

config_setting(
    name = "windows",
    constraint_values = ["@platforms//os:windows"],
    visibility = [":__subpackages__"],
)

config_setting(
    name = "perfcounters",
    define_values = {"pfm": "1"},
    visibility = [":__subpackages__"],
)

cc_library(
    name = "benchmark",
    srcs = glob(
        ["src/*.cc", "src/*.h"],
        exclude = ["src/benchmark_main.cc"],
    ),
    hdrs = [
        "include/benchmark/benchmark.h",
        "include/benchmark/benchmark_api.h",
        "include/benchmark/counter.h",
        "include/benchmark/export.h",
        "include/benchmark/macros.h",
        "include/benchmark/managers.h",
        "include/benchmark/registration.h",
        "include/benchmark/reporter.h",
        "include/benchmark/state.h",
        "include/benchmark/statistics.h",
        "include/benchmark/sysinfo.h",
        "include/benchmark/types.h",
        "include/benchmark/utils.h",
    ],
    copts = select({
        ":windows": MSVC_COPTS,
        "//conditions:default": COPTS,
    }),
    defines = [
        "BENCHMARK_STATIC_DEFINE",
        "BENCHMARK_VERSION=\\\"1.9.5\\\"",
    ] + select({
        ":perfcounters": ["HAVE_LIBPFM"],
        "//conditions:default": [],
    }),
    includes = ["include"],
    linkopts = select({
        ":windows": ["-DEFAULTLIB:shlwapi.lib"],
        "//conditions:default": ["-pthread"],
    }),
    linkstatic = True,
    local_defines = [
        "_FILE_OFFSET_BITS=64",
        "_LARGEFILE64_SOURCE",
        "_LARGEFILE_SOURCE",
    ],
    visibility = ["//visibility:public"],
    deps = select({
        ":perfcounters": ["@libpfm"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "benchmark_main",
    srcs = ["src/benchmark_main.cc"],
    hdrs = [
        "include/benchmark/benchmark.h",
        "include/benchmark/benchmark_api.h",
        "include/benchmark/counter.h",
        "include/benchmark/export.h",
        "include/benchmark/macros.h",
        "include/benchmark/managers.h",
        "include/benchmark/registration.h",
        "include/benchmark/reporter.h",
        "include/benchmark/state.h",
        "include/benchmark/statistics.h",
        "include/benchmark/sysinfo.h",
        "include/benchmark/types.h",
        "include/benchmark/utils.h",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [":benchmark"],
)

cc_library(
    name = "benchmark_internal_headers",
    hdrs = glob(["src/*.h"]),
    visibility = ["//test:__pkg__"],
)
