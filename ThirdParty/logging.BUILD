# Overlay BUILD for the KhwarizmiAnalytix/Logging submodule (ThirdParty/Logging).
# Do not edit files inside that submodule — this overlay lives in this repo, mirroring
# the same technique XSigma itself uses for its own ThirdParty/logging.BUILD.
# This repo depends on @logging//:Logging only. Tracks the submodule's own
# BUILD.bazel (include/ headers, src/ implementations).
load("//bazel:logging.bzl", "logging_copts", "logging_defines", "logging_linkopts")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "logging_hdrs",
    srcs = glob(
        [
            "include/*.h",
            "include/common/*.h",
            "include/util/*.h",
            "include/logger/*.h",
            "include/backend/*.h",
        ],
        allow_empty = True,
    ),
)

filegroup(
    name = "logging_srcs",
    # Implementation lives under src/ (mirrors CMakeLists.txt). The per-backend
    # files are compile-time guarded, so globbing all of them is safe.
    srcs = glob(
        [
            "src/*.cpp",
            "src/backend/*.cpp",
        ],
        allow_empty = True,
    ),
)

cc_library(
    name = "logging_lib",
    srcs = [
        ":logging_srcs",
    ],
    hdrs = [":logging_hdrs"],
    copts = logging_copts(),
    defines = logging_defines() + select({
        "//bazel:shared_libs": ["LOGGING_SHARED_DEFINE"],
        "//conditions:default": ["LOGGING_STATIC_DEFINE"],
    }),
    local_defines = select({
        "//bazel:shared_libs": ["LOGGING_BUILDING_DLL"],
        "//conditions:default": [],
    }),
    # No "." in includes: the package root is the staged repo root, so Bazel's
    # default -iquote already resolves "include/..." headers.
    # "include/logger" stays: sources include its headers by bare name (e.g.
    # "logger_verbosity_enum.h" from within include/logger/logger.h itself).
    includes = [
        "",
        "include/logger",
    ],
    linkopts = logging_linkopts() + select({
        "@platforms//os:windows": ["dbghelp.lib"],
        "//conditions:default": [],
    }),
    linkstatic = select({
        "//bazel:shared_libs": False,
        "//conditions:default": True,
    }),
    deps = [
        "@fmt//:fmt",
    ] + select({
        "//bazel:disable_magic_enum": [],
        "//conditions:default": ["@magic_enum//:magic_enum"],
    }) + select({
        "//bazel:logging_glog": ["@glog//:glog"],
        "//bazel:logging_native": [],
        "//bazel:logging_spdlog": ["@spdlog//:spdlog"],
        "//conditions:default": [],
    }),
    alwayslink = False,
)

cc_library(
    name = "Logging",
    deps = [":logging_lib"],
    visibility = ["//visibility:public"],
)
