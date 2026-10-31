"""System-installed ROCm/HIP — mirrors Cmake/hip.cmake + Cmake/rocm_prefix.cmake.

HIP/ROCm is a Unix (Linux) technology: the ROCm toolkit and hipcc are not
distributed or validated for Windows in this project (see the WIN32 guard in
Cmake/hip.cmake). On Windows this repository always generates a stub that
fails analysis with that same message — never a bare "repository not defined"
error — the moment //bazel:enable_hip is actually selected.

On Unix, ROCM_PATH / HIP_PATH / /opt/rocm[-*] are checked in that order for an
installed ROCm SDK root (same precedence as Cmake/rocm_prefix.cmake). When
found, its include/ and lib/ are wrapped in a cc_library exposing the HIP host
runtime only — mirrors PROJECT_HIP_LIBRARIES=hip::host in Cmake/hip.cmake,
since hip::device-equivalent --offload-arch flags belong on device TUs only,
not leaked PUBLIC. When not found, the stub fails analysis with a message
pointing at ROCM_PATH/HIP_PATH, mirroring hip.cmake's "HIP not found"
FATAL_ERROR.

Either way, builds that never select //bazel:enable_hip are unaffected: the
failure lives in a rule implementation that only runs when :hip is analyzed.

Usage in MODULE.bazel:
    load("//bazel:hip_repository.bzl", "hip_repository")
    hip_repository(name = "local_config_hip")
"""

_WINDOWS_MESSAGE = (
    "MEMORY_GPU_BACKEND=hip is not supported on Windows in this project " +
    "(HIP/ROCm is Unix-only here). Use MEMORY_GPU_BACKEND=cuda on Windows, " +
    "or build on Linux for HIP."
)

_NOT_FOUND_MESSAGE = (
    "HIP not found. Please install ROCm/HIP and ensure ROCM_PATH or " +
    "HIP_PATH points at the SDK root (e.g. /opt/rocm)."
)

_STUB_TEMPLATE = """\
load("@xsigma_memory//bazel:hip_checks.bzl", "hip_unavailable")

package(default_visibility = ["//visibility:public"])

hip_unavailable(
    name = "hip",
    message = {message},
)
"""

_BUILD_TEMPLATE = """\
package(default_visibility = ["//visibility:public"])

# include/, lib/ are symlinks into the detected ROCm SDK root (see
# hip_repository.bzl). Host runtime only, matching hip::host in
# Cmake/hip.cmake -- device offload flags are not modeled here.

cc_library(
    name = "hip",
    hdrs = glob([
        "include/hip/*.h",
        "include/hip/**/*.h",
    ], allow_empty = True),
    includes = ["include"],
    defines = ["__HIP_PLATFORM_AMD__=1"],
    linkopts = ["-L{lib_dir}", "-lamdhip64"],
)
"""

def _find_rocm_root(repository_ctx):
    for var in ("ROCM_PATH", "HIP_PATH"):
        val = repository_ctx.os.environ.get(var, "")
        if val and val not in ("/usr", "/usr/local", "/"):
            return val

    result = repository_ctx.execute(["sh", "-c", "ls -d /opt/rocm /opt/rocm-* 2>/dev/null"])
    for candidate in result.stdout.splitlines():
        candidate = candidate.strip()
        if candidate:
            return candidate
    return None

def _hip_repository_impl(repository_ctx):
    repository_ctx.file("WORKSPACE", 'workspace(name = "local_config_hip")\n')

    if repository_ctx.os.name.lower().startswith("windows"):
        repository_ctx.file("BUILD.bazel", _STUB_TEMPLATE.format(message = repr(_WINDOWS_MESSAGE)))
        return

    rocm_root = _find_rocm_root(repository_ctx)
    if not rocm_root:
        repository_ctx.file("BUILD.bazel", _STUB_TEMPLATE.format(message = repr(_NOT_FOUND_MESSAGE)))
        return

    src = repository_ctx.path(rocm_root)
    if not src.exists:
        repository_ctx.file("BUILD.bazel", _STUB_TEMPLATE.format(message = repr(_NOT_FOUND_MESSAGE)))
        return

    for sub in ("include", "lib"):
        child = src.get_child(sub)
        if child.exists:
            repository_ctx.symlink(child, sub)
    repository_ctx.file("BUILD.bazel", _BUILD_TEMPLATE.format(lib_dir = str(src.get_child("lib"))))

hip_repository = repository_rule(
    implementation = _hip_repository_impl,
    attrs = {},
    local = True,
    environ = ["ROCM_PATH", "HIP_PATH"],
    doc = "Wraps a system-installed ROCm/HIP SDK found via ROCM_PATH/HIP_PATH/opt-rocm, " +
          "or fails analysis with a clear message when unavailable (Windows, or not found).",
)
