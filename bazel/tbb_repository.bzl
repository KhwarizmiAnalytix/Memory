"""System-installed oneTBB — mirrors cmake/tbb_memory.cmake's find_package pattern.

Checks TBB_ROOT / TBB_DIR / TBBROOT environment variables for an installed
oneTBB package and creates a Bazel repository that wraps it.  No vendored
ThirdParty/tbb submodule is needed.

Only the three needed subdirectories (include/, lib/, redist/) are symlinked
into the repository root — the TBB installation's own BUILD.bazel is never
exposed as a Bazel subpackage.  Target paths follow the oneTBB binary-release
package layout:
  lib/intel64/vc14/tbbmalloc.lib   (Windows import lib)
  redist/intel64/vc14/tbbmalloc.dll
  include/tbb/scalable_allocator.h  (and other headers)

When TBB_ROOT is not set, stub (empty) targets are written so that builds
with --define=memory_enable_tbb=false load @tbb without error.  Builds with
--define=memory_enable_tbb=true will fail at compile time with a clear message
pointing to the missing env var.

Usage in WORKSPACE.bazel:
    load("//bazel:tbb_repository.bzl", "tbb_repository")
    tbb_repository(name = "tbb")
"""

_BUILD_TEMPLATE = """\
package(default_visibility = ["//visibility:public"])

# include/, lib/, redist/ are symlinks to the TBB installation root's
# corresponding subdirectories (see tbb_repository.bzl).  Paths are identical
# to the original ThirdParty/tbb.BUILD so existing @tbb//:tbbmalloc deps work.

cc_import(
    name = "tbbmalloc_prebuilt",
    hdrs = [],
    interface_library = "lib/intel64/vc14/tbbmalloc.lib",
    shared_library = "redist/intel64/vc14/tbbmalloc.dll",
)

cc_library(
    name = "tbbmalloc",
    hdrs = glob([
        "include/tbb/scalable_allocator.h",
        "include/tbb/tbb_allocator.h",
        "include/tbb/cache_aligned_allocator.h",
    ], allow_empty = True),
    includes = ["include"],
    deps = [":tbbmalloc_prebuilt"],
)

cc_import(
    name = "tbb_prebuilt",
    hdrs = [],
    interface_library = "lib/intel64/vc14/tbb12.lib",
    shared_library = "redist/intel64/vc14/tbb12.dll",
)

cc_library(
    name = "tbb",
    hdrs = glob([
        "include/tbb/*.h",
        "include/oneapi/tbb/*.h",
        "include/oneapi/tbb/detail/*.h",
    ], allow_empty = True),
    includes = ["include"],
    deps = [
        ":tbb_prebuilt",
        ":tbbmalloc",
    ],
)
"""

_STUB_TEMPLATE = """\
# TBB not found — TBB_ROOT / TBB_DIR / TBBROOT is not set.
# Stub targets allow @tbb to load for builds where memory_enable_tbb=false
# (the default).  Any build with --define=memory_enable_tbb=true will fail at
# compile time; set TBB_ROOT and re-run to fix it.
#
# RECOMMENDED INSTALLATION METHODS:
#
# 1. CI / official binaries (recommended):
#    .github/workflows/install/install-deps-windows.ps1 -WithTbb
#    (downloads oneTBB Windows binaries and sets TBB_ROOT)
#
# 2. Using vcpkg:
#    vcpkg install tbb:x64-windows
#    set TBB_ROOT=<vcpkg root>/installed/x64-windows
#
# 3. Manual installation:
#    - Download from: https://github.com/uxlfoundation/oneTBB/releases
#    - Extract and set TBB_ROOT (e.g. set TBB_ROOT=C:/tbb)

package(default_visibility = ["//visibility:public"])

cc_library(name = "tbbmalloc", hdrs = [], srcs = [])
cc_library(name = "tbb", hdrs = [], srcs = [])
"""

def _tbb_repository_impl(repository_ctx):
    # Check environment variables in priority order —
    # mirrors cmake/tbb_memory.cmake: TBB_ROOT → TBB_DIR → TBBROOT (Intel oneAPI).
    tbb_root = None
    for var in ("TBB_ROOT", "TBB_DIR", "TBBROOT"):
        val = repository_ctx.os.environ.get(var, "")
        if val:
            tbb_root = val
            break

    repository_ctx.file("WORKSPACE", 'workspace(name = "tbb")\n')

    if not tbb_root:
        # Emit stubs so that --define=memory_enable_tbb=false builds (the
        # default) continue to load @tbb without error.
        repository_ctx.file("BUILD.bazel", _STUB_TEMPLATE)
        return

    src = repository_ctx.path(tbb_root)
    if not src.exists:
        fail("TBB_ROOT points to a non-existent path: %s" % tbb_root)

    # Symlink only the three subdirectories we need so that any BUILD.bazel
    # inside the TBB installation root is never picked up as a Bazel subpackage.
    # Paths in BUILD_TEMPLATE are relative to these symlinks (include/,
    # lib/intel64/vc14/, redist/intel64/vc14/) — identical to the old
    # ThirdParty/tbb.BUILD so existing @tbb//:tbbmalloc deps keep working.
    for sub in ("include", "lib", "redist"):
        child = src.get_child(sub)
        if child.exists:
            repository_ctx.symlink(child, sub)
    repository_ctx.file("BUILD.bazel", _BUILD_TEMPLATE)

tbb_repository = repository_rule(
    implementation = _tbb_repository_impl,
    attrs = {},
    local = True,
    environ = ["TBB_ROOT", "TBB_DIR", "TBBROOT"],
    doc = "Wraps a system-installed oneTBB package found via TBB_ROOT/TBB_DIR/TBBROOT.",
)
