"""Wrap installed SDKs without hardcoded versions or undeclared include files.

Repository rules are fetched only when their optional targets are selected.
Environment inputs are tracked so changing an install prefix invalidates the repo.
"""

_LOAD = 'load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")\npackage(default_visibility = ["//visibility:public"])\n'

def _root(ctx, variables, marker, defaults = [], required = True):
    candidates = [ctx.os.environ[v] for v in variables if ctx.os.environ.get(v)]
    candidates += defaults
    for candidate in candidates:
        path = ctx.path(candidate)
        # TBB_DIR commonly points at lib/cmake/TBB rather than the prefix.
        for _ in range(7):
            if path.get_child(marker).exists:
                return path
            if path.dirname == path:
                break
            path = path.dirname
    if not required:
        return None
    fail("%s: installation not found; set %s to a prefix containing %s" % (
        ctx.name,
        " or ".join(variables),
        marker,
    ))

def _find(root, directories, names, required = True):
    for directory in directories:
        for name in names:
            path = root.get_child(directory + "/" + name)
            if path.exists:
                return path
    if required:
        fail("Missing %s under %s (searched %s)" % (", ".join(names), root, directories))
    return None

def _shared(root, directories, stem, os_name):
    names = [stem + ".dll"] if "windows" in os_name else ["lib" + stem + ".dylib", "lib" + stem + ".so"]
    # Prefer the SONAME-level filename (e.g. .so.2 or .2.dylib) over the
    # unversioned development symlink, so runfiles contain the loader's name.
    for directory in directories:
        folder = root.get_child(directory)
        if not folder.exists:
            continue
        versioned = []
        for path in folder.readdir():
            name = path.basename
            if name.startswith("lib" + stem + ".so.") or (name.startswith("lib" + stem + ".") and name.endswith(".dylib")):
                versioned.append(name)
        if versioned:
            names = sorted(versioned, key = _filename_length) + names
    return _find(root, directories, names)

def _filename_length(name):
    return len(name)

def _import(ctx, name, root, directories, static_names, stem, dll_dirs = []):
    static = _find(root, directories, static_names, required = False)
    if static:
        out = "lib/" + static.basename
        ctx.symlink(static, out)
        return 'cc_import(name = %r, static_library = %r)\n' % (name, out)
    shared = _shared(root, dll_dirs or directories, stem, ctx.os.name)
    out = "lib/" + shared.basename
    ctx.symlink(shared, out)
    attrs = 'shared_library = %r' % out
    if "windows" in ctx.os.name:
        interface = _find(root, directories, [stem + ".lib"])
        interface_out = "lib/" + interface.basename
        ctx.symlink(interface, interface_out)
        attrs += ', interface_library = %r' % interface_out
    return 'cc_import(name = %r, %s)\n' % (name, attrs)

def _headers(ctx, root):
    ctx.symlink(root.get_child("include"), "include")

def _tbb_impl(ctx):
    # @tbb is named from an unselected select() branch, so Bazel loads it even
    # when memory_enable_tbb is false. A missing install must stay loadable.
    root = _root(
        ctx,
        ["TBB_ROOT", "TBB_DIR", "TBBROOT"],
        "include/tbb/scalable_allocator.h",
        ["/opt/homebrew/opt/tbb", "/usr/local", "/usr"],
        required = False,
    )
    if not root:
        ctx.file(
            "BUILD.bazel",
            _LOAD +
            "# TBB not found. Set TBB_ROOT, TBB_DIR, or TBBROOT to a prefix containing\n" +
            "# include/tbb/scalable_allocator.h before enabling memory_enable_tbb.\n" +
            'cc_library(name = "tbbmalloc")\n' +
            'cc_library(name = "tbb")\n',
        )
        return
    _headers(ctx, root)
    directories = ["lib", "lib64", "lib/x86_64-linux-gnu", "lib/aarch64-linux-gnu", "lib/intel64/vc14"]
    dll_dirs = ["bin", "redist/intel64/vc14", "redist/intel64/vc14_release"]
    build = _LOAD
    build += _import(ctx, "tbbmalloc_import", root, directories, ["libtbbmalloc.a"], "tbbmalloc", dll_dirs if "windows" in ctx.os.name else [])
    build += 'cc_library(name = "tbbmalloc", hdrs = glob(["include/**/*.h"]), includes = ["include"], deps = [":tbbmalloc_import"])\n'
    ctx.file("BUILD.bazel", build)

tbb_repository = repository_rule(
    implementation = _tbb_impl,
    local = True,
    environ = ["TBB_ROOT", "TBB_DIR", "TBBROOT"],
)

def _cuda_impl(ctx):
    if "mac" in ctx.os.name:
        fail("CUDA is supported on Linux and Windows; select memory_gpu_backend=metal on macOS")
    root = _root(ctx, ["CUDA_ROOT", "CUDA_PATH", "CUDA_HOME"], "include/cuda_runtime.h", ["/usr/local/cuda", "/opt/cuda"])
    _headers(ctx, root)
    directories = ["lib64", "lib/x64", "lib", "targets/x86_64-linux/lib", "targets/aarch64-linux/lib"]
    build = _LOAD
    if "windows" in ctx.os.name:
        driver = _find(root, directories, ["cuda.lib"])
    else:
        driver = _find(root, ["lib64/stubs", "lib64", "targets/x86_64-linux/lib/stubs", "targets/aarch64-linux/lib/stubs"], ["libcuda.so"])
    driver_out = "lib/cuda.lib" if "windows" in ctx.os.name else "lib/cuda.ifso"
    ctx.symlink(driver, driver_out)
    build += 'cc_import(name = "driver", interface_library = %r, system_provided = True)\n' % driver_out
    build += _import(ctx, "runtime", root, directories, ["cudart_static.lib", "libcudart_static.a"], "cudart", ["bin", "bin/x64"] if "windows" in ctx.os.name else [])
    system_libs = ["advapi32.lib"] if "windows" in ctx.os.name else ["-ldl", "-lpthread", "-lrt"]
    build += 'cc_library(name = "cuda", hdrs = glob(["include/**"]), includes = ["include"], deps = [":driver", ":runtime"], linkopts = %r)\n' % system_libs
    ctx.file("BUILD.bazel", build)

cuda_repository = repository_rule(
    implementation = _cuda_impl,
    local = True,
    environ = ["CUDA_ROOT", "CUDA_PATH", "CUDA_HOME"],
)

def _hip_impl(ctx):
    if "windows" in ctx.os.name or "mac" in ctx.os.name:
        fail("Memory's HIP backend requires Linux ROCm")
    root = _root(ctx, ["ROCM_ROOT", "ROCM_PATH", "HIP_PATH"], "include/hip/hip_runtime.h", ["/opt/rocm"])
    _headers(ctx, root)
    build = _LOAD + _import(ctx, "runtime", root, ["lib", "lib64"], [], "amdhip64")
    build += 'cc_library(name = "hip", hdrs = glob(["include/**"]), includes = ["include"], deps = [":runtime"], linkopts = ["-ldl", "-lpthread"])\n'
    ctx.file("BUILD.bazel", build)

hip_repository = repository_rule(
    implementation = _hip_impl,
    local = True,
    environ = ["ROCM_ROOT", "ROCM_PATH", "HIP_PATH"],
)

def _profiler_impl(ctx):
    root = _root(ctx, ["PROFILER_ROOT"], "include/profiler.h")
    _headers(ctx, root)
    # A shared installation encapsulates Profiler's private Kineto/ITT deps.
    # Static Profiler packages require CMake's ProfilerStaticThirdParty metadata.
    build = _LOAD + _import(ctx, "profiler_import", root, ["lib", "lib64"], [], "Profiler", ["bin", "lib"] if "windows" in ctx.os.name else [])
    build += 'cc_library(name = "Profiler", hdrs = glob(["include/**/*.h"]), includes = ["include"], defines = ["PROFILER_SHARED_DEFINE"], deps = [":profiler_import"])\n'
    ctx.file("BUILD.bazel", build)

profiler_repository = repository_rule(
    implementation = _profiler_impl,
    local = True,
    environ = ["PROFILER_ROOT"],
)
