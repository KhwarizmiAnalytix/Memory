"""Wrap the installed CUDA host runtime and driver imports on Linux/Windows."""

def _cuda_repository_impl(repository_ctx):
    toolkit = repository_ctx.os.environ.get("CUDA_PATH") or repository_ctx.os.environ.get("CUDA_HOME")
    if not toolkit:
        nvcc = repository_ctx.which("nvcc")
        if nvcc:
            toolkit = str(nvcc.realpath.dirname.dirname)
        elif repository_ctx.os.name.lower().startswith("windows"):
            toolkit = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2"
        else:
            toolkit = "/usr/local/cuda"

    root = repository_ctx.path(toolkit)
    if not root.get_child("include/cuda.h").exists:
        fail("CUDA toolkit not found at %s. Set CUDA_PATH or CUDA_HOME to the SDK root." % toolkit)

    for subdir in ("include", "lib", "lib64"):
        path = root.get_child(subdir)
        if path.exists:
            repository_ctx.symlink(path, subdir)
    repository_ctx.template("BUILD.bazel", repository_ctx.attr.build_file, {})

cuda_repository = repository_rule(
    implementation = _cuda_repository_impl,
    attrs = {
        "build_file": attr.label(default = "//bazel:cuda.BUILD", allow_single_file = True),
    },
    environ = ["CUDA_PATH", "CUDA_HOME", "PATH"],
    local = True,
)
