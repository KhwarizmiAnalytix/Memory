"""Analysis-time failure for an unavailable HIP/ROCm toolkit.

hip_repository.bzl generates a BUILD file loading hip_unavailable() instead of
a real cc_library when ROCm isn't usable (Windows, or no ROCm install found on
Unix). fail() inside a rule implementation only runs when that target is
actually analyzed, which happens only when //bazel:enable_hip is the chosen
select() arm — so builds that don't request the hip backend never pay for it,
matching Cmake/hip.cmake's configure-time FATAL_ERROR being conditioned the
same way.
"""

def _hip_unavailable_impl(ctx):
    fail(ctx.attr.message)

hip_unavailable = rule(
    implementation = _hip_unavailable_impl,
    attrs = {
        "message": attr.string(mandatory = True),
    },
    doc = "Fails analysis with `message` when depended upon.",
)
