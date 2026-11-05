"""Compatibility entry point for the installed oneTBB repository rule.

The rule is implemented in installed_repositories.bzl. When TBB_ROOT, TBB_DIR,
TBBROOT, and the standard prefixes do not contain oneTBB, it writes stub
@tbb targets so a default memory_enable_tbb=false build can still load @tbb.
"""

load(":installed_repositories.bzl", _tbb_repository = "tbb_repository")

tbb_repository = _tbb_repository
