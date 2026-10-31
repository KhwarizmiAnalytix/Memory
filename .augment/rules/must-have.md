---
type: "always_apply"
---

## Cross-Platform Compatibility Rule

All code **must be platform-independent** and **not rely on machine-specific configurations**.

### Requirements:
- ✅ Use **standard libraries** or **well-maintained cross-platform dependencies**.
- ✅ Avoid hardcoded file paths, environment variables, or OS-specific commands.
- ✅ Scripts must run consistently on **Linux, macOS, and Windows** unless explicitly scoped otherwise — this repo is actively built and tested on Windows (see `README.md`), so don't assume a Unix-only tool is available without a guard (e.g. `Cmake/hip.cmake`'s `WIN32` guard, mirrored by `bazel/hip_repository.bzl`, for HIP/ROCm being Unix-only here).
- ✅ Provide **containerization (e.g., Docker)** or reproducible environments when platform parity is critical.
- ✅ Use **relative paths** instead of absolute ones.
- ✅ Ensure all build scripts, tests, and tooling work in **CI/CD pipelines** independent of local machines.
- ✅ Document any unavoidable platform-specific requirements and provide fallbacks or alternatives.

### Enforcement:
- Changes are reviewed for cross-platform assumptions.
- Both build systems (CMake and Bazel — see `project-build` and `session-checklist` skills) are expected to build and test on each supported platform.
- Violations must be fixed before merge.
