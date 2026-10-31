---
type: "always_apply"
---

Do not modify any code or build files within third-party submodules or external dependencies. This includes:

- Any files within third-party submodule directories under `ThirdParty/` (e.g. `ThirdParty/mimalloc`, `ThirdParty/googletest`, `ThirdParty/benchmark`, `ThirdParty/Logging` — all managed as git submodules, see `.gitmodules`)
- `CMakeLists.txt` files that belong to third-party libraries
- Source code (`.cpp`, `.h`, `.hpp`, etc.) in third-party dependencies
- Configuration files within third-party library directories
- Build scripts or setup files that are part of external dependencies

Only modify this project's own files (`include/`, `src/`, `Testing/`, `Scripts/`, `Cmake/`, `bazel/`, and project-level `CMakeLists.txt` / `BUILD.bazel` / `MODULE.bazel`). If integration with a third-party library needs adjustment, make changes only to this project's own CMake configuration files (`Cmake/`), Bazel wiring (`bazel/`), or wrapper code — not to the third-party library files themselves.

If a third-party library has issues or needs modifications, document the issue and suggest alternative approaches such as:
- Adjusting this project's CMake or Bazel configuration to work with the library as-is
- Creating wrapper classes or adapter code in `include/`/`src/`
- Suggesting a different version or alternative library
- Reporting the issue upstream to the library maintainers
