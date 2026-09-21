# Contributing to Logging

## Building

```bash
git submodule update --init --recursive
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
```

Switch backend / options with `-DLOGGING_BACKEND=SPDLOG|GLOG|NATIVE|LOGURU` and any
`-DLOGGING_ENABLE_*` flag documented in [README.md](README.md#cmake-options). Bazel:
`bazel test //...` (see [README.md](README.md#bazel-flags) for `--define` flags).

## Before opening a PR

- Run the test suite against at least the default backend (`LOGURU`); if your change
  touches backend-dispatch code, build all four (`LOGURU`, `SPDLOG`, `GLOG`, `NATIVE`).
- For changes touching memory or object lifetime, build with a sanitizer:
  ```bash
  cmake -S . -B build-asan -G Ninja -DCMAKE_BUILD_TYPE=Debug \
    -DLOGGING_ENABLE_SANITIZER=ON -DLOGGING_SANITIZER_TYPE=address
  cmake --build build-asan && ctest --test-dir build-asan --output-on-failure
  ```
- Add or update a test under `Testing/Cxx/` for behavior changes.
- Keep public CMake/Bazel option changes documented in `README.md`.

## Coverage locally

```bash
cmake -S . -B build-cov -G Ninja -DCMAKE_BUILD_TYPE=Debug -DLOGGING_ENABLE_COVERAGE=ON
cmake --build build-cov
ctest --test-dir build-cov --output-on-failure
lcov --capture --directory build-cov --output-file coverage.info --ignore-errors mismatch,negative,gcov,source
lcov --remove coverage.info '*/ThirdParty/*' '*/Testing/*' '/usr/*' --ignore-errors unused --output-file coverage.info
genhtml coverage.info --output-directory coverage_html
```

CI runs the same steps (`.github/workflows/ci.yml`, `coverage` job) and uploads to Codecov.

## CI

Every PR runs: CMake build+test on Linux (gcc/clang × all backends), macOS, and Windows;
a Bazel build+test; Clang ASan/UBSan; and the coverage job above. See
`.github/workflows/ci.yml`.
