---
name: project-build
description: Cheat sheet for configuring, building, and testing this repo via Scripts/setup.py (CMake) and Scripts/setup_bazel.py (Bazel) — dotted-token syntax, sanitizers, coverage, GPU backends. Use whenever the user asks to build, configure, test, or run a sanitizer/coverage pass on this repo.
---

# project-build

This repo is configured/built/tested through `Scripts/setup.py` (CMake) and
`Scripts/setup_bazel.py` (Bazel) — see root `/CLAUDE.md`. Run both from the
`Scripts/` directory. `python3 setup.py --help` / `python3 setup_bazel.py --help`
are the source of truth; this is a distilled cheat sheet of the common cases.
The `session-checklist` skill expects both build systems to build and test
clean for any non-trivial change.

## CMake (`setup.py`)

### Core invocation shape

Dotted tokens chain together, case-insensitive, in any order:
```
python3 setup.py <config|build|test>.<compiler/generator>.<feature tokens...> [--flag=value ...]
```

```
# Standard dev loop: configure + build + run tests, Ninja + Clang (defaults)
python3 setup.py config.build.test

# Release + GCC
python3 setup.py config.build.test.gcc.release

# macOS + Xcode
python3 setup.py config.build.test.xcode

# Coverage (analysis runs automatically)
python3 setup.py config.build.test.coverage

# GPU backend, with TBB
python3 setup.py config.build.test.metal.tbb
python3 setup.py config.build.test --gpu.cuda
```

### GPU backend selection

`--gpu.none|.cuda|.hip|.metal` (or bare `cuda`/`hip`/`metal` as a dotted
token). HIP is Unix-only — fails fast with a clear message on Windows by
policy, see `Cmake/hip.cmake`'s `WIN32` guard (mirrored in Bazel by
`bazel/hip_repository.bzl`). Metal is Apple-only.

### Sanitizers

```
python3 setup.py config.build.test --sanitizer.address
python3 setup.py config.build.test --sanitizer.undefined
python3 setup.py config.build.test --sanitizer.thread
```
Also: `--sanitizer.memory` (Clang only), `--sanitizer.leak`.

### Static analysis (clang-tidy)

`clangtidy` / `fix` tokens — see the dedicated `clang-tidy` skill for the two
available mechanisms (build-integrated vs. lintrunner), since fix-mode
rewrites source files and needs to be scoped carefully.

### Other useful tokens

`static` (static instead of default shared libs), `test`, `examples`,
`gtest`, `benchmark`, `profiler`, `mimalloc`/`mimallocstats`, `memkind`
(Linux), `tbb`, `numa` (Linux), `icecc`, `cache`/`cache_type`, `iwyu`,
`spell`, `valgrind`, `linker.<mold|lld|gold|lld-link>`, `cxxstd` (cxx11/14/17/20/23),
`lto[.<thin|full|ipo|auto>]`, `cppcheck`.

## Bazel (`setup_bazel.py`)

Equivalent shape, same actions:
```
python3 setup_bazel.py config.build.test
python3 setup_bazel.py build.test.release
python3 setup_bazel.py build.test.cuda
python3 setup_bazel.py build.test.tbb
python3 setup_bazel.py test
python3 setup_bazel.py coverage          # llvm-profdata/llvm-cov + genhtml, Clang only
python3 setup_bazel.py clean.build.test.release
```

`--define` flags this maps to (see `bazel/BUILD.bazel` config_settings):
`cuda`/`hip`/`metal` → `memory_gpu_backend`; `tbb` → `memory_enable_tbb=true`
(needs `TBB_ROOT` env var — see `bazel/tbb_repository.bzl`); `nomimalloc` →
`memory_enable_mimalloc=false` (enabled by default); `numa`, `memkind`,
`profiler`, `shared` map similarly.

Not available on the Bazel side (use Bazel flags directly, or the CMake side
for these): `icecc`, `examples`, `sanitizer.*` (no sanitizer `--config` is
defined in this repo's `.bazelrc`), `cache` (use `--disk_cache=<path>` or
`--remote_cache=<url>` instead).

`clangtidy`/`iwyu` on the Bazel side currently only print an aspect-invocation
hint (no `.bzl` aspect wired up yet) — use the CMake side's `clangtidy` token,
or lintrunner, for an actual clang-tidy pass; see the `clang-tidy` skill.

## Coverage

CMake: `python3 setup.py config.build.test.coverage` — analysis runs
automatically, no separate step needed. Coverage thresholds/targets are
configured in `codecov.yml` (project/patch `auto` target, 1% regression
threshold — not a fixed numeric floor).

Bazel: `python3 setup_bazel.py coverage` (Clang only; lcov report).
