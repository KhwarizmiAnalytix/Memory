---
name: session-checklist
description: End-of-session / pre-commit checklist for this repo — verify CMake and Bazel build definitions stay in sync, build and test both build systems, and run lintrunner and fix issues. Use when wrapping up a session that made non-trivial changes, or before creating a commit.
---

# session-checklist

Run this before ending a session that made non-trivial changes (new files,
new build/CMake/Bazel config, or a multi-file change) or before creating a
commit. It's a process expectation, not a CI gate — a comment tweak or a
one-line doc fix doesn't need the full sequence, but skip it deliberately,
not by default.

## 1. CMake ↔ Bazel sync check

This repo ships two parallel, independently-maintained build systems
(`Scripts/setup.py` for CMake, `Scripts/setup_bazel.py` for Bazel — see the
root `/CLAUDE.md`). A file or option added to one and not the other builds
fine on whichever system you happened to test and silently breaks (or
silently drops coverage/tests) on the other. There is no automated diff tool
for this yet, so check by reading the relevant files side by side for
anything touched this session:

- **Sources/headers are globbed differently on each side — this is the main
  hazard.** CMake's `file(GLOB_RECURSE ...)` in `CMakeLists.txt` walks all of
  `src/` and `include/` recursively, so a new subdirectory is picked up
  automatically. `BUILD.bazel`'s `memory_srcs`/`memory_hdrs` `filegroup`s use
  an explicit, non-recursive list of glob patterns instead (`src/*.cpp`,
  `src/common/*.cpp`, `src/helper/*.cpp`, `src/profiler/*.cpp`, plus
  `src/gpu/*.cpp` gated behind the GPU-backend `select()`s). **A new
  top-level subdirectory under `src/` needs a new glob line added to
  `BUILD.bazel` manually**, or it silently won't compile under Bazel while
  still working fine under CMake.
- **Feature/config tokens**: a new CMake option (`MEMORY_ENABLE_X`) needs a
  matching `.bazelrc` `--define=` config_setting in `bazel/BUILD.bazel`, and
  the corresponding token handled in `Scripts/setup_bazel.py`. The HIP
  backend gap fixed this session (`@local_config_hip` referenced in
  `BUILD.bazel` but never defined in `MODULE.bazel`, until
  `bazel/hip_repository.bzl` was added) is exactly this class of bug — the
  CMake side (`Cmake/hip.cmake`) had the guard; the Bazel side didn't, until
  now.
- **New dependency / `ThirdParty/` repo wiring**: a new vendored submodule or
  system-detected library needs, on both sides — CMake (`find_package`/
  `Cmake/*.cmake` module) and Bazel (`MODULE.bazel` repo rule or
  `local_repository`/`new_local_repository`, plus a `bazel/*.BUILD` or
  `bazel/*_repository.bzl` if it's not checked in with its own build file;
  see `bazel/cuda_repository.bzl`, `bazel/hip_repository.bzl`,
  `bazel/tbb_repository.bzl` for the system-detection pattern this repo
  uses).

## 2. Build and test — both build systems

CMake (see the `project-build` skill for the full flag cheat sheet):
```
cd Scripts
python3 setup.py config.build.test
```

Bazel:
```
cd Scripts
python3 setup_bazel.py config.build.test
```

Both builds must succeed and both test suites (`//Testing/Cxx:MemoryCxxTests`
on the Bazel side) must pass — a change that only builds under one build
system is not done. If the change touches a GPU backend, build/test it under
each backend that's actually exercisable on the current machine (e.g.
`--gpu.cuda` / `build.test.cuda` on a machine with the CUDA Toolkit
installed) as well as the default `none` backend.

## 3. Lint before commit

```
lintrunner -m main
```
`-m/--merge-base-with main` catches everything that differs from `main`,
not just the current working-tree diff — needed because prior uncommitted
work can still be sitting in the tree. Apply the mechanical fixes first:
```
lintrunner -a -m main
```
then hand-fix whatever `-a` can't patch (docstring length, missing type
annotations, genuine lint errors). Two known pre-existing items are *not*
yours to silently fix as a drive-by — confirm they're still the same (don't
assume, re-check), and only touch them if the user asks:
- `Scripts/setup.py`'s `import colorama` triggers the `IMPORT_LINTER`
  "disallowed import" lint. Pre-existing on `main`; fixing it means removing
  colorama from a widely-used script — a real but separate refactor.
- The `BAZEL_LINTER` "Advice" failure
  (`bazel query 'kind(http_archive, //external:*)'` exits with "no such
  package 'external' ... WORKSPACE file is disabled, please migrate to
  Bzlmod") is a bug in `lint_tool`'s `bazel_linter` adapter — it queries a
  WORKSPACE-era construct this repo doesn't have (pure Bzlmod, no
  `WORKSPACE`/`WORKSPACE.bazel` file). Not fixable by changing repo content.

Two CMake linters in this repo's `.lintrunner.toml` (`cmakelint` via the
`CMAKE` code, and `cmake-format` via `CMAKEFORMAT`) can disagree on
`AND (`/`OR (` spacing when a line starts with the operator immediately
followed by a parenthesized sub-expression — `cmakelint` misreads the
operator as a command name. If the parens are redundant (AND binds tighter
than OR in CMake), just drop them. If they're load-bearing, hoist the
sub-expression into a `set(_var TRUE/FALSE)` guard before the `if()` so no
line starts with `AND (`/`OR (` at all. No file in this repo currently hits
this (only `ThirdParty/` vendored CMake does, which is excluded from both
linters) — but watch for it in any new `if()` you write with a parenthesized
`AND`/`OR` sub-expression at the start of a continuation line.

## Reporting

State plainly which of the three checks passed, which didn't, and why —
don't say "done" without naming what was actually run.
