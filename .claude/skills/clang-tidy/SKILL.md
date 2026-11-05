---
name: clang-tidy
description: Run and interpret clang-tidy on this repo — two separate mechanisms (setup.py build-integrated vs. lintrunner), how fix-mode rewrites source, and how to add a per-directory suppression when a check misfires on a specific file. Use when the user asks to run clang-tidy, fix static-analysis/clang-tidy warnings, or add a clang-tidy suppression.
---

# clang-tidy

This repo has **two independent ways to run clang-tidy** — they use different
binaries and different scoping, don't assume they agree or that fixing one
satisfies the other. (Bazel has no clang-tidy integration yet — `setup_bazel.py`'s
`clangtidy`/`iwyu` tokens currently only print a hint that an aspect `.bzl`
setup is needed.)

## 1. Build-integrated (via `setup.py`, CMake `CXX_CLANG_TIDY`/`C_CLANG_TIDY`)

```
cd Scripts
python3 setup.py config.build.clangtidy              # check only, warnings as errors
python3 setup.py config.build.clangtidy.fix           # check AND auto-fix
```
- Uses whatever `clang-tidy` is found on `PATH` (`find_program` in
  `Cmake/clang_tidy.cmake`) — configure fails fast if none is found.
- Runs as part of compilation, target by target; `-warnings-as-errors=*` so
  a clang-tidy finding fails the build.
- `.fix` adds `-fix-errors -fix`, which **rewrites source files in place**
  (`memory_target_clang_tidy()` in `Cmake/clang_tidy.cmake` warns about this).
  **Never run `.fix` on the whole tree with uncommitted, unrelated changes
  in the working directory** — commit or stash first so an unwanted rewrite
  is a trivial `git diff`/`git checkout` away from being undone, and diff
  the result afterward rather than trusting it blindly.
- Header filter is `^<root>/(include|src)/.*`, with a ThirdParty exclude
  filter added when clang-tidy is 19+ (older clang-tidy rejects
  `--exclude-header-filter`; the header filter alone already keeps
  ThirdParty out of the report on older versions too).

## 2. lintrunner (`clang-tidy` from PATH, CI-style check)

```
lintrunner --take CLANGTIDY
lintrunner -a --take CLANGTIDY       # apply available auto-fixes, if any
```
- Resolves `clang-tidy` via PATH (`shutil.which`, code `CLANGTIDY` in
  `.lintrunner.toml`) — **not** a separately pinned/downloaded binary.
  This is deliberate: clang-tidy's bundled resource-dir/builtin headers are
  tightly version-coupled to libc++'s internal header chaining, so a pinned
  binary even a few major versions behind the toolchain that actually
  produced `compile_commands.json` breaks *every* file with bogus "system
  header not found" errors. Make sure whatever `clang-tidy` is first on
  `PATH` matches (or is very close to) the compiler actually used for the
  build.
- `--build_dir` has no hardcoded default in `.lintrunner.toml`; the adapter
  (`lint_tool.adapters.clangtidy_linter`, an installed package — not a file
  in this repo) auto-detects the newest top-level `build_ninja*` (or
  `build`/`build_*`/`out`/`cmake-build-*`) directory with a
  `compile_commands.json`. In practice: whichever configure/build you ran
  most recently is what lintrunner checks against. Pass `--build_dir`
  explicitly to pin a specific one (e.g. for a future CI job).
- **`compile_commands.json` only ever has entries for compiled `.cpp`/`.cc`/
  `.c` files — never headers**, since headers aren't their own translation
  unit. The adapter mirrors how the real build actually validates headers
  (only via inclusion from a `.cpp`, using `--header-filter` — never
  standalone): a `.h`/`.hpp` with no TU entry is silently skipped, not
  flagged. A header's findings surface when a `.cpp` that includes it gets
  checked.
- A `.cpp` file *should* always have a `compile_commands.json` entry once
  the build directory covers it. If one doesn't, the adapter fails loudly
  (a `compile-error` lint message) instead of silently passing — a clean
  result for a `.cpp` means clang-tidy actually analyzed it, not just that
  nothing matched. If you see `compile-error` on many `.cpp` files, the
  auto-detected build directory doesn't cover them; rebuild so it does.
- Scope: lintrunner's `CLANGTIDY` `include_patterns` cover `include/**` and
  `Testing/**`/`Tests/**` only — **not `src/**`** (`src/*.cpp` is checked by
  the build-integrated path instead, since that's what actually compiles
  it). `exclude_patterns` drops `ThirdParty/**` and `bazel-*/**`.
- Not currently run in CI (`.github/workflows/ci.yml` / `toolchains.yml`
  have no lintrunner or clang-tidy job — the root `.clang-tidy` file itself
  says "Local/IDE use. Not yet wired into CI" in its header comment). Treat
  "passes clang-tidy" as a self-discipline check for now, not something CI
  will catch for you.

## The check set is a deliberate allowlist — don't casually add to it

Root `.clang-tidy` enables `bugprone-*`, `performance-*`, a handful of
specific `modernize-*`/`readability-*` checks, then disables a few with an
inline comment explaining why (e.g. `-bugprone-stringview-nullptr` because
clang-tidy 22 access-violates on that check against
`ThirdParty/Logging`'s vendored `magic_enum.hpp` for TUs that include it).
Read the comment above a disabled check before re-enabling it or adding a
new blanket disable at the root — a narrower, file-scoped fix is almost
always more correct (see below). It also sets
`bugprone-reserved-identifier.AllowedIdentifiers` for TBB's reserved
`__TBB_NO_IMPLICIT_LINKAGE` macro name, which can't be renamed.

## When a check misfires on one file: scope the suppression, don't disable it repo-wide

No directory-local `.clang-tidy` override exists in this repo yet. If you
hit a real crash or an unfixable false positive confined to one file or
directory, add a directory-local `.clang-tidy` starting with
`InheritParentConfig: true` so it narrows rather than replaces the root
ruleset — with a comment stating the specific reason — rather than
disabling the check for the whole repo.

## `ThirdParty/` is already excluded

Both mechanisms exclude `ThirdParty/` by default (the header/exclude filter
in `Cmake/clang_tidy.cmake`; lintrunner's `exclude_patterns` in
`.lintrunner.toml`) — this is consistent with the `ThirdParty.md` rule's
hard rule of never modifying vendored code. If clang-tidy output ever
references a `ThirdParty/` path, that's a config regression to flag, not
something to silence by editing vendored code.
