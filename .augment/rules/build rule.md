---
type: "always_apply"
description: "build"
---

rule: build_process_rule
description: >
  Ensures the correct build sequence for this repository (see root
  `/CLAUDE.md` "Build and test" and the `project-build` skill):
  1. Change directory to 'Scripts'.
  2. Inspect `python3 setup.py --help` before adding any feature flag you
     haven't used in this session yet — the dotted-token/flag surface is
     large and changes across sessions.
  3. Run 'python3 setup.py config.build.test' for CMake, or
     'python3 setup_bazel.py config.build.test' for Bazel.
  4. Scope to the library/feature you touched (e.g. a GPU backend token)
     rather than always running the full default configuration.

steps:
  - step:
      action: cd
      target: Scripts
      comment: "Navigate to the Scripts directory before building."

  - step:
      action: run
      command: python3 setup.py --help
      comment: "Check available tokens/flags before adding one not already used this session."

  - step:
      action: run
      command: python3 setup.py config.build.test
      comment: "CMake: configure, build, and run tests in one pass."

  - step:
      action: run
      command: python3 setup_bazel.py config.build.test
      comment: "Bazel: configure, build, and run tests in one pass (see session-checklist skill — both build systems must pass)."
