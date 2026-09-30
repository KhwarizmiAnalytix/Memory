# Validation Manifest Template

Use this template to record every test run, benchmark, and validation. This prevents conflating untested historical features with accepted behavior.

## Run metadata

- **Date:** YYYY-MM-DD HH:MM UTC
- **Branch/Commit:** `git log --oneline -1`
- **Test harness:** `ctest`, custom script, CI job, etc.

### Build configuration

- **CMake version:** `cmake --version`
- **Compiler:** `c++ --version` (exact command)
- **C++ standard:** (e.g., C++17, C++20)
- **Build type:** Debug / Release / RelWithDebInfo
- **Sanitizer:** None / AddressSanitizer / MemorySanitizer / UBSan / ThreadSanitizer

### Platform

- **OS:** macOS / Linux / Windows
- **Kernel version:** `uname -r`
- **CPU:** model, core count, NUMA topology (if relevant)
- **RAM:** total available

### GPU configuration (if applicable)

- **Backend:** CUDA / HIP / Metal / None
- **Hardware:** GPU model(s), compute capability/RDNA generation
- **Driver version:** `nvidia-smi --query-gpu=driver_version` / `hipconfig --version`
- **Runtime version:** CUDA 12.x / HIP 5.x / Metal (version from OS)
- **Default stream mode:** legacy / per-thread (if build supports choice)

### CPU allocator

- **Primary:** mimalloc / TBB / platform malloc
- **Relevant options:**
  - `MEMORY_ENABLE_MIMALLOC`: ON/OFF
  - `MEMORY_ENABLE_TBB`: ON/OFF
  - `MEMORY_ENABLE_NUMA`: ON/OFF

## Test Results

### Unit tests

| Test Category | Passed | Skipped | Failed | Notes |
|---|---|---|---|---|
| Ownership (data_ptr, retained_ptr, data_view) | | | | |
| Copy/clone semantics | | | | |
| Execution context validation | | | | |
| Allocation/deallocation | | | | |
| GPU transfers (if backend enabled) | | | | |
| Pinned memory (CUDA/HIP only) | | | | |
| Workspace/arena reuse | | | | |
| Error handling (OOM, invalid args) | | | | |
| **Total** | | | | |

**Command run:** 
```
cmake -B build [OPTIONS]
cmake --build build
ctest --test-dir build --output-on-failure
```

### Benchmark Results (if run)

| Workload | Operation | Count | Time (µs) | Notes |
|---|---|---|---|---|
| Cold/warm allocation | allocate | 1000 | | Same size/alignment; explicit warmup |
| | deallocate | 1000 | | |
| Multi-thread contention | shared allocator, N threads | | | N = 2,4,8,16,32 |
| Mixed lifetimes | seeded replay | | | Long-lived + temporaries |
| Multi-stream | concurrent copies | | | Same-stream vs cross-stream |

**Command run:**
```
ctest --test-dir build --tests-regex Benchmark
```

### Feature checklist

- [ ] `data_ptr` move-only semantics work (copy deleted, move transfers ownership)
- [ ] `data_ptr::clone()` allocates and copies
- [ ] `retained_ptr` shallow copy preserves backing
- [ ] `data_view` does not retain owner
- [ ] `copy_token::wait()` blocks until completion
- [ ] `copy_token::ready()` queries without blocking
- [ ] GPU stream recording (CUDA/HIP) defers reuse correctly
- [ ] Pinned buffer lifecycle (if CUDA/HIP) ✗ or supported
- [ ] `gpu_workspace` reuse after reset
- [ ] `empty_cache()` flushes unused blocks
- [ ] Error handling: `std::bad_alloc` on OOM
- [ ] Error handling: `std::invalid_argument` on zero size
- [ ] Documentation examples compile and run

### Known limitations / skipped cases

| Limitation | Reason | Impact |
|---|---|---|
| e.g., no Metal hardware | Apple-only | Metal allocator untested |
| e.g., no multi-GPU | CI runner constraint | Peer access paths untested |

## Issues encountered

- **Issue #1:** [Brief description]
  - Root cause: [Investigation result]
  - Fix applied / Workaround / Deferred to phase X
  - Reference: [Commit hash, PR, issue link]

## Acceptance decision

- [ ] **PASS** — All required tests pass; no unsupported guarantees in documentation
- [ ] **PASS with caveats** — Passes but see limitations section
- [ ] **BLOCKED** — Cannot run required tests; reason: [hardware / tooling / dependency]
- [ ] **FAIL** — Tests failed; see issues section

**Gate decision:** [Phase X accepted / held / requires rework]

**Reviewer notes:** [Name, date]

---

## How to use this template

1. Copy this template for each validation run
2. Fill in metadata **before** running tests
3. Record exact test commands and output (for reproducibility)
4. If any test is skipped, document why (missing hardware, dependency issue, etc.)
5. Do **not** assume historical test results are current — test and record
6. Link to this manifest from the implementation review or PR description

---

## Example (completed run)

See `validation_manifest_2026-09-30_cpu_only.md` for a sample filled-in manifest.
