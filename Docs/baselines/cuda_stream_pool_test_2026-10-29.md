# CUDA stream-pool test failure and fix - 2026-10-29

## Failure

At Memory revision `4601c1d`,
`CudaCachingAllocator.data_ptr_uses_allocation_stream_pool` passed on its first
invocation but failed when repeated within the same process. An isolated run of
20 repetitions passed iteration 1 and failed iterations 2 through 20. A broader
48-test hardware run first encountered the same assertion on iteration 3.

On isolated iteration 2, the assertions reported:

| Driver-allocation count | Expected | Actual |
|---|---:|---:|
| After allocating twice on stream A | 3 | 2 |
| After allocating once on stream B | 4 | 2 |

The full unit suite passed before this stress check: 516 tests across six CTest
entries, with no skips. Five fresh-process runs of the 48 hardware tests also
passed. The failure depended on retained state within one process.

## Cause and fix

The test uses `caching_allocator_for_device(0)`, a process-wide allocator. Its free
blocks survive a test invocation. CUDA/HIP can recycle destroyed stream handles,
allowing a subsequent invocation to reuse those blocks. The test assumed that a
new stream handle required a new backing allocation, even when the cache already
contained a matching free block. The lower driver-allocation count is consistent
with cache reuse; the assertion did not establish an allocator defect.

The fix in
[`TestCudaCachingAllocator.cpp`](../../Testing/Cxx/TestCudaCachingAllocator.cpp)
calls `cache.empty_cache()` before taking the driver-allocation baseline. It also
flushes the test's cached blocks before destroying its streams. The original
assertions remain: two allocations on stream A require one driver allocation,
and an allocation on stream B requires a second. Production allocator behavior
is unchanged.

## Environment and reproduction

Windows, NVIDIA GeForce RTX 4060 Ti (8 GB), driver 591.86, CUDA Toolkit 13.2.78,
Clang 22.1.2, Release, C++20, CUDA backend. The existing `build_cuda_pytorch`
configuration has LibTorch enabled and Profiler and TBB disabled. Testing needed
`BUILD_TESTING=ON` as well as `MEMORY_ENABLE_TESTING=ON`; CUDA needed
`MEMORY_LTO_MODE=off`. These are local build-cache settings.

From the repository root in PowerShell, using that configured build:

```powershell
cmake -S . -B build_cuda_pytorch -DBUILD_TESTING=ON -DMEMORY_ENABLE_TESTING=ON -DMEMORY_LTO_MODE=off
cmake --build build_cuda_pytorch --target MemoryCxxTests --parallel 8
Push-Location build_cuda_pytorch/bin
./MemoryCxxTests.exe --gtest_filter=CudaCachingAllocator.data_ptr_uses_allocation_stream_pool --gtest_repeat=20
Pop-Location
```

The test must run all repetitions in one process to exercise the original
failure. Separate invocations start with fresh allocator state.

## Validation after the fix

The same machine and build configuration produced these results after rebuilding
`MemoryCxxTests` with the fix:

| Check | Result |
|---|---|
| Isolated stream-pool test, 100 repetitions in one process | 100 passed |
| 48 CUDA hardware tests, 10 repetitions in one process | 480 passed, no skips |
| Full unit/runtime suite, `ctest -LE Benchmark` | All 516 tests passed across six CTest entries, no skips |
| NVIDIA Compute Sanitizer memcheck on the 48 hardware tests | 48 passed; `ERROR SUMMARY: 0 errors`; exit 0 |
| Clang-format on the edited lines | Passed |
| Clang-tidy with the actual CUDA compile commands, filtered to the edited lines | Passed; no diagnostics on those lines |
| Codespell, newline checks, Markdown link target, and diff whitespace | Passed |

Additional commands, from the repository root in PowerShell:

```powershell
ctest --test-dir build_cuda_pytorch --output-on-failure -LE Benchmark
Push-Location build_cuda_pytorch/bin
$hardwareFilter = 'CudaCachingAllocator.*:PinnedMemoryRuntime.*:Phase4Hardware.*:Phase6Hardware.*'
./MemoryCxxTests.exe --gtest_filter=CudaCachingAllocator.data_ptr_uses_allocation_stream_pool --gtest_repeat=100
./MemoryCxxTests.exe "--gtest_filter=$hardwareFilter" --gtest_repeat=10
& 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/compute-sanitizer/compute-sanitizer.exe' --tool memcheck --error-exitcode 99 ./MemoryCxxTests.exe "--gtest_filter=$hardwareFilter"
Pop-Location
```

Raw local evidence is retained under the ignored `build_cuda_pytorch/` directory:
`cuda_validation_stream_repeat.log` records the original failure;
`cuda_stream_pool_fix_repeat.log`, `cuda_stream_pool_fix_hardware.log`,
`cuda_stream_pool_fix_ctest.log`, and `cuda_stream_pool_fix_memcheck.log` record
the post-fix checks. These logs are local build artifacts, not committed files.

Bazel validation was attempted with
`bazel build //Testing/Cxx:MemoryCxxTests --define=memory_gpu_backend=cuda`.
It stopped before compilation while resolving `rules_pkg/1.0.1` from
`https://bcr.bazel.build/`: Java reported `PKIX path building failed` and
`unable to find valid certification path to requested target`. Bazel validation
remains unavailable until that registry certificate trust issue is resolved.

A whole-file clang-format check also reports two pre-existing violations in
`tracks_successful_allocations_and_frees`; the original `HEAD` version has the
same violations. The edited region passes. Clang-tidy suppressed existing
diagnostics outside the edited lines. Validation covers this CUDA test fix on
one GPU; HIP hardware was not available.
