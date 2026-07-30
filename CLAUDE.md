# Memory

Allocators, memory pools, and GPU memory management. See root `/CLAUDE.md`
for general coding/testing/build rules — this file only covers what's
specific to this library.

## GPU feature-guard macro: `MEMORY_HAS_CUDA` / `MEMORY_HAS_HIP`

All CUDA/HIP-conditional code in `gpu/` must be guarded with
`MEMORY_HAS_CUDA` / `MEMORY_HAS_HIP`, defined by CMake from the selected
`--gpu_backend=`. **Not** `PROJECT_HAS_CUDA`/`PROJECT_HAS_HIP` — those
symbols don't exist anywhere in this repo, so code guarded by them compiles
out silently and the GPU path never actually runs. This exact bug hit 13
test files here before being fixed in commit `f15cf987`; if you touch a
`#if` guard in `gpu/` or `Testing/Cxx/TestGpu*.cpp`, double-check it's
`MEMORY_HAS_*` before assuming the branch is live.

## `try`/`catch` is allowed in `gpu/`, by exception

Root `/CLAUDE.md` bans `try`/`catch` in new application code by default,
but `gpu/gpu_allocator_tracking.cpp` and `gpu/gpu_memory_transfer.cpp`
legitimately catch `std::exception` around calls into the CUDA/HIP runtime,
which throws on driver-level failures. This is an intentional boundary
around a third-party API, not a lapse — don't "clean it up" to
return-value-only error handling as a drive-by change, and match this
pattern (catch at the CUDA/HIP call boundary, translate to the project's
own error/result type immediately) if you add new GPU runtime calls.
