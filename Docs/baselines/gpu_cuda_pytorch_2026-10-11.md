# GPU CUDA allocation benchmark against PyTorch — 2026-10-11

Windows 11, RTX 4060 Ti, NVIDIA driver 591.86 (CUDA 13.1), CUDA Toolkit 13.2. Memory built with
`MEMORY_GPU_BACKEND=cuda` using Clang 22.1.2 as the CUDA compiler (`Cmake/cuda.cmake`'s
Clang-direct-CUDA path, not nvcc), Release, C++20, ThinLTO off for this run. LibTorch is `torch`
2.14.1+cu130 (pip wheel, `pip install torch --index-url https://download.pytorch.org/whl/cu130`),
installed into an isolated venv (`.venv_libtorch/`, not tracked) rather than the project's main
Python environment, to avoid an unrelated pip crash in that environment's dependency scan. Memory
commit `58e8694` plus the uncommitted CMake/benchmark changes this run added (see below). Profiler
and TBB disabled.

Each timed iteration enqueues an allocation and its matching free on the CUDA caching allocator;
neither side synchronizes the device, so these are **host submission times** (time to enqueue
against the warm cache), not GPU-visible completion times — the same convention
`BenchmarkCudaCachingAllocator.cpp`'s existing warm-path benchmarks already use. Iterations are
explicitly bounded (`->Iterations(5000)`) rather than left to Google Benchmark's own convergence,
matching that file's documented reason: unbounded iteration counts on this allocator have been
observed to reproduce a rare, pre-existing, timing-dependent crash on long allocate/free churn
(`Docs/memory_runtime_implementation_plan.md` Appendix B), independent of which allocator is under
test.

Two levels, both warm (cache already populated before the timed loop):

- **Level 1 — raw allocator core**: `gpu::cuda_caching_allocator::allocate/deallocate` vs
  `c10::cuda::CUDACachingAllocator::raw_alloc/raw_delete`.
- **Level 2 — owning buffer object**: `memory::data_ptr<float>(numel, device_enum::CUDA)` vs
  `torch::empty({numel}, ...device(kCUDA))`.

Seven repetitions per case, 0.2 s minimum measurement time, 0.1 s minimum warmup, randomized
interleaving. Table reports median CPU time in nanoseconds per allocate/free pair. Speedup =
PyTorch / Memory. CV is the coefficient of variation across the 7 repetitions.

| Level | Bytes/elements | Memory ns | PyTorch ns | Speedup | CV Memory / PyTorch |
|---|---:|---:|---:|---:|---:|
| Raw | 4,096 B | 96 | 380 | 3.96× | 8.3% / 7.9% |
| Raw | 32,768 B | 98 | 370 | 3.78× | 17.7% / 2.9% |
| Raw | 262,144 B | 98 | 359 | 3.66× | 7.8% / 2.4% |
| Raw | 2,097,152 B | 95 | 374 | 3.94× | 11.9% / 21.9% |
| Raw | 4,194,304 B | 95 | 382 | 4.02× | 5.6% / 3.3% |
| Owning | 16 floats | 77 | 850 | 11.04× | 0.6% / 3.5% |
| Owning | 64 floats | 77 | 855 | 11.10× | 2.2% / 4.8% |
| Owning | 512 floats | 77 | 897 | 11.65× | 2.7% / 3.5% |
| Owning | 4,096 floats | 74 | 861 | 11.64× | 2.7% / 3.5% |
| Owning | 32,768 floats | 77 | 899 | 11.68× | 23.6% / 4.1% |
| Owning | 262,144 floats | 73 | 907 | 12.42× | 2.9% / 2.6% |

The 32,768-element rows (both Raw and Owning) have CV at or above about 15% on at least one side
and should be read as indicative, not conclusive, same caveat as the CPU baseline. All other rows
are under 12% CV. No GPU-side synchronization or completion-latency numbers are reported here —
only host enqueue cost; a bandwidth-bound comparison would need a different benchmark design
(timed `cudaDeviceSynchronize`/event-based completion).

Aggregate statistics (individual repetitions were not retained):
[gpu_cuda_pytorch_2026-10-11.json](gpu_cuda_pytorch_2026-10-11.json).

## What this run needed that didn't exist before

This is plan task **8.6**'s CUDA half ("Comparisons: raw driver vs Memory cache vs PyTorch CUDA
allocator"), previously not started ("no PyTorch build or version pin"). Getting a working build
required three fixes, none of which touch Memory's allocator or transfer code:

1. **New benchmark code**: `Testing/Cxx/BenchmarkPyTorchComparisonGpu.cpp` had only a Metal/MPS
   section; added a CUDA section (both levels, since unlike MPS, PyTorch does expose a public raw
   CUDA allocator API).
2. **LibTorch's CUDA version check is nvcc-only.** `Caffe2/public/cuda.cmake` asserts
   `CMAKE_CUDA_COMPILER_VERSION == CUDAToolkit_VERSION` unconditionally. That holds for nvcc but not
   for this project's Clang-as-CUDA-compiler setup (`Cmake/cuda.cmake:23`, deliberate), where
   `CMAKE_CUDA_COMPILER_VERSION` is Clang's own version. Memory's own `cuda.cmake` already narrows
   the equivalent check to `CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA"` for exactly this reason;
   `Testing/Cxx/CMakeLists.txt` now shadows `CMAKE_CUDA_COMPILER_VERSION` with
   `CUDAToolkit_VERSION` (a directory-local regular variable, not a cache write) right before
   `find_package(Torch)`, matching that existing rationale instead of patching the vendored file.
3. **`torch::cuda::is_available()` silently returned `false`** even with a working driver and
   cudart (confirmed by a minimal out-of-tree repro: raw `cudaGetDeviceCount` succeeded in the same
   binary while `torch::cuda::device_count()` returned 0, no error, no warning). Root cause:
   `torch_cuda`'s own exported `INTERFACE_LINK_LIBRARIES` carries
   `-INCLUDE:?warp_size@cuda@at@@YAHXZ` specifically to force-link the CUDA hooks registration
   static initializer; without it, an otherwise-unreferenced registration object gets dropped and
   PyTorch falls back to its CPU-only stub hooks. This flag was not reaching the benchmark target
   through the normal `target_link_libraries(torch)` graph, so `Testing/Cxx/CMakeLists.txt` now adds
   it directly via `target_link_options` when `torch_cuda` exists.

## Limits

- One GPU, one driver/toolkit version, one compiler (Clang 22.1.2 as the CUDA compiler). Not run:
  HIP/ROCm (no AMD hardware), multi-GPU, nvcc-as-compiler (would not need fix #2 above), MSVC.
- Host submission time only, as stated above — not a GPU-bandwidth or kernel-launch comparison.
- LibTorch came from a venv outside the repo's normal dependency tree (`ThirdParty/`), chosen over
  vendoring because of CI reproduce-by-pip-install precedent in `cpu_pytorch_2026-10-06.md`; a CI
  job would need the same `pip install torch --index-url https://download.pytorch.org/whl/cu130`
  step plus the `CMAKE_PREFIX_PATH` below.
- `benchmark_memory_pytorchcomparisongpu`'s CTest registration (`--benchmark_min_time=0.01s`) was
  not exercised in this run; only the direct binary invocation below was.

## Reproduce from the repository root

```sh
# Isolated venv so this doesn't touch the main Python environment:
python -m venv .venv_libtorch
.venv_libtorch/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu130

cmake -S . -B build_cuda_pytorch -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DMEMORY_GPU_BACKEND=cuda -DMEMORY_ENABLE_LIBTORCH=ON \
  -DCMAKE_PREFIX_PATH="$(pwd)/.venv_libtorch/Lib/site-packages/torch/share/cmake" \
  -DMEMORY_ENABLE_PROFILER=OFF -DMEMORY_ENABLE_TBB=OFF
cmake --build build_cuda_pytorch --target benchmark_memory_pytorchcomparisongpu -j 8
build_cuda_pytorch/bin/benchmark_memory_pytorchcomparisongpu \
  --benchmark_min_time=0.2s --benchmark_min_warmup_time=0.1 \
  --benchmark_repetitions=7 --benchmark_enable_random_interleaving=true \
  --benchmark_report_aggregates_only=true \
  --benchmark_out=Docs/baselines/gpu_cuda_pytorch_2026-10-11.json --benchmark_out_format=json
```
