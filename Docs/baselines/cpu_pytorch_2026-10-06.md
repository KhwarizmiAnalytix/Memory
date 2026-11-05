# CPU allocation benchmark against PyTorch — 2026-10-06

Apple M4 Pro, 14 cores (10 performance, 4 efficiency), 24 GB RAM; macOS 26.6.2. Single-thread C++ benchmark, PyTorch/LibTorch 2.13.0, mimalloc 3.1, Clang 22.1.2, C++20 Release with ThinLTO. Memory commit `2ed125025a64f1d062bfffcd19928716b6be0bff`. GPU, TBB, NUMA, profiler, sanitizers and coverage disabled.

Each timed iteration includes allocation and destruction/free. Both sides request uninitialized buffers; memory contents are not touched. Raw comparison uses 64-byte alignment. Owning comparison uses float32 buffers: Memory data_ptr versus a full PyTorch tensor, whose metadata and functionality are more extensive. These are allocation timings, not compute or memory bandwidth measurements. Batch timings include vector creation, allocation of all buffers, and release of all buffers.

Seven repetitions per case; 0.2 s minimum measurement time, 0.1 s minimum warmup, randomized interleaving. Table reports median CPU time in nanoseconds per allocation/free pair (batch totals divided by batch count). Speedup = PyTorch / Memory. CV is the coefficient of variation across repetitions.

| Operation | Bytes/buffer | Batch | Memory ns | PyTorch ns | Speedup | CV Memory / PyTorch |
|---|---:|---:|---:|---:|---:|---:|
| Owning | 16 | 1 | 6.94 | 78.86 | 11.36× | 1.9% / 2.0% |
| Owning | 64 | 1 | 8.07 | 81.30 | 10.07× | 1.5% / 1.4% |
| Owning | 256 | 100 | 16.23 | 82.85 | 5.10× | 23.0% / 3.2% |
| Owning | 256 | 1,000 | 15.24 | 85.30 | 5.60× | 2.5% / 1.3% |
| Owning | 512 | 1 | 16.95 | 90.40 | 5.33× | 1.1% / 1.6% |
| Owning | 4,096 | 1 | 17.05 | 89.67 | 5.26× | 2.4% / 1.9% |
| Owning | 32,768 | 1 | 215.70 | 293.90 | 1.36× | 5.2% / 1.3% |
| Owning | 262,144 | 1 | 219.60 | 300.50 | 1.37× | 1.8% / 1.0% |
| Raw | 64 | 1 | 5.41 | 14.82 | 2.74× | 2.2% / 1.1% |
| Raw | 512 | 1 | 5.47 | 14.62 | 2.67× | 2.1% / 1.7% |
| Raw | 1,024 | 100 | 9.85 | 20.44 | 2.07× | 15.4% / 14.5% |
| Raw | 1,024 | 1,000 | 13.94 | 23.06 | 1.65× | 3.5% / 2.4% |
| Raw | 4,096 | 1 | 15.33 | 24.54 | 1.60× | 1.6% / 1.8% |
| Raw | 32,768 | 1 | 17.86 | 27.23 | 1.52× | 2.6% / 1.5% |
| Raw | 262,144 | 1 | 215.10 | 224.30 | 1.04× | 1.8% / 2.0% |
| Raw | 1,048,576 | 1 | 222.20 | 232.00 | 1.04× | 29.2% / 1.5% |

macOS prevented affinity control and CPU frequency discovery. The JSON frequency value is invalid and must not be used. Clocks and power state were uncontrolled; this is one host run. The 100-buffer batch cases and the Raw 1 MiB single-allocation case have substantial variability (CV ≥ 15%), so differences in those rows should not be treated as conclusive.

Aggregate statistics (individual repetitions were not retained): [cpu_pytorch_2026-10-06.json](cpu_pytorch_2026-10-06.json).

Reproduce from the repository root:

```sh
cmake -S . -B build_cpu_pytorch -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DMEMORY_GPU_BACKEND=none -DMEMORY_ENABLE_LIBTORCH=ON \
  -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
  -DMEMORY_ENABLE_PROFILER=OFF -DMEMORY_ENABLE_TBB=OFF
cmake --build build_cpu_pytorch --target benchmark_memory_pytorchcomparisoncpu -j 8
build_cpu_pytorch/bin/benchmark_memory_pytorchcomparisoncpu \
  --benchmark_min_time=0.2s --benchmark_min_warmup_time=0.1 \
  --benchmark_repetitions=7 --benchmark_enable_random_interleaving=true \
  --benchmark_report_aggregates_only=true \
  --benchmark_out=Docs/baselines/cpu_pytorch_2026-10-06.json --benchmark_out_format=json
```
