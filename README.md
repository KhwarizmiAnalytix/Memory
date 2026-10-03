# Memory Library – GPU/CPU Allocation

[![CI](https://github.com/KhwarizmiAnalytix/Memory/actions/workflows/ci.yml/badge.svg)](https://github.com/KhwarizmiAnalytix/Memory/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/KhwarizmiAnalytix/Memory/branch/main/graph/badge.svg)](https://codecov.io/gh/KhwarizmiAnalytix/Memory)
[![License: GPL v3 / Commercial](https://img.shields.io/badge/license-GPL--3.0--or--later%20%2F%20commercial-blue.svg)](LICENSE)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](CMakeLists.txt)

**What it is:** Unified C++17/20 memory-allocation library for GPU and CPU with three ownership models and compile-time backend selection (CUDA, HIP, Metal, or CPU-only).

**Key features:**
- 🔒 **Three ownership models:** `data_ptr<T>` (unique), `retained_ptr<T>` (shared), `data_view<T>` (borrowed)
- 🧠 **Unified allocator:** `allocator<T>` routes CPU/GPU automatically
- 🎮 **GPU backends:** CUDA, HIP, Metal with PyTorch-style segment caching
- 💻 **CPU backends:** mimalloc, TBB, or platform malloc
- ⚡ **Async-safe:** `copy_token` for async GPU↔CPU transfers
- 📊 **Profiler-friendly:** Memory statistics, snapshots, event reporting

---

## Quick Start

### Build

```bash
# CUDA GPU support
cmake -B build -DMEMORY_GPU_BACKEND=cuda
cmake --build build && ctest --test-dir build

# HIP (AMD)
cmake -B build -DMEMORY_GPU_BACKEND=hip

# Metal (Apple)
cmake -B build -DMEMORY_GPU_BACKEND=metal

# CPU-only
cmake -B build
```

### First Program

```cpp
#include "allocator.h"
#include "common/data_ptr.h"
#include "common/execution_context.h"

using namespace memory;

int main() {
    // Allocate on GPU device 0
    auto ctx = execution_context::cuda(0);
    data_ptr<float> tensor(1000, ctx);

    // Use tensor.data(), tensor.size()
    // Memory freed automatically ✅
    return 0;
}
```

---

## Three Ownership Models – Choose One

### 1. **data_ptr<T>** – Single owner (most common)

```cpp
data_ptr<float> tensor(1000, execution_context::cuda(0));

// Copy is deleted — data_ptr is move-only
// auto copy = tensor;  // ❌ Compilation error

// Explicit deep clone
auto copy = tensor.clone();  // ✅ Allocates new memory

// Move = transfers ownership (cheap)
auto moved = std::move(tensor);  // ✅ Fast
// tensor is now empty
```

**Use when:** You own the lifetime, single component
**Note:** Copying is not implicit — use `clone()` for explicit deep-copy

---

### 2. **retained_ptr<T>** – Shared ownership (multi-consumer)

```cpp
// Allocate shared
retained_ptr<float> data = allocator<float>::allocate(
    10_million, execution_context::cuda(0)
);

// Multiple holders (shallow copy, no allocation)
auto ref1 = data;  // ✅ Cheap
auto ref2 = data;  // ✅ Cheap

// Slicing (no copy, just a window)
auto slice1 = data.slice(0, 5_million);
auto slice2 = data.slice(5_million, 5_million);

// Memory freed when ALL holders release
```

**Use when:**
- Multiple kernels/stages share data
- Async copy operations
- Multi-thread access

---

### 3. **data_view<T>** – Borrow (temporary access)

```cpp
data_ptr<float> owner(1000, ctx);

// Borrow for temporary use
{
    data_view<float> view = owner;  // Doesn't own
    process(view);  // Use it
}  // view destroyed; doesn't affect owner
```

**Use when:** Temporary access, kernel arguments, owner guaranteed alive

---

## Common Patterns

### Pattern 1: Async GPU→CPU Copy

```cpp
retained_ptr<float> gpu_data = allocator<float>::allocate(nbytes, ctx_gpu);
retained_ptr<float> cpu_data = allocator<float>::allocate(nbytes, ctx_cpu);

// Async copy (doesn't block; records stream dependencies)
copy_token token = allocator<float>::copy_async(gpu_data, cpu_data);

// Do other work while copy happens
do_other_work();

// Wait for completion before accessing host buffer
token.wait();  // Synchronizes the operation's stream
process(cpu_data);  // Now safe to read
```

**Why retained_ptr:** 
- Both endpoints must survive until `token.wait()` returns
- `retained_ptr` prevents accidental premature deallocation
- The caching allocator defers GPU buffer reuse via `record_stream()` automatically

---

### Pattern 2: Multi-Kernel Pipeline

```cpp
retained_ptr<float> features = feature_extractor(input);

// Each kernel holds a reference (no copies)
enqueue_kernel_a(features, stream_a);  // refcount++
enqueue_kernel_b(features, stream_b);  // refcount++

// Memory freed when BOTH kernels + features release
```

---

### Pattern 3: Operator Workspace

```cpp
// Reusable scratch slab (no malloc per iteration)
gpu_workspace workspace(4_MiB, execution_context::cuda(0));

for (int iter = 0; iter < 1000; ++iter) {
    float* scratch = workspace.acquire<float>(100_KiB);
    my_kernel(scratch, 100_KiB);
    workspace.release();  // Reset cursor, reuse allocation
}
```

---

## Architecture Overview

The [design and implementation plan](Docs/memory_runtime_implementation_plan.md)
is the single design document: architecture, contracts, performance rules,
phased roadmap and current status.

Current structure:

```
data_ptr<T> / retained_ptr<T> / data_view<T>
  └─ allocator<T> (static facade)
      ├─ CPU path → cpu::memory_allocator (mimalloc / TBB / platform aligned malloc)
      └─ GPU path → gpu::caching_allocator_for_device(i)   (per-device registry)
          ├─ cuda_caching_allocator (CUDA or HIP)
          └─ metal_caching_allocator (Metal)
```

**What the GPU cache does:**
- PyTorch-style segment cache: 512 B request rounding; 2 MiB segments for small
  requests, 20 MiB for 1–10 MiB, 2 MiB-rounded large segments
- Per-stream free pools; cross-stream reuse deferred with CUDA/HIP events (`record_stream`)
- OOM recovery: flush cached segments and retry once
- Lock dropped around the driver `malloc`

The plan's target architecture (§4) adds a byte-level `storage_handle` under the
typed handles and removes the registry lookup from the free path.

---

## When to Use What

| Scenario | Use | Why |
|----------|-----|-----|
| **Single owner** | `data_ptr` | Simplest, most efficient |
| **Multi-consumer** | `retained_ptr` | No deep-copies |
| **Async copy** | `retained_ptr` + `copy_token` | Token holds references |
| **Temporary access** | `data_view` | Zero-cost borrow |
| **Scratch space** | `gpu_workspace` | Reusable slab |

---

## Error Handling

```cpp
try {
    auto ptr = allocator<float>::allocate(huge_size, ctx);
} catch (std::bad_alloc const& e) {
    // Flush cache and retry
    gpu::empty_cache(device_index);
    auto ptr = allocator<float>::allocate(smaller_size, ctx);
}
```

**Common errors:**
- `std::bad_alloc` – OOM (after cache flush + retry)
- `std::invalid_argument` – Invalid copy arguments (null endpoint with a non-zero count, unsupported backend combination), invalid adoption (empty deleter, null base with non-zero capacity, non-null base with zero capacity)
- `std::overflow_error` – Byte-count overflow (copy extents, adoption capacity, workspace `acquire<T>`)
- `logging::exception` – Violated preconditions, in every build type: a pointer the cache does not own, a double free, a bad alignment, `gpu_workspace::rebind` while slices are live
- `std::runtime_error` – `copy_token::wait()` on a failed operation

Validation happens before anything is submitted: a copy that throws one of the argument errors has not started, and nothing it reserved is left behind.

### Cleanup failures never throw

Destructors and deleters cannot throw. A failure during cleanup (for example a driver error while freeing, or a pinned buffer that had to be quarantined) is counted instead and reported through `cleanup_diagnostic`:

```cpp
auto const before = cleanup_diagnostic::failure_count(cleanup_source::gpu_cache);
// ... workload ...
if (cleanup_diagnostic::failure_count(cleanup_source::gpu_cache) != before) { /* investigate */ }
```

Sources are `data_ptr`, `retained_ptr`, `pinned_buffer` and `gpu_cache`. `cleanup_diagnostic::set_handler` installs an optional `noexcept` function pointer that runs on the failing thread; it must not allocate or call back into the allocator, and it must stay valid for the life of the process (or be cleared with `nullptr`). The allocator never calls it while holding its own lock.

**Adopting foreign memory:** `allocate_adopted` and `retained_ptr::adopt` require an explicit deleter. If adoption throws, the deleter has not run and you still own the pointer.

---

## Copy Token — Async Operations

### Important: Token Lifecycle

**`copy_token` does NOT retain endpoints.** The caller is responsible for keeping both source and destination buffers alive until `token.wait()` completes:

```cpp
{
    retained_ptr<float> gpu_buf = allocator<float>::allocate(1000, ctx_gpu);
    retained_ptr<float> cpu_buf = allocator<float>::allocate(1000, ctx_cpu);
    
    copy_token token = allocator<float>::copy_async(gpu_buf, cpu_buf);
    
    // ✅ CORRECT: wait before dropping buffers
    token.wait();
}  // Buffers destroyed after token.wait()

// ❌ INCORRECT: buffers destroyed while copy is pending
{
    retained_ptr<float> gpu_buf = allocator<float>::allocate(1000, ctx_gpu);
    retained_ptr<float> cpu_buf = allocator<float>::allocate(1000, ctx_cpu);
    copy_token token = allocator<float>::copy_async(gpu_buf, cpu_buf);
}  // token.wait() not called — copy may still be pending!
```

**Stream recording:** GPU buffers automatically have `record_stream()` called before submission, so the caching allocator defers their reuse until the stream completes. Pageable (non-pinned) CPU buffers require manual synchronization.

### Token results are stable

All copies of a token share one operation. The first terminal result (complete or failed) is recorded once and every copy, on every thread, then sees the same result, whatever work is queued on the stream afterwards:

- `state()` returns `pending`, `complete` or `failed` without blocking or throwing.
- `ready()` is true only for `complete`.
- `wait()` returns on `complete`, blocks while `pending`, and throws `std::runtime_error` on `failed`, every time it is called.
- You cannot force a state from user code; only the library decides that an operation completed.

If submission fails after the driver was asked to move data, the library waits on the submitting stream; if it cannot prove the stream is idle, the operation is quarantined and a retained copy keeps its owners alive. It never assumes the copy did not start from the error code alone.

---

## Pinned Host Transfers (CUDA/HIP)

Fast GPU↔CPU copies without page-lock contention:

```cpp
#include "common/pinned_buffer.h"

memory::pinned_buffer<float> host(count, device=0);
host.data()[0..count) = ...;  // Fill on CPU

// Async copy (no page-lock overhead)
host.copy_to_device_async(device_pointer, stream);
// Destruction waits for stream completion before reusing buffer
```

**API:** `copy_to_device_async()`, `copy_from_device_async()`
**Note:** Metal uses shared buffers instead; CPU-only has no pinned memory

---

## Project Structure

```
Memory/
├── CMakeLists.txt           # Build config (MEMORY_ENABLE_*, MEMORY_GPU_BACKEND)
├── BUILD.bazel              # Bazel build
├── Cmake/                   # CUDA, HIP, Metal, TBB, NUMA toolchain modules
├── include/
│   └── memory/
│       ├── allocator.h      # Main: allocator<T>
│       ├── common/          # data_ptr, data_view, retained_ptr, copy_token
│       ├── gpu/             # CUDA/HIP/Metal caching allocators
│       └── helper/          # CPU allocators, pinned memory
└── Testing/Cxx/             # Tests & benchmarks
```

---

## Build Configuration

### Essential CMake Options

| Option | Default | Purpose |
|--------|---------|---------|
| **`MEMORY_GPU_BACKEND`** | `none` | `cuda`, `hip`, `metal`, or `none` (CPU-only). Sets `MEMORY_HAS_CUDA`, `MEMORY_HAS_HIP`, `MEMORY_HAS_METAL` |
| `MEMORY_CXX_STANDARD` | 20 | C++ version: 11, 17, 20, 23 |
| `MEMORY_ENABLE_TESTING` | ON | Build tests |
| `MEMORY_ENABLE_BENCHMARK` | ON | Build benchmarks |
| `MEMORY_ENABLE_MIMALLOC` | ON | Use mimalloc for CPU |

### Optional Features

| Option | Default | Purpose |
|--------|---------|---------|
| `MEMORY_ENABLE_TBB` | OFF | TBB scalable allocator |
| `MEMORY_ENABLE_NUMA` | OFF | NUMA-aware allocation |
| `MEMORY_ENABLE_MIMALLOC_STATS` | OFF | Runtime stats reporting |
| `MEMORY_ENABLE_COVERAGE` | OFF | Coverage instrumentation |
| `MEMORY_ENABLE_SANITIZER` | OFF | Sanitizer (address/memory/thread/undefined) |

**See `CMakeLists.txt` for complete list**

---

## Bazel Build (Alternative to CMake)

```bash
# CUDA support
bazel build //:Memory --define memory_enable_cuda=1

# Convenience configs
bazel build //:Memory --config=cuda
bazel build //:Memory --config=hip
bazel build //:Memory --config=tbb
```

See `bazel/memory.bzl` and root `.bazelrc` for available flags.

---

## Continuous integration and coverage

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push and pull request to `main`, and on manual dispatch. One workflow badge covers the whole matrix. The [Codecov](https://codecov.io/gh/KhwarizmiAnalytix/Memory) badge is the Linux line-coverage number for first-party sources.

| Job | Where | What it checks |
|-----|--------|----------------|
| **Linux CPU** | Ubuntu, gcc and clang | Release, plus one Debug build per compiler. `MEMORY_GPU_BACKEND=none` |
| **macOS CPU** | macOS, AppleClang | Release, CPU backend |
| **Windows CPU** | Windows, MSVC | Release, CPU backend |
| **Bazel** | Ubuntu, macOS, Windows | `bazel build //...` and `bazel test //...` (CPU backend) |
| **Sanitizers** | Ubuntu and macOS, Clang | ASan and UBSan, Debug, CPU backend |
| **Coverage** | Ubuntu, macOS, Windows | Instrumented CPU-backend tests; HTML report uploaded as `coverage-html-<os>` |
| **Linux CUDA** | Self-hosted NVIDIA GPU | Release tests when `GPU_RUNNERS_AVAILABLE` is `true` |
| **Linux HIP** | Self-hosted AMD GPU | Release tests when `GPU_RUNNERS_AVAILABLE` is `true` |

Hosted jobs exercise the CPU allocator. CUDA and HIP run on labeled self-hosted runners (`self-hosted`, `linux`, `gpu`, plus `cuda` or `rocm`) only while the repository variable `GPU_RUNNERS_AVAILABLE` is `true`. Metal stays a local Apple build.

### Coverage

`MEMORY_ENABLE_COVERAGE=ON` instruments the library and runs the CPU test suite. CI collects reports with [coverage-tool](https://github.com/KhwarizmiAnalytix/coverage-tool):

| Platform | Compiler | Collector |
|----------|----------|-----------|
| Linux | gcc | gcov + lcov |
| macOS | Homebrew LLVM clang | llvm-cov |
| Windows | MSVC | OpenCppCoverage |

Linux uploads `build/coverage_filtered.info` to Codecov. macOS and Windows keep HTML artifacts only. [`codecov.yml`](codecov.yml) ignores `ThirdParty/`, `Testing/`, and build trees. Project and patch status use an automatic target with a 1% threshold.

Local Linux run, matching CI:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DMEMORY_GPU_BACKEND=none \
  -DMEMORY_ENABLE_TESTING=ON \
  -DMEMORY_ENABLE_BENCHMARK=OFF \
  -DMEMORY_ENABLE_COVERAGE=ON
cmake --build build
ctest --test-dir build --output-on-failure
pip install "git+https://github.com/KhwarizmiAnalytix/coverage-tool.git"
coverage-tool --build=build --compiler=gcc
```

Open `build/coverage_report/html/index.html`. On macOS pass `--compiler=clang` (Homebrew LLVM, so `llvm-cov` matches the compiler). On Windows pass `--compiler=msvc` and install OpenCppCoverage first. The reported percentage is CPU-backend coverage; CUDA, HIP, and Metal paths are outside that number.

---

## Understanding the Code

**For users:** See the [**Dependency Graph**](https://claude.ai/artifact/3zHXeRs5FKbxvU6tW5vfqE) for visual architecture
**For integration:** Focus on `allocator<T>`, `data_ptr<T>`, `retained_ptr<T>`, `copy_token`
**For optimization:** See `device_handle_cache` (thread-local LRU) and segment caching strategy

**Key files:**
- `allocator.h` – Unified allocation facade
- `common/data_ptr.h` – Unique ownership
- `common/retained_ptr.h` – Shared ownership
- `gpu/caching_allocator.h` – GPU dispatcher
- `gpu/cuda_caching_allocator.h` – CUDA/HIP backend
- `gpu/metal/metal_caching_allocator.h` – Metal backend

---

## FAQ

**Q: When should I use data_ptr vs retained_ptr?**
A: Use `data_ptr` by default (single owner). Switch to `retained_ptr` for:
- Multiple kernels/stages needing same data
- Async copy operations
- Sharing across threads

**Q: Does copying data_ptr allocate new memory?**
A: Copying is deleted (`data_ptr` is move-only). Call `clone()` for a deep copy; use `std::move()` or `data_view` otherwise.

**Q: Can I mix CUDA and HIP?**
A: No, choose one at build time (compile-time exclusive).

**Q: How do I avoid memory leaks?**
A: Destructors handle cleanup. Just avoid double-free:
- Don't manually deallocate
- Don't copy data_ptr unnecessarily
- Don't keep data_view longer than owner

**Q: Does a zero-size allocation have an identity?**
A: No. A zero-size, default-constructed or moved-from `data_ptr` is empty and has an invalid `allocation_id`; there is no memory and no allocation lifetime. Real ids come from one process-wide generator and are never reset or reused.

**Q: What if I run out of GPU memory?**
A: Allocators throw `std::bad_alloc`. Call `gpu::empty_cache(device_index)` to free unused cached blocks.

---

## Known Issues

**GPU cache churn crash (held, unconfirmed).** A benchmark that creates many `cuda_caching_allocator` instances and runs millions of allocate/free pairs crashed intermittently with an access violation on the maintainers' test machine. That machine also crashes a control program with no Memory code and has processor machine-check events, so the crash is not attributed to Memory and has not been cleared either. It needs a rerun on healthy hardware. Until then, treat the GPU caching allocator as unaccepted for long-running churn workloads. Details, reproduction and next steps: [plan Appendix B](Docs/memory_runtime_implementation_plan.md).

---

## Next Steps

- Read test files in `Testing/Cxx/` for more examples
- Read the [design and implementation plan](Docs/memory_runtime_implementation_plan.md) for architecture, status and next work
- Enable `-DMEMORY_HAS_PROFILER=1` to monitor memory usage
- Use `gpu::memory_stats()` and `gpu::memory_snapshot()` for profiling
