# Memory Library – GPU/CPU Allocation

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

// Copy = allocates new memory (expensive)
auto copy = tensor;  // ❌ Deep clone

// Move = transfers ownership (cheap)
auto moved = std::move(tensor);  // ✅ Fast
// tensor is now empty
```

**Use when:** You own the lifetime, single component

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

// Async copy (doesn't block)
copy_token token = allocator<float>::copy_async(gpu_data, cpu_data);

// Do other work while copy happens
do_other_work();

// Check completion
if (token.ready()) {
    process(cpu_data);  // Safe to read
}

// Or wait
token.wait();
```

**Why retained_ptr:** Token holds references; memory survives async operation

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

See **[Dependency Graph](https://claude.ai/artifact/3zHXeRs5FKbxvU6tW5vfqE)** for visual architecture:

```
allocator<T> (unified facade)
  ├─ CPU path → cpu::memory_allocator (mimalloc/TBB/malloc)
  └─ GPU path → caching_allocator_for_device()
      ├─ [1st call] Register in global registry
      └─ [cached] Thread-local device handle cache (LRU)
          └─ cuda_caching_allocator (CUDA/HIP)
             OR metal_caching_allocator (Metal)
```

**Key optimizations:**
- Thread-local device cache (8-slot LRU) avoids mutex on 90%+ allocations
- PyTorch-style segment cache (512B–20MiB segments)
- Stream-aware reuse (CUDA events defer cross-stream reuse)
- OOM recovery (flush cache + retry)

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
- `std::invalid_argument` – Zero size, size mismatch
- `std::overflow_error` – Integer overflow on alignment
- `std::logic_error` – Double-free, nested capture

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
A: Yes, copy = deep-clone. Use `move()` or `data_view` to avoid it.

**Q: Can I mix CUDA and HIP?**
A: No, choose one at build time (compile-time exclusive).

**Q: How do I avoid memory leaks?**
A: Destructors handle cleanup. Just avoid double-free:
- Don't manually deallocate
- Don't copy data_ptr unnecessarily
- Don't keep data_view longer than owner

**Q: What if I run out of GPU memory?**
A: Allocators throw `std::bad_alloc`. Call `gpu::empty_cache(device_index)` to free unused cached blocks.

---

## Next Steps

- Read test files in `Testing/Cxx/` for more examples
- Check `Docs/` folder for detailed design documentation
- Enable `-DMEMORY_HAS_PROFILER=1` to monitor memory usage
- Use `gpu::memory_stats()` and `gpu::memory_snapshot()` for profiling
