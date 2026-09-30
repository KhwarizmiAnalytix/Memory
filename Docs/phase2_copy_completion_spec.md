# Phase 2: Transfer Completion and Context Semantics

**Status:** Design specification (not yet implemented)  
**Baseline:** commit `b4626e5`  
**Scope:** CPU-independent design; GPU implementation deferred to hardware validation

## 1. Overview

Phase 2 establishes that:
1. Synchronous copies (`copy_sync()`) complete before returning
2. Async copies record device and stream state correctly
3. Completion is operation-specific, not stream-wide
4. Device activation is correct for multi-device scenarios
5. Failure modes are predictable and safe

---

## 2. Current state (gaps)

### 2.1 `copy_sync()` does not guarantee completion

**Current code:** `allocator<T>::copy()` without a token  
**Issue:** Returns without establishing host-visible completion

```cpp
// Current behavior (incorrect)
allocator<float>::copy(src, nbytes, dst, CPU, GPU, 0, 0, stream);
// Returns immediately — destination may not be visible on host yet

// Expected behavior (Phase 2 target)
allocator<float>::copy_sync(src, nbytes, dst, CPU, GPU, 0, 0, stream);
// Returns only after GPU→CPU transfer is visible on host
```

### 2.2 `copy_token` identifies the stream, not the operation

**Current code:**  
```cpp
copy_token token = allocator<float>::copy_async(from, to);
// token stores: {device, stream}
// token.wait() calls: cudaStreamSynchronize(stream)
```

**Issue:** Later work on the same stream makes `token.ready()` true even if *this specific copy* hasn't completed (stream barrier semantics, not operation semantics).

### 2.3 Device context not explicitly validated

**Current code:**  
```cpp
cudaStreamSynchronize(static_cast<cudaStream_t>(ctx_.stream));
```

**Issue:** No explicit device activation before stream call; relies on CUDA's current-context tracking.

### 2.4 Submission errors not distinguished from completion errors

**Current code:** No error handling in `copy_async()` or on submission path.  
**Issue:** Cannot distinguish "copy never started" from "copy started but completion failed".

---

## 3. Design specification

### 3.1 Submission and completion (§2.1)

**Definition:**
- **Submission:** Copy operation queued with GPU driver (call returns)
- **Completion:** Result visible to consumer (host memory written, producer stream past copy)

**Requirement:** Synchronous `copy_sync()` must establish completion before returning.

#### 3.1.1 Sync copy behavior

**New method:**
```cpp
// Copy-and-wait semantics: returns only after completion
template <typename T>
static void copy_sync(
    T const*           src,
    size_t             count,
    T*                 dst,
    device_enum        from_type,
    device_enum        to_type,
    int                from_index = 0,
    int                to_index   = 0,
    stream_t           stream     = nullptr
);
```

**Preconditions:**
- Pointers are valid and aligned per Phase 3
- Count and alignment do not overflow
- Devices are accessible (compute capability, driver support)
- For GPU→CPU: stream is null or per-thread default ✓, not a user stream that may not exist (spec in Phase 2.4)

**Implementation strategy:**
1. **CPU→CPU:** `std::memcpy` or backend-native copy, return (synchronous)
2. **CPU→GPU or GPU→GPU:** 
   - Validate device, stream, and alignment
   - Submit work to the operation's device stream
   - Record completion marker (operation-specific event, not stream query)
   - Wait for that marker before returning
3. **GPU→CPU:**
   - If stream is null: use a fixed safe stream (e.g., legacy default)
   - If stream is user-provided: caller responsible for stream lifetime
   - Submit copy
   - Wait for stream completion before returning

**Completion guarantee:** After `copy_sync()` returns:
- For GPU→CPU: host destination is visible and stable
- For CPU→GPU: device destination is written and visible to kernels on the same stream
- For GPU→GPU: destination stream has finished the transfer

#### 3.1.2 Async copy behavior (no change to signature, fixes semantics)

**Existing method (corrected semantics):**
```cpp
template <typename T>
static copy_token copy_async(
    retained_ptr<T> const& from,
    retained_ptr<T> const& to,
    stream_t               stream = nullptr
);
```

**Change:** Record a **completion marker** (not just stream state).

**Pre-submission:**
- Call `record_stream()` on both GPU endpoints (existing behavior ✓)
- Validate pointers, count, alignment, devices

**Submission:**
- Submit copy to device/stream pair
- Record a **per-operation event** (CUDA `cudaEvent_t`, not just stream reference)
- Store event in `copy_token` (and stream/device for fallback)

**Postsubmission:**
- If submission fails (before work queued): return error (see §3.5)
- If event recording fails: fallback to stream wait; quarantine on repeat failure (Phase 1.4)

**Return:** `copy_token` holding operation's completion event + context

---

### 3.2 Operation-specific completion (§2.3)

**Current token:** Stores `{device_type, device_index, stream}` only

**Corrected token:** Stores:
- `device_type` – backend (CUDA, HIP, Metal)
- `device_index` – device ID
- `stream` – the stream used (for fallback / diagnostics)
- `event` (CUDA/HIP only) – operation-specific event, or null for CPU
- `pending` – state: submitted / complete / failed (Phase 2.7)

**Methods:**
```cpp
bool ready() const noexcept;  // Returns true iff completion marker is complete
void wait() const noexcept;   // Blocks until completion marker fires
completion_state state() const noexcept;  // pending, complete, failed
```

**Behavior change:**
- `token.ready()` observes the **operation's event**, not the stream
- Later stream work does **not** make `token.ready()` true
- Unrelated kernels on the same stream do not affect token completion

**Benefit:** Caller can check `token.ready()` without false positives from unrelated work.

---

### 3.3 Device context handling (§2.4)

**Problem:** Multi-device programs may have different current contexts on each thread.

**Solution:** Explicit device activation on the operation's device.

#### 3.3.1 CUDA/HIP (Recommended)

**At submission:**
```cpp
cudaSetDevice(device_index);  // Activate operation's device
cudaMemcpyAsync(...);         // Submit on this device's stream
// Implicit restore not guaranteed; caller manages push/pop if needed
```

**At completion (token.wait()):**
```cpp
if (ctx_.is_gpu()) {
    cudaSetDevice(ctx_.device_index);
    cudaEventSynchronize(event_);  // Wait on the right device
}
```

**Open issue:** Whether to restore the prior device context. Phase 2 spec does **not** save/restore; Phase 8 (API) may define a RAII guard.

#### 3.3.2 Default stream identity

**Problem:** CUDA has two "default stream" modes:
- **Legacy mode:** One default stream per device (not per-thread)
- **Per-thread mode:** Each thread has its own default stream

**Requirement:** Clarify which mode is supported.

**Recommendation:**
- Support only **legacy default stream** (null pointer = device's default stream) in Phase 2 core
- Document per-thread mode as unsupported or experimental
- Reject user streams that cannot be validated (Phase 2.5)

---

### 3.4 Stream validation and selection (§2.4, 2.5)

**Current behavior:** Copy with `stream=nullptr` on GPU uses a context-dependent stream (unclear).

**Phase 2 requirement:**

| Scenario | Behavior |
|----------|----------|
| CPU↔CPU | Ignore stream; always synchronous |
| CPU→GPU, stream=nullptr | Use device's legacy default stream |
| GPU→GPU same device, stream=nullptr | Use device's legacy default stream |
| GPU→GPU peer, stream=nullptr | Use destination device's default stream |
| GPU→CPU, stream=nullptr | Use source device's default stream (or legacy CPU) |
| User stream provided | Use as-is; caller ensures stream lifetime |

**Validation (Phase 2.1):**
- Do **not** validate stream pointer (it may be valid or invalid until submission)
- At submission time, detect if stream is invalid and fail with `std::invalid_argument`

**Reject unsupported combos before submission:**
- Per-thread default stream with null pointer ✗ → emit `std::logic_error` or document as unsupported
- Cross-device copy without peer-access permission ✗ (Phase 7 with graphs; Phase 2 may reject)

---

### 3.5 Submission failure handling (§2.7)

**Distinguish two failure points:**

1. **Pre-submission failure** (validation, memory, etc.)
   - `std::bad_alloc` – OOM on metadata allocation
   - `std::invalid_argument` – size overflow, bad pointer
   - `std::logic_error` – unsupported context combo

2. **Post-submission failure** (driver error after work queued)
   - Submission succeeded but event recording failed
   - No exception thrown; operation incomplete, possibly unsafe
   - Fallback: wait on stream (slower, catches the error)
   - If stream wait also fails: quarantine (Phase 1.4)

**Implementation:**
```cpp
// Pre-submission validation
if (count > max_size_t / sizeof(T)) {
    throw std::overflow_error("...");
}

// Submission
cudaMemcpyAsync(...);  // May fail (driver OOM)
if (cudaGetLastError() != cudaSuccess) {
    throw std::runtime_error("...");  // Phase 2: decide if this is thrown or fallback
}

// Event recording (post-submission)
cudaEventRecord(event);
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    // Fallback: wait on stream (may be slow)
    // Or: quarantine (Phase 1.4)
}
```

**Phase 2 decision:** Throw on pre-submission errors; fallback or quarantine on post-submission errors (details in Phase 1 review).

---

### 3.6 Clone semantics (§2.6)

**Current behavior:**
```cpp
data_ptr<T> copy = original.clone();  // Synchronous deep-copy
```

**Questions for Phase 3 (not Phase 2):**
- Does `clone()` establish GPU→CPU completion? (Yes, it's CPU-visible)
- Should there be an async clone? (Recommend: `clone_async()` returns `data_ptr + copy_token`)

**Phase 2 scope:** Keep `clone()` synchronous. Async clone deferred to Phase 3.

---

### 3.7 Copy with foreign/borrowed pointers (§2.5, 3.6)

**Current code:**
```cpp
// Raw pointer copy (not retained)
allocator<float>::copy(raw_src, count, raw_dst, CPU, GPU, 0, 0, stream);
```

**Issue:** Raw pointers cannot be validated against the caching allocator; `record_stream()` may fail.

**Phase 2 approach:**
- Borrow API remains (callers use at their own risk)
- No validation of borrowed pointers at submission
- GPU pointer must be in the caching allocator's registry (submit succeeds but copy fails if interior/foreign)
- Document: "Borrowed GPU pointers must be base allocations; interior pointers fail without recovery"

**Phase 3:** Introduce optional pointer validation with adoption metadata.

---

## 4. Implementation roadmap

### Step 1: Specify `copy_sync()` API and semantics (this document)

### Step 2: Add operation-specific completion marker to `copy_token`

- Define `enum class completion_state { pending, complete, failed }`
- Add `event_` field to `copy_token` (CUDA/HIP only)
- Implement `state()`, update `ready()` and `wait()` to query event

### Step 3: Implement `copy_sync()`

- Add synchronous wrapper around `copy_async()` + `token.wait()`
- Fix device activation for multi-device scenarios
- Add pre-submission validation

### Step 4: Validate with CPU-only tests

- CPU→CPU copies
- Borrowed pointer copies (CPU)
- Error cases: bad_alloc, invalid_argument, overflow

### Step 5: GPU testing (Phase 2, hardware-dependent)

- CUDA/HIP stream-specific completion
- Event handling under failure
- Multi-device context switching

---

## 5. Acceptance criteria

- [ ] `copy_sync()` established before return (observe with intentional delay)
- [ ] `copy_token::ready()` true only after operation completes (not after later stream work)
- [ ] Device context managed correctly for multi-device scenarios
- [ ] Pre-submission errors throw with correct type
- [ ] Post-submission errors fall back or quarantine (Phase 1)
- [ ] No unsupported default-stream modes accidentally enabled
- [ ] Documentation clarifies token does not retain endpoints
- [ ] CPU-only tests pass (including borrowed-pointer error cases)
- [ ] Hardware tests confirm stream/event ordering (CUDA/HIP when available)

---

## 6. Open issues for Phase 2+

| Issue | Resolution path |
|-------|---|
| Device context save/restore | Phase 8 (API stability) |
| Per-thread default stream support | Phase 7 or later (optional feature) |
| Cross-device peer copies | Phase 7 (driver pools) or Phase 2 reject |
| Async clone API | Phase 3 (storage identity) |
| Ringbuffer trace recording | Phase 5 (accounting) |
