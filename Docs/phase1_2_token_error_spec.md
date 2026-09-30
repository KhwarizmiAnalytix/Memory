# Phases 1–2: Token Lifecycle and Error Semantics

**Status:** Design specification for token/error behavior  
**Baseline:** commit `b4626e5`  
**Scope:** CPU-independent specification; GPU validation deferred

## 1. Overview

Token lifecycle and error handling establish:
1. Clear error reporting for allocation, submission, and completion failures
2. Token state machine: pending → complete / failed
3. Safe token discard (does not leak resources)
4. Error boundaries: pre-submission vs. post-submission
5. Quarantine states for failures that may leave unsafe allocations

---

## 2. Token state machine

### 2.1 States

```
┌─────────────┐
│  created    │
│  (default)  │
└──────┬──────┘
       │
       ├──→ CPU copy (immediate)
       │    └──→ ✅ complete
       │
       └──→ GPU submission
            │
            ├──→ ✅ pending
            │    └──→ (operation in flight)
            │         │
            │         ├──→ ✅ complete (event fires)
            │         └──→ ❌ failed (event error)
            │
            └──→ ❌ failed (submission error)
```

### 2.2 Detailed state definitions

| State | Meaning | Queries return | Wait behavior |
|-------|---------|---|---|
| **created** | Default-constructed token (CPU no-op) | `ready()==true` | returns immediately |
| **pending** | GPU work enqueued; event recorded | `ready()==false` | blocks until event fires |
| **complete** | Event has fired; operation done | `ready()==true` | returns immediately |
| **failed** | Event error or submission failed | `ready()==false` (if not yet checked) | blocks, then throws |

### 2.3 Transitions

```cpp
copy_token token;  // created
// [submit GPU copy]
token.state() == completion_state::pending;

// Option A: operation completes
token.state() == completion_state::complete;  // read-only after first query

// Option B: operation/event fails
token.state() == completion_state::failed;
token.wait();  // throws std::runtime_error
```

---

## 3. Error categories

### 3.1 Allocation and setup errors (pre-submission)

**Thrown before GPU call; operation never started.**

| Error | When | What to do |
|-------|------|---|
| `std::bad_alloc` | Event/metadata allocation fails | Flush cache (`empty_cache()`) and retry |
| `std::invalid_argument` | Size overflow, bad alignment, count mismatch | Fix caller; check types |
| `std::overflow_error` | Integer overflow in `count * sizeof(T)` | Use smaller sizes |
| `std::logic_error` | Unsupported device/stream combo | Use supported combo or wait for Phase 7 |

**Guarantees:**
- Storage is not modified
- No operation queued on GPU
- Token is either created or failed

### 3.2 Submission errors (at GPU call)

**Thrown when driver call fails; operation may or may not be queued.**

| Error | When | State |
|---|---|---|
| `std::runtime_error` | `cudaMemcpyAsync` fails | Pre-submission (safe, storage unmodified) |
| `std::runtime_error` | `cudaEventRecord` fails | Post-submission (unsafe; see Phase 1.4) |

**Detail:** Driver OOM, permission denied, stream invalid (stale pointer).

**Handling strategy:**
- Pre-submission (before `cudaMemcpyAsync`): safe to retry or fail
- Post-submission (after `cudaMemcpyAsync`): work may be queued; must wait or quarantine

### 3.3 Completion errors (from token.wait())

**Exceptions thrown when waiting for a failed operation.**

```cpp
copy_token token = ...;  // Submission succeeded
// [GPU error occurs during operation]
token.wait();  // May throw std::runtime_error
```

**Possible errors:**
- Event query returns error
- Stream synchronization fails (rare)

**Handling:** Cannot undo; destination may be partially written.

---

## 4. Detailed API specification

### 4.1 `copy_token` class definition

```cpp
namespace memory {

enum class completion_state {
    pending,   // Awaiting completion
    complete,  // Operation done
    failed     // Error occurred
};

class copy_token {
public:
    // Default construction: CPU no-op (immediately complete)
    copy_token() noexcept = default;

    // From context (internal constructor for allocator)
    explicit copy_token(execution_context ctx) noexcept;

    // Copyable (share outcome with multiple waiters)
    copy_token(copy_token const&) noexcept             = default;
    copy_token& operator=(copy_token const&) noexcept  = default;

    // Movable (transfer ownership)
    copy_token(copy_token&&) noexcept                 = default;
    copy_token& operator=(copy_token&&) noexcept      = default;

    // Query completion state
    // Returns immediately; does not block
    // For GPU ops: queries the recorded event (if any)
    bool ready() const noexcept;

    // Return detailed state: pending, complete, or failed
    // For CPU ops: always returns 'complete'
    // For GPU ops: queries event or stream (see Phase 1.4)
    completion_state state() const noexcept;

    // Block until operation completes
    // Returns when event fires or stream reaches this operation
    // Throws std::runtime_error if event has error status
    void wait() const;  // May throw

    // Get execution context (for diagnostics)
    execution_context context() const noexcept { return ctx_; }

private:
    execution_context ctx_{execution_context::cpu()};
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
    cudaEvent_t event_{nullptr};
    mutable completion_state cached_state_{completion_state::pending};
#endif
};

}  // namespace memory
```

### 4.2 Behavior specifications

#### 4.2.1 `ready()` semantics

```cpp
bool token.ready() const noexcept;
```

**For CPU operations:**
- Always returns `true` (no-op copy already done)

**For GPU operations:**
- Queries the operation's event (not the stream)
- Returns `true` if event has fired
- Returns `false` if event is pending or has error status
- Does **not** throw; returns false on query error

**Guarantees:**
- No blocking
- Multiple `ready()` calls return same result while pending
- Once `true`, remains `true` forever

**False positive fix:** Only `true` iff *this specific operation* is done, not just any work on the stream.

#### 4.2.2 `state()` semantics

```cpp
completion_state token.state() const noexcept;
```

**Returns:**
- `completion_state::complete` – operation succeeded and is done
- `completion_state::pending` – operation still in flight
- `completion_state::failed` – operation or event recording failed

**For CPU operations:**
- Always returns `complete`

**For GPU operations:**
- Queries event status (not stream)
- Returns `failed` if event has error or event recording failed
- Caches result (first call observes state; subsequent calls return cached state)

**Guarantees:**
- No blocking
- Idempotent (same state after each call)

#### 4.2.3 `wait()` semantics

```cpp
void token.wait() const;  // May throw std::runtime_error
```

**For CPU operations:**
- Returns immediately (no-op)

**For GPU operations:**
- If state is `complete`: returns immediately
- If state is `pending`:
  - Blocks until event fires
  - Queries event result
  - If event succeeded: returns
  - If event has error: throws `std::runtime_error`
- If state is `failed`:
  - Returns immediately; operation already failed (no throw)
  - Or: throws if error details available

**Guarantees:**
- After successful `wait()`, caller knows destination is stable
- Exception on `wait()` does not clear the error (calling `wait()` again re-throws)
- Safe to call multiple times

**Error reporting:**
- Throw `std::runtime_error` with diagnostic message
- Include event status, device, stream in message (Phase 5 telemetry)

---

## 5. Allocation error handling

### 5.1 Pre-submission contract

**The caller's responsibility:**

```cpp
try {
    copy_token token = allocator<float>::copy_async(from, to);
    // If we reach here, submission succeeded (or is pending if GPU)
}
catch (std::bad_alloc const&) {
    // Flush cache and retry
    gpu::empty_cache(device_index);
    // Try again
}
catch (std::invalid_argument const&) {
    // Fix the input: size, alignment, pointer
}
catch (std::logic_error const&) {
    // Unsupported combo (e.g., per-thread default stream without support)
    // Use different device/stream or upgrade
}
```

**After exception, both endpoints are untouched.**

### 5.2 Post-submission contract

**Failure after work queued (driver OOM, event errors):**

This is a Phase 1 issue. Placeholder for Phase 2:

```cpp
copy_token token = ...;  // Submission succeeded
// [GPU error]
token.state() == completion_state::failed;
try {
    token.wait();  // Destination may be partially written
}
catch (std::runtime_error const&) {
    // Error details in message
    // Destination in unknown state
    // Allocator may retry; quarantine if repeated
}
```

**Handling:** Documented in Phase 1.4 (quarantine); Phase 2 establishes the API only.

---

## 6. Token discard (addressed in Phase 3)

**Current issue:** If token is destroyed while operation is pending, what happens?

**Phase 2 assumption:** Caller keeps token alive.

**Phase 3 solution:** Optional pending-operation service retains tokens until completion (work deferred).

---

## 7. Multi-threaded token use

### 7.1 Token sharing

```cpp
copy_token token = ...;
std::thread worker([token]() {
    if (token.ready()) {
        // Check on another thread
    }
    token.wait();  // Wait from another thread
});
```

**Supported:** Tokens are thread-safe for querying and waiting (no mutable state except cached_state, which is guarded by idempotency).

**Not supported in Phase 2:** Default-stream identity if different threads have different default streams (see Phase 2.3.2).

### 7.2 Copy with cross-thread endpoints

**Not a Phase 2 concern.** Phase 3 specifies ownership; Phase 2 assumes caller's responsibility.

---

## 8. Default-constructed tokens (CPU no-ops)

```cpp
copy_token default_token;  // No copy submitted
default_token.ready();      // true
default_token.state();      // complete
default_token.wait();       // no-op
```

**Use case:** Placeholder for conditional copying.

```cpp
copy_token maybe_copy;
if (condition) {
    maybe_copy = allocator<T>::copy_async(...);
}
maybe_copy.wait();  // Safe whether or not copy happened
```

---

## 9. Diagnostics and telemetry

### 9.1 Error messages (Phase 5+ detail)

Include in `std::runtime_error`:
- Device index
- Stream pointer (as-is, not dereferenced)
- Event status code
- Whether it's pre/post-submission

### 9.2 Profiler integration (Phase 5+)

- Record token creation, completion, and failure
- Track pending token count (diagnostics for unbounded queuing)
- Report error rates and retry counts

---

## 10. Acceptance criteria (Phases 1–2)

- [ ] Token state machine documented and tested
- [ ] Pre-submission errors throw with correct type
- [ ] Post-submission errors reported (Phase 1.4 specifies quarantine)
- [ ] `ready()` queries operation's event, not stream
- [ ] `state()` returns accurate completion status
- [ ] `wait()` blocks until operation completes
- [ ] Default-constructed token is immediate no-op
- [ ] Multi-threaded token sharing is safe
- [ ] Error messages include diagnostic context
- [ ] CPU-only tests pass (no GPU required)
- [ ] GPU tests pass when hardware available (Phase 2 hardware validation)

---

## 11. Related phases

| Phase | Dependency | What's added |
|---|---|---|
| Phase 1 | This spec | Failure injection; quarantine; diagnostics hooks |
| Phase 2 | This spec | Actual implementation of state machine + error handling |
| Phase 3 | Phases 1–2 | Token lifetime management (service retaining abandoned tokens) |
| Phase 5 | Phase 2 | Telemetry and error reporting |
