# Phase 3: Storage Identity, Ownership, and Retained Async Lifetime

**Status:** Design specification for storage resource contract  
**Baseline:** commit `b4626e5`  
**Scope:** CPU-independent spec; GPU implementation with Phase 2

## 1. Overview

Phase 3 establishes:
1. Unique identity per allocation lifetime (reused addresses get new identities)
2. Byte resource contract: allocation, ownership, adoption, slicing
3. Retained storage: keeps endpoints alive through async operations
4. Token discard safety: bounded pending-operation service
5. Borrowed pointers with explicit limitations

---

## 2. Current state (gaps)

### 2.1 No allocation identity

**Current:** Pointers are the identity. Address reuse means loss of tracking.

```cpp
auto ptr1 = allocator<T>::allocate(100, ctx);
allocator<T>::free(ptr1, ...);
auto ptr2 = allocator<T>::allocate(100, ctx);
// ptr1 == ptr2 after deallocation + reallocation
// No way to tell if ptr2 is a new allocation or reuse
```

**Issue:** Telemetry, retained operations, and debugging cannot distinguish lifecycle.

### 2.2 `view()` does not retain owner

**Current code:** `data_view<T>` does not hold a reference to `data_ptr<T>`.

```cpp
data_ptr<float> owner(1000, ctx);
data_view<float> view = owner;
// owner.~data_ptr() → allocator::free()
// view.data() now points to deallocated memory
```

**Issue:** Silently unsafe if owner destroyed while view is in use.

**Design:** (Intentional per project CLAUDE.md) View is borrowed; caller ensures owner lifetime.

### 2.3 `retained_ptr` exists but lifecycle not specified

**Current:** Can be created with `adopt()` but:
- Adoption failure behavior unclear
- Retained slices don't track original allocation
- No specified shared-owner reference counting

### 2.4 Async operations do not retain endpoints

**Current:** `copy_async()` does not hold references.

```cpp
retained_ptr<T> src = ..., dst = ...;
copy_token token = allocator<T>::copy_async(src, dst);
// User drops both ptrs while token is pending
// Allocator may reuse destination before copy finishes
```

**Issue:** Even with `retained_ptr`, caller must manually keep references alive.

---

## 3. Design specification

### 3.1 Allocation identity (§3.2)

**Requirement:** Every allocation lifetime has a unique identity.

#### 3.1.1 Identity definition

```cpp
namespace memory {

struct allocation_id {
    uint64_t value;  // Unique within process lifetime
    
    bool operator==(allocation_id const&) const noexcept;
    bool operator!=(allocation_id const&) const noexcept;
};

}  // namespace memory
```

**Uniqueness:**
- Incremented on every `allocate()` call
- Never reused (no wraparound / reset)
- Process-wide (not per-device or per-allocator)
- Stable across address reuse (new address = new identity)

**Scope:**
- Valid from first successful allocation through final `free()`
- Survives allocation address being reused for different data
- Not tied to `data_ptr` or `retained_ptr` lifetime (handles may disappear)

#### 3.1.2 Identity tracking in ownership types

**`data_ptr<T>`:**
```cpp
struct data_ptr<T> {
    allocation_id id() const noexcept;  // Unique per allocation lifetime
    // ... other members
};
```

**`retained_ptr<T>`:**
```cpp
struct retained_ptr<T> {
    allocation_id id() const noexcept;  // Same as underlying allocation
    // ... other members
};
```

**`data_view<T>`:**
```cpp
struct data_view<T> {
    allocation_id id() const noexcept;  // Derived from owner (Phase 3.1.3)
    // ... other members
};
```

#### 3.1.3 Slicing preserves identity

**Requirement:** A slice (view or retained slice) carries the original allocation's identity.

```cpp
auto full = allocator<float>::allocate(1000, ctx);
allocation_id full_id = full.id();

auto slice = full.slice(100, 500);
allocation_id slice_id = slice.id();

// full_id == slice_id  // Same underlying allocation
slice.offset() == 100;   // But offset is tracked separately
slice.size() == 500;
```

**Benefit:** Telemetry, debugging, and Phase 1's quarantine can track allocation through all slices.

---

### 3.2 Byte resource contract (§3.1)

**Resource type: block of contiguous bytes with metadata.**

#### 3.2.1 Allocation request

```cpp
namespace memory::allocator {

template <typename T>
static data_ptr<T> allocate(
    size_t count,              // Element count (not bytes)
    execution_context ctx,     // Device, stream
    size_t alignment = alignof(T)  // Alignment requirement
);

}  // namespace memory::allocator
```

**Input validation:**
- `count`: Non-negative; overflow check: `count * sizeof(T)` fits in `size_t`
- `alignment`: Power-of-two, ≤ max(system alignment, 64 for AVX-512)
- `ctx`: Valid device index for the backend

**Output:**
- Unique `allocation_id`
- Pointer aligned to ≥ `max(alignment, backend_default)`
- Capacity ≥ `count * sizeof(T)` bytes
- Device index, stream, type stored in owner

**Exceptions:**
- `std::invalid_argument`: count == 0 or invalid alignment
- `std::overflow_error`: `count * sizeof(T)` overflows
- `std::bad_alloc`: Driver OOM (after cache flush)

#### 3.2.2 Owned vs. borrowed storage

**Owned:** Created by `allocate()` or `adopt()`, deallocated by `free()`.

**Borrowed:** Created by `borrow()` or constructor from raw pointer; caller manages lifetime.

```cpp
// Owned (allocator manages)
data_ptr<float> owner(100, ctx);  // allocate()
auto adopted = allocator<float>::adopt(raw_ptr, count, ctx);  // adopt()

// Borrowed (caller manages)
data_view<float> borrowed = owner.view();  // Non-owning reference
auto borrowed2 = allocator<float>::borrow(raw_ptr, count, ctx);  // Raw borrow (no validation)
```

**Guarantee:** Borrowed storage requires explicit caller lifetime management. Documentation must be clear.

#### 3.2.3 Adoption contract (§3.3)

**New method (Phase 3):**

```cpp
template <typename T>
static retained_ptr<T> allocate_adopted(
    T* foreign_ptr,            // Caller-allocated pointer
    size_t count,              // Element count
    device_enum type,          // Device type (GPU backend must match)
    int device_index = 0,      // Device ID
    std::function<void(T*)> deleter = [](T* p) { /* default */ }
);
```

**Contract:**
- Takes ownership of `foreign_ptr`
- Calls `deleter(foreign_ptr)` on destruction
- Assigns new `allocation_id`
- Tracks `count * sizeof(T)` as the allocation size
- If adoption fails (allocation, bad pointer): deleter NOT called

**Supported deleters:**
- `nullptr` → no-op deleter (memory was static/external)
- Lambda / `std::function` → called once on destruction
- CPU `delete[]` → wrapped in safe deleter

**Unsupported (Phase 2/3):**
- GPU device pointers adopted for use with CPU code (type mismatch)
- Interior pointers (adoption succeeds but point may not resolve)

**Error handling:**
- `std::invalid_argument`: null pointer, bad deleter state
- `std::runtime_error`: adoption fails (e.g., device mismatch)

---

### 3.3 Retained storage and async lifetime (§3.4, 3.5)

**Problem:** Without retained ownership, async operations may start on data that's freed.

#### 3.3.1 Retained transfer API

**Spec: Use `retained_ptr` for async operations that keep endpoints alive.**

```cpp
// Current (caller-managed lifetime, from Phase 2)
copy_token allocator<T>::copy_async(
    data_ptr<T> const& from,  // Caller must keep alive until token.wait()
    data_ptr<T> const& to
);

// New (retained ownership model, Phase 3)
copy_token allocator<T>::copy_async_retained(
    retained_ptr<T> const& from,  // Operation retains reference
    retained_ptr<T> const& to     // Operation retains reference
);
```

**Behavior:**
- Takes shared ownership via reference count
- Increments refcount during submission
- Decrements refcount when operation completes (or is discarded)
- Endpoints guaranteed valid through completion

**Benefit:** User drops `retained_ptr`s; async operation keeps them alive.

#### 3.3.2 Retained operation service (§3.5)

**Problem:** Token destroyed while operation pending → storage leak.

**Solution: Optional bounded service.**

```cpp
namespace memory {

class retained_operation_service {
public:
    // Singleton per process
    static retained_operation_service& instance();
    
    // Enqueue an operation token for automatic cleanup
    void enqueue(copy_token&& token, size_t priority = 0);
    
    // Poll pending operations; return count of completed
    size_t poll() noexcept;
    
    // Block until all pending complete or timeout
    void wait_all(std::chrono::milliseconds timeout);
    
    // Diagnostics
    size_t pending_count() const noexcept;
    void set_max_pending(size_t limit);  // Backpressure control
};

}  // namespace memory
```

**Behavior:**
- User can opt-in: `service.enqueue(std::move(token))`
- Service polls tokens asynchronously (thread pool, background task, or manual `poll()`)
- When token completes, service discards it (refcount drops)
- Max pending size enforced; `enqueue()` waits or fails if full
- At shutdown: drain all pending operations

**Benefits:**
- Prevents unbounded pending-operation queue
- Caller can use RAII guard: `auto guard = service.enqueue(std::move(token))`
- Transparent integration with token API

**Phase 3 scope:** Design and API only. Phase 4+ implements background polling.

---

### 3.4 Borrowed pointer limitations (§3.6)

**Explicit constraints for unsupported scenarios.**

#### 3.4.1 Borrowed GPU pointers

**Allowed:**
- Base allocations from `allocate()` (registered in caching allocator)
- Explicit `data_view<T>` of owned allocation

**Disallowed:**
- Interior pointers (offsets into allocations not tracked)
- Foreign GPU pointers (from external CUDA malloc, TensorFlow, etc.)
- Peer-device pointers (GPU0 buffer accessed as GPU1 buffer)

**Behavior:**
- Submission validation does **not** check borrowed pointers
- If borrowed pointer is invalid: submit succeeds, but GPU operation fails (event error)
- Phase 1.4 quarantines the allocation if repeated
- Documentation must warn: "Borrowed GPU pointers must be base allocations"

#### 3.4.2 Borrowed CPU pointers

**Allowed:**
- Any valid host pointer
- Stack, heap, static memory, pinned buffers

**Disallowed:**
- Pointers that move (e.g., `std::string` data, growable vectors)
- Pointers to memory freed during async operation

**Behavior:**
- Copy submission does not validate lifetime
- If pointer becomes invalid: GPU→CPU copy silently corrupts
- Phase 3+ could add optional callback-based lifetime tracking
- Documentation: "Caller ensures buffer lifetime through completion"

---

### 3.5 Typed storage (§3.7)

**Phase 3 scope: Trivial types only.**

```cpp
template <typename T>
class data_ptr {
    // Requires: std::is_trivial_v<T> (no constructor/destructor)
    static_assert(std::is_trivial_v<T>, "...");
    
    // Allocate uninitialized storage
    data_ptr(size_t count, execution_context ctx)
        : /* ... allocated but not constructed ... */
    {
        static_assert(std::is_trivial_v<T>);
    }
};
```

**Enforcement:**
- Compile-time check via `static_assert`
- No implicit construction/destruction
- `clone()` does `std::memcpy`
- Copy/move do not call any T methods

**Non-trivial types (deferred):**
- Would require arena or allocator-aware containers (Phase 4+)
- Not a goal of core Phase 3

---

## 4. Implementation roadmap

### Step 1: Define `allocation_id` and storage_identity.h

### Step 2: Add identity tracking to `data_ptr`, `retained_ptr`, `data_view`

### Step 3: Implement `adopt()` / `allocate_adopted()` for foreign storage

### Step 4: Specify retained transfer semantics (no code change yet)

### Step 5: Design retained_operation_service API (skeleton only)

### Step 6: CPU-only tests for identity, adoption, slicing

### Step 7: GPU tests for retained transfers (Phase 2 hardware)

---

## 5. Acceptance criteria

- [ ] Every allocation has unique identity; addresses may be reused
- [ ] Slicing preserves identity with separate offset/size tracking
- [ ] Adoption contract specified; foreign storage supported
- [ ] `data_view` documented as non-owning borrow
- [ ] Retained transfer API supports keeping endpoints alive
- [ ] Retained operation service API exists (skeleton)
- [ ] Borrowed GPU pointers must be base allocations (documented, tested)
- [ ] Borrowed CPU pointers lifetime caller's responsibility (clear docs)
- [ ] Trivial-type requirement enforced or documented
- [ ] No silent double-free or premature reuse of adopted storage
- [ ] CPU-only tests pass (allocation, adoption, slicing identity)
- [ ] GPU tests pass when hardware available

---

## 6. Open issues for Phase 3+

| Issue | Resolution |
|---|---|
| Interior pointer validation | Phase 4 (optional adoption metadata) |
| Peer-device access | Phase 7 (graphs, device mapping) |
| Non-trivial types | Phase 4+ (general object containers) |
| Async clone | Phase 3 (add `clone_async()` returning `data_ptr + copy_token`) |
| NUMA-aware identity | Phase 2+ (allocator placement hints) |
| Shared ownership refcount impl | Phase 3 (std::shared_ptr? custom?) |

---

## 7. Related documentation

- Phase 1: Quarantine states for unsafe allocations
- Phase 2: Completion guarantees for retained transfers
- Phase 4: Arena/workspace identity integration
- Phase 5: Telemetry identity tracking and loss detection
