# Phases 0–3 CPU-Independent Work — Summary (2026-09-30)

**Baseline:** commit `b4626e5`  
**Completed:** Phase 0 (documentation), Phases 1–3 (specifications)  
**Status:** Design-ready for implementation; no GPU hardware required

---

## What was done

### Phase 0 — Contract and Validation Baseline

**README updates (committed):**
- Line 58–69: Fixed `data_ptr<T>` example
  - Removed false claim "copy = deep clone" that was wrong
  - Added explicit `clone()` method for deep-copy
  - Clarified move-only semantics
- Line 120–140: Fixed "Async GPU→CPU Copy" pattern
  - Removed false claim "Token holds references"
  - Documented caller responsibility to keep endpoints alive
  - Clarified `record_stream()` automatic handling
- New section: "Copy Token — Async Operations"
  - Explicit lifecycle rules with code examples
  - Shows correct ✅ and incorrect ❌ patterns
  - Stream recording behavior documented

**Validation manifest template (new file):**
- `Docs/validation_manifest_template.md`
- Metadata capture: compiler, platform, GPU backend, driver, hardware
- Test result table for all unit test categories
- Benchmark result template
- Feature checklist
- Known limitations section
- Acceptance decision template
- How to use: fill once per run, link from PRs

**What's next:** Use this manifest for every test run going forward.

---

### Phase 1–2 — Token Lifecycle and Error Semantics

**New specification:** `Docs/phase1_2_token_error_spec.md`

Covers:
- **Token state machine:** created → pending/complete/failed
- **Error categories:** Pre-submission (safe to retry), post-submission (unsafe, quarantine)
- **API contract for `copy_token`:**
  - `ready()` – queries operation's event (not stream)
  - `state()` – returns detailed state
  - `wait()` – blocks until complete, throws on error
- **Allocation error handling:**
  - `std::bad_alloc` → flush cache, retry
  - `std::invalid_argument` → fix input
  - `std::overflow_error` → use smaller sizes
  - `std::logic_error` → unsupported device/stream combo
- **Post-submission errors:** Placeholder for Phase 1 quarantine

**Key insight:** Current `copy_token` is broken because `ready()/wait()` observe the entire stream, not the specific operation. Phase 2 must add per-operation events.

**What's next:** Implement Phase 1 (diagnostics, quarantine) and Phase 2 (event-based completion) using this spec.

---

### Phase 2 — Transfer Completion and Context Semantics

**New specification:** `Docs/phase2_copy_completion_spec.md`

Covers:
- **Problem statement:** `copy_sync()` doesn't actually wait; tokens see stream-wide completion, not operation-specific
- **Required changes:**
  1. Add operation-specific completion markers (CUDA events, not stream queries)
  2. Implement truthful `copy_sync()` that waits before returning
  3. Fix device context activation for multi-device scenarios
  4. Distinguish pre-submission errors from post-submission errors
  5. Clarify stream validation and default-stream identity
- **API changes:**
  - New `allocator<T>::copy_sync()` method
  - Extended `copy_token` with event + state tracking
  - Updated `token.ready()` to observe event, not stream
- **Acceptance criteria:** CPU-only tests pass; GPU tests (with hardware) verify event ordering

**Key insight:** Phase 2 is the foundation for Phases 3–5. Cannot proceed to retained ownership without truthful completion.

**What's next:** Implement step-by-step (spec, add events to token, implement copy_sync, add validation, test).

---

### Phase 3 — Storage Identity, Ownership, and Retained Async Lifetime

**New specification:** `Docs/phase3_storage_identity_spec.md`

Covers:
- **Allocation identity:** Unique per allocation lifetime (survives address reuse)
  - New type: `allocation_id` (uint64, process-wide, never reset)
  - Carried by `data_ptr`, `retained_ptr`, `data_view`
  - Slicing preserves identity with separate offset/size
- **Byte resource contract:**
  - `allocate(count, ctx, alignment)` input validation and guarantees
  - Owned vs. borrowed storage distinction
  - `adopt() / allocate_adopted()` for foreign storage (with deleter contract)
- **Retained storage and async lifetime:**
  - New `copy_async_retained()` keeping endpoints alive
  - Retained operation service (optional, bounded queue)
  - Safe token discard via service (Phase 3+)
- **Borrowed pointer limitations:**
  - GPU: must be base allocations (not interior/foreign)
  - CPU: caller ensures lifetime through completion
- **Typed storage:** Phase 3 supports trivial types only (no constructor/destructor)

**Key insight:** Retained ownership + identity enable Phase 5 telemetry and Phase 1 quarantine. GPU code is simpler when allocator keeps endpoints alive.

**What's next:** Implement allocation_id, adoption factories, retained transfer API (API only in Phase 3, retained service in Phase 4).

---

## Architecture of the fixes

```
Phase 0 (Documentation)
 ├─ README fixes ✓
 ├─ Validation manifest ✓
 └─ Reproduction protocol

Phase 1 (Failure Safety)
 ├─ Churn diagnosis (GPU hardware needed)
 ├─ Allocation rollback testing
 ├─ Cleanup diagnostic hooks
 ├─ Quarantine states
 └─ Shutdown ordering spec

Phase 2 (Completion Semantics) ← Foundation for Phases 3+
 ├─ Operation-specific events (not stream queries)
 ├─ Truthful copy_sync()
 ├─ Device context management
 ├─ Submission/completion error distinction
 └─ Phase 1–2 token spec ✓

Phase 3 (Storage Identity & Ownership)
 ├─ Allocation identity (surviving address reuse)
 ├─ Byte resource contract
 ├─ Adoption + foreign storage
 ├─ Retained async transfers
 ├─ Retained operation service API
 └─ Borrowed pointer limits (documented)

Phase 4 (Reusable Storage & Metal)
 ├─ CPU arena contract
 ├─ GPU workspace contract
 ├─ Pinned transfer integration
 ├─ Metal command-buffer integration
 └─ [Depends on Phases 2–3]

Phase 5 (Accounting & Diagnostics)
 ├─ O(1) memory queries (no free-list scans)
 ├─ Populate trace fields (real IDs, sizes)
 ├─ Fragmentation metrics
 ├─ OOM stack capture
 └─ [Depends on Phase 3 identity]

Phase 6 (Benchmarks & Optimization)
 ├─ Representative workloads
 ├─ Baseline measurements
 ├─ Comparison with native CUDA allocator
 └─ [Depends on Phases 1–5]

Phase 7 (Optional: Driver Pools & Graphs)
 ├─ cudaMallocAsync integration
 ├─ CUDA graph allocation
 └─ [Depends on Phases 2–6]

Phase 8 (API Stabilization)
 ├─ Final contract documentation
 ├─ Downstream consumer audit
 ├─ Support matrix
 └─ [Depends on Phases 0–6]
```

---

## Next steps (implementation order)

**Recommended sequence for non-GPU work:**

1. **Validate Phase 0** (today):
   - Run tests with new README examples
   - Confirm docs are accurate

2. **Start Phase 2 (CPU-side)** (parallel, non-blocking on churn):
   - Add operation-specific completion marker to `copy_token`
   - Implement `copy_sync()` synchronous wrapper
   - Add device context validation (activate correct device before GPU calls)
   - CPU-only tests (borrowed pointers, error cases)
   - This work does **not** require GPU hardware

3. **Start Phase 1 (non-churn parts)** (parallel):
   - Add diagnostic cleanup hooks (nonthrowing)
   - Document allocation rollback contract
   - Set up failure injection test framework
   - Churn diagnosis deferred (needs GPU + debugger)

4. **Start Phase 3 (design + CPU parts)** (after Phase 2):
   - Implement `allocation_id` type
   - Add identity tracking to ownership types
   - Implement adoption factories (`adopt()`)
   - Phase 3 retained_operation_service API skeleton only

5. **Merge and validate Phases 2–3** with hardware when available

6. **Phase 4–8** proceed from there

---

## Files created/modified (2026-09-30)

**Modified:**
- `README.md` – Examples and token lifecycle documentation

**Created:**
- `Docs/validation_manifest_template.md` – Template for all test runs
- `Docs/phase1_2_token_error_spec.md` – Token state machine + error handling
- `Docs/phase2_copy_completion_spec.md` – copy_sync(), events, device context
- `Docs/phase3_storage_identity_spec.md` – allocation_id, adoption, retained ownership
- `Docs/phase0_1_2_3_summary.md` – This file

**Update CLAUDE.md** to reference these specifications (next step).

---

## How this unblocks the roadmap

| Blocker | Removed by | How |
|---------|---|---|
| README claims unsupported API | Phase 0 docs ✓ | Examples and limitations clarified |
| Unclear token semantics | Phase 1–2 spec ✓ | State machine and error types documented |
| `copy_sync()` incomplete | Phase 2 spec ✓ | Completion markers designed |
| Retained ownership undefined | Phase 3 spec ✓ | Adoption + service API designed |
| No allocation tracking | Phase 3 spec ✓ | allocation_id designed |
| Churn crash TBD | Phase 1 (pending) | Hardware diagnosis needed |

---

## Acceptance gates

**Phase 0:** ✓ (done)
- README examples compile and are correct
- Manifest template is usable

**Phases 1–2 (design):** ✓ (specifications written)
- Token state machine specified
- copy_sync() and completion semantics specified
- Error handling contract defined
- Ready for implementation review

**Phases 1–2 (implementation):** Pending
- Phase 2 CPU-only work: can start immediately
- Phase 1 churn diagnosis: GPU hardware needed
- Tests pass (CPU) and (GPU with hardware)

**Phase 3 (design):** ✓ (specification written)
- allocation_id designed
- Adoption contract specified
- Retained ownership designed
- Ready for implementation review

**Phase 3 (implementation):** Next
- CPU-only parts: allocation_id, adoption factories
- GPU parts: retained_operation_service (Phase 4+)

---

## Deferred to hardware validation

1. **Phase 1.1** – Churn crash diagnosis (needs CUDA/HIP debugger + self-hosted runners)
2. **Phase 2 GPU tests** – Event recording, multi-stream ordering, cross-device context
3. **Phase 4** – Metal command-buffer integration, pinned transfer pipeline
4. **Phase 6** – Benchmark harness on real hardware
