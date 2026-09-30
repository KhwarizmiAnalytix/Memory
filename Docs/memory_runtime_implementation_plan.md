# Memory runtime implementation plan

Updated: 2026-09-30. Source baseline: `b4626e5` on local `main`.

This is the current implementation roadmap. It supersedes the delivery order and
completion claims in the [September 22 plan](cpu_gpu_memory_plan.md) and the
[September 28 review](cpu_gpu_memory_review.md), which remain historical evidence.
The September 29 [benchmark analysis](cuda_benchmark_analysis.md) is historical
measurement commentary, not evidence that the runtime is production-ready.

This update changes documentation only. No new build, sanitizer, benchmark, or
GPU runtime results are claimed. Findings below are either source observations
at the baseline commit or explicitly identified reports from earlier reviews.
Downstream Tensor, LinearAlgebra, and Vectorization implementations have not been
audited as part of this update.

## 1. Objective and scope

Deliver a small CPU/GPU storage runtime with explicit allocation, ownership,
transfer, completion, reclamation, and accounting contracts. Preserve the CPU
dispatch and GPU segment caches. Complete and validate existing components before
adding allocator policies or reorganizing public headers.

Memory owns byte storage, alignment, placement, execution context identification,
unique/shared/borrowed handles, pinned staging, reusable workspaces, deferred
reclamation, and optional telemetry. Numerical consumers own tensor shape,
strides, expressions, kernels, BLAS dispatch, and autograd. Tensor and
LinearAlgebra may each depend directly on Memory.

The core release does not require a driver-pool backend, graph pools, growing
virtual-address segments, managed memory, a new CPU allocator, or a wholesale
directory restructure. Those are independent extensions with their own gates.

## 2. Corrected starting point

| Area | Baseline evidence | Remaining work |
|---|---|---|
| Unique ownership | `data_ptr` is move-only and has `clone()` | Document the compatibility break; specify GPU clone completion; audit downstream callers |
| Borrowed views | Owner-derived slices preserve the allocation base | Complete transfer support for slices; specify foreign borrowed-memory limitations |
| Shared ownership | `retained_ptr::adopt()` and retained slices exist | Connect ownership to async operations; validate adoption metadata and deleter contracts |
| Sync transfers | `copy_sync()` forwards to `copy(..., nullptr)`; GPU implementation submits async work | Establish actual host-visible completion before return |
| Async tokens | `copy_token` stores only a context; queries/waits on the stream | Operation-specific completion, device selection, error reporting, retained endpoints, safe token discard |
| Pinned storage | Constrained buffers, deferred reclamation, and backing limits exist | Protect both transfer endpoints; bounded transfer slots and failure/backpressure policy |
| Reusable storage | `cpu_arena` and `gpu_workspace` exist | Alignment/arithmetic checks, Release rebind contract, completion-safe reuse |
| Native GPU caches | Segments, splitting/coalescing, stream tracking, budgets, and fault fixes exist | Close the reported churn crash; revalidate failure and shutdown paths |
| Metal | Heap accounting and explicit completion-token bookkeeping exist | Real command-buffer integration, race/failure tests, Apple hardware validation |
| Telemetry | Bounded trace ring and extended schema exist | Populate allocation IDs/requested sizes; cheap basic counters; accounting invariants |
| Allocator handles | Thread-local `device_handle_cache` exists | Lifecycle design and measured integration; facade still uses registry lookup |
| Driver pools/graphs | Async allocation wrapper and graph-pool skeleton exist | Build/API integration, capability checks, lifetime semantics, hardware acceptance |
| Documentation | README describes unavailable retained-copy APIs and deleted implicit copying | Compile working examples and remove unsupported guarantees |
| Hardware CI | Workflow declares self-hosted CUDA/HIP jobs | Verify runners actually execute required cases; job definitions alone are not pass evidence |

Source anchors:
[ownership](../include/common/data_ptr.h),
[shared storage](../include/common/retained_ptr.h),
[copy implementation](../include/allocator.h),
[token](../include/common/copy_token.h),
[workspace](../include/gpu/gpu_workspace.h),
[native cache](../src/gpu/cuda_caching_allocator.cpp),
[trace schema](../include/profiler/gpu_memory_snapshot.h).

The reported allocator-churn corruption is still open. Its root cause is not
established; neither allocator culpability nor benchmark-harness culpability
should be assumed. Bounded benchmark iterations are a mitigation, not closure.

## 3. Delivery rules and dependencies

Track each work package through **planned → implemented → integrated → tested →
accepted**. Record a blocked validation separately, including required hardware
or tooling. A header, unit test, or successful compile does not establish a
runtime contract. Historical test counts are not current test results.

Every package records its commit, supported configurations, exact commands,
executed/skipped cases, artifacts, known limitations, and gate decision. Use the
existing test infrastructure and extend it where contracts require coverage.

| Phase | Outcome | Dependency | Completion evidence |
|---|---|---|---|
| 0 | Accurate contract/status baseline and reproducible validation setup | None | Feature inventory, corrected examples, reproduction protocol |
| 1 | Observable, failure-safe cleanup and churn diagnosis | Phase 0 | Root-cause evidence, regression, fault/sanitizer results |
| 2 | Truthful transfer completion and context semantics | Phase 0; Phase 1 cleanup contract | Ordering, device, completion, and error tests |
| 3 | Integrated byte storage and retained async lifetime | Phase 2; Phase 1 failure handling | Ownership/adoption tests and delayed-completion tests |
| 4 | Validated arenas, staging, and Metal completion integration | Relevant Phase 2–3 contracts | Reuse, bounds, budget, and backend runtime results |
| 5 | Trustworthy accounting, identity, and bounded diagnostics | Phase 3 identity/resource contract | Invariant, lifecycle, loss, and overhead evidence |
| 6 | Representative benchmark suite and measured native optimizations | Harness starts in Phase 0; acceptance follows Phases 1–5 | Reproducible raw data, comparison report, regression budgets |
| 7 | Optional driver-pool and graph features | Phases 2–6 | Backend-specific lifetime, capture, and performance acceptance |
| 8 | Stable core API and documented support matrix | Phases 0–6 for core; Phase 7 only for advertised extensions | Consumer builds, package checks, actual CI/hardware artifacts |

The correctness phases may progress independently of hardware diagnosis when
they do not depend on the unresolved failure. The crash remains a release blocker
for the affected configuration; it does not prohibit useful CPU work, contract
design, or deterministic runtime-shim tests. Performance harness construction
starts early, but performance and readiness claims wait for correctness gates.

## 4. Phase 0 — Establish the contract and validation baseline

**Purpose:** prevent further work from targeting features that already exist or
assuming guarantees that have not been implemented.

Work packages:

- **0.1 Contract inventory.** For every public allocate/free/copy/clone/adopt/view/
  workspace operation, specify bytes versus elements, alignment, supported
  backend, ownership, ordering, completion, thread safety, and failure behavior.
  Distinguish synchronous host copies, synchronous cross-device copies, borrowed
  async submission, and retained async submission.
- **0.2 Documentation repair.** Update README examples to the current move-only
  owner; explain migration from implicit copies to `clone()`. Remove the claim
  that today's token retains endpoints and examples using nonexistent overloads.
  Fix stale source paths and the missing design-document reference in `CLAUDE.md`.
  Keep proposed APIs visibly separate from supported APIs.
- **0.3 Validation manifest.** Record compiler, platform, CPU allocator, GPU
  backend, default-stream mode, build type, sanitizer, driver/runtime versions,
  hardware, and effective allocator configuration for every run. Confirm which
  self-hosted jobs have usable hardware and publish skipped cases explicitly.
- **0.4 Reproduction setup.** Preserve the historical churn command and bounded
  baseline. Define the uncapped reproduction, process repetition count, stopping
  criteria, seed where applicable, and crash-dump collection before modifying the
  suspected implementation. Keep diagnosis runs distinct from performance runs.
- **0.5 Benchmark audit.** List measured operations and timing boundaries for
  each existing case. Flag multi-stream versus multi-thread coverage, vector
  allocation overhead, unsupported fragmentation conclusions, inconsistent ratios,
  and the Debug-to-Release estimates that lack direct measurements.

Primary files: `README.md`, `CLAUDE.md`, `Docs/`, `Testing/Cxx/`, and
`.github/workflows/ci.yml`.

Acceptance: every supported example compiles in its stated configuration; a
current feature/status table exists; the reproduction protocol and validation
manifest can be followed by another developer. Do not relabel untested historical
features as accepted during this inventory.

## 5. Phase 1 — Close failure and cleanup hazards

**Purpose:** prove that failed allocation, submission, event recording, cleanup,
and shutdown cannot silently turn unsafe storage into reusable storage.

Work packages:

- **1.1 Diagnose allocator churn.** Reproduce the documented failure with a
  working debugger or sanitizer. Capture stack traces, allocator/segment state,
  build/runtime configuration, and benchmark transition. Compare a harness-free
  reproduction with the original benchmark without assuming either is at fault.
  If a toolchain or harness fault is identified, preserve evidence and correct
  the benchmark/environment before establishing a new baseline.
- **1.2 Validate allocation transactions.** Audit in-flight budget reservation,
  driver allocation, retry, metadata insertion, and rollback. Inject failures at
  each boundary, including device activation and event allocation/recording.
  Preserve prior fixes for NaN, overflow, budget shrink during retry, and partial
  event failure. Audit cleanup paths for stale references after map erasure.
- **1.3 Define cleanup observability.** Keep destructors nonthrowing. Design an
  explicit cleanup operation that reports failure and a minimal nonthrowing
  diagnostic hook/counter for destruction failures. The hook must not depend on
  allocating through Memory, recursive logging, or taking the same allocator
  locks. Specify handler lifetime and concurrent replacement policy.
- **1.4 Preserve unsafe allocations.** Define quarantine states and counters.
  If completion cannot be established, retain the allocation and sufficient
  metadata to prevent reuse. Do not equate a swallowed exception with successful
  cleanup. Specify what information remains available for diagnosis.
- **1.5 Define runtime lifetime.** Document stream, token, pool, allocator, and
  device shutdown ordering. Specify behavior for pending work at explicit runtime
  shutdown and avoid relying on uncontrolled cross-library static destruction.

Primary files: `src/gpu/cuda_caching_allocator.cpp`,
`src/helper/pinned_memory_allocator.cpp`, ownership destructors,
`include/gpu/device_guard.h`, `Testing/CudaCachingAllocator/`, and
`Testing/PinnedRuntime/`.

Validation: extend existing shims for first/partial event failures, allocation and
retry failures, rollback, cleanup diagnostics, and quarantine. Run supported CPU
ASan/UBSan configurations; use real CUDA/HIP runs for actual event ordering.

Acceptance: the churn issue has an evidence-backed disposition and regression;
unsafe blocks are never recycled in injected-failure cases; budget reservations
roll back exactly once; cleanup failures are observable. Predeclare repeated
stress duration and counts, and report results. A sequence of clean runs without
root-cause evidence does not by itself close the historical crash.

## 6. Phase 2 — Make transfer completion and contexts precise

**Purpose:** make API names and return values correspond to actual completion.

Work packages:

- **2.1 Separate submission from completion.** Extract internal transfer routing
  from the allocator facade without changing unrelated allocation policy.
  Preserve facade wrappers during migration. Validate backend labels, device
  indices, extents, byte arithmetic, null endpoints, and unsupported combinations
  before submission; define zero-byte and overlapping-copy behavior.
- **2.2 Repair `copy_sync()`.** Explicitly establish transfer completion before
  returning. Select the correct device and transfer stream; do not rely on a
  null stream or pageable-memory staging to make submission synchronous. Prefer
  operation/stream completion over device-wide synchronization where possible.
- **2.3 Introduce operation-specific completion.** Record a backend completion
  marker after successful submission. Have `ready()`/`wait()` observe that marker,
  so unrelated later work cannot change whether this operation has completed.
  Specify completion-marker resource lifetime, token copy/move semantics, and
  error representation. Separate pending, complete, and failed states.
- **2.4 Correct device and default-stream handling.** Activate and restore the
  operation's device for backend calls. Define legacy and per-thread default
  stream identity, including cross-thread token use. Either support a mode with
  the required identity tracking or reject it explicitly. A bare null pointer
  must not ambiguously identify different per-thread streams in a shared cache.
- **2.5 Specify ordering separately from reclamation.** `record_stream()` delays
  reuse; it does not make a consumer wait for a producer. Define explicit
  dependencies for source production, destination reuse, cross-stream copies,
  and peer copies. The caller must retain valid endpoints through submission;
  concurrent host destruction is not made safe by stream registration alone.
- **2.6 Resolve clone semantics.** Recommended target: `clone()` returns a fully
  copied buffer; an explicitly named async clone returns its owned result plus
  completion state. Record the compatibility/performance consequences before
  changing behavior. Apply the same documented completion model to constructors
  that copy from pointers or views.
- **2.7 Handle submission failure.** Distinguish failure before submission from
  failure to record completion after work may have been submitted. The latter
  must fall back to a proven-safe wait or quarantine; it cannot release storage
  under an assumed completed operation.

Primary files: `include/allocator.h`, `include/common/copy_token.h`,
`include/common/execution_context.h`, `include/common/data_ptr.h`,
`include/gpu/gpu_runtime.h`, and transfer tests.

Validation: delay GPU work deliberately; verify sync calls return only after
completion, and a completed token remains complete while later stream work is
pending. Exercise nonblocking streams, both supported default-stream modes,
current-device mismatch, cross-thread waits, multiple devices where available,
and injected query/wait/event errors. Shims verify call ordering and failure
branches; actual hardware verifies completion and visibility.

Acceptance: synchronous copies satisfy their completion promise; async tokens
identify one operation, report errors, and address the correct device. Unsupported
device/stream combinations fail before submission. No retained-endpoint guarantee
is advertised until Phase 3 is accepted.

## 7. Phase 3 — Connect byte storage, ownership, and asynchronous lifetime

**Purpose:** keep storage alive for outstanding work and give all ownership and
telemetry paths a coherent allocation identity.

Work packages:

- **3.1 Define a minimal byte resource contract.** Specify allocation requests
  using bytes, alignment, device, memory kind, and execution context. Handles
  preserve resource/deleter provenance, base, requested extent, known capacity,
  alignment, and identity. Distinguish known capacity from caller-asserted foreign
  capacity. Use typed adapters over this contract and preserve static CPU
  dispatch. Unsupported kinds/capabilities fail explicitly.
- **3.2 Centralize identity.** Assign an identity per allocation lifetime; reuse
  of an address creates a new identity. Owners, retained slices, borrowed views
  where provenance is known, operations, and trace records refer to that same
  allocation. Keep offsets/extents separate from bases. Specify identity scope
  across shared libraries and avoid forcing shared ownership on every CPU owner.
- **3.3 Define allocation and adoption factories.** Provide explicit supported
  paths to unique and retained storage, including transfer of an existing unique
  owner into retained storage if supported. Specify who owns the pointer when
  control-block allocation fails, nonnull/zero-capacity adoption, empty deleters,
  byte overflow, and custom deleter state. Never create independent owning
  control blocks for the same allocation implicitly.
- **3.4 Implement retained transfer overloads.** Retain source and destination
  before submission; register allocator-managed bases, not slice addresses.
  Support retained pageable host storage, tracked GPU storage, pinned endpoints,
  and explicitly adopted foreign storage according to their capabilities.
  Validate extents before accessing either endpoint.
- **3.5 Make token discard safe.** An outstanding retained operation must keep
  endpoint references even if all user tokens disappear. Transfer ownership to
  a bounded pending-operation service until completion or explicitly choose and
  document a blocking token-destruction policy. Recommended target: a bounded
  service with explicit poll/drain and try/fail or wait backpressure; no silent
  unbounded queue. Completion failures retain/quarantine unsafe state. Define
  drain and shutdown behavior using Phase 1's runtime lifetime contract.
- **3.6 Preserve a borrowed path.** Raw-pointer and borrowed-view operations
  remain useful, but their caller-managed lifetime and provenance requirements
  must be visible. Resolve known sliced bases before submission. Reject unsupported
  foreign/interior GPU pointers early rather than discovering them in allocator
  bookkeeping after work has started.
- **3.7 Specify typed storage.** Recommended initial scope: uninitialized storage
  for supported trivial transfer types, with explicit alignment validation.
  Do not silently promise general C++ object construction/destruction. Check
  `alignof(T)` as well as requested alignment. General object containers require
  a separately specified facility.
- **3.8 Measure shared-owner representation.** Retain `std::function` until
  measurements justify replacement. If changed, include owned deleter-state
  destruction and exception safety in the function-pointer/context design.

Primary files: ownership headers, `storage_identity.h`, allocation/transfer
interfaces, `TestRetainedPtr.cpp`, `TestStorageIdentity.cpp`, `TestDataView.cpp`,
and `TestCopyToken.cpp`.

Validation: destroy all external owners and user tokens while a deliberately
delayed copy is pending; both endpoints must remain valid until completion.
Test nested slices, address reuse, cross-thread reference release, external
deleters called exactly once, adoption failure, dropped tokens, service capacity,
and shutdown with pending/failed operations.

Acceptance: retained async operations pass end-to-end lifetime tests; borrowed
operations have explicit limits; all allocation identity consumers agree; no
double-free, premature reuse, unbounded pending queue, or leaked ownership due
to a normal submission failure. Fault quarantine is separately accounted.

## 8. Phase 4 — Validate reusable storage and backend completion integration

**Purpose:** complete the existing arenas, workspaces, staging, and Metal paths.

Work packages:

- **4.1 CPU arena contract.** Validate power-of-two alignment and supported
  maxima; account for backing-address alignment when aligning suballocations.
  Check growth arithmetic and capacity bounds. Specify reset invalidation,
  exhaustion, zero-capacity lazy initialization, and thread confinement. Keep
  optional NUMA/page placement explicit and avoid migrating pages shared with
  unrelated small allocations.
- **4.2 GPU workspace contract.** Define whether the API permits multiple live
  slices, and align documentation with that choice. Validate typed alignment and
  the compiled backend. Enforce required rebind preconditions in Release, or
  explicitly designate unchecked APIs. Track the original allocation resource
  independently of the current use context. Distinguish cursor reset from safe
  cross-stream reuse; require a completion dependency or proven quiescence.
- **4.3 Pinned transfer integration.** Protect both the host staging block and
  GPU endpoint through Phase 3's operation contract. Maintain separate live,
  pending, reusable, and quarantined pinned-byte accounting. Define total backing
  limits, alignment overhead, cache trimming, and budget-shrink behavior.
- **4.4 Bounded staging pipeline.** Start with a configurable small number of
  reusable slots. Define try/fail versus wait behavior when all slots are busy;
  never overwrite an in-flight slot. Measure pageable versus pinned transfers
  and overlap separately from allocation throughput.
- **4.5 Metal completion integration.** Connect allocation use to actual command
  buffer completion, including registration/completion races and failed command
  buffers. Reclaim only after every registered consumer finishes. Preserve heap
  capacity accounting and test existing-heap reuse, standalone fallback, trimming,
  and budget admission. Keep synchronous shared-buffer copies explicit; do not
  imply CPU/GPU visibility without the required ordering.

Primary files: `cpu_arena.h`, `gpu_workspace.h`, `pinned_buffer.h`, pinned
allocator implementation, Metal allocator/binding implementation, and their tests.

Acceptance: repeated fixed-shape operations reuse backing after warmup; no reset
or rebind enables premature reuse; configured backing/slot limits hold under
delayed completion and errors. Run CPU alignment/bounds cases without hardware;
CUDA/HIP and Metal acceptance require real backend execution. Do not invent an
Apple test result from the existence of completion bookkeeping.

## 9. Phase 5 — Complete accounting and allocation diagnostics

**Purpose:** make statistics reliable enough to diagnose failures and compare
allocator policies without materially distorting the allocation path.

Work packages:

- **5.1 Define accounting layers.** Track requested live bytes, live block
  capacity, pending capacity, reusable capacity, quarantine, in-flight driver
  reservations, and backing separately. For native segment pools verify:

  `segment backing = live capacity + pending capacity + reusable capacity + quarantined capacity`

  Metal unused heap capacity, pinned padding, and virtual reservation versus
  mapped backing require separate accounting layers. CPU RSS is process-wide;
  shared host/GPU backing must not be counted twice.
- **5.2 Make basic queries O(1).** Maintain required basic counters at state
  transitions; do not route allocated/reserved queries through free-list scans.
  Keep detailed fragmentation snapshots explicitly more expensive. Define whether
  multi-field reads are coherent snapshots or independently sampled counters.
- **5.3 Populate trace fields.** Pass real allocation IDs and requested sizes to
  history recording instead of the zero-filling legacy overload. Preserve the
  original allocation identity/address through free request, delayed completion,
  coalescing, quarantine, and address reuse.
- **5.4 Bound collection cost.** Keep off/counters/trace/diagnostic modes explicit.
  Preserve the preallocated ring, publish loss and coverage, and export outside
  allocator locks. Preallocate emergency OOM evidence and ensure optional
  telemetry cannot make a successful allocation/free fail. Distinguish policy
  budget rejection from driver OOM and metadata failure.
- **5.5 Define fragmentation metrics.** Report internal waste as live capacity
  minus requested live bytes, with a documented denominator and zero case.
  Report inactive split bytes and largest reusable blocks per relevant stream/
  pool. Do not label all cached or pending bytes as fragmentation. Define cache
  hit/miss denominators, zero-size treatment, retries, and failed requests.

Validation: replay allocation/free histories, test reused addresses and coalesced
blocks, reconcile counters after fault injection, overflow the trace ring, and
exercise profiler start/stop with existing allocations. Validate accounting after
quarantine and budget updates, not just normal allocation/free loops.

Acceptance: basic reads do not scan blocks; replay pairs allocation lifecycles;
accounting invariants hold; loss/unavailable data are explicit; telemetry failures
cannot corrupt allocator state. Retain the earlier proposed overhead budgets of
5% for counters and 10% for sampled trace only as provisional targets, to be
accepted against named workloads and a measured baseline.

## 10. Phase 6 — Measure representative workloads and optimize native paths

**Purpose:** establish trustworthy evidence before introducing more allocator
complexity. Build the harness from Phase 0 onward; accept results after relevant
correctness and accounting gates pass.

Required workloads:

| Workload | Required controls and observations |
|---|---|
| Cold/warm allocation and free | Separate timings; same sizes/alignment; explicit warmup and cache state |
| Host-thread contention | 1/2/4/8/16/32 threads sharing one device allocator; barriers; cross-thread frees |
| Mixed lifetimes/fragmentation | Seeded replay; long-lived blocks plus temporaries; selective frees; changing sizes |
| Multi-stream | Actual consumers and delayed completion; same-stream and cross-stream uses, not just round-robin API calls |
| Pressure/OOM | Policy cap and driver failure; retries, trims, pending and quarantine behavior |
| Workspace reuse | Fixed/changing scratch demands; steady-state driver calls; end-to-end operation time |
| Transfers | Pageable/pinned H2D and D2H; bounded pipeline; useful overlap and total backing |
| Telemetry | Identical workload with off/counters/sampled trace/diagnostic modes |

Collect allocation/free p50/p95/p99 with documented sampling overhead; repeated
benchmark means are not per-operation percentiles. Report throughput, lock wait,
driver time/call count, cache hits/misses, peak backing, requested/capacity waste,
pending/reuse delay, and synchronization. Label allocation-byte rate separately
from actual transfer bandwidth.

Preallocate harness bookkeeping, keep setup/teardown timing consistent, preserve
actual sizes for each deallocation, and record seeds and raw data. Explain clock
basis and operation counts so latency, throughput, ratios, and units reconcile.
Keep bounded historical runs separate from new unconstrained stress results.

Comparisons use the same abstraction level: raw Memory cache versus raw CUDA
cache/pool APIs, and owning Memory objects versus owning tensor objects as a
separate comparison. Include native cache, direct allocation, and the optional
driver pool when implemented; add a version-pinned PyTorch CUDA allocator
comparison where practical. Existing MPS owning-object comparisons do not
substitute for a CUDA allocator comparison. CPU comparisons use matching
alignment/initialization and actual available backends.

Optimize one measured source of cost at a time: stable resource-handle lookup,
incremental counters, metadata reuse, bounded event polling, and then finer lock
granularity if justified. Before integrating thread-local handle caching, prove
allocator lifetime and invalidation across all participating threads; clearing
one thread's cache is insufficient for a destroyed shared resource. Bound polling
without starving pending frees, and specify how budget pressure forces progress.

Acceptance: raw artifacts are reproducible; benchmark conclusions match what was
measured; every optimization preserves correctness and memory budgets and improves
the named workload beyond measured noise. A feature with no repeatable benefit
may remain unintegrated. Do not extrapolate Debug results to Release or infer
fragmentation resilience from allocation latency alone.

## 11. Phase 7 — Optional driver pools and graph-aware allocation

**Purpose:** add alternatives only after the common storage/transfer contracts
can express them. This phase is not required for the native-cache core release.

Work packages:

- **7.1 Integrate backend selection.** Wire an actual build option and explicit
  initialization-time backend selection. Query runtime/device support, reject
  unavailable modes, and define resource provenance so an allocation is always
  freed through its originating backend. Do not silently switch live resources.
- **7.2 Define pool ownership and policy.** Specify owned versus default/shared
  pools, release thresholds, trim behavior, budget semantics, and statistics.
  Keep native-cache counters distinct from unsupported driver-pool measurements;
  do not fabricate parity.
- **7.3 Validate ordering and access.** Enforce allocation-before-use and
  last-use-before-free across streams. Treat peer access permission separately
  from execution dependencies. Resolve async free errors through Phase 1's
  diagnostics and lifetime policy. Validate HIP support independently.
- **7.4 Complete graph semantics.** Define capture, replay, allocation lifetime,
  pool ownership, address stability where required, and graph destruction. A
  capture-scope helper or allocation registry is not sufficient proof. Reject
  unsupported capture modes before mutating state.
- **7.5 Compare against the native cache.** Run Phase 6 workloads and publish
  both performance and backing-memory tradeoffs. Keep growing virtual-address
  segments deferred unless traces demonstrate an unmet need.

Primary files: `cuda_malloc_async_allocator.h`, `gpu_graph_pool.h`, backend
dispatch, build configuration, capability reporting, and backend runtime tests.

Acceptance: ordering, token discard, peer/multi-device use, errors, shutdown, and
capture/replay pass on each advertised backend. Benchmark evidence supports the
use cases for enabling it. Unsupported configurations fail explicitly; optional
features remain experimental until their own gates pass.

Backend references:
[NVIDIA stream-ordered allocator contract](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html),
[CUDA synchronization behavior](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html),
[CUDA execution and default streams](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html).

## 12. Phase 8 — Stabilize the public API and release evidence

**Purpose:** make the supported behavior usable across consuming projects and
packages, with support claims that match executed validation.

Work packages:

- Publish the final byte allocation, ownership, transfer, workspace, error, and
  capability contracts. Document API/ABI changes and the move-only migration.
  Preserve compatibility wrappers where they do not perpetuate unsafe promises.
- Audit real downstream uses of copy/clone, borrowed slices, GPU registration,
  workspace reuse, and externally adopted buffers. Numerical operations should
  register their uses and express dependencies internally. Report projects not
  inspected; do not infer integration success from Memory-only tests.
- Add install-and-consume checks for namespaced headers, CMake exported targets,
  Bazel, static/shared builds, and one compatible runtime across plugin/library
  boundaries. Verify optional Profiler/Logging integration does not duplicate
  device registries or introduce an unconditional expensive CPU path.
- Publish a support matrix distinguishing compile-tested, shim-tested, and
  hardware-tested configurations. Initially audit the existing Linux CPU,
  self-hosted CUDA/HIP, macOS CPU, Windows CPU, and Bazel jobs. Add a Metal runtime
  job when hardware is available. Do not assign Windows CUDA or Metal a release
  tier solely because they appear in an aspirational matrix.
- Compile documented examples as part of relevant checks. Keep the canonical
  status in this plan until replaced by a release status document; link historical
  reviews rather than maintaining competing current completion tables.

Core release gate: Phases 0–6 are accepted for each advertised core configuration;
the documented churn issue has an evidence-backed disposition; completion and
lifetime tests pass; no unsupported ownership promise remains in documentation;
consumer/package checks pass; actual hardware artifacts support backend claims.
Phase 7 features may ship disabled/experimental or be excluded without blocking
the core. Enabling an extension as supported requires its independent acceptance.

## 13. Initial implementation queue and review record

The first implementation changes should remain small enough to review:

1. Correct README examples/status and add the validation manifest/reproduction
   protocol (Phase 0).
2. Capture the churn failure under working tooling; submit a focused regression
   and fix only after identifying the cause (Phase 1).
3. Implement and test truthful `copy_sync()` completion and context validation
   (Phase 2); this work can proceed while churn diagnosis awaits hardware.
4. Specify operation/token/error behavior, then implement event-specific
   completion and its failure handling (Phases 1–2).
5. Introduce the minimum storage provenance/identity contract and retained
   transfer integration, including discarded-token ownership (Phase 3).
6. Complete existing reusable-storage and telemetry gates before accepting
   native-cache optimization or driver-pool performance claims (Phases 4–7).

For each package, append or link a review record containing: package ID, intended
contract, implementation commit, evidence artifacts, tested configurations,
skipped/blocked cases, migration impact, and accepted/open gate decisions. No
calendar estimate is assigned until hardware availability, downstream scope, and
the churn diagnosis are known.
