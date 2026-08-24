# ADR-0008: Compose row-wise SIMD and SIMT stages in one selected kernel

- Status: Accepted
- Date: 2026-08-22

> **Scope extension:**
> [ADR-0009](0009-register-rowwise-hybrid-recipes-and-add-layernorm.md)
> generalizes the mixed-stage registration contract and adds LayerNorm. The
> same-kernel execution and evidence-invalidation rules below remain in force.

## Context

[ADR-0007](0007-explicit-ast-gated-rowwise-simd-dispatch.md) introduced the
explicit `dav-3510-rowwise-simd-v1` recipe, its AST proof, the pre-launch
selector, and the versioned target-support ABI. Its first implementation used
the source-level spelling `RowwiseSimdFacadeV1::Try*Simd`, which made the
selected and fallback paths look like a whole-operator SIMD-or-SIMT choice.

That model does not express the intended 950PR programming capability. Within
a supported Softmax or RMSNorm call, regular vector math is a good fit for
SIMD, while contiguous element indexing and output staging can remain SIMT.
The selected implementation should therefore compose both execution
styles inside one operator kernel instead of treating them as mutually
exclusive whole-operator implementations.

The v1 target support has not yet been released or committed as a stable
external contract. Its public C launch signatures, selector predicates, DSO
names, and SONAMEs do not need to change to express this internal stage
composition.

## Decision

### Amend the pre-release v1 source facade without changing the C ABI

The canonical generated-code entry is now:

- `RowwiseHybridFacadeV1::TrySoftmaxHybrid`; and
- `RowwiseHybridFacadeV1::TryRmsNormHybrid`.

It returns `HybridTryResult`, an alias of
`SimdTryResult { bool handled; aclError status; }`. The legacy
`RowwiseSimdFacadeV1::Try*Simd` spelling remains as a source alias for already
generated review artifacts. New conversions use only the Hybrid spelling.

The public C symbols, their arguments, the three DSO names, their `.so.1`
SONAMEs, and `ASCIFY_DAV_C310_ROWWISE_SIMD_ABI_VERSION == 1` remain unchanged.
This is an internal implementation and pre-release source-facade amendment,
not an incompatible ABI revision.

### Keep the outer branch as pre-launch support gating

The generated ownership contract remains:

```text
handled == false  -> selector miss; execute the retained whole-SIMT launch
handled == true   -> the selected mixed runtime owns the call; return status
```

This branch does not choose between an all-SIMD operator and an all-SIMT
operator. It decides only whether the call satisfies the versioned support
contract before any target kernel is launched. A selected launch error remains
final and must not fall through to SIMT, because a failed launch may already
have observable effects.

Selector misses keep the original translated SIMT implementation unchanged.
The existing shape, alignment, extent, aliasing, epsilon, and pointer checks
therefore remain the boundary between selected mixed execution and the
whole-SIMT fallback.

### Compose SIMD math and SIMT output staging inside each selected kernel

Every selected target kernel uses an `__aicore__` shell that invokes both SIMD
and SIMT vector functions in the same operator execution. Stage ownership is:

| Operator | SIMD-owned stages | SIMT-owned stages |
|---|---|---|
| Softmax | row maximum, exponentiation, sum reduction, normalization division | contiguous normalized-value staging from UB to the output-local tile, including element indexing and tail copy |
| RMSNorm plain | square accumulation, sum reduction, square root, division, normalization | contiguous normalized-value staging from UB to the output-local tile, including element indexing and tail copy |
| RMSNorm affine | square accumulation, sum reduction, square root, division, normalization, per-column affine multiply | contiguous normalized-value staging from UB to the output-local tile, including element indexing and tail copy |

The ordinary GM-to-UB and UB-to-GM data-movement pipeline remains owned by the
kernel shell and is not reclassified as a general SIMT adapter capability.
The current SIMT stage is specifically a contiguous UB output-staging loop.

Version 1 does not implement or claim masked, gathered, scattered, strided,
shifted, or otherwise non-contiguous adapter lowering. Those source forms
remain outside the AST proof or on the retained SIMT fallback. Expanding this
stage boundary requires its own semantic proof, runtime implementation,
correctness cases, and performance evidence.

### Treat dynamic UB capacity as part of the selected-launch contract

The second kernel-launch argument must provide enough dynamic UB for every
buffer allocated by the selected kernel's `TPipe`. Passing `0` or `nullptr`
is invalid: the mixed kernel shares UB across its SIMD and SIMT
stages and would otherwise access UB outside the launch allocation.

The version 1 launch budgets are fixed at 163,904 bytes for Softmax recompute,
65,600 bytes for RMSNorm cached, and 147,520 bytes for RMSNorm plain row-batch.
Changing a kernel's `TPipe` allocation requires updating and revalidating its
launch budget in the same source change. The host static gate rejects a `0`
or `nullptr` dynamic-UB argument, and selected-route device correctness
tests exercise all three non-zero launch budgets.

### Invalidate predecessor implementation evidence

Changing a selected kernel from an all-SIMD device path to intra-kernel
SIMD+SIMT composition changes its binary, UB budget, synchronization, and
performance characteristics. Build, correctness, and performance evidence
collected for predecessor runtime sources or generated facade calls cannot be
attributed to this implementation.

Promotion requires a fresh evidence chain built from one frozen source state:

1. build the current frontend and regenerate the Softmax, RMSNorm, and
   dependency outputs with the Hybrid facade;
2. build all three current dav-3510 target-support DSOs and the generated main
   CCE;
3. rerun correctness for selector hits and misses, both RMSNorm routes, and
   affine RMSNorm; and
4. rerun performance against the declared A800 corpus using the current
   binaries and sampling protocol.

Historical measurements may remain as predecessor context, but they must be
labelled historical and must not be presented as results for the current
source or binary hashes.

## Alternatives considered

### Keep a whole-operator SIMD-or-SIMT branch

Rejected because a selected call would not demonstrate intra-operator mixed
programming and would leave the intended element-wise staging work outside the
selected kernel.

### Lower arbitrary CUDA primitives independently to SIMD or SIMT

Not selected for version 1. General per-primitive partitioning needs a broader
data-flow, memory-effect, synchronization, and adapter proof than the current
closed Softmax/RMSNorm recipes provide.

### Introduce a v2 C ABI immediately

Rejected because no public launch signature or selector contract changes, the
v1 implementation is still pre-release, and the legacy source facade can be
retained without changing the deployed ABI.

### Describe the selected runtime as supporting general SIMT adapters

Rejected because the current SIMT stage only performs contiguous output
staging. Mask, gather, scatter, and non-contiguous claims would exceed the
implemented and tested boundary.

## Consequences

- New generated CCEs make the mixed intent explicit through the Hybrid facade.
- A selected Softmax or RMSNorm call executes both SIMD and SIMT stages inside
  one target kernel; a selector miss still executes the retained whole-SIMT
  launch.
- The stable v1 C symbols and deployment layout remain usable by existing
  generated review artifacts through the legacy source alias.
- The additional UB staging buffer, synchronization, and SIMD/SIMT transition
  can change latency, so performance must be measured rather than inferred.
- Dynamic UB capacity is part of the selected-kernel launch contract; kernel
  buffer changes and their launch budgets must be reviewed and validated
  together.
- Documentation and reports must distinguish source recognition, hybrid-route
  coverage, whole-SIMT fallback coverage, correctness, and measured
  performance.
- All other ADR-0007 decisions about explicit activation, AST proof,
  fail-closed handling, selectors, deployment, and launch ownership remain in
  force.
