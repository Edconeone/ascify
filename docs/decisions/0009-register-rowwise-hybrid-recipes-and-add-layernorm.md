# ADR-0009: Register row-wise hybrid recipes and validate extensibility with LayerNorm

- Status: Accepted
- Date: 2026-08-22

## Context

[ADR-0007](0007-explicit-ast-gated-rowwise-simd-dispatch.md) and
[ADR-0008](0008-compose-rowwise-simd-and-simt-stages-in-one-kernel.md)
established a correct same-kernel SIMD+SIMT path for Softmax and RMSNorm.
Those implementations shared an execution model, but the public facade and
stage contract still named only those two operator families. Adding another
operator would have required another bespoke facade path without any
compile-time proof that its runtime actually contained both execution styles.

The base Ascify translator is not tied to those operators. The opt-in hybrid
path, however, must not be described as generally extensible until a third
semantic family can use common registration and dispatch machinery while
retaining its own fail-closed source proof and runtime selector.

## Decision

### Add an operator-independent compile-time recipe registry

`rowwise_hybrid_registry_v1.hpp` defines engine, stage, and recipe
descriptors plus `DispatchRegisteredHybrid<Recipe>`. A registered recipe must
declare:

- MTE load as its first stage and MTE store as its last stage;
- at least one SIMD stage;
- exactly one SIMD-to-SIMT handoff, with one or more SIMT stages ending in
  output adaptation; and
- no interior MTE stage or SIMD stage after the SIMT handoff.

`RowwiseHybridFacadeV1::Try<Recipe>(...)` is the common generated-code and
extension entry. Named Softmax, RMSNorm, and LayerNorm members delegate to
this entry. A new row-wise recipe can therefore register a typed `Try`
implementation and stage descriptor without changing the registry or generic
facade.

The registry validates execution composition, not operator equivalence.
Shape, layout, alias, type, numerical, and side-effect proofs remain the
responsibility of the operator-specific frontend proof and runtime selector.

### Add LayerNorm as a third end-to-end semantic family

The explicit frontend recognizes a LayerNorm forward kernel only when its AST
and data flow prove all of the following:

- FP16 row-major input/output adapters with FP32 computation;
- Welford mean, second central moment, and count are used consistently;
- variance is derived from the proved second moment and count;
- epsilon and reciprocal square root feed centered normalization;
- mean and inverse-variance outputs are distinct and written by the proved
  lane-zero leader; and
- row/column coordinates and their aliases are not mutated before use.

Names, namespaces, file paths, and source hashes are not recognition inputs.
Renaming kernels, wrappers, and local variables must preserve recognition;
changing the variance source, centering, output leader, auxiliary outputs, or
coordinate mutability must fail closed and leave the original translated SIMT
launch in place.

The generated wrapper calls
`RowwiseHybridFacadeV1::TryLayerNormHybrid`. A selector miss executes the
retained original SIMT launch. A selector hit transfers ownership to one
dav-3510 kernel in which:

- MTE moves the row between GM and UB;
- SIMD computes mean, centered variance, and reciprocal square root; and
- SIMT applies centered normalization and maps each element into the
  output-local tile, including indexed tail-safe staging.

### Extend ABI v1 additively

Add `ascify950_layernorm_reg_cached_launch_v1` in a fourth independent DSO,
`libascify950_layernorm_reg_cached_v1.so.1`. Existing v1 symbols and signatures
do not change. The LayerNorm selector requires positive non-overflowing
extents, finite positive epsilon, aligned non-aliasing auxiliary outputs,
disjoint-or-exact input/output spans, columns divisible by 8, and at most 8192
columns.

The selected kernel uses 32,832 bytes of dynamic UB. Any change to its buffer
layout, selector, launch budget, or numerical implementation invalidates its
build, correctness, and performance evidence.

### Preserve an explicit generalization boundary

This decision makes recipe registration and mixed-stage dispatch reusable; it
does not introduce automatic SIMD/SIMT partitioning for arbitrary CUDA. A new
operator family still requires:

1. a semantic AST/data-flow proof and adversarial fail-closed mutations;
2. a registered stage descriptor and typed facade entry;
3. a selector and versioned ABI implementation;
4. a dav-3510 runtime build; and
5. host, device-correctness, and performance evidence.

Unsupported layouts, masked or non-contiguous adapters, backward kernels, and
unproved algebra remain on the ordinary translated SIMT path.

## Alternatives considered

### Add LayerNorm through another bespoke facade

Rejected because it would demonstrate one more hard-coded operator without
providing a reusable composition contract.

### Treat stage descriptors as sufficient semantic proof

Rejected because a declarative stage list cannot prove source equivalence,
memory effects, aliases, or numerical outputs. The frontend proof and runtime
selector remain mandatory.

### Attempt arbitrary per-primitive partitioning now

Not selected. General partitioning requires a broader memory-effect IR,
synchronization placement, liveness and UB-capacity planning, and code
generation for non-contiguous adapters. Claiming that capability from three
row-wise families would exceed the implementation and evidence.

## Consequences

- Softmax, RMSNorm, and LayerNorm use one registered mixed-stage dispatch
  contract.
- A compile-time synthetic recipe test proves that registry extension does not
  require modifying the generic facade.
- LayerNorm supplies the third independent end-to-end operator proof rather
  than another spelling of Softmax or RMSNorm.
- The LayerNorm device gate covers both direct ABI invocation and invocation
  through an unedited Ascify-generated wrapper and its typed adapters.
- Runtime deployment now contains four independently registered DSOs.
- Documentation and reporting must distinguish reusable recipe
  infrastructure from the currently recognized operator families.
- ADR-0007 and ADR-0008 remain authoritative for activation, ownership,
  fallback, and same-kernel execution; their two-operator and three-DSO scope
  statements are superseded by this additive decision.
