# ADR-0017: Wrap row-wise kernel analysis in static proofs

## Status

Accepted for M0

## Date

2026-08-24

## Context

The dav-c310 row-wise recipe already proves the semantics of three CUDA
kernel families: Softmax, RMSNorm, and LayerNorm.  Before this change, the
analyzers communicated success to direct-wrapper recognition through three
untyped sets of canonical `FunctionDecl` pointers.  Softmax choice evidence
travelled in a fourth, parallel map.  This was sufficient for the three
existing recipes, but it made two important facts implicit:

- which semantic analyzer had established the kernel result; and
- which definition and source provenance the result belonged to.

Growing the set of supported CUDA kernels directly on top of parallel pointer
sets would make it easier to combine evidence from the wrong family or lose
the link between analyzer evidence and wrapper emission.

M0 must remain behavior-preserving.  It must not add a new Hybrid recipe,
change generated wrapper text, extend the runtime ABI/DSO surface, or import
the reporting-only paper Stage IR into the product path.

## Decision

Introduce an internal `StaticKernelProof` analyzer-to-emitter contract.  A
proof records:

- the canonical function declaration;
- the analyzed definition declaration and its source range;
- whether that definition came from the main file, a user header, or a system
  header; and
- one typed semantic family: Softmax, RMSNorm, or LayerNorm.

`StaticKernelProofRegistry` owns those proofs during
`DavC310TargetRecipe::finalize`.  Existing analyzers remain the source of
truth: M0 creates a proof only after the same Softmax, RMSNorm, or LayerNorm
analysis has succeeded.  Existing Softmax enum-choice bindings are attached
to the Softmax proof record instead of being passed independently to the
wrapper prover.

Direct-wrapper recognition now consumes the proof registry.  It still
requires exactly one proven family for the launched canonical kernel, and all
existing wrapper, launch-domain, status, template, and effect checks remain
unchanged.

The proof type stays under `src/`; it is not a public lowering IR and is not
part of the target recipe headers or runtime ABI.

## Alternatives Considered

### Keep three pointer sets and add more parallel maps

This has the smallest immediate diff, but family identity and source
provenance remain implicit.  Each new analyzer would increase the chance of
misaligned parallel state.

### Introduce the paper Hybrid Stage IR into product conversion

That IR is currently a reporting/checker artifact without a production
extractor, consumer, or lowering contract.  Making it an emitter dependency
would expand M0 into an ABI and lowering change without device evidence.

### Replace all analyzer internals with a new generic analysis framework

That is a possible later step, but it is too broad for a behavior-preserving
M0.  The current analyzers have substantial family-specific semantic gates
that should be retained while the typed boundary is established.

## Consequences

- Wrapper emission has one typed source of kernel-family truth.
- Softmax semantic-choice evidence is owned by the corresponding proof.
- Source identity and provenance are available for future fail-closed
  generalization without changing the current generated text.
- Analyzer internals are still family-specific.  M0 does not yet define a
  generic vector-stage decomposition, discover a new operator family, or
  lower an arbitrary proof to a new Hybrid recipe.
- The release gate includes a standalone proof-invariant unit test and static
  ownership checks.  Existing row-wise mutation/golden gates remain the
  behavior-preservation oracle.
