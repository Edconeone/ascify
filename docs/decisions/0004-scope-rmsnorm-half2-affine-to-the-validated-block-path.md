# ADR-0004: Scope RMSNorm half2 affine stores to the validated block path

- Status: Accepted
- Date: 2026-07-30

## Context

RMSNorm affine must preserve two FP16 rounding points: normalize in FP32 and
round to FP16, then multiply by an FP16 weight and round the result to FP16.
Packing the final independent multiplies with `__hmulx2` materially improves
the 950PR block-per-row kernel.

An unconditional experiment also applied that form to the warp kernels. The
4096/8192-column block cases passed, but three 32/64-column affine cases
produced wrong output. Type compatibility alone is therefore not sufficient
evidence for a global half2 rewrite with the current CCEC target.

## Decision

- Keep scalar FP16 affine stores in warp, scalar-tail, and streaming kernels.
- Use `__hmulx2` only in the aligned, cached, one-row-per-block affine path for
  `rows <= 256` and `4096 <= cols <= 8192`.
- Preserve the two rounding stages by constructing the normalized `half2`
  from two independently computed FP32 products before the packed FP16
  weight multiply.
- Gate the route with the existing base/stride/tail alignment checks.
- Require the dedicated block/warp boundary table in addition to the general
  RMSNorm oracle before changing this route.
- Do not add a type-only half2 affine transformation to Ascify. A future tool
  rule requires target policy, control-flow/path proof, and an operator oracle.

## Consequences

- The final `128×8192` affine result improves from about 204 GB/s to
  316–322 GB/s without relaxing the OneFlow accuracy contract.
- Small warp shapes retain the validated scalar implementation.
- The negative small-shape evidence remains part of the design rationale,
  preventing a later cleanup from widening the optimization accidentally.
- Ascify can treat this implementation as a measured target pattern, not yet
  as a generally safe rewrite.
