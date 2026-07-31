# ADR-0002: Use a thin CUDA-to-ACL compatibility layer

- Status: Accepted
- Date: 2026-07-30

## Context

Several CUDA and CANN APIs have equivalent intent but different signatures.
Blind identifier replacement currently produces code that is either rejected
by CCEC or, in the occupancy case, compiles as a wrong comma expression.
Global token rewriting such as `double` to `float` also destroys overload,
specialization, and SFINAE identity.

The goal is to strengthen Ascify without replacing its existing translation
framework or retaining an operator-specific post-processing script.

## Decision

Keep simple same-signature renames in the existing maps. Route signature or
semantic adaptations through one small installed compatibility header:

- preserve CUDA call signatures at translated call sites;
- adapt device attributes, last-error queries, occupancy estimates, and
  function attributes inside the compatibility layer;
- use CANN 9.1 hardware SIMT shuffle and math APIs;
- reject or trap unsupported masked-warp semantics rather than silently
  changing them;
- preserve language type identity and alignment instead of global
  `double -> float` or `__align__` deletion.

AST rules are limited to cases where context is required, such as device FP64
lowering and CUDA shuffle argument semantics. Operator thresholds and tiling
choices remain outside this compatibility layer.

## Consequences

- Softmax and RMSNorm share the same API adaptations.
- Generated code has one explicit, testable dependency instead of a regex
  fixup script.
- Unsupported CUDA semantics remain visible at compile or runtime.
- New compatibility rules require rewrite golden tests and a CCEC compile
  gate before being treated as supported.
