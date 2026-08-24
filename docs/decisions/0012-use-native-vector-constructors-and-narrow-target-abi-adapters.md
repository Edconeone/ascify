# ADR 0012: Use native vector constructors and narrow target-ABI adapters

- Status: Accepted; coherent target syntax/link regression is a release gate
- Date: 2026-08-24
- Scope: coherent CANN 9.1 beta3 target, with a public-8.5 host-stub regression gate

## Context

The frozen CUDA Samples vector corpus exposed four target-ABI gaps after
frontend conversion and quoted local-header closure: CUDA-style vector
constructors, source-default `dim3`, one-argument `cudaEventRecord`, and
`cudaStreamCreateWithFlags`.

These gaps do not have one common textual fix. Coherent CANN already owns the
CUDA-style vector types and constructors. CCEC's global `dim3` lacks CUDA's
source-default `(1,1,1)` construction. ACL 1.17 exposes a current-default-stream
query, while the admitted public-8.5 surface does not. CUDA and ACL both use
the numeric flag value one for different stream semantics.

## Decision

1. On the admitted legacy-beta3 header family, include the target-owned
   `<simt_api/vector_functions.h>` when present. Ascify does not redeclare
   vector structs or `make_*` constructors.
2. Rewrite an uninitialized `dim3` declaration to explicit `(1,1,1)` only
   when its canonical record comes from a system CUDA header and the variable
   is host-local, written directly in the main file, non-macro, non-`extern`,
   and followed by `,` or `;`. Global, device, kernel-local, shared, array,
   attributed, explicitly initialized, macro-produced, and user-defined
   declarations remain unchanged.
3. Keep explicit-stream event recording unchanged. For the no-stream form,
   use `aclrtCtxGetCurrentDefaultStream` only on the legacy beta3 / ACL 1.17+
   branch. Other admitted header families return
   `ACL_ERROR_FEATURE_UNSUPPORTED` without naming that newer API.
4. Preserve CUDA stream flag names in the Ascify namespace. Default stream
   creation uses the existing stream shim. Nonblocking creation returns
   `ACL_ERROR_FEATURE_UNSUPPORTED` before creating a stream; unknown bits
   return `ACL_ERROR_RT_PARAM_INVALID`.
5. Keep occupancy policy, CUDA Samples helper utilities, device ranking,
   Driver VMM, and compression policy outside this change.

## Alternatives rejected

- Redeclaring `float4` or `make_float4`: conflicts with target-owned ABI and
  introduces ODR risk.
- Redefining CCEC's `dim3`: broadens the change into compiler launch ABI.
- Rewriting every uninitialized `dim3`: changes extern, global, device, or
  shared declarations. The accepted rule is deliberately host-local.
- Passing CUDA flag value one into ACL stream configuration: the identical
  number denotes different semantics.
- Guessing occupancy, device ranking, VMM, or compression support: none has a
  proved CUDA-to-ACL contract in this experiment.

## Consequences

Each release source must repeat a clean LLVM build, the complete release gate,
separate legacy/public-8.5 host probes, coherent target syntax/object/link
probes, and focused whole-TU target-syntax regressions from one exact manifest.
No device correctness, Hybrid coverage, or performance result follows from
this decision.
