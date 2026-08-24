# ADR 0014: Lower only parameter-root 32-bit global atomics

- Status: Accepted; target syntax is a release gate
- Date: 2026-08-24
- Scope: CUDA device atomics rewritten to the legacy AIV global-memory atomic surface

## Context

CUDA uses one source spelling such as `atomicAdd` for addresses in different
memory spaces. The admitted CANN SIMT interface exposes separate global-memory
atomic intrinsics, and selecting that interface for a shared, local or otherwise
unproved pointer would change semantics. A textual API map cannot recover the
address space reliably.

The current target evidence covers signed and unsigned 32-bit integer global
atomics. It does not establish float, 64-bit, arbitrary pointer-alias, shared
memory or device-global-variable lowering.

## Decision

Rewrite a CUDA atomic call only when the Clang AST proves all of these facts:

1. The direct callee is the exact CUDA system-header declaration for
   `atomicAdd`, `atomicSub`, `atomicExch`, `atomicMax`, `atomicMin`, `atomicInc`,
   `atomicDec`, `atomicCAS`, `atomicAnd`, `atomicOr` or `atomicXor`.
2. The call is inside the current `__global__` function and is spelled directly
   in the main file, not produced by a macro or routed through a helper.
3. The first argument is rooted directly at a pointer parameter of that same
   kernel. Only parentheses, implicit casts, address-of array subscripting, and
   pointer addition by an integral expression may appear between the parameter
   and the final address. A local pointer alias is not a proof.
4. The canonical parameter, pointer, scalar arguments and return type are all
   the same admitted 32-bit signed or unsigned integer type. `atomicInc` and
   `atomicDec` require unsigned 32-bit values.

An admitted call is rewritten to a typed `ascify::atomic_*_global` wrapper. On
the verified legacy SIMT header family, that wrapper calls the corresponding
target global-memory intrinsic. On a header family without an admitted global
atomic surface, template instantiation fails with an explicit diagnostic.

Every call that misses the proof is left unchanged. This is deliberate: a
later target compiler error is safer than fabricating a global-memory cast for
a shared, local, aliased or unsupported address.

## Alternatives rejected

- Textually map every CUDA atomic to one target intrinsic: the source spelling
  does not prove its memory space.
- Cast every pointer to global memory: that can silently redirect shared or
  local operations and is therefore unsound.
- Infer a local alias from its initializer: proving that the alias remains
  unchanged requires a broader data-flow analysis and is outside this rule.
- Admit float or 64-bit variants because CUDA declares them: declaration
  availability is not target semantic evidence.

## Consequences

The rule covers common kernel-parameter counters and array updates while
remaining intentionally incomplete. Direct device-global variables, shared
variables, local variables, helper parameters, mutable aliases, macro calls,
float and 64-bit variants do not receive the wrapper. Target compilation,
device contention/order correctness and performance remain separate claims.

## Verification

- `tests/rewrite/global_atomic_input.cu` and `global_atomic.expected` require
  direct parameter-root rewrites for all eleven admitted operation spellings
  and require shared, local, alias, float, device-global, helper and macro cases
  to remain unchanged.
- `tests/rewrite/global_atomic_compat_test.cpp` executes the typed wrapper
  contract against controllable legacy host stubs.
- `tests/rewrite/global_atomic_fail_closed_test.cpp` requires an explicit
  compile failure when the admitted target atomic header is unavailable.
- `tests/rewrite/check_simt_compat.sh` binds those fixtures into the release
  gate. A focused CCEC probe from the exact release source is additionally
  required because a whole CUDA Samples translation can stop at an earlier,
  unrelated helper API before reaching the atomic expression.
