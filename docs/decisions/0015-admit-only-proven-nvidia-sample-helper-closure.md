# ADR 0015: Admit only proven NVIDIA sample-helper closure

## Status

Accepted

## Date

2026-08-24

## Context

Several CUDA Samples vector translation units include `Common/helper_cuda.h`
only for error handling. Keeping the complete helper header in a recursively
closed translation unit also imports host utilities and CUDA APIs that Ascify
does not implement. Removing the include by filename or rewriting every
same-spelled macro would be unsafe: projects can own a different
`helper_cuda.h`, NVIDIA's generic `checkCudaErrors` accepts non-runtime status
domains, preprocessor uses can escape normal macro-expansion callbacks, and an
active macro can differ from official-looking text present under `#if 0`.

The narrow requirement is to close `checkCudaErrors` and
`getLastCudaError` when their CUDA-to-ACL semantics are proven, while retaining
all unsupported helper dependencies as explicit target boundaries.

## Decision

Ascify automatically considers a proof-gated NVIDIA sample-helper closure when
the main translation unit directly includes a recognized helper. There is no
separate command-line switch for this closure. The following rules decide
whether the complete transaction may commit.

### Header and active-definition provenance

A direct include is a candidate only when the resolved file:

- is named `helper_cuda.h`;
- has the `COMMON_HELPER_CUDA_H_` include guard;
- contains the NVIDIA helper `check` template and `__getLastCudaError`
  declaration; and
- contains the two admitted official macro contracts.

The recognizer uses this structural fingerprint, not a copyright marker or
year. At every expansion, the active Clang `MacroInfo` body must also have the
exact token shape:

```text
checkCudaErrors(p) -> check((p), #p, __FILE__, __LINE__)
getLastCudaError(p) -> __getLastCudaError(p, __FILE__, __LINE__)
```

Whitespace and the parameter spelling are irrelevant; token order and
identity are not. Thus, official text in an excluded conditional region cannot
authorize a different active definition.

### `checkCudaErrors` status-domain proof

The macro argument must be a direct call whose declaration comes from a system
CUDA header. Its function name must be in the small return-domain allowlist,
and Ascify's runtime map must route that name to an `ascify::` implementation
returning `aclError`.

The allowlist covers only implemented CUDA Runtime status functions. It is not
a general CUDA API allowlist. CUDA Driver `CUresult`, library-specific status,
occupancy, and arbitrary integer-returning calls fail closed even when wrapped
by NVIDIA's generic helper macro.

The replacement passes the expression as a value argument to
`sampleCheckCudaErrors`, so side effects occur exactly once. Diagnostics retain
the expression spelling, source file, source line, numeric status, and
available ACL error string.

### `getLastCudaError` consume/reset contract

The replacement calls `sampleGetLastCudaError`, which invokes
`ascify::cudaGetLastError()` exactly once. This consumes and clears a pending
Ascify lifecycle error or delegates to ACL's consuming thread-level query. It
must not use `cudaPeekAtLastError`.

### Include-removal transaction

Closure is allowed only for exactly one removable include. It must resolve to
the recognized helper and be written in the main file as the exact direct
`helper_cuda.h` literal. A second direct include, macro-expanded include,
relative-path spelling, or transitive include makes the complete transaction
unsafe.

Before removing the include, Ascify requires:

- every admitted macro use to have a proven active definition and direct
  source-token invocation;
- every `checkCudaErrors` call to have a proven Runtime status domain;
- no raw preprocessor use in the main file or any external local/user header;
- no helper macro expansion or residual helper declaration use in an external
  header, including `findCudaDevice`;
- no pre-existing reserved replacement macro from source or command line;
- completion of the main-file raw-token audit through EOF; and
- successful staging of all macro and include edits against a copy of the
  complete replacement set.

No helper edit enters the shared replacement set during the AST or raw pass.
Only after all proofs pass does Ascify publish the copied replacement set,
semantic ranges, counters, and header state in one assignment. Early target
recipe returns and staging conflicts therefore leave the helper transaction
uncommitted.

The include is replaced by `<ascify/ascify_cuda_compat.hpp>` only when that
header is not already present. Any failed proof keeps the helper include and
unsupported call. User headers and symbols with the same names are untouched.

### Explicit non-goals

This decision does not implement or emulate:

- `findCudaDevice`;
- CUDA occupancy APIs;
- CUDA Driver or VMM APIs;
- compression helpers; or
- compressible allocation.

Those surfaces remain software or hardware boundaries until a separate
semantic decision and evidence package exists.

## Alternatives considered

### Remove every include named `helper_cuda.h`

Rejected. The basename does not prove ownership or which declarations remain
in use.

### Rewrite the two macro names wherever they appear

Rejected. User-defined same-name macros are valid, and NVIDIA's generic status
wrapper spans incompatible return domains.

### Copy all NVIDIA helper implementation into Ascify

Rejected. That would silently introduce device-selection, occupancy, Driver,
and other behaviors beyond the semantic evidence.

### Treat any integer result as `aclError`

Rejected. Equal representation does not establish equal status domain or
error-string semantics.

## Consequences

Proven CUDA Samples translation units that use only the two admitted error
macros can close automatically. This closure is independent of the optional
frontend-compatibility profile and recursive quoted-local-header conversion;
those features may still be required for other dependencies in the same
translation unit. Remaining helper dependencies stay visible: for example, a
translation unit using `findCudaDevice` retains the helper and fails at that
unsupported boundary instead of becoming an assisted success.

The cost is conservative coverage and a maintained Runtime-status allowlist.
That cost is intentional: a new status family or helper macro requires a new
semantic proof, negative tests, and target evidence rather than a name-only
extension.
