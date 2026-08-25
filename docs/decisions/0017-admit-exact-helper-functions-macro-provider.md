# ADR 0017: Admit only the exact CUDA Samples macro provider

## Status

Accepted

## Date

2026-08-24

## Context

NVIDIA CUDA Samples commit
`b7c5481c556c3fe98db060207ecaa41a4b9a9abc` lets several Common helper
headers establish policy macros. A sample such as `simpleAtomicIntrinsics`
includes `helper_functions.h` before `helper_cuda.h`. The exact
`helper_functions.h` includes `helper_image.h` first, so the later frozen
consumers see two active definitions from `helper_image.h`: `EXIT_WAIVED=2`
in `helper_string.h` and `helper_cuda.h`, and `MAX(a,b)=((a>b)?a:b)` in
`helper_cuda.h`.

Ascify's frozen helper transaction previously rejected that definition because
the defining header was neither one of the two frozen consumer files nor an
initially trusted system file. Keeping `helper_cuda.h` then also kept
`findCudaDevice` and every error-check macro, blocking the otherwise supported
translation.

A general exception for either macro name, for a header basename, or for a
later `system_header` classification would let project policy enter a
transaction whose helper include is removed. The exception therefore needs
a source identity and provenance boundary as strict as the existing frozen
helper proof.

## Decision

Recognize the exact b7c files by basename, byte size, and SHA-256:

- `helper_functions.h`: 2,358 bytes,
  `3fdcd18e41ffc2a9c88ade3595384e9cd05a2d84f80b86a2d5982035ca79c426`;
- `helper_image.h`: 28,739 bytes,
  `bc1fe7921bafad278ffa2e4bc8a99c18825208b9f5f47842a7cf7e86cae8b3f1`.

Record only the exact `helper_image.h` provider's physical `FileEntry`
identity, and only on its first `EnterFile` callback. Do not record
remapped/overridden buffers, zero identities, later aliases, alternative
Common helpers, or files whose exact role is not proven at that first entry.
Also require the exact `helper_functions.h` root to have been seen.

Admit exactly two active `#ifndef` dependencies:

- object-like, one-token `EXIT_WAIVED`, whose token is the numeric spelling
  `2`, in already frozen `helper_cuda.h` or `helper_string.h`;
- function-like, two-parameter `MAX`, whose token body is exactly
  `((a > b) ? a : b)` modulo the parameter identifiers, in already frozen
  `helper_cuda.h` only.

Each definition must come from the first-entry exact `helper_image.h` identity
and still match that provider's frozen bytes at the use site. No provider role
is added to the frozen function-call, declaration, attribute, or general macro
trust masks.

For these two names, do not fall back to compiler-predefined, builtin, or
generic initial-system-header trust. Exact definitions from the original
frozen `helper_cuda.h`/`helper_string.h` pair remain valid, preserving the
direct `helper_cuda.h` case.

## Alternatives considered

### Trust either body regardless of origin

Rejected. A project header or command-line definition would silently influence
a removed helper transaction.

### Trust `helper_functions.h` by basename

Rejected. A local shadow can use the same name and macro while carrying
arbitrary surrounding behavior.

### Trust only the exact `helper_functions.h` definition location

Rejected. In the unmodified include order, exact `helper_image.h` defines the
macro before control reaches the definition in `helper_functions.h`; this
would not fix the real sample.

### Admit helper_string, helper_timer, or helper_functions as alternatives

Rejected. They are not the active provider in the canonical sample graph.
Adding alternative providers without a real positive would broaden the trust
surface beyond the observed dependency.

### Trust any initially systemic definition

Rejected for these macros. They are CUDA Samples policy values, and an
arbitrary `-isystem` input is not proof that it belongs to the frozen Samples
closure.

## Consequences

- The unmodified official helper-functions include order can complete the
  existing all-or-nothing `findCudaDevice`/error-check transaction.
- Command-line definitions, project headers, redefinitions, same-name shadows,
  same-size changed provider bodies, `system_header` promotion/re-entry,
  `#line` filename spoofing, and remapped files fail closed and keep
  `helper_cuda.h`.
- Updating CUDA Samples requires deliberately refreshing every affected
  frozen size/hash and its positive and negative fixtures.
- On LLVM versions without SHA-256 support, this admission remains disabled,
  matching the existing frozen-profile behavior.
