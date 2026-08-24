# ADR-0011: Keep narrow frontend admission explicit and separate from parser shims

## Status

Accepted; clean-build and translated-fixture regression are release gates

## Date

2026-08-24

## Scope

The opt-in `ascify-admitted-v1` parser profile and its include-provenance
boundary. This decision does not admit new target semantics.

## Context

The CUDA parser installation used by the vector-corpus experiment reaches
NVIDIA's `<cooperative_groups.h>` and then fails on a missing `nv/target`
dependency. An external declaration-only parser shim lets more translation
units reach Ascify's rewrite pass, but it declares a much wider CUDA surface
than the target has proved. Counting that shim as product support would mix a
parser experiment with Ascify's supported conversion boundary.

The audited CUDA Samples block kernels need only four declarations while they
are parsed: `thread_block`, `this_thread_block()`, `thread_block::sync()`, and
the exact free overload `sync(const thread_block&)`. The target-side wrapper is
separate and already lowers this exact block barrier to the verified legacy
CANN primitive. There is no corresponding evidence for generic group sync,
tile or coalesced-group barriers, `cooperative_groups/reduce.h`, ballot,
dynamic shared memory, or CUDA's internal `nv/target` implementation surface.

## Decision

Add an explicit product option:

```text
--frontend-compat=none|ascify-admitted-v1
```

The default is `none`. It leaves the CUDA installation and include search
unchanged. `ascify-admitted-v1` prepends one installed, Ascify-owned include
root containing only:

- `cooperative_groups::thread_block`;
- `cooperative_groups::this_thread_block()`;
- `thread_block::sync()`; and
- `sync(const thread_block&)`.

The profile contains a poison `cooperative_groups/reduce.h`, so a source
requesting reduction fails at the frontend boundary. It does not provide an
`nv/target` stub, a generic sync template, tiles, coalesced groups, reduce,
ballot, shuffle, or dynamic-shared declarations.

CMake installs the profile below Ascify's private `libexec` tree. A build-tree
binary uses the same checked-in source directory through a compile-time path.
Before Clang parses the source, Ascify requires the exact versioned manifest,
the admission header and the reduction poison. Their byte counts and fixed
SHA-256 identities are published in the manifest, while the executable compares
their complete bytes against its compiled-in profile identities. No required
file may be a symlink, and the profile may contain no unmanifested file or
directory. If an installed profile exists but fails that audit, Ascify rejects
it instead of silently falling back to the build-tree copy. Source-tree fallback
is accepted only when the running executable resolves inside the build directory
recorded into that binary; relocated or installed binaries require their
installed private profile.

The preprocessor callback then treats the resolved `FileEntryRef` as the source
of truth. An admitted `cooperative_groups.h` must resolve canonically to the
verified profile file. A quoted local shadow is rejected. Every
`cooperative_groups/*` subheader is rejected even if Clang could otherwise
continue searching and find a CUDA Toolkit copy. This closes both the
missing-poison fallback and unknown-subheader fallback paths.

Experiment results keep three mutually exclusive lane identifiers:

1. `official-dependencies`: no product compatibility option and no shim;
2. `ascify-admitted-v1`: the explicit product profile and no external shim;
3. `wide-parser-shim`: the external declaration-only shim and the default
   product profile.

Lane results are frontend-translation evidence only. Their numerators are not
merged, and none is a target-compile, device-correctness, or performance
result.

## Alternatives considered

### Enable the profile by default

Rejected. This would silently replace an official CUDA dependency and change
the historical frontend denominator.

### Vendor NVIDIA's complete cooperative-groups headers

Rejected. It would inherit internal dependencies such as `nv/target` and
expose semantics that have not been admitted on the AIV target.

### Promote the external parser shim into the product

Rejected. Its deliberately broad declarations are useful for diagnosis but
do not encode Ascify's target contract.

### Declare a generic `sync(Group)` adapter

Rejected. Host parsing would then accept tile and coalesced-group calls even
though only block synchronization has a verified target primitive.

## Consequences

The product can measure a narrow, reproducible cooperative-groups frontend
lane without claiming the wide parser shim. Existing invocations retain the
official dependency behavior because the option defaults to `none`.

The profile is intentionally incomplete. Reduction and every unlisted group
surface fail during parsing. The closed manifest means that extending the
profile requires a new reviewed manifest and exact byte identity rather than
dropping a new header into an installed directory. A successful
product-profile translation
still requires independent target syntax, link, device-correctness, and
performance gates before it can contribute to a broader conversion or
performance claim.

## Verification

`tests/frontend_compat/check_frontend_compat.sh` checks the exact manifest and
header surface, option default, install wiring, positive block syntax, and
negative generic-sync, tile, and reduction cases. With `ASCIFY_BINARY` set,
`tests/run_release_checks.sh` also exercises the real translator against
unknown `scan` and `memcpy_async` subheaders, a quoted local shadow, and an
unadmitted path spelling. It also runs isolated installed-profile negatives for
a same-size header mutation, a manifest-version mutation, an extra file, a
missing required poison and a wholly missing profile. Every rejection must
return nonzero and publish no output. Corpus lane accounting remains an external
experiment-harness responsibility; it is not installed as a product release
test.
