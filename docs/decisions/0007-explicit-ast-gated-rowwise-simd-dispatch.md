# ADR-0007: Use explicit AST-gated SIMD dispatch with versioned target support

- Status: Accepted
- Date: 2026-08-21

> **Amendment:**
> [ADR-0008](0008-compose-rowwise-simd-and-simt-stages-in-one-kernel.md)
> changes the canonical generated source facade and the selected kernel's
> internal execution from an all-SIMD path to same-kernel SIMD+SIMT stage
> composition. The activation, AST proof, selector, C ABI, deployment, and
> launch-ownership decisions in this ADR remain in force.
>
> **Scope extension:**
> [ADR-0009](0009-register-rowwise-hybrid-recipes-and-add-layernorm.md)
> adds the common recipe registry, LayerNorm, and a fourth DSO without changing
> the activation or ownership rules here.

## Context

[ADR-0005](0005-add-versioned-dav-c310-rowwise-recipes.md) established the
whole-operator semantic-recognition proof used for the existing dav-c310
row-wise recipes. That proof, which does not prescribe one execution style,
can identify supported Softmax and RMSNorm launch wrappers without
depending on project, file, kernel, or wrapper names. It also preserves the
translated SIMT body whenever a target recipe declines a call.

The high-performance target-support implementations have a different deployment
boundary from an ordinary source rewrite. Their device code must be compiled
with the dav-3510 CANN toolchain, retained in separate device-registration
artifacts, and linked or loaded together with the generated main CCE. Enabling
that dependency implicitly under the existing `dav-c310-vec + fast` pair would
change the link contract of existing users without a distinct opt-in.

Publishing preselected whole-file payloads is also not a compiler contract. A
payload selected by a source hash does not show that `ascify-clang` recognized
the input AST, inserted a dispatch at the proved wrapper, or retained the
source-derived SIMT fallback.

## Decision

### Explicit activation

Add the public option:

```text
--target-recipe=none|dav-3510-rowwise-simd-v1
```

`none` is the default. It does not introduce the externally linked target ABI
into generated code and preserves the pre-existing conversion behavior.

`dav-3510-rowwise-simd-v1` is valid only with both:

```text
--target-policy=dav-c310-vec --simt-math=fast
```

The frontend rejects an unknown recipe or any other policy/math combination.
The `dav-c310-vec` spelling remains the converter policy name; `dav-3510` in
the recipe name identifies the deployed 950PR target-support contract.

### AST rewrite and fallback

Explicit mode reuses the adapter, data-flow, control-flow, effect, geometry,
and direct-wrapper proofs from ADR-0005. It does not use an operator name or a
whole-file source hash as the rewrite trigger.

User `ascify.semantic.*` annotations are not accepted as whole-kernel target
proof in explicit mode. FP32 primitive template specializations must have an
inspectable body. The proof reads the complete raw source body, including
branches excluded from the current AST, and requires every textual conditional
branch in that body to contain only the same exact exp/divide/rsqrt return.
Reserved macros, pre-instrumented adapter markers, input-defined dispatch
declarations, and input-defined launch-ABI symbols fail closed. Feature macros
that affect the proved source branch must agree between conversion and CCE
build; the release evidence records aligned `OF_*_USE_FAST_MATH` values.
Toolkit-version macros may differ only when they select the same validated
branch. These checks prevent an invalid inactive branch, annotated identity,
side-effecting helper, or forged dispatch from authorizing a target takeover.
Explicit mode also rejects non-system GNU or Microsoft assembler declarations
and statements. Skipped non-system source is scanned for assembler tokens, so
an inactive include cannot publish an alias for a protected `_launch_v1`
symbol after the AST proof has completed.

For each proved wrapper, the generated main CCE:

- includes
  `ascify/target/dav_c310/rowwise_simd_recipes.hpp` exactly when a hybrid
  dispatch is inserted;
- calls the closed `RowwiseHybridFacadeV1::TrySoftmaxHybrid` or
  `RowwiseHybridFacadeV1::TryRmsNormHybrid` member with the proved adapter and
  launch arguments;
- returns immediately when the result has `handled == true`; and
- leaves the original translated SIMT launch after the prologue.

The canonical facade result name is `HybridTryResult`, an alias of
`SimdTryResult { bool handled; aclError status; }` in the versioned namespace
and re-exported as `ascify::target::dav_c310::HybridTryResult`. The legacy
`RowwiseSimdFacadeV1::Try*Simd` entry remains a source-compatible alias for
predecessor generated artifacts; neither legacy spelling describes the
selected kernel's current execution composition.

Softmax dispatch is inserted at the proved direct-wrapper entry and remains
guarded to the Softmax, not LogSoftmax, choice. RMSNorm dispatch is inserted
after the proved launch-geometry/error block and before the unique original
launch.

The injected target headers are macro-isolated. If the main source starts with
a conventional `#ifndef` / `#define` outer guard, the frontend discovers that
guard from the raw main-file tokens and temporarily undefines it only around
the start-of-file target include. The original guard state is then restored
before the source body, including when the `#define` has a value. This prevents
an injected dependency with the same guard spelling from suppressing the
translated operator body.

The ownership rule is strict:

```text
handled == false  -> selector miss; execute the retained SIMT path
handled == true   -> the selected mixed runtime owns the call; return status
```

The outer branch is a pre-launch support gate, not an all-SIMD versus all-SIMT
operator choice. As defined by ADR-0008, a selected kernel uses SIMD for the
proved vector math and SIMT for contiguous UB output staging. A selector miss
still runs the retained whole-SIMT implementation. This ownership rule also
includes a selected launch error: `handled == true` with a non-success status
must not run the fallback afterward, because doing so could execute the
operator twice or write the output after a partial failed launch.

The Softmax runtime repeats the device and launch-geometry attribute queries
that the original direct wrapper would perform before launching. It propagates
`aclrtGetDevice`, vector-core-count, and maximum-threads-per-vector-core errors
unchanged. The query is per call and has no guessed default or process-global
cache, so a successful query on one device cannot mask an error or device
change on a later call.

### Versioned target support

The target support is a separate build under `runtime/dav_3510/rowwise/`.
ABI version 1 fixes the public launch signatures, selector contract, and three
device-registration artifacts:

| Route | Public C symbol | DSO linker name | SONAME |
|---|---|---|---|
| Softmax recompute hybrid | `ascify950_softmax_reg_recompute_launch_v1` | `libascify950_softmax_reg_recompute_v1.so` | `libascify950_softmax_reg_recompute_v1.so.1` |
| RMSNorm cached hybrid | `ascify950_rmsnorm_reg_cached_launch_v1` | `libascify950_rmsnorm_reg_cached_v1.so` | `libascify950_rmsnorm_reg_cached_v1.so.1` |
| RMSNorm plain row-batch hybrid | `ascify950_rmsnorm_reg_plain_rowbatch_launch_v1` | `libascify950_rmsnorm_reg_plain_rowbatch_v1.so` | `libascify950_rmsnorm_reg_plain_rowbatch_v1.so.1` |

Every DSO has CMake `VERSION 1.0.0` and `SOVERSION 1`. The version suffix is
part of both the C symbol and linker name, while `.so.1` is the runtime SONAME.
Keeping the launches in separate DSOs also keeps their device-registration
domains separate when all three libraries are linked.

The generated main CCE includes the ABI/selector facade but does not contain
those device implementations. Consumers must compile the target support for
`dav-3510`, link or deploy the required runtime libraries, and build the main
CCE against the same public ABI and compatible CANN SDK. A source conversion
that has not completed those steps is not an end-to-end qualified conversion.

The repository harness uses `ROWWISE_SIMD_RUNTIME_DIR` as the directory that
contains all three `.so` linker-name files, their `.so.1` SONAME files, and
the corresponding implementations. Its build and smoke scripts verify each
ELF SONAME and require both names to resolve to the same file. The build adds
the exact `-lascify950_*_v1` dependencies to the link; the smoke runner
prepends the directory to process-local `LD_LIBRARY_PATH`.

The facade performs the first runtime-domain selection. Each public
`_launch_v1` DSO entry then calls the same route predicate again before any
kernel launch. This deliberate boundary check protects direct ABI callers and
rejects an invalid call with `ACL_ERROR_RT_PARAM_INVALID`. It does not turn a
selected launch error into a selector miss: the facade has already transferred
ownership, so the main CCE returns the error without launching SIMT.

The selector is versioned with the ABI. Expanding a shape range, changing
aliasing or alignment rules, changing a launch signature, or replacing a
kernel requires review and new correctness/performance evidence. An
incompatible public contract requires a new ABI or recipe version rather than
a silent change to `v1`.

The host release gate compiles direct-ABI negative cases, checks that every
runtime entry references its corresponding v1 predicate, and checks the CMake
`VERSION`/`SOVERSION` declarations. The 950PR build verification must build all
three DSOs, then link the Softmax check ELF only to its Softmax DSO and the
RMSNorm check ELF only to its two RMSNorm DSOs. Exact dynamic-symbol sets,
exports, and ELF SONAMEs are checked with symbol and `readelf` inspection. This
is a build/link boundary, not a device-performance result.

Full-fixture conversion is published through the wrapper only after its v3
structural inspection finds one target-support header and verifies that every recipe
call has the exact generated guard plus `handled -> return status` ownership
contract in a distinct function body with a later direct-scope SIMT launch in
that same wrapper. This report is an integrity check, not a replacement for
the frontend AST/CFG proof. The inspection removes comments, literals,
preprocessor directives, and unproved conditional regions before counting;
generated and input kernel-launch counts must match. Release negatives mutate
a math helper into an annotated identity, mutate all three forward wrappers
into annotated non-status wrappers, independently corrupt each side of a
conditional primitive, and introduce reserved semantic-callee/injected-name
macros; the explicit recipe must reject every case without
publishing output. The wrapper also rejects source that already contains
recipe code, so a pre-existing dispatch cannot be attributed to the current
conversion.

The report scanner's lexical contract follows the translation details needed
for fail-closed publication: invalid UTF-8, NUL bytes, a leading UTF-8 BOM,
and legal trigraph spellings are rejected in raw source and output bytes; standard and
Clang whitespace-extended phase-2
splices are removed before comments; comments and literals are blanked without
confusing C++ digit separators with character literals; physical lines split
only at CR/LF; and digraph punctuation is normalized before scope and
directive analysis. The negative suite includes inactive trigraph directives,
trigraph/spaced comment splices, spaced return/goto splices, BOM/NUL-prefixed
inactive branches, VT/FF macro bodies, and Softmax/RMSNorm dispatch hidden by
digraph braces.

The negative gate additionally covers direct facade declarations and
specializations, file- and block-scope assembler, assembler in skipped
non-system dependencies, injected-header guard collisions (including valued
outer guards), protected ABI/version macros, and exact device-query error
propagation. The guard probes preprocess the translated operator body and
syntax-compile the actual injected prologue.

### Supported source and runtime domain

The source proof is limited to forward FP16 row-wise operators with FP32
compute, exact contiguous row-major adapters, and the complete proved
Softmax or RMSNorm data-flow graph. It covers:

- ordinary forward Softmax with direct load/store;
- forward RMSNorm with a direct store; and
- forward affine RMSNorm when the concrete caller store proves the exact
  per-column FP16 weight multiplication.

It does not cover LogSoftmax, backward kernels, masking, bias, clamp, skip
connections, arbitrary custom adapters, non-contiguous strides, unsupported
types, or a semantically similar source form outside the proved AST/CFG normal
forms. Those cases remain unchanged SIMT or fail closed at recognition.

Runtime hybrid selection is narrower than source recognition. Version 1 checks
pointer validity, extents, aliasing, alignment where required, finite positive
RMSNorm epsilon, and its versioned shape limits. A recognized and rewritten
operator can therefore execute the selected mixed kernel for some calls and
the retained whole-SIMT path for others. The selected SIMT stage is limited to
contiguous UB output staging; version 1 makes no mask, gather, scatter, stride,
or other non-contiguous adapter claim.

### Evidence and reporting

Report these quantities separately:

1. **Structural recognition rate**: AST-proved in-scope operator
   implementation units divided by predefined in-scope implementation units.
2. **Rewrite success rate**: recognized units with a valid hybrid dispatch and
   retained SIMT fallback divided by recognized units.
3. **End-to-end qualified conversion rate**: in-scope units that complete
   rewrite, target-support build/link, CANN main-CCE build, and device
   correctness divided by in-scope units.
4. **Hybrid route coverage**: correctness-passing runtime cases that select
   the mixed target kernel divided by all correctness-passing in-scope runtime
   cases.

The counting unit is an operator implementation unit, not each of its three
direct launch wrappers. Wrapper insertion counts are a separate topology
check. Hybrid route coverage is not a source conversion rate. Legacy schema
fields may retain `simd` in their names for compatibility.

Performance numbers must identify the source corpus, shape corpus, precision,
build and runtime identities, CANN version, device, sampling protocol, and
aggregation. A ratio of `1.23x` means 123% of the stated baseline, or a 23%
increase; it must not be reported as a 123% increase. No performance result is
attributed to a newly generated binary before its own build, correctness, and
measurement gates pass.

ADR-0008 invalidates predecessor generated-output, DSO-build, correctness, and
performance attribution for the selected runtime. Promotion of the amended v1
requires fresh Hybrid-facade conversion plus same-source build, correctness,
and performance evidence.

Generated headers, runtime binaries, raw CSV, logs, and manifests remain
outside Git under the evidence policy in
[ADR-0006](0006-version-conversion-inputs-and-separate-run-evidence.md).

## Alternatives considered

### Enable the target support whenever `dav-c310-vec + fast` is selected

Rejected because it would silently add new headers, unresolved launch symbols,
and runtime deployment requirements to an existing option pair.

### Publish sealed whole-file SIMD/SIMT payloads

Rejected because selecting a deposited file is not an AST conversion and does
not prove dispatch placement, fallback retention, or compatibility with a
changed source revision.

### Inline all target implementation source into the generated CCE

Not selected for version 1 because the measured implementations require
separate ASC/DPP compilation and device-registration boundaries. A future
self-contained artifact format may be introduced under a different explicit
contract.

### Fall back to SIMT after a selected target launch error

Rejected because a failed selected launch may already have side effects. Only
a pre-launch selector miss may execute the SIMT fallback.

## Consequences

- Existing users keep their current output unless they explicitly select the
  new recipe.
- A reviewable main CCE now demonstrates actual AST recognition, hybrid
  dispatch insertion, and retained source-derived fallback.
- Deployment requires version-matched target-support headers and separately
  built runtime libraries; the main CCE is not self-contained.
- A selector miss is expected behavior and does not make the source rewrite a
  failure; a selected call composes SIMD math and contiguous SIMT output
  staging inside one target kernel.
- A selected launch error is returned directly and is a correctness/runtime
  failure, not a reason to launch SIMT a second time.
- New operator forms or selector domains require positive, adversarial
  negative, build, correctness, and performance evidence before promotion.
