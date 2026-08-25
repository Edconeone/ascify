# ADR 0016: Proof-gate single-device CUDA Sample selection

## Status

Accepted; extends ADR 0015 only for the `findCudaDevice` surface

## Date

2026-08-24

## Context

Some official NVIDIA CUDA Samples call
`findCudaDevice(int, const char **)` before otherwise translatable runtime and
kernel code. The helper's default CUDA behavior examines device properties and
ranks GPUs by an estimated GFLOPS score. Ascify has neither an equivalent
cross-architecture ranking contract nor evidence that a CUDA SM-based score is
meaningful for an AIV-only target. Copying that policy, returning zero without
checking the runtime, or rewriting a same-spelled project function would make
unsupported behavior look successful.

The useful, provable deployment case is narrower: exactly one logical device
is visible, and the program either gives no device selector or explicitly asks
for logical device zero. Closing that case removes one host-helper dependency
without claiming CUDA device-ranking equivalence.

## Decision

Ascify rewrites a call to its owned
`::ascify::sampleFindCudaDevice(int, const char **)` only after three
independent proofs succeed. This rewrite joins the existing NVIDIA
sample-helper closure and commits atomically with every admitted macro rewrite
and the `helper_cuda.h` include replacement.

### Frozen source profile and active-definition AST proof

The declaration must resolve to a definition in the structurally recognized
NVIDIA `helper_cuda.h` described by ADR 0015. Structural provenance is
necessary but is not enough for this rewrite: the actual source files that own
the `findCudaDevice`, `checkCmdLineFlag`, `getCmdLineArgumentInt`,
`gpuDeviceInit`, `gpuGetMaxGflopsDeviceId`, `check`, and
`_ConvertSMVer2ArchName` definitions must also match one complete frozen
profile by basename, byte length, and SHA-256 digest.

The digest is computed over `SourceManager::getBufferData()` for the spelling
`FileID` that produced each resolved definition. Ascify does not reopen the
filesystem path: VFS overlays, remapped buffers, and a file changed after
parsing therefore cannot make the proof observe different bytes from the AST.
LLVM's `Support/SHA256.h` first appears in LLVM 13. On older supported LLVM
builds Ascify returns no frozen-profile match, so this feature alone remains
disabled rather than weakening identity or raising the whole project's LLVM
minimum version.

The production profile pins NVIDIA cuda-samples commit
`b7c5481c556c3fe98db060207ecaa41a4b9a9abc`:

- `helper_cuda.h`: 28,177 bytes,
  `997f9ac1f8e5f8e5f45f8b11eebab5b89305dee7430b90654bafe62283cffee1`;
- `helper_string.h`: 15,079 bytes,
  `26e988c97fb3d77d498e384c685177ed7966e41d5d58ebc9b7d3d696859f5e57`.

The real-frontend positive fixture vendors those exact upstream bytes; no
synthetic helper hash is admitted by production code. A same-named header, a
declaration from the frozen file whose resolved definition lives elsewhere,
or any byte change to an active dependency body fails closed. This prevents an
otherwise identical helper from gaining telemetry or another side effect
while still passing a name-and-call-count check.

File identity alone does not freeze preprocessing. For every macro expansion
and active conditional-macro dependency inside either frozen header, the
preprocessor callback also proves that the macro definition comes from the
same frozen profile, a file whose first `EnterFile` callback already carried a
system characteristic, or a compiler builtin. A project, preinclude, or
command-line macro that changes `STRNCASECMP`, `printf`, `exit`, a runtime
call, or a transitive helper body rejects the whole transaction. This rule is
based on definition provenance rather than a list of trusted macro names.

The proof also walks the resolved direct-call graph rooted at the exact frozen
`findCudaDevice` definition. A call may resolve only to a definition in the
same paired frozen profile or to a function whose complete redeclaration chain
comes from files recorded as system at entry. This covers, among others,
`strlen`, `strchr`, `atoi`, `strncasecmp`, `printf`, `exit`, CUDA Runtime
calls, and the helper-to-helper calls reached through
`checkCmdLineFlag`, `getCmdLineArgumentInt`, `gpuDeviceInit`, and
`gpuGetMaxGflopsDeviceId`. A normal user definition before or after the frozen
header therefore cannot inherit a trusted canonical declaration and silently
add behavior that the replacement would remove.

Every redeclaration's source-backed attributes are part of the same proof.
Their application location must be in the frozen profile or an initially
trusted system file; only genuinely implicit compiler attributes with no
source location are admitted without one. This rejects symbol redirection and
aliasing introduced from user source, including `#pragma redefine_extname`,
`AsmLabelAttr`, alias, weak-reference, and IFUNC-style attributes.

The Ascify executable, Clang resource directory, CUDA SDK, sysroot, and
system-include graph are a trusted configuration boundary; this feature does
not cryptographically attest them. Main-file source, `-D` definitions,
ordinary headers and preincludes, VFS-remapped helper contents, and `#line`
presumed filenames remain untrusted. A compiler predefinition is admitted only
when its actual spelling `FileID` equals the preprocessor's predefines
`FileID`, and a command-line definition in that buffer is still rejected. This
prevents `#line 1 "<built-in>"` in an ordinary preinclude from forging builtin
provenance.

System trust is recorded from the file's stable `FileEntry` unique identity
and characteristic on its first `EnterFile` callback, then mapped to each
actual `FileID`. A later `#pragma clang system_header` or
`#pragma GCC system_header` notification, including a second entry that gains
a new `FileID`, cannot promote an identity first seen as ordinary. Imaginary
files, invalid identities, and SourceManager content overrides are not trusted.
This avoids relying on the mutable current
`SourceManager::isInSystemHeader()` classification.

The same immutable entry-time identity check is used by the pre-existing
`checkCudaErrors` runtime-status-domain admission. A same-named runtime
overload or redeclaration from an ordinary pragma-marked header cannot acquire
CUDA Runtime status semantics merely because its current SourceManager
characteristic says system.

Clang classifies directories supplied with `-isystem` as system headers, so a
caller must not mark an attacker-controlled directory as system and then treat
this proof as adversarial-input isolation. User-provided `-isystem` therefore
enlarges the trusted configuration and is outside this guarantee. Extending
the threat model past that boundary requires a separate
header-search-origin manifest.

Adding a new upstream helper version requires a reviewed source diff, a pinned
upstream revision, both byte lengths and hashes, a positive real-frontend
fixture, and a dependency-body mutation negative. Updating a path or function
name alone cannot extend admission.

After the source profile is established, the active Clang AST must prove all
of the following:

- an inline, non-template function with the exact canonical signature
  `int(int, const char **)`;
- a three-statement outer body consisting of integer `devID = 0`, one
  device-selection `if` with an `else`, and `return devID`;
- a condition that directly calls `checkCmdLineFlag(argc, argv, "device")`;
- the complete official explicit-device branch: the
  `getCmdLineArgumentInt` result is assigned to `devID`, both negative-result
  tests read that same variable and execute only the official diagnostic plus
  `exit(EXIT_FAILURE)`, and the `gpuDeviceInit(devID)` result is assigned back
  to `devID`; and
- the complete official automatic branch: the
  `gpuGetMaxGflopsDeviceId()` result is assigned to `devID`,
  `cudaSetDevice(devID)` and both compute-capability queries remain wrapped by
  the recognized four-argument `check` expansion, their outputs feed the
  official summary print, and no other statement is present.

The direct helper calls in this AST must also use unqualified, non-macro source
tokens. The `printf` and `exit` callees in both failure blocks and the final
summary additionally resolve to their unqualified declarations in files that
entered preprocessing as system headers, and every redeclaration must retain
that provenance. A preinclude macro that merely redirects those names back to
`::printf` or `::exit` is still rejected.

The proof matches every statement and nested call in both branches, not only a
count of selected function names. An extra call, comma expression, changed
assignment target, discarded result, unchecked runtime call, wrong device
argument, altered failure block, or extra declaration rejects the definition.

Text hidden by preprocessing cannot satisfy this proof. A recognized-looking
header with a changed active selection body also fails closed.

### Direct call-site proof

The use must be a direct `CallExpr` in a main-file host function. After only
parentheses and implicit casts are removed, its callee must still be an
unqualified `DeclRefExpr` to the proven canonical definition. The callee
location must be a non-macro source token whose exact spelling is
`findCudaDevice`; even an explicit `::findCudaDevice` qualifier is rejected.

Ascify replaces only that callee token. Both arguments and the expression
context remain unchanged, so ignored returns, assigned returns, and directly
returned results preserve their ordinary C++ behavior. Macro-produced calls,
preprocessor aliases, references, function pointers, indirect calls, using
aliases, device/global calls, external-header uses, and same-spelled
project-owned functions remain outside the admitted set.

`sampleFindCudaDevice` is a reserved generated token. If source, a user
header, or the command line ever defines it as a macro, the complete helper
transaction is rejected. Raw-token auditing also treats every unproved
`findCudaDevice` occurrence as a residual dependency.

### Runtime contract

The owned helper deliberately implements a smaller contract:

- unrelated command-line arguments are ignored;
- no device selector, or exactly one literal `--device=0`, is admitted;
- any other device selector spelling or value, a duplicate selector, malformed
  `argc`/`argv`, or a null argument fails with a diagnostic and
  `EXIT_FAILURE`;
- `cudaGetDeviceCount` must succeed and report exactly one visible logical
  device; zero or multiple visible devices fail;
- `cudaSetDevice(0)` must succeed; and
- success returns logical device ID zero.

The helper never calls or emulates `gpuGetMaxGflopsDeviceId`, never fabricates
CUDA SM properties, and never ranks devices. Query and bind errors retain an
explicit ACL-backed diagnostic before failure.

## Alternatives considered

### Map `findCudaDevice` by spelling

Rejected. The name is not an ownership or semantic proof, and projects may
provide their own function.

### Copy NVIDIA's GFLOPS ranking

Rejected. Its CUDA architecture inputs do not establish a portable ordering
for Ascend logical devices, and Ascify lacks a validated equivalent.

### Always return zero

Rejected. It would silently accept no-device, multi-device, and failed-bind
states while pretending selection succeeded.

### Accept every selector that NVIDIA's string helper accepts

Rejected. The only evidenced explicit selection is logical zero. Broader
spellings and IDs would enlarge the compatibility claim without a multi-device
contract.

## Consequences

Official CUDA Samples with the proven helper definition and direct host calls
can close this dependency while preserving return-value use. One unsafe helper
use still prevents every helper edit and keeps the original include, which
makes the unsupported boundary visible.

The result is source-closure and target-syntax progress, not a whole-program
runnability claim. In particular, the current `simpleAtomicIntrinsics`
translation still rejects `cudaStreamNonBlocking`, and target syntax for GM
atomics does not by itself prove contended old-value semantics on a device.
Those boundaries require separate implementation and device evidence.

Host tests cover default and explicit-zero success plus zero/multiple devices,
other or duplicate selectors, query failure, and bind failure. Real-frontend
fixtures cover returned and discarded-result call contexts plus the qualified,
macro, alias, indirect, external-header, altered-definition,
non-NVIDIA-function, and output-macro fail-closed cases. Preinclude negatives
cover direct `printf`/`exit` redirects, a transitive `STRNCASECMP` redirect,
`#line` builtin-name spoofing, Clang/GCC `system_header` self-promotion, and
pragma-marked header re-entry with a new `FileID`, and non-macro user
definitions of `strlen` and `atoi`, plus a `redefine_extname` redirect of
`strlen`.
A frozen `helper_cuda.h` paired with a telemetry-mutated `helper_string.h`
proves that dependency-file identity is enforced separately from the outer
`findCudaDevice` AST.

The altered source fixtures for an extra side effect, discarded or wrongly
assigned internal selection results, a changed signature, an unchecked bind,
and a wrong-device bind all stop at the frozen source-profile gate. They do
not claim independent runtime coverage of each defense-in-depth AST matcher
branch; such a claim would require a matcher-unit harness that can exercise an
AST directly without admitting an unsafe production hash. When
`ASCIFY_BINARY` is supplied, the top-level release gate invokes the complete
real-frontend suite with the binary, CUDA path, and Clang resource directory
rather than stopping at host/static checks.
