# ADR-0005: Add versioned dav-c310 row-wise target recipes

- Status: Accepted
- Date: 2026-07-30

## Context

The measured Softmax and RMSNorm implementations require one coordinated
choice of row routing, packed FP16 memory access, FP32 reduction, SIMT math,
launch geometry, and affine-store handling. Local token substitutions cannot
establish that an arbitrary CUDA kernel, adapter, or host dispatcher has the
same behavior. Intercepting a higher-level dispatcher can also bypass
log-softmax, masking, bias, clamp, skip, or caller-defined store semantics.

RMSNorm affine is a cross-file contract: `rms_norm.cuh` supplies the
normalization kernel while the caller supplies the store that optionally
multiplies each FP16 result by a per-column FP16 weight. The optimization may
therefore be selected only after both the kernel semantics and the concrete
load/store adapters have been proved.

## Decision

### Activation, versioning, and fallback

Run `DavC310TargetRecipe` only for the exact option pair
`--target-policy=dav-c310-vec --simt-math=fast`. The default remains
`portable + precise`.

Install the target implementation under `ascify::target::dav_c310::v1`.
Generated code may use the unversioned aliases, while an artifact that needs a
stable semantic contract can name `v1` explicitly. Production configuration
is fixed inside the public target header and protected by macro save/restore;
tuning translation units require an explicit caller-configuration opt-in.

All AST edits are insert-only. A proven Softmax direct launch wrapper receives
an entry prologue that calls `TrySoftmax`. A proven RMSNorm direct launch
wrapper first executes its original launch-geometry block and exact error
return, then calls `TryRmsNorm` before the original CUDA launch. A handled
result returns the target status. `NotHandled` executes the remaining converted
wrapper unchanged, including its original CUDA launch and error propagation.

### Strict adapter contract

Emit a target marker only for a main-file class-template definition whose
complete constructor, fields, and packet access method prove:

- exact owner type with no base classes;
- FP16 storage and FP32 compute;
- exact data pointer and row-stride mapping;
- `(row * stride + column) / pack` packet addressing;
- one direct global-memory packet transfer and one canonical
  `0..pack-1` cast-copy loop;
- no volatile state, shifted pointer, transformed element, extra write,
  hidden call, or deferred/dead-path evidence.

An affine store additionally requires the same-column FP16 weight packet and
the exact `static_cast<DST>(src[i]) * weight[i]` branch. The marker exposes an
exact-owner tag, storage/compute aliases, data and stride accessors, the affine
constant, and the weight accessor when affine is proved. Inheritance cannot
reuse another record's proof.

The target adapter overload accepts only each argument's exact marked owner
type, half load/store pointers, `Compute == float`, contiguous strides equal
to `columns`, a live affine weight when required, and a float inverse-RMS
output.

### Semantic, control-flow, and effect proof

Classify `exp`, division, and reciprocal square root only through system
CUDA/C declarations, direct division syntax, recursively proved forwarding
wrappers, or explicit semantic annotations.

For Softmax, prove the connected value flow
`load -> max -> value - max -> exp -> same sum -> divide -> store`. For
RMSNorm, prove
`load -> square -> same sum/mean -> +epsilon -> rsqrt -> inverse output and
normalized store`. Reduction/storage roots and template bindings must remain
the same throughout the flow.

All evidence statements must be executable in a compatible control context.
Evidence inside an uncalled lambda, constant-false branch, false conditional
arm, short-circuited operand, unreachable loop, or mutually exclusive branch
is rejected. Opposite comparisons and different enum choices cannot be
spliced into one proof.

The kernel effect proof permits only the proved adapter operations,
classified math/reduction operations, local computation, the exact
inverse-RMS write, and side-effect-free helpers or supported system builtins.
Global/parameter writes, atomics, traps, volatile state, assembly, allocation,
exceptions, and unresolved calls reject the recipe.

### Direct launch-wrapper placement

Inject only into a host wrapper that has one unconditional launch of a proved
semantic global and an exact template/argument mapping for load, store,
compute type, rows, columns, stream, shared memory, and RMSNorm epsilon and
inverse output.

The wrapper effect proof requires this source order:

1. one proved launch-geometry setup call;
2. `status != success` followed by `return status`;
3. the unique unconditional CUDA launch;
4. `return cudaPeekAtLastError()`.

The geometry helper may mutate only its designated grid output and local
state. Every successful CFG exit must follow the unique positive grid write;
an earlier failure return must remain provably nonzero. CUDA runtime and
utility calls require system/builtin provenance. Atomics, volatile reads,
loops, switches, lambdas, conditional launches, extra calls, persistent
state, or other host side effects reject injection.

Softmax geometry contains only non-dependent system calls, so its target
recipe may run at wrapper entry after the complete helper and wrapper effects
have been proved safe to bypass. RMSNorm geometry contains a dependent
occupancy call that can acquire ADL candidates at template instantiation.
The recipe therefore preserves the exact two-statement geometry scope
`{ status = layer_norm::GetNumBlocks(...); if (status != success) return
status; }` and inserts only after its closing brace.

Softmax injection is guarded by the proved `kSoftmax` enum choice, so
`kLogSoftmax` retains the converted path. The accepted OneFlow-shaped input
has exactly these six direct-wrapper placements:

- Softmax:
  `LaunchSoftmaxWarpImpl`, `LaunchSoftmaxBlockSMemImpl`,
  `LaunchSoftmaxBlockUncachedImpl`;
- RMSNorm:
  `LaunchRmsNormWarpImpl`, `LaunchRmsNormBlockSMemImpl`,
  `LaunchRmsNormBlockUncachedImpl`.

The three Softmax placements are wrapper-entry prologues. The three RMSNorm
placements are after the exact geometry scope and before the unique launch.
No recipe is inserted into `DispatchSoftmax`, `DispatchLogSoftmax`, or
`LaunchRmsNorm`.

### Runtime domain

Both recipes require non-null stream/input/output, positive rows and columns,
`columns <= 32768`, and a non-overflowing `rows * columns`. RMSNorm also
requires rows and columns representable by the original `int` kernel
arguments, finite positive epsilon, non-null float inverse output, and a
non-null affine weight when affine is active. A failed precondition returns
`NotHandled`.

## Formal conversion evidence

`run_910_conversion_v3.sh` packages the exact Ascify binary, recipe
source/header, four inputs, four outputs, stdout/stderr logs, exact argv, and
SHA256 values in a non-overwriting `ascify-conversion-evidence-v3` directory
and manifest. The validator requires this output topology:

| Conversion unit | Target include | Load marker | Store marker | Softmax entry | RMSNorm post-geometry |
|---|---:|---:|---:|---:|---:|
| `softmax` | 1 | 1 | 1 | 3 | 0 |
| `layer_norm` | 0 | 1 | 1 | 0 | 0 |
| `rmsnorm` | 1 | 0 | 0 | 0 | 3 |
| `rmsnorm_adapter` | 0 | 0 | 1 | 0 | 0 |

Placement validation requires one exact entry binding in each Softmax direct
wrapper and one exact post-geometry/pre-launch binding in each RMSNorm direct
wrapper. It also requires zero unexpected recipe calls, zero calls in the
forbidden dispatchers, and no `kLogSoftmax` reference in a Softmax direct
wrapper. The same topology must appear in the generated files, manifest, and
single recipe summary line in each stderr log.

An independent real-source mutation gate changes one proof-critical property
per case. Its evidence TSV binds the Ascify binary, harness, Softmax,
RMSNorm/layer-norm inputs, and recipe source/header hashes. The accepted matrix
contains 29 Softmax and 9 RMSNorm cases covering renamed kernels/wrappers,
coverage gaps, reducer/data-flow spoofs, observable effects, geometry CFG,
macro insertion points, generated-name collisions, and member wrappers.

On 950PR, `run_formal_recipe_v3.sh` uses a staged protocol. `prepare` holds the
result lock, revalidates conversion evidence, freezes all build inputs, builds
direct and native controls side-by-side, and atomically publishes an immutable
binary bundle without reserving a device. It then performs a fresh strict-idle
selection and enters `measure` with the inherited result and device locks.
`measure` rechecks the frozen inputs and bundle, uses `SKIP_BUILD=1`, runs 42
Softmax plus 18 RMSNorm boundary cases and all tuning shapes, isolates preheat,
and executes direct-A/native/direct-B without interleaved build or correctness
work. While retaining the same device lock, it polls only for delayed runtime
process cleanup between phases; each phase still begins with the complete
strict-idle snapshot. `resume` is permitted only for an already prepared set and skips
rebuilding before fresh selection.

Binary-bundle schema v2 records the canonical build-input snapshot path and
SHA256. The bundle directory, manifest, and eight binaries are read-only; each
phase and the final bracket revalidate the snapshot-to-bundle binding.

The formal set records every run manifest, rejects a direct A/B spread above
1.05, requires every direct/native geometric center to be at least 0.90, and
requires each Softmax, RMSNorm-plain, and RMSNorm-affine group geomean to be at
least 0.95. Every phase validates exact pre/post device snapshots. These prove
idle endpoints and exclude cooperating project runs through the device lock;
they do not prove interval-wide exclusion of a non-cooperating scheduler.

## Consequences

- A successful direct conversion dispatches an unedited generated header into
  the versioned packaged target implementation. The same-round native control
  is a thin entry into that implementation, so performance parity demonstrates
  reproducibility of the deposited recipe implementation, not independent
  low-level synthesis from arbitrary CUDA.
- Unsupported code or runtime inputs retain the original converted behavior.
- Project-specific file, kernel, and wrapper names are not recipe triggers;
  renamed-kernel/wrapper mutations and forbidden-dispatcher gates cover this
  property for the accepted source family.
- Affine RMSNorm requires converting and staging its caller adapter together
  with `layer_norm.cuh` and `rms_norm.cuh`.
- Expanding the supported domain or adapter/kernel shapes requires new
  positive and adversarial negative evidence; it is not implied by the
  current recipe.
- Semantics-preserving source forms outside the currently proved
  AST/data-flow/CFG normal forms fail closed; rename coverage does not imply
  unrestricted syntactic generality.
