# ascify-clang (ASCIFY)

**ascify-clang** is a Clang-based source translator that takes CUDA C/C++ and rewrites it toward **DPP**-oriented APIs and headers (for example `acl/acl.h` and runtime entry points such as `aclrtMalloc`). It is built as a standalone tool (`ascify-clang`) on top of the same LLVM/Clang your project links against.

The upstream Clang driver is still used for parsing and CUDA sema; ascify-specific options control the rewrite, statistics, and auxiliary outputs (Perl/Python maps, Markdown/CSV docs).

## Execution modes and current scope

Ascify keeps one broad compatibility path and adds a proof-gated performance
path. Both start from the same CUDA source translation; the Hybrid path is an
explicit extension, not a second standalone translator.

| Mode | How to select it | What Ascify emits | Current role |
|------|------------------|-------------------|--------------|
| Pure SIMT | Default recipe (`--target-recipe=none`) | CUDA API and kernel constructs mapped to the DPP/SIMT compatibility layer | Broad semantic coverage and whole-operator fallback |
| SIMD+SIMT Hybrid | Add `--target-recipe=dav-3510-rowwise-simd-v1` with the validated target policy | A proved facade call plus the original translated SIMT launch | Accelerated regular row-wise stages for supported Softmax, RMSNorm, and LayerNorm cases |

The Hybrid recipe is **proof-gated**. It currently accepts contiguous
row-major FP16/FP32 forward paths whose complete data flow matches a registered
Softmax, RMSNorm, or LayerNorm recipe. A selector miss continues through the
retained whole-operator SIMT kernel. Masked, strided, gather/scatter, backward,
unsupported-type, or otherwise unproved paths remain SIMT; Ascify does not
claim automatic SIMD partitioning for arbitrary CUDA programs.

The names describe two different layers: `dav-c310-vec` is the existing
translator policy identifier, while `dav-3510-rowwise-simd-v1` selects the
versioned 950PR target runtime and its selector/ABI contract.

### Who controls each part of a selected Hybrid kernel?

| Part | Responsibility |
|------|----------------|
| AIV scalar control | Assigns blocks to rows or row batches, drives tile loops, and issues queue/barrier operations. This orchestration is neither the SIMD stage nor the SIMT stage. |
| MTE2 / MTE3 | Move data between GM and UB (`GM -> UB` / `UB -> GM`). |
| SIMD stages | Execute proved regular vector math and reductions on the local tile, such as cast, max/sum reduction, `exp`, `rsqrt`, divide, and affine multiply. |
| SIMT stages | Preserve `threadIdx.x` ownership, indexing/tail behavior, and normalized-local-to-output-local adaptation inside a selected kernel; on a recipe miss, the retained pure-SIMT kernel performs the whole operator. |

“SIMD load/store” and “SIMT load/store” in compute microbenchmarks refer to
their local access instructions. They do not replace or own the MTE2/MTE3
GM-to-UB and UB-to-GM transfers.

## Requirements

- **CMake** 3.16.8 or newer
- **LLVM and Clang** installed or built from source, with CMake config files available (the directory you pass as `CMAKE_PREFIX_PATH` is usually the LLVM **build** tree, e.g. `llvm-project/build`)
- A **C++ toolchain** that can compile for the machine where you configure ascify (often the same Clang you built LLVM with, or the system compiler)
- **Ninja** (recommended) or another CMake generator
- For translating real CUDA inputs: a **CUDA toolkit** layout on disk (see `--cuda-path`) and Clang’s **resource directory** (see `--clang-resource-directory`) so CUDA headers and builtins resolve correctly

LLVM must be built with the backends and default triple you need (for example **X86** plus a non-empty **`LLVM_DEFAULT_TARGET_TRIPLE`** if you compile on x86_64 Linux). A Clang that only ships AArch64/NVPTX cannot drive native host compiles for CMake’s compiler tests.

## Build

1. Point `LLVM_PROJECT_PATH` (or equivalent) at your **llvm-project** checkout whose **build** directory already contains `find_package(LLVM)` / `find_package(Clang)` metadata.

2. Configure and build:

   ```bash
   export LLVM_PROJECT_PATH=/path/to/llvm-project
   ./build.sh
   ```

`build.sh` configures an out-of-tree `build/`, sets `CMAKE_PREFIX_PATH` to
`$LLVM_PROJECT_PATH/build`, selects compilers via `ASCIFY_CC` / `ASCIFY_CXX`,
and builds with two jobs by default. Override `BUILD_DIR`, `INSTALL_ROOT`,
`LLVM_BUILD_DIR`, or `ASCIFY_BUILD_JOBS` as needed. Override the compiler if
your stage-1 LLVM cannot target the host:

```bash
export ASCIFY_CC=/usr/bin/clang
export ASCIFY_CXX=/usr/bin/clang++
./build.sh
```

Set `ASCIFY_LINKER` only when a specific linker is required. If it is unset,
`build.sh` uses `${LLVM_BUILD_DIR}/bin/lld` when present and otherwise lets the
compiler select the system linker.

Install (optional):

```bash
cmake --install build
```

## Quick start: choose a conversion mode

For the validated pure-SIMT policy, omit the target recipe:

```bash
./build/ascify-clang input.cuh \
  --target-policy=dav-c310-vec \
  --simt-math=fast \
  --cuda-path=/path/to/cuda \
  --clang-resource-directory=/path/to/llvm/lib/clang/23 \
  -o output_simt.cuh
```

For the proof-gated SIMD+SIMT path, add the explicit 950PR recipe:

```bash
./build/ascify-clang input.cuh \
  --target-policy=dav-c310-vec \
  --simt-math=fast \
  --target-recipe=dav-3510-rowwise-simd-v1 \
  --cuda-path=/path/to/cuda \
  --clang-resource-directory=/path/to/llvm/lib/clang/23 \
  -o output_hybrid.cuh
```

The Hybrid output dispatches to separately built target support. Build those
four versioned libraries with the dav-3510 CANN toolchain:

```bash
export ASCEND_HOME_PATH=/path/to/cann
cmake -S runtime/dav_3510/rowwise \
  -B build/rowwise-simd-v1 \
  -DCMAKE_ASC_ARCHITECTURES=dav-3510
cmake --build build/rowwise-simd-v1
```

Source conversion alone does not prove target compilation, device correctness,
route selection, or performance. The complete build/link and 950PR verification
procedure is in the
[row-wise SIMD+SIMT conversion guide](docs/rowwise-simd-conversion.md).

### CMake options

| Option | Default | Meaning |
|--------|---------|--------|
| `ASCIFY_CLANG_TESTS` | OFF | Build Ascify and register the CTest release gate |
| `ASCIFY_CLANG_TESTS_ONLY` | OFF | Register host-only release checks without LLVM/Clang |
| `ASCIFY_INSTALL_CLANG_HEADERS` | ON | Install Clang resource headers with ascify |
| `ASCIFY_INCLUDE_IN_ASCIFY_SDK` | OFF | Windows SDK-style integration (restricted platform) |

## Usage

Typical invocation passes ascify flags first, then **`--`**, then normal Clang flags (omit `--` if there are no extra Clang arguments):

```bash
ascify-clang [ascify-options] -- [clang-options] <inputs>
```

Minimal ingredients for CUDA sources:

- **`--cuda-path=<dir>`** — root of the CUDA installation (headers, `nvvm`, etc.).
- **`--clang-resource-directory=<dir>`** — parent of Clang’s `include/` tree (contains `__clang_cuda_runtime_wrapper.h` and related runtime pieces). This is usually `…/lib/clang/<major>` inside your LLVM build or install.

Example (adjust paths and Clang major version):

```bash
./build/ascify-clang examples/vector_add.cu \
  --cuda-path=/path/to/user-owned/cuda \
  --clang-resource-directory=/path/to/llvm-project/build/lib/clang/23 \
  -o /tmp/vector_add.cpp
```

Where `--clang-resource-directory` specifies `lib/clang/23`, which is in the **install directory of llvm-project**, that is, the `-DCMAKE_INSTALL_PREFIX` of llvm-project. Write to a file with **`-o`**, a directory tree with **`-o-dir`**, or inspect only with **`-examine`** (combines `-no-output` and `-print-stats`).

### Useful ascify options

| Flag | Role |
|------|------|
| `-o`, `-o-dir` | Output file or directory |
| `-inplace` | Rewrite source in place (optional backup unless `-no-backup`) |
| `-cuda-gpu-arch=sm_XX` | GPU arch for CUDA compilation (repeatable) |
| `-experimental` | Allow experimentally supported DPP mappings (otherwise warnings) |
| `-print-stats` / `-print-stats-csv` | Translation statistics |
| `-perl` / `-python` | Emit ascify-perl / Python map artifacts (see also `-o-ascify-perl-dir`, `-o-python-map-dir`) |
| `-md` / `-csv` | Documentation export (`-doc-format` refines layout) |
| `-local-headers` / `-local-headers-recursive` | Process quoted local includes |
| `--target-policy=portable\|dav-c310-vec` | Select conservative output or opt in to validated dav-c310 SIMT rewrites |
| `--simt-math=precise\|fast` | Keep precise semantics or enable guarded fast-SIMT transformations |
| `--target-recipe=none\|dav-3510-rowwise-simd-v1` | Explicitly enable the versioned, externally linked row-wise SIMD+SIMT hybrid dispatch; default `none` |
| `--no-lower-device-double-params` | Preserve by-value scalar `double` parameters on CUDA `__global__` functions |
| `-versions` | Print supported third-party version range |

Full list: run **`ascify-clang --help`**.

`portable` and `precise` are the defaults. The currently validated 950PR
performance policy is explicitly opt-in:

```bash
ascify-clang input.cuh \
  --target-policy=dav-c310-vec \
  --simt-math=fast \
  --cuda-path=/path/to/cuda \
  --clang-resource-directory=/path/to/llvm/lib/clang/23 \
  -o output.cuh
```

Under that policy, Ascify may replace a semantically proven full-mask,
32-lane add-reduction loop with the native warp reduction helper, and may tag
pure binary sum/max/min functors for `aclcub::BlockReduce`. Partial masks,
sub-warps, side effects, dependent calls, unsupported data types, and
unproven functors retain the compatibility fallback.

### Explicit row-wise SIMD+SIMT hybrid dispatch

The dav-3510 row-wise hybrid path has an additional explicit opt-in:

```bash
ascify-clang input.cuh \
  --target-policy=dav-c310-vec \
  --simt-math=fast \
  --target-recipe=dav-3510-rowwise-simd-v1 \
  --cuda-path=/path/to/cuda \
  --clang-resource-directory=/path/to/llvm/lib/clang/23 \
  -o output.cuh
```

`--target-recipe=none` is the default and does not add the externally linked
row-wise target ABI to generated code. The explicit recipe is rejected unless
both `dav-c310-vec` and `fast` are selected.

In explicit mode, Ascify inserts the closed `RowwiseHybridFacadeV1` entry only
when it proves all of the following in the main-file AST:

- an exact packed row-major FP16-to-FP32 load and FP32-to-FP16 direct or
  per-column affine store;
- exact-owner adapter metadata, so derived classes cannot inherit the proof;
- the complete Softmax, RMSNorm, or LayerNorm primitive/data-flow graph, including
  resolved `exp`/divide/`rsqrt` semantics;
- a supported shape/type contract at runtime.

Explicit hybrid mode does not accept `ascify.semantic.*` annotations as the
semantic or status-type proof. It inspects the FP32 helper specialization,
reads the complete source body rather than only the preprocessor-selected AST
branch, and requires every textual conditional branch in that body to contain
only the corresponding `exp`, divide, or `rsqrt` return. It also verifies the
wrapper return type and rejects identity or side-effecting helpers. Reserved
macros, pre-instrumented adapter markers, input-defined dispatch declarations,
and input-defined launch-ABI symbols fail closed. Feature macros that affect a
proved source branch must match between conversion and CCE build. The recorded
release evidence aligns both `OF_*_USE_FAST_MATH` values; unrelated toolkit
version macros may differ only when they select the same validated branch. The
v3 report records the converter arguments and frontend/generator digests.
Non-system GNU/MS assembler is rejected, including assembler tokens in skipped
dependencies, so source cannot alias a protected `_launch_v1` symbol. Injected
headers are macro-shielded; a conventional main-file include guard is
temporarily undefined only around the injected include and restored before the
translated body.

The generated main CCE calls
`RowwiseHybridFacadeV1::TrySoftmaxHybrid` or
`RowwiseHybridFacadeV1::TryRmsNormHybrid` or
`RowwiseHybridFacadeV1::TryLayerNormHybrid` and retains its original
translated SIMT launch. The outer `handled` branch is a pre-launch support gate: a
selector miss continues into that whole-SIMT launch, while a selected call is
owned by one target kernel that uses both execution styles. It is not an
all-SIMD versus all-SIMT operator choice. A selected error is returned and
never launches the fallback a second time.

Inside a selected Softmax kernel, SIMD performs row maximum,
exponentiation/sum reduction, and normalization division. Inside selected
RMSNorm kernels, SIMD performs square/sum reduction, square root, division,
normalization, and the optional affine multiply. For LayerNorm, SIMD performs
mean, centered variance, and reciprocal square root; SIMT applies centered
normalization and output-local indexing. In Softmax and RMSNorm, SIMT performs
contiguous normalized-value staging from UB to the output-local tile. None of
the current recipes provides mask, gather, scatter, stride, or other
non-contiguous adapter lowering.

Mixed kernels require an explicit dynamic-UB launch capacity in the second
launch argument, large enough for the complete `TPipe` allocation. The
current budgets are 163,904 bytes for Softmax recompute, 65,600 bytes for
RMSNorm cached, 147,520 bytes for RMSNorm plain row-batch, and 32,832 bytes for
LayerNorm cached. `0` and `nullptr` are invalid; the host static gate rejects
both forms, and selected-route device correctness covers the current budgets.

The Softmax runtime queries the current device, vector-core count, and maximum
threads per vector core on every selected call. Query failures are propagated
unchanged; the runtime does not substitute a fixed core count or reuse a
process-global result from another device.

The facade reports that decision as `HybridTryResult`, an alias of the legacy
`SimdTryResult { bool handled; aclError status; }` type. ABI v1 exports four C
symbols: `ascify950_softmax_reg_recompute_launch_v1`,
`ascify950_rmsnorm_reg_cached_launch_v1`,
`ascify950_rmsnorm_reg_plain_rowbatch_launch_v1`, and
`ascify950_layernorm_reg_cached_launch_v1`. Their respective DSO linker names
are `libascify950_softmax_reg_recompute_v1.so`,
`libascify950_rmsnorm_reg_cached_v1.so`,
`libascify950_rmsnorm_reg_plain_rowbatch_v1.so`, and
`libascify950_layernorm_reg_cached_v1.so`. Their SONAMEs append `.1` to those
linker names (`VERSION 1.0.0`, `SOVERSION 1`). Each public DSO entry
repeats the corresponding v1 selector-domain check before launching a kernel,
so a direct ABI call cannot bypass the shape, alignment, extent, or aliasing
guard.

The mixed target-support device implementations are built separately from
`runtime/dav_3510/rowwise/` and linked or deployed with the generated main CCE.
Source conversion alone therefore does not establish CANN build, device
correctness, hybrid routing, or performance. RMSNorm affine also requires
converting the caller file that defines its store adapter together with
`layer_norm.cuh` and `rms_norm.cuh`; an unknown external store is never
inferred from field names or layout.

The intra-kernel stage change invalidates predecessor build, correctness, and
performance attribution. Current claims require regenerated Hybrid-facade
output and freshly built, checked, and measured target-support binaries.

For the existing 950PR harness, set `ROWWISE_SIMD_RUNTIME_DIR` to the
unchanged CMake build/install `lib` directory containing all four versioned
`.so` linker-name files, all four `.so.1` SONAME files, and their
implementations. The harness verifies each ELF SONAME and that its linker and
SONAME paths resolve to the same file. The build script links the Softmax check
ELF only to the Softmax DSO and the RMSNorm check ELF only to the two RMSNorm
DSOs; the dedicated LayerNorm check exposes only its LayerNorm launch symbol.
The script rejects any extra or missing row-wise `_launch_v1` dynamic symbol.
The smoke runner prepends the same directory to its process-local
`LD_LIBRARY_PATH`.

See [the explicit row-wise SIMD+SIMT conversion guide](docs/rowwise-simd-conversion.md)
for the ABI, build/link steps, selector domains, verification gates, and
conversion/performance reporting definitions. The explicit activation and AST
proof are recorded in
[ADR-0007](docs/decisions/0007-explicit-ast-gated-rowwise-simd-dispatch.md),
and the same-kernel stage composition is recorded in
[ADR-0008](docs/decisions/0008-compose-rowwise-simd-and-simt-stages-in-one-kernel.md).
The common recipe registry and third-family LayerNorm extension are recorded
in
[ADR-0009](docs/decisions/0009-register-rowwise-hybrid-recipes-and-add-layernorm.md).

## Tests

Run the dependency-free host gate:

```bash
sh tests/run_release_checks.sh
```

It checks rewrite contracts and runs the host Python suites. To re-translate and
compare all golden fixtures, also set `ASCIFY_BINARY`, `ASCIFY_CUDA_PATH`, and
`ASCIFY_CLANG_RESOURCE_DIRECTORY`; see [CONTRIBUTING.md](CONTRIBUTING.md).
The binary-enabled gate also converts the complete OneFlow fixtures through
the generation wrapper, validates the v3 dispatch/ownership/fallback report,
rejects pre-instrumented input, and rejects annotated identity-helper and
non-status-wrapper mutations. It also corrupts each side of a conditional
primitive in turn and verifies that an invalid inactive branch cannot authorize
hybrid conversion. Report acceptance structurally requires every recipe call to
form the generated ownership contract in a distinct wrapper with a later
direct-scope SIMT launch; the generated and input launch counts must match.

The shared 950PR harness and its legacy in-header row-wise recipe replay are
documented in
[tests/softmax_rmsnorm_950/README.md](tests/softmax_rmsnorm_950/README.md).
That older replay is useful as a SIMT baseline but does not enable the new
`--target-recipe` or build the four hybrid target-support DSOs. Use the
explicit conversion guide above for the v1 SIMD+SIMT workflow.

## Examples

- **`examples/vector_add.cu`** — CUDA runtime sample (`cuda_runtime.h`).
- **`examples/vector_add.cu.dpp`** — same logic after ascify-style rewrites toward ACL/DPP-style APIs (`acl/acl.h`, `aclrt*`).

Use these to sanity-check your CUDA path, resource dir, and GPU arch flags.

## Repository layout

| Path | Role |
|------|------|
| `acl_cub/` | The ACL version of the cub library includes the reduction of Warps within the Block |
| `src/` | Frontend action, CUDA→DPP maps, CLI, statistics |
| `include/ascify/` | Thin CUDA-signature compatibility layer installed with Ascify |
| `runtime/dav_3510/rowwise/` | Separately built versioned row-wise SIMD+SIMT target support |
| `tests/rewrite/` | Positive/negative semantic rewrite and compatibility-header gates |
| `tests/softmax_rmsnorm_950/` | Unified 950PR correctness, hardware probe, and benchmark harness |
| `docs/decisions/` | Architecture decisions for conversion and target-policy boundaries |
| `examples/` | Sample CUDA and translated-style sources |
| `build.sh` | Environment-driven out-of-tree configure and build |
| `run.sh` | Environment-driven wrapper for one translation command |
