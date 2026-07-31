# ascify-clang (ASCIFY)

**ascify-clang** is a Clang-based source translator that takes CUDA C/C++ and rewrites it toward **DPP**-oriented APIs and headers (for example `acl/acl.h` and runtime entry points such as `aclrtMalloc`). It is built as a standalone tool (`ascify-clang`) on top of the same LLVM/Clang your project links against.

The upstream Clang driver is still used for parsing and CUDA sema; ascify-specific options control the rewrite, statistics, and auxiliary outputs (Perl/Python maps, Markdown/CSV docs).

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

The same opt-in policy can emit the versioned dav-c310 FP16 row-wise recipe
when it proves all of the following in the main-file AST:

- an exact packed row-major FP16-to-FP32 load and FP32-to-FP16 direct or
  per-column affine store;
- exact-owner adapter metadata, so derived classes cannot inherit the proof;
- the complete Softmax or RMSNorm primitive/data-flow graph, including
  resolved `exp`/divide/`rsqrt` semantics;
- a supported shape/type contract at runtime.

Here, “recipe direct” means an unedited generated header whose proved wrapper
dispatches to the packaged `dav_c310::v1` target implementation. It is distinct
from the legacy generic rewrite and does not claim to synthesize the validated
16-byte, half2, cache, or block-per-row implementation from arbitrary CUDA.

The generated Softmax prologue calls the target recipe at wrapper entry.
RMSNorm preserves its proved launch-geometry block and error return, then calls
the target recipe before the original launch. `NotHandled` preserves the
remaining translated body as fallback. RMSNorm affine requires converting the
caller file that defines its store adapter as well as `layer_norm.cuh` and
`rms_norm.cuh`; an unknown external store is never inferred
from field names or layout.

## Tests

Run the dependency-free host gate:

```bash
sh tests/run_release_checks.sh
```

It checks rewrite contracts and runs 49 Python tests. To re-translate and
compare all golden fixtures, also set `ASCIFY_BINARY`, `ASCIFY_CUDA_PATH`, and
`ASCIFY_CLANG_RESOURCE_DIRECTORY`; see [CONTRIBUTING.md](CONTRIBUTING.md).

The complete two-host 910C conversion and 950PR correctness/performance replay
is documented in
[tests/softmax_rmsnorm_950/README.md](tests/softmax_rmsnorm_950/README.md).

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
| `tests/rewrite/` | Positive/negative semantic rewrite and compatibility-header gates |
| `tests/softmax_rmsnorm_950/` | Unified 950PR correctness, hardware probe, and benchmark harness |
| `docs/decisions/` | Architecture decisions for conversion and target-policy boundaries |
| `examples/` | Sample CUDA and translated-style sources |
| `build.sh` | Environment-driven out-of-tree configure and build |
| `run.sh` | Environment-driven wrapper for one translation command |
