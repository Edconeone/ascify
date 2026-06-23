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

2. From an empty build directory:

   ```bash
   export LLVM_PROJECT_PATH=/path/to/llvm-project
   mkdir -p build && cd build
   bash ../build.sh
   ninja
   ```

`build.sh` sets `CMAKE_PREFIX_PATH` to `$LLVM_PROJECT_PATH/build`, picks compilers via `ASCIFY_CC` / `ASCIFY_CXX` (defaulting to that tree’s `clang`/`clang++`), and configures an install prefix under `ascify_install` next to the build dir. Override the compiler if your stage-1 LLVM cannot target the host:

```bash
export ASCIFY_CC=/usr/bin/clang
export ASCIFY_CXX=/usr/bin/clang++
bash ../build.sh
```

Install (optional):

```bash
ninja install
```

### CMake options

| Option | Default | Meaning |
|--------|---------|--------|
| `ASCIFY_CLANG_TESTS` | OFF | Build lit-based tests (needs `lit`) |
| `ASCIFY_CLANG_TESTS_ONLY` | OFF | Tests only; expects `ascify-clang` already built |
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
  --cuda-path=/usr/local/cuda \
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
| `-versions` | Print supported third-party version range |

Full list: run **`ascify-clang --help`**.

## Examples

- **`examples/vector_add.cu`** — CUDA runtime sample (`cuda_runtime.h`).
- **`examples/vector_add.cu.dpp`** — same logic after ascify-style rewrites toward ACL/DPP-style APIs (`acl/acl.h`, `aclrt*`).

Use these to sanity-check your CUDA path, resource dir, and GPU arch flags.

## Repository layout

| Path | Role |
|------|------|
| `acl_cub/` | The ACL version of the cub library includes the reduction of Warps within the Block |
| `src/` | Frontend action, CUDA→DPP maps, CLI, statistics |
| `examples/` | Sample CUDA and translated-style sources |
| `build.sh` | Reference CMake configure for out-of-tree build |
| `run.sh` | Minimal wrapper example (local paths; copy and edit) |

