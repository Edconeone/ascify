# Softmax / RMSNorm 910C-to-950PR replay

This harness validates forward FP16 Softmax and RMSNorm (plain and affine).
It excludes LogSoftmax, backward kernels, and A800 comparison.

`direct` means an unedited Ascify-generated header whose proved wrapper calls
the versioned `ascify::target::dav_c310::v1` implementation. `native` is a thin
control entry into the same implementation. The comparison proves that a
clean conversion reproduces the deposited target recipe; it does not claim
that Ascify synthesizes the low-level kernel from arbitrary CUDA.

## Repository and work roots

Use one checkout and one ignored work tree on each host:

```bash
export REPO_ROOT="$(pwd -P)"
export WORK_ROOT="${REPO_ROOT}/.work/softmax_rmsnorm_950"
```

All generated headers, converter binaries, build outputs, locks, manifests,
CSV evidence, and logs stay below `WORK_ROOT`. CANN, LLVM, CUDA parsing headers,
drivers, and global shell configuration are external and must not be modified.

The three OneFlow conversion inputs are versioned in
`tests/fixtures/oneflow/`; their origin, local RMSNorm patch, and SHA256 values
are recorded in [the fixture manifest](../fixtures/oneflow/README.md).

## Host-only gate

```bash
sh tests/run_release_checks.sh
```

This runs four static rewrite contracts and 49 Python tests. With a built
translator, it also re-translates the golden fixtures:

```bash
ASCIFY_BINARY=build/ascify-clang \
ASCIFY_CUDA_PATH=/path/to/user-owned/cuda \
ASCIFY_CLANG_RESOURCE_DIRECTORY=ascify_install/include/ascify \
sh tests/run_release_checks.sh
```

## 910C: build and convert

Configure a user-owned LLVM/Clang tree and CUDA parsing layout:

```bash
export LLVM_PROJECT_PATH=/path/to/llvm-project
export CUDA_ROOT=/path/to/user-owned/cuda
export CLANG_RESOURCE_DIRECTORY="${REPO_ROOT}/ascify_install/include/ascify"

./build.sh
cmake --install build --prefix "${REPO_ROOT}/ascify_install"
```

Create a new evidence set. The script refuses to overwrite an existing set:

```bash
WORK_ROOT="${WORK_ROOT}" \
CUDA_ROOT="${CUDA_ROOT}" \
CLANG_RESOURCE_DIRECTORY="${CLANG_RESOURCE_DIRECTORY}" \
tests/softmax_rmsnorm_950/scripts/run_910_conversion_v3.sh
```

The command fixes:

```text
--target-policy=dav-c310-vec
--simt-math=fast
-Iinputs
-std=c++17
```

It packages the exact converter binary, recipe source/header, four inputs,
four outputs, argv, logs, and hashes. The required generated topology is:

| Unit | target include | load marker | store marker | `TrySoftmax` | `TryRmsNorm` |
|---|---:|---:|---:|---:|---:|
| `softmax` | 1 | 1 | 1 | 3 | 0 |
| `layer_norm` | 0 | 1 | 1 | 0 | 0 |
| `rmsnorm` | 1 | 0 | 0 | 0 | 3 |
| `rmsnorm_adapter` | 0 | 0 | 1 | 0 | 0 |

Softmax calls must be in the Warp, BlockSMem, and BlockUncached direct launch
wrappers. RMSNorm calls must be after the complete `GetNumBlocks` geometry and
error-return block and before the unique launch. Dispatchers, LogSoftmax, and
all other locations must contain no recipe call.

Run the fail-close mutation matrix against the same commit and inputs:

```bash
python3 -B tests/rewrite/check_dav_c310_rowwise_mutations.py \
  --ascify build/ascify-clang \
  --softmax tests/fixtures/oneflow/oneflow/core/cuda/softmax.cuh \
  --rms-norm tests/fixtures/oneflow/oneflow/core/cuda/rms_norm.cuh \
  --layer-norm tests/fixtures/oneflow/oneflow/core/cuda/layer_norm.cuh \
  --recipe-source src/DavC310TargetRecipe.cpp \
  --recipe-header src/DavC310TargetRecipe.h \
  --cuda-path "${CUDA_ROOT}" \
  --clang-resource-directory "${CLANG_RESOURCE_DIRECTORY}" \
  --include-dir tests/fixtures/oneflow \
  --work-dir "${WORK_ROOT}/recipe_mutations" \
  --converter-cwd "${REPO_ROOT}"
```

The accepted matrix is 29 Softmax plus 9 RMSNorm cases. Exit status zero means
38/38 passed.

## Transfer the immutable conversion evidence

From a machine with SSH aliases for both hosts:

```bash
export ASCIFY_CONVERT_HOST=910C
export ASCIFY_TEST_HOST=950PR
export CONVERT_REPO=/path/to/ascify-on-910c
export TEST_REPO=/path/to/ascify-on-950pr

ssh "${ASCIFY_CONVERT_HOST}" \
  "tar -C '${CONVERT_REPO}/.work/softmax_rmsnorm_950/conversion' -cf - evidence_v3" |
ssh "${ASCIFY_TEST_HOST}" \
  "mkdir -p '${TEST_REPO}/.work/softmax_rmsnorm_950/conversion' &&
   tar -C '${TEST_REPO}/.work/softmax_rmsnorm_950/conversion' -xf -"
```

Both checkouts must resolve to the same Git commit. Do not edit any generated
header after transfer.

## 950PR: stage, build, verify, and benchmark

Set the user-owned CANN package explicitly:

```bash
export REPO_ROOT="$(pwd -P)"
export WORK_ROOT="${REPO_ROOT}/.work/softmax_rmsnorm_950"
export CANN_ROOT=/path/to/user-owned/cann

mkdir -p "${WORK_ROOT}/generated"
cp -a "${WORK_ROOT}/conversion/evidence_v3/outputs/." \
  "${WORK_ROOT}/generated/"

export ASCIFY_BINARY_SHA256="$(
  sha256sum "${WORK_ROOT}/conversion/evidence_v3/converter/ascify-clang" |
  awk '{print $1}'
)"

FORMAL_TAG=repro_v3 \
ASCIFY_BINARY_SHA256="${ASCIFY_BINARY_SHA256}" \
CANN_ROOT="${CANN_ROOT}" \
WORK_ROOT="${WORK_ROOT}" \
tests/softmax_rmsnorm_950/scripts/run_formal_recipe_v3.sh
```

Do not set `DEVICE`. After building, the script selects and locks a healthy
device with zero compute and HBM-bandwidth utilization.

The formal run fixes:

- direct and native `correctness.csv`: 42 Softmax and 18 RMSNorm cases each;
- direct and native `unified_tune.csv`: 5 Softmax and 10 RMSNorm cases each;
- `WARMUP=20`, `SAMPLES=50`, `INNER_REPEATS=20`;
- direct-A, native, direct-B ordering with no interleaved build or correctness;
- a bounded process-cleanup poll between phases, followed by the unchanged
  strict idle pre-snapshot;
- direct A/B spread at most `1.05`;
- every shape direct/native geometric center at least `0.90`;
- Softmax, RMSNorm plain, and RMSNorm affine group geomean at least `0.95`.

The immutable binary bundle and run manifests are written below:

```text
.work/softmax_rmsnorm_950/results/manifests/
```

Derive ordinary arithmetic throughput and separate SFU call rates:

```bash
python3 -B tests/softmax_rmsnorm_950/tools/derive_work_metrics.py \
  "${WORK_ROOT}/results/perf_history.csv" \
  --shape-manifest tests/softmax_rmsnorm_950/shapes/unified_tune.csv \
  --output "${WORK_ROOT}/results/perf_metrics_v1.csv"
```

The calculation does not count `exp` or `rsqrt` as FLOPs. Probe instructions
are in [probes/README.md](probes/README.md); measured conversion and tuning
results are in
[the tuning report](../../docs/softmax-rmsnorm-950pr-tuning-report.md).
