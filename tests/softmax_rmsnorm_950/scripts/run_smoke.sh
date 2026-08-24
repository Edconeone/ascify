#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEST_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${TEST_ROOT}/../.." && pwd)"

WORK_ROOT="${WORK_ROOT:-${REPO_ROOT}/.work/softmax_rmsnorm_950}"
GENERATED_ROOT="${GENERATED_ROOT:-${WORK_ROOT}/generated}"
BIN_DIR="${BIN_DIR:-${WORK_ROOT}/bin}"
RESULT_DIR="${RESULT_DIR:-${WORK_ROOT}/results}"
CANN_ROOT="${CANN_ROOT:-}"
ROWWISE_SIMD_RUNTIME_DIR="${ROWWISE_SIMD_RUNTIME_DIR:-}"

validate_rowwise_runtime_dir() {
  local runtime_dir="$1"
  local base
  local linker_file
  local soname_file
  local actual_soname
  if ! command -v readelf >/dev/null 2>&1; then
    echo "readelf is required to validate the row-wise SIMD runtime" >&2
    return 1
  fi
  for base in \
    libascify950_softmax_reg_recompute_v1 \
    libascify950_rmsnorm_reg_cached_v1 \
    libascify950_rmsnorm_reg_plain_rowbatch_v1 \
    libascify950_layernorm_reg_cached_v1; do
    linker_file="${runtime_dir}/${base}.so"
    soname_file="${runtime_dir}/${base}.so.1"
    if [[ ! -f "${linker_file}" || ! -f "${soname_file}" ]]; then
      echo "missing row-wise SIMD linker-name or SONAME file:" >&2
      echo "  ${linker_file}" >&2
      echo "  ${soname_file}" >&2
      return 1
    fi
    actual_soname="$(
      readelf -d -- "${linker_file}" |
        sed -n 's/^.*(SONAME).*\[\([^]]*\)\].*$/\1/p'
    )"
    if [[ "${actual_soname}" != "${base}.so.1" ]]; then
      echo "unexpected row-wise SIMD SONAME in ${linker_file}: ${actual_soname}" >&2
      return 1
    fi
    if [[ "$(readlink -f -- "${linker_file}")" != \
          "$(readlink -f -- "${soname_file}")" ]]; then
      echo "row-wise SIMD linker-name and SONAME files resolve differently:" >&2
      echo "  ${linker_file}" >&2
      echo "  ${soname_file}" >&2
      return 1
    fi
  done
}
SHAPES="${SHAPES:-${TEST_ROOT}/shapes/correctness.csv}"
MATH_MODE="${MATH_MODE:-production-fast}"
SOFTMAX_VARIANT="${SOFTMAX_VARIANT:-converted}"
SOFTMAX_BLOCK_THREADS="${SOFTMAX_BLOCK_THREADS:-adaptive}"
SOFTMAX_BLOCK_ROW_MAX="${SOFTMAX_BLOCK_ROW_MAX:-2048}"
RMSNORM_VARIANT="${RMSNORM_VARIANT:-converted}"
RMSNORM_BLOCK_ROW_THREADS="${RMSNORM_BLOCK_ROW_THREADS:-512}"
RMSNORM_BLOCK_ROW_AFFINE_THREADS="${RMSNORM_BLOCK_ROW_AFFINE_THREADS:-256}"
TIER="${TIER:-smoke}"
WARMUP="${WARMUP:-5}"
SAMPLES="${SAMPLES:-20}"
INNER_REPEATS="${INNER_REPEATS:-10}"
CHECK_MAX_ELEMENTS="${CHECK_MAX_ELEMENTS:-4000000}"
OP_FILTER="${OP_FILTER:-both}"
FORMAL_MANIFEST="${FORMAL_MANIFEST:-0}"
FORMAL_SET_ID="${FORMAL_SET_ID:-}"
FORMAL_PHASE="${FORMAL_PHASE:-}"
FORMAL_EXECUTION_KIND="${FORMAL_EXECUTION_KIND:-}"
FORMAL_VARIANT_MODE="${FORMAL_VARIANT_MODE:-}"
FORMAL_EXPECT_SOFTMAX="${FORMAL_EXPECT_SOFTMAX:-}"
FORMAL_EXPECT_RMSNORM="${FORMAL_EXPECT_RMSNORM:-}"
ASCIFY_CONVERSION_MANIFEST="${ASCIFY_CONVERSION_MANIFEST:-${WORK_ROOT}/conversion/evidence_v3/manifest.json}"
FORMAL_BUILD_INPUT_SNAPSHOT="${FORMAL_BUILD_INPUT_SNAPSHOT:-}"
FORMAL_BINARY_BUNDLE_MANIFEST="${FORMAL_BINARY_BUNDLE_MANIFEST:-}"
FORMAL_BOUNDARY_SHAPES="${FORMAL_BOUNDARY_SHAPES:-}"
FORMAL_TUNE_SHAPES="${FORMAL_TUNE_SHAPES:-}"
FORMAL_DEVICE_UTIL_MAX="${FORMAL_DEVICE_UTIL_MAX:-}"
FORMAL_DEVICE_HBM_BW_MAX="${FORMAL_DEVICE_HBM_BW_MAX:-}"

case "${OP_FILTER}" in
  both)
    RUN_SOFTMAX=1
    RUN_RMSNORM=1
    SELECTED_OPS="softmax and rms_norm"
    ;;
  softmax)
    RUN_SOFTMAX=1
    RUN_RMSNORM=0
    SELECTED_OPS="softmax"
    ;;
  rms_norm)
    RUN_SOFTMAX=0
    RUN_RMSNORM=1
    SELECTED_OPS="rms_norm"
    ;;
  *)
    echo "OP_FILTER must be both, softmax, or rms_norm" >&2
    exit 2
    ;;
esac

if [[ -z "${CANN_ROOT}" ]]; then
  echo "CANN_ROOT must point to a user-owned CANN installation" >&2
  exit 2
fi

if [[ -n "${ROWWISE_SIMD_RUNTIME_DIR}" ]]; then
  validate_rowwise_runtime_dir "${ROWWISE_SIMD_RUNTIME_DIR}"
  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH="${ROWWISE_SIMD_RUNTIME_DIR}:${LD_LIBRARY_PATH}"
  else
    export LD_LIBRARY_PATH="${ROWWISE_SIMD_RUNTIME_DIR}"
  fi
fi

if [[ -z "${ASCIFY_DEVICE:-}" || -z "${ASCIFY_DEVICE_LOCK:-}" ]]; then
  # select_device holds its project-local flock across this exec and all children.
  REQUESTED_DEVICE="${ASCIFY_DEVICE:-${DEVICE:-}}"
  unset ASCIFY_DEVICE ASCIFY_DEVICE_LOCK
  if [[ "${FORMAL_MANIFEST}" == "1" ]]; then
    if [[ -n "${REQUESTED_DEVICE}" ]]; then
      exec env UTIL_MAX=0 HBM_BW_MAX=0 DEVICE="${REQUESTED_DEVICE}" \
        "${SCRIPT_DIR}/select_device.sh" -- "$0" "$@"
    fi
    exec env UTIL_MAX=0 HBM_BW_MAX=0 \
      "${SCRIPT_DIR}/select_device.sh" -- "$0" "$@"
  fi
  if [[ -n "${REQUESTED_DEVICE}" ]]; then
    DEVICE="${REQUESTED_DEVICE}" exec "${SCRIPT_DIR}/select_device.sh" -- "$0" "$@"
  fi
  exec "${SCRIPT_DIR}/select_device.sh" -- "$0" "$@"
fi
if [[ ! "${ASCIFY_DEVICE}" =~ ^[0-9]+$ ]]; then
  echo "ASCIFY_DEVICE must be a non-negative integer" >&2
  exit 2
fi
if [[ ! "${CHECK_MAX_ELEMENTS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "CHECK_MAX_ELEMENTS must be a positive integer" >&2
  exit 2
fi
if [[ ! "${FORMAL_MANIFEST}" =~ ^[01]$ ]]; then
  echo "FORMAL_MANIFEST must be 0 or 1" >&2
  exit 2
fi
if [[ "${FORMAL_MANIFEST}" == "1"
      && ( "${FORMAL_DEVICE_UTIL_MAX}" != "0"
           || "${FORMAL_DEVICE_HBM_BW_MAX}" != "0" ) ]]; then
  echo "formal runs require zero compute and HBM bandwidth thresholds" >&2
  exit 2
fi

MODE_SUFFIX="${MATH_MODE//-/_}"
case "${SOFTMAX_VARIANT}" in
  converted) SOFTMAX_BIN_SUFFIX="" ;;
  native-half2-compact) SOFTMAX_BIN_SUFFIX="_native_half2_compact" ;;
  native-half2-hybrid) SOFTMAX_BIN_SUFFIX="_native_half2_hybrid" ;;
  *)
    echo "unsupported SOFTMAX_VARIANT: ${SOFTMAX_VARIANT}" >&2
    exit 2
    ;;
esac
case "${RMSNORM_VARIANT}" in
  converted) RMSNORM_BIN_SUFFIX="" ;;
  native)
    case "${RMSNORM_BLOCK_ROW_THREADS}" in
      128|256|512|1024) ;;
      *)
        echo "RMSNORM_BLOCK_ROW_THREADS must be 128, 256, 512, or 1024" >&2
        exit 2
        ;;
    esac
    case "${RMSNORM_BLOCK_ROW_AFFINE_THREADS}" in
      128|256|512|1024) ;;
      *)
        echo "RMSNORM_BLOCK_ROW_AFFINE_THREADS must be 128, 256, 512, or 1024" >&2
        exit 2
        ;;
    esac
    RMSNORM_BIN_SUFFIX="_native"
    ;;
  *)
    echo "unsupported RMSNORM_VARIANT: ${RMSNORM_VARIANT}" >&2
    exit 2
    ;;
esac
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_softmax_rmsnorm_${TIER}_${MODE_SUFFIX}_$$}"
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "RUN_ID may contain only letters, digits, '.', '_', and '-': ${RUN_ID}" >&2
  exit 2
fi
ACCURACY_CSV="${ACCURACY_CSV:-${RESULT_DIR}/accuracy_history_v2.csv}"
PERF_CSV="${PERF_CSV:-${RESULT_DIR}/perf_history.csv}"
MANIFEST_DIR="${RESULT_DIR}/manifests"
MANIFEST="${MANIFEST_DIR}/${RUN_ID}.tsv"
EVIDENCE="${MANIFEST_DIR}/${RUN_ID}.csv"
DEVICE_SNAPSHOT="${MANIFEST_DIR}/${RUN_ID}.device.txt"
RUNTIME_CONFIG="${MANIFEST_DIR}/${RUN_ID}.runtime_config.tsv"
MANIFEST_OUTPUT="${MANIFEST}"
EVIDENCE_OUTPUT="${EVIDENCE}"
DEVICE_SNAPSHOT_OUTPUT="${DEVICE_SNAPSHOT}"
RUNTIME_CONFIG_OUTPUT="${RUNTIME_CONFIG}"
if [[ "${FORMAL_MANIFEST}" == "1" ]]; then
  MANIFEST_OUTPUT="${MANIFEST}.partial.$$"
  EVIDENCE_OUTPUT="${EVIDENCE}.partial.$$"
  DEVICE_SNAPSHOT_OUTPUT="${DEVICE_SNAPSHOT}.partial.$$"
  RUNTIME_CONFIG_OUTPUT="${RUNTIME_CONFIG}.partial.$$"
fi

mkdir -p "${RESULT_DIR}"

if [[ "${FORMAL_MANIFEST}" == "1" ]]; then
  if [[ -z "${FORMAL_SET_ID}" || ! "${FORMAL_SET_ID}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "FORMAL_SET_ID is required and must be a safe identifier" >&2
    exit 2
  fi
  if [[ -z "${FORMAL_PHASE}" || ! "${FORMAL_PHASE}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "FORMAL_PHASE is required and must be a safe identifier" >&2
    exit 2
  fi
  case "${FORMAL_EXECUTION_KIND}" in
    accuracy)
      if [[ "${CHECK_ONLY:-0}" != "1" || "${SKIP_CHECK:-0}" == "1" ]]; then
        echo "formal accuracy runs require CHECK_ONLY=1 and SKIP_CHECK=0" >&2
        exit 2
      fi
      ;;
    perf)
      if [[ "${SKIP_CHECK:-0}" != "1" || "${CHECK_ONLY:-0}" == "1" ]]; then
        echo "formal perf runs require SKIP_CHECK=1 and CHECK_ONLY=0" >&2
        exit 2
      fi
      ;;
    *)
      echo "FORMAL_EXECUTION_KIND must be accuracy or perf" >&2
      exit 2
      ;;
  esac
  case "${FORMAL_VARIANT_MODE}" in
    direct)
      if [[ "${SOFTMAX_VARIANT}" != "converted"
            || "${RMSNORM_VARIANT}" != "converted" ]]; then
        echo "FORMAL_VARIANT_MODE=direct requires both converted variants" >&2
        exit 2
      fi
      ;;
    native)
      if [[ "${SOFTMAX_VARIANT}" != "native-half2-hybrid"
            || "${RMSNORM_VARIANT}" != "native" ]]; then
        echo "FORMAL_VARIANT_MODE=native requires native-half2-hybrid/native" >&2
        exit 2
      fi
      ;;
    *)
      echo "FORMAL_VARIANT_MODE must be direct or native" >&2
      exit 2
      ;;
  esac
  if [[ ! "${FORMAL_EXPECT_SOFTMAX}" =~ ^[1-9][0-9]*$
        || ! "${FORMAL_EXPECT_RMSNORM}" =~ ^[1-9][0-9]*$ ]]; then
    echo "formal expected row counts must be positive integers" >&2
    exit 2
  fi
  if [[ ! "${ASCIFY_BINARY_SHA256:-}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "formal runs require ASCIFY_BINARY_SHA256 as 64 lowercase hex digits" >&2
    exit 2
  fi
  if [[ -z "${ASCIFY_CONVERSION_SET_ID:-}"
        || ! "${ASCIFY_CONVERSION_SET_ID}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "formal runs require a safe ASCIFY_CONVERSION_SET_ID" >&2
    exit 2
  fi
  for conversion_digest in \
    "${ASCIFY_CONVERSION_MANIFEST_SHA256:-}" \
    "${ASCIFY_INPUT_SOFTMAX_SHA256:-}" \
    "${ASCIFY_INPUT_LAYER_NORM_SHA256:-}" \
    "${ASCIFY_INPUT_RMSNORM_SHA256:-}" \
    "${ASCIFY_INPUT_RMSNORM_ADAPTER_SHA256:-}"; do
    if [[ ! "${conversion_digest}" =~ ^[0-9a-f]{64}$ ]]; then
      echo "formal conversion evidence digests must be 64 lowercase hex digits" >&2
      exit 2
    fi
  done
  if [[ "${SOFTMAX_BLOCK_THREADS}" != "adaptive"
        || "${SOFTMAX_BLOCK_ROW_MAX}" != "2048"
        || "${RMSNORM_BLOCK_ROW_THREADS}" != "512"
        || "${RMSNORM_BLOCK_ROW_AFFINE_THREADS}" != "256" ]]; then
    echo "formal v3 effective tuning is fixed to adaptive/2048/512/256" >&2
    exit 2
  fi
  if [[ ! -f "${ASCIFY_CONVERSION_MANIFEST}" ]]; then
    echo "missing portable Ascify conversion manifest: ${ASCIFY_CONVERSION_MANIFEST}" >&2
    exit 1
  fi
  if [[ -z "${FORMAL_BUILD_INPUT_SNAPSHOT}"
        || ! -f "${FORMAL_BUILD_INPUT_SNAPSHOT}" ]]; then
    echo "formal runs require FORMAL_BUILD_INPUT_SNAPSHOT" >&2
    exit 1
  fi
  for formal_file in \
    "${FORMAL_BINARY_BUNDLE_MANIFEST}" \
    "${FORMAL_BOUNDARY_SHAPES}" \
    "${FORMAL_TUNE_SHAPES}"; do
    if [[ -z "${formal_file}" || ! -f "${formal_file}" ]]; then
      echo "formal runs require binary-bundle and canonical shape artifacts" >&2
      exit 1
    fi
  done
  if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required for formal result validation" >&2
    exit 1
  fi
  if ! command -v flock >/dev/null 2>&1; then
    echo "flock is required for formal result validation" >&2
    exit 1
  fi
  if ! { true >&9; } 2>/dev/null || ! flock -n 9; then
    echo "formal runs require the inherited select_device fd 9 lock" >&2
    exit 1
  fi
  if [[ -z "${FORMAL_RESULT_LOCK:-}" ]] \
      || ! { true >&8; } 2>/dev/null || ! flock -n 8; then
    echo "formal runs require the inherited shared-result fd 8 lock" >&2
    exit 1
  fi
  "${SCRIPT_DIR}/validate_formal_run.py" lock \
    --fd 9 \
    --declared "${ASCIFY_DEVICE_LOCK}" \
    --expected "${WORK_ROOT}/locks/device-${ASCIFY_DEVICE}.lock"
  "${SCRIPT_DIR}/validate_formal_run.py" lock \
    --fd 8 \
    --declared "${FORMAL_RESULT_LOCK}" \
    --expected "${RESULT_DIR}/.formal-results.lock"
  "${SCRIPT_DIR}/validate_formal_run.py" freeze \
    --formal-contract \
    --check "${FORMAL_BUILD_INPUT_SNAPSHOT}"
  "${SCRIPT_DIR}/validate_formal_run.py" binary-bundle-check \
    --manifest "${FORMAL_BINARY_BUNDLE_MANIFEST}" \
    --formal-set-id "${FORMAL_SET_ID}" \
    --build-input-snapshot "${FORMAL_BUILD_INPUT_SNAPSHOT}"
  "${SCRIPT_DIR}/validate_formal_run.py" preflight \
    --accuracy-csv "${ACCURACY_CSV}" \
    --perf-csv "${PERF_CSV}" \
    --manifest-dir "${MANIFEST_DIR}" \
    --run-id "${RUN_ID}"
  "${SCRIPT_DIR}/validate_formal_run.py" run-contract \
    --formal-phase "${FORMAL_PHASE}" \
    --execution-kind "${FORMAL_EXECUTION_KIND}" \
    --shapes "${SHAPES}" \
    --tier "${TIER}" \
    --variant-mode "${FORMAL_VARIANT_MODE}" \
    --math-mode "${MATH_MODE}" \
    --expected-softmax "${FORMAL_EXPECT_SOFTMAX}" \
    --expected-rmsnorm "${FORMAL_EXPECT_RMSNORM}" \
    --warmup "${WARMUP}" \
    --samples "${SAMPLES}" \
    --inner-repeats "${INNER_REPEATS}" \
    --check-max-elements "${CHECK_MAX_ELEMENTS}" \
    --device-util-max "${FORMAL_DEVICE_UTIL_MAX}" \
    --device-hbm-bw-max "${FORMAL_DEVICE_HBM_BW_MAX}"
fi

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  "${SCRIPT_DIR}/build.sh" all "${MATH_MODE}"
fi

CHECK_BINARY="${BIN_DIR}/softmax_check_${MODE_SUFFIX}${SOFTMAX_BIN_SUFFIX}"
BENCH_BINARY="${BIN_DIR}/softmax_bench_${MODE_SUFFIX}${SOFTMAX_BIN_SUFFIX}"
RMS_CHECK_BINARY="${BIN_DIR}/rmsnorm_check_${MODE_SUFFIX}${RMSNORM_BIN_SUFFIX}"
RMS_BENCH_BINARY="${BIN_DIR}/rmsnorm_bench_${MODE_SUFFIX}${RMSNORM_BIN_SUFFIX}"
if [[ "${SKIP_CHECK:-0}" != "1"
      && ( ( "${RUN_SOFTMAX}" == "1" && ! -x "${CHECK_BINARY}" )
           || ( "${RUN_RMSNORM}" == "1" && ! -x "${RMS_CHECK_BINARY}" ) ) ]]; then
  echo "correctness binaries are missing; expected:" >&2
  if [[ "${RUN_SOFTMAX}" == "1" ]]; then echo "  ${CHECK_BINARY}" >&2; fi
  if [[ "${RUN_RMSNORM}" == "1" ]]; then echo "  ${RMS_CHECK_BINARY}" >&2; fi
  exit 1
fi
if [[ "${CHECK_ONLY:-0}" != "1"
      && ( ( "${RUN_SOFTMAX}" == "1" && ! -x "${BENCH_BINARY}" )
           || ( "${RUN_RMSNORM}" == "1" && ! -x "${RMS_BENCH_BINARY}" ) ) ]]; then
  echo "benchmark binaries are missing; expected:" >&2
  if [[ "${RUN_SOFTMAX}" == "1" ]]; then echo "  ${BENCH_BINARY}" >&2; fi
  if [[ "${RUN_RMSNORM}" == "1" ]]; then echo "  ${RMS_BENCH_BINARY}" >&2; fi
  exit 1
fi
if [[ "${CHECK_ONLY:-0}" == "1" && "${SKIP_CHECK:-0}" == "1" ]]; then
  echo "CHECK_ONLY=1 and SKIP_CHECK=1 cannot be used together" >&2
  exit 2
fi
if [[ ! -f "${CANN_ROOT}/set_env.sh" ]]; then
  echo "missing user-owned CANN environment: ${CANN_ROOT}/set_env.sh" >&2
  exit 1
fi
if [[ ! -f "${SHAPES}" ]]; then
  echo "missing shapes file: ${SHAPES}" >&2
  exit 1
fi
if ! command -v sha256sum >/dev/null 2>&1; then
  echo "sha256sum is required to create the run manifest" >&2
  exit 1
fi

manifest_value() {
  if [[ "$1" == *$'\t'* || "$1" == *$'\n'* || "$2" == *$'\t'* || "$2" == *$'\n'* ]]; then
    echo "manifest values must be single-line TSV-safe strings" >&2
    return 1
  fi
  printf 'value\t%s\t%s\t\n' "$1" "$2"
}

manifest_file() {
  local label="$1"
  local path="$2"
  local digest_line
  local digest
  [[ -f "${path}" ]] || return 0
  digest_line="$(sha256sum -- "${path}")"
  digest="${digest_line%% *}"
  printf 'file\t%s\t%s\t%s\n' "${label}" "${path}" "${digest}"
}

manifest_required_file() {
  local label="$1"
  local path="$2"
  if [[ ! -f "${path}" ]]; then
    echo "missing required formal manifest artifact: ${label}=${path}" >&2
    return 1
  fi
  manifest_file "${label}" "${path}"
}

manifest_common_file() {
  if [[ "${FORMAL_MANIFEST}" == "1" ]]; then
    manifest_required_file "$1" "$2"
  else
    manifest_file "$1" "$2"
  fi
}

manifest_file_as() {
  local label="$1"
  local recorded_path="$2"
  local hash_path="$3"
  local digest_line
  local digest
  if [[ ! -f "${hash_path}" ]]; then
    echo "missing manifest hash source: ${label}=${hash_path}" >&2
    return 1
  fi
  digest_line="$(sha256sum -- "${hash_path}")"
  digest="${digest_line%% *}"
  printf 'file\t%s\t%s\t%s\n' "${label}" "${recorded_path}" "${digest}"
}

SOFTMAX_NATIVE_SOURCE="${TEST_ROOT}/kernels/softmax_native_kernel.cce"
RMSNORM_NATIVE_SOURCE="${TEST_ROOT}/kernels/rmsnorm_native_kernel.cce"
GENERATED_SOFTMAX="${GENERATED_ROOT}/oneflow/core/cuda/softmax.cuh"
GENERATED_RMSNORM="${GENERATED_ROOT}/oneflow/core/cuda/rms_norm.cuh"
GENERATED_LAYER_NORM="${GENERATED_ROOT}/oneflow/core/cuda/layer_norm.cuh"
GENERATED_RMSNORM_ADAPTER="${GENERATED_ROOT}/ascify950/rmsnorm_affine_store.cuh"
TARGET_RECIPE_HEADER="${REPO_ROOT}/include/ascify/target/dav_c310/rowwise_norm_recipes.hpp"
TARGET_SOFTMAX_IMPL="${REPO_ROOT}/include/ascify/target/dav_c310/detail/softmax_fp16_impl.hpp"
TARGET_RMSNORM_IMPL="${REPO_ROOT}/include/ascify/target/dav_c310/detail/rmsnorm_fp16_impl.hpp"
ROWWISE_SIMD_ABI_HEADER="${REPO_ROOT}/include/ascify/target/dav_c310/rowwise_simd_abi.h"
ROWWISE_SIMD_SELECTORS_HEADER="${REPO_ROOT}/include/ascify/target/dav_c310/rowwise_simd_selectors_v1.hpp"
ROWWISE_SIMD_RECIPES_HEADER="${REPO_ROOT}/include/ascify/target/dav_c310/rowwise_simd_recipes.hpp"
ACLCUB_HEADER="${REPO_ROOT}/acl_cub/aclcub.hpp"
ASCIFY_RECIPE_SOURCE="${REPO_ROOT}/src/DavC310TargetRecipe.cpp"
ASCIFY_RECIPE_HEADER="${REPO_ROOT}/src/DavC310TargetRecipe.h"
RMSNORM_ADAPTER_INPUT="${TEST_ROOT}/inputs/rmsnorm_affine_store.cuh"
RUNNER_COMMON_HEADER="${TEST_ROOT}/common/runner_common.hpp"
SOFTMAX_CHECK_SOURCE="${TEST_ROOT}/softmax_check.cce"
RMSNORM_CHECK_SOURCE="${TEST_ROOT}/rmsnorm_check.cce"
SOFTMAX_BENCH_SOURCE="${TEST_ROOT}/softmax_bench.cce"
RMSNORM_BENCH_SOURCE="${TEST_ROOT}/rmsnorm_bench.cce"
BUILD_SCRIPT="${SCRIPT_DIR}/build.sh"
RUN_SMOKE_SCRIPT="${SCRIPT_DIR}/run_smoke.sh"
SELECT_DEVICE_SCRIPT="${SCRIPT_DIR}/select_device.sh"
FORMAL_RUNNER_SCRIPT="${SCRIPT_DIR}/run_formal_recipe_v3.sh"
VALIDATOR_SCRIPT="${SCRIPT_DIR}/validate_formal_run.py"
VALIDATOR_TEST="${SCRIPT_DIR}/test_validate_formal_run.py"
SAFE_LOCK_HELPER="${SCRIPT_DIR}/safe_lock_exec.py"
DERIVE_WORK_METRICS_TOOL="${TEST_ROOT}/tools/derive_work_metrics.py"
DERIVE_WORK_METRICS_TEST="${TEST_ROOT}/tools/test_derive_work_metrics.py"
CANN_SET_ENV="${CANN_ROOT}/set_env.sh"
CCEC_BINARY="${CCEC:-${CANN_ROOT}/tools/bisheng_compiler/bin/ccec}"
CCEC_VERSION="unavailable"
if [[ -x "${CCEC_BINARY}" ]]; then
  CCEC_VERSION="$(
    set +u
    # shellcheck disable=SC1090
    source "${CANN_SET_ENV}" >/dev/null
    "${CCEC_BINARY}" --version 2>&1 | sed -n '1p'
  )" || CCEC_VERSION="unavailable"
fi
if [[ "${FORMAL_MANIFEST}" == "1" ]]; then
  "${VALIDATOR_SCRIPT}" conversion \
    --manifest "${ASCIFY_CONVERSION_MANIFEST}" \
    --ascify-binary-sha256 "${ASCIFY_BINARY_SHA256}" \
    --recipe-source "${ASCIFY_RECIPE_SOURCE}" \
    --recipe-header "${ASCIFY_RECIPE_HEADER}" \
    --staged-softmax "${GENERATED_SOFTMAX}" \
    --staged-layer-norm "${GENERATED_LAYER_NORM}" \
    --staged-rmsnorm "${GENERATED_RMSNORM}" \
    --staged-rmsnorm-adapter "${GENERATED_RMSNORM_ADAPTER}"
fi
RUN_START_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
RUN_HOSTNAME="$(hostname)"

mkdir -p -- "${MANIFEST_DIR}"
capture_device_state() {
  local phase="$1"
  printf 'phase=%s\n' "${phase}"
  printf 'utc_time=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'hostname=%s\n' "${RUN_HOSTNAME}"
  printf 'run_id=%s\n' "${RUN_ID}"
  printf 'device=%s\n' "${ASCIFY_DEVICE}"
  printf '[mapping]\n'
  npu-smi info -m
  printf '[health]\n'
  npu-smi info -t health -i "${ASCIFY_DEVICE}"
  printf '[usages]\n'
  npu-smi info -t usages -i "${ASCIFY_DEVICE}"
  printf '[processes]\n'
  npu-smi info -t proc-mem -i "${ASCIFY_DEVICE}"
}
if [[ "${FORMAL_MANIFEST}" == "1" ]]; then
  capture_device_state pre > "${DEVICE_SNAPSHOT_OUTPUT}"
  "${VALIDATOR_SCRIPT}" device-snapshot \
    --snapshot "${DEVICE_SNAPSHOT_OUTPUT}" \
    --device "${ASCIFY_DEVICE}" \
    --run-id "${RUN_ID}" \
    --hostname "${RUN_HOSTNAME}" \
    --phase pre
fi
{
  printf 'kind\tname\tvalue_or_path\tsha256\n'
  manifest_value manifest_schema_version "2"
  manifest_value formal_manifest "${FORMAL_MANIFEST}"
  manifest_value run_id "${RUN_ID}"
  manifest_value formal_set_id "${FORMAL_SET_ID:-none}"
  manifest_value formal_phase "${FORMAL_PHASE:-none}"
  manifest_value execution_kind "${FORMAL_EXECUTION_KIND:-mixed}"
  manifest_value expected_softmax_rows "${FORMAL_EXPECT_SOFTMAX:-0}"
  manifest_value expected_rmsnorm_rows "${FORMAL_EXPECT_RMSNORM:-0}"
  manifest_value utc_time "${RUN_START_UTC}"
  manifest_value hostname "${RUN_HOSTNAME}"
  manifest_value device "${ASCIFY_DEVICE}"
  manifest_value device_lock "${ASCIFY_DEVICE_LOCK:-unknown}"
  manifest_value formal_result_lock "${FORMAL_RESULT_LOCK:-unknown}"
  manifest_value device_util_max "${FORMAL_DEVICE_UTIL_MAX:-unknown}"
  manifest_value device_hbm_bw_max "${FORMAL_DEVICE_HBM_BW_MAX:-unknown}"
  manifest_value cann_root "${CANN_ROOT}"
  manifest_value ccec_version "${CCEC_VERSION}"
  manifest_value work_root "${WORK_ROOT}"
  manifest_value generated_root "${GENERATED_ROOT}"
  manifest_value ascify_binary_sha256 "${ASCIFY_BINARY_SHA256:-unknown}"
  manifest_value ascify_target_policy "dav-c310-vec"
  manifest_value ascify_simt_math "fast"
  manifest_value ascify_conversion_set_id "${ASCIFY_CONVERSION_SET_ID:-unknown}"
  manifest_value ascify_input_softmax_sha256 "${ASCIFY_INPUT_SOFTMAX_SHA256:-unknown}"
  manifest_value ascify_input_layer_norm_sha256 \
    "${ASCIFY_INPUT_LAYER_NORM_SHA256:-unknown}"
  manifest_value ascify_input_rmsnorm_sha256 "${ASCIFY_INPUT_RMSNORM_SHA256:-unknown}"
  manifest_value ascify_input_rmsnorm_adapter_sha256 \
    "${ASCIFY_INPUT_RMSNORM_ADAPTER_SHA256:-unknown}"
  manifest_value ascify_conversion_manifest_sha256 \
    "${ASCIFY_CONVERSION_MANIFEST_SHA256:-unknown}"
  manifest_value bin_dir "${BIN_DIR}"
  manifest_value result_dir "${RESULT_DIR}"
  manifest_value accuracy_csv "${ACCURACY_CSV}"
  manifest_value perf_csv "${PERF_CSV}"
  manifest_value math_mode "${MATH_MODE}"
  manifest_value softmax_variant "${SOFTMAX_VARIANT}"
  manifest_value softmax_block_threads "${SOFTMAX_BLOCK_THREADS}"
  manifest_value softmax_block_row_max "${SOFTMAX_BLOCK_ROW_MAX}"
  manifest_value softmax_grid_cap "auto-aiv-x32"
  manifest_value rmsnorm_variant "${RMSNORM_VARIANT}"
  manifest_value rmsnorm_block_row_threads "${RMSNORM_BLOCK_ROW_THREADS}"
  manifest_value rmsnorm_block_row_affine_threads "${RMSNORM_BLOCK_ROW_AFFINE_THREADS}"
  manifest_value rmsnorm_grid_cap "auto-aiv-x32"
  manifest_value op_filter "${OP_FILTER}"
  manifest_value tier "${TIER}"
  manifest_value skip_build "${SKIP_BUILD:-0}"
  manifest_value skip_check "${SKIP_CHECK:-0}"
  manifest_value check_only "${CHECK_ONLY:-0}"
  manifest_value warmup "${WARMUP}"
  manifest_value samples "${SAMPLES}"
  manifest_value inner_repeats "${INNER_REPEATS}"
  manifest_value check_max_elements "${CHECK_MAX_ELEMENTS}"
  manifest_common_file shapes "${SHAPES}"
  if [[ "${SKIP_CHECK:-0}" != "1" ]]; then
    if [[ "${RUN_SOFTMAX}" == "1" ]]; then
      manifest_file softmax_check_binary "${CHECK_BINARY}"
    fi
    if [[ "${RUN_RMSNORM}" == "1" ]]; then
      manifest_file rmsnorm_check_binary "${RMS_CHECK_BINARY}"
    fi
  fi
  if [[ "${CHECK_ONLY:-0}" != "1" ]]; then
    if [[ "${RUN_SOFTMAX}" == "1" ]]; then
      manifest_file softmax_bench_binary "${BENCH_BINARY}"
    fi
    if [[ "${RUN_RMSNORM}" == "1" ]]; then
      manifest_file rmsnorm_bench_binary "${RMS_BENCH_BINARY}"
    fi
  fi
  manifest_common_file ccec_binary "${CCEC_BINARY}"
  manifest_common_file cann_set_env "${CANN_SET_ENV}"
  manifest_common_file softmax_native_source "${SOFTMAX_NATIVE_SOURCE}"
  manifest_common_file rmsnorm_native_source "${RMSNORM_NATIVE_SOURCE}"
  manifest_common_file generated_softmax_header "${GENERATED_SOFTMAX}"
  manifest_common_file generated_rmsnorm_header "${GENERATED_RMSNORM}"
  manifest_common_file generated_layer_norm_header "${GENERATED_LAYER_NORM}"
  manifest_common_file generated_rmsnorm_adapter "${GENERATED_RMSNORM_ADAPTER}"
  manifest_common_file rmsnorm_adapter_input "${RMSNORM_ADAPTER_INPUT}"
  manifest_common_file dav_c310_recipe_header "${TARGET_RECIPE_HEADER}"
  manifest_common_file dav_c310_softmax_impl "${TARGET_SOFTMAX_IMPL}"
  manifest_common_file dav_c310_rmsnorm_impl "${TARGET_RMSNORM_IMPL}"
  manifest_common_file rowwise_simd_abi_header "${ROWWISE_SIMD_ABI_HEADER}"
  manifest_common_file rowwise_simd_selectors_header "${ROWWISE_SIMD_SELECTORS_HEADER}"
  manifest_common_file rowwise_simd_recipes_header "${ROWWISE_SIMD_RECIPES_HEADER}"
  if [[ -n "${ROWWISE_SIMD_RUNTIME_DIR}" ]]; then
    manifest_common_file rowwise_simd_softmax_runtime_dso \
      "$(readlink -f -- "${ROWWISE_SIMD_RUNTIME_DIR}/libascify950_softmax_reg_recompute_v1.so")"
    manifest_common_file rowwise_simd_rmsnorm_cached_runtime_dso \
      "$(readlink -f -- "${ROWWISE_SIMD_RUNTIME_DIR}/libascify950_rmsnorm_reg_cached_v1.so")"
    manifest_common_file rowwise_simd_rmsnorm_plain_runtime_dso \
      "$(readlink -f -- "${ROWWISE_SIMD_RUNTIME_DIR}/libascify950_rmsnorm_reg_plain_rowbatch_v1.so")"
  fi
  manifest_common_file ascify_recipe_source "${ASCIFY_RECIPE_SOURCE}"
  manifest_common_file ascify_recipe_header "${ASCIFY_RECIPE_HEADER}"
  manifest_common_file runner_common_header "${RUNNER_COMMON_HEADER}"
  if [[ "${SKIP_CHECK:-0}" != "1" ]]; then
    manifest_common_file softmax_check_source "${SOFTMAX_CHECK_SOURCE}"
    manifest_common_file rmsnorm_check_source "${RMSNORM_CHECK_SOURCE}"
  fi
  if [[ "${CHECK_ONLY:-0}" != "1" ]]; then
    manifest_common_file softmax_bench_source "${SOFTMAX_BENCH_SOURCE}"
    manifest_common_file rmsnorm_bench_source "${RMSNORM_BENCH_SOURCE}"
  fi
  manifest_common_file build_script "${BUILD_SCRIPT}"
  manifest_common_file run_smoke_script "${RUN_SMOKE_SCRIPT}"
  manifest_common_file select_device_script "${SELECT_DEVICE_SCRIPT}"
  manifest_common_file formal_runner_script "${FORMAL_RUNNER_SCRIPT}"
  manifest_common_file validator_script "${VALIDATOR_SCRIPT}"
  manifest_common_file validator_test "${VALIDATOR_TEST}"
  manifest_common_file safe_lock_helper "${SAFE_LOCK_HELPER}"
  manifest_common_file derive_work_metrics_tool "${DERIVE_WORK_METRICS_TOOL}"
  manifest_common_file derive_work_metrics_test "${DERIVE_WORK_METRICS_TEST}"
  manifest_common_file ascify_conversion_manifest "${ASCIFY_CONVERSION_MANIFEST}"
  manifest_common_file formal_build_input_snapshot \
    "${FORMAL_BUILD_INPUT_SNAPSHOT}"
  manifest_common_file formal_binary_bundle_manifest \
    "${FORMAL_BINARY_BUNDLE_MANIFEST}"
  manifest_common_file aclcub_header "${ACLCUB_HEADER}"
  manifest_common_file boundary_shapes "${FORMAL_BOUNDARY_SHAPES}"
  manifest_common_file tune_shapes "${FORMAL_TUNE_SHAPES}"
} > "${MANIFEST_OUTPUT}"

FORMAL_COMPLETE=0
formal_exit_notice() {
  local status=$?
  if [[ "${FORMAL_MANIFEST}" == "1" && "${FORMAL_COMPLETE}" != "1" ]]; then
    echo "[formal] run is incomplete; inspect ${MANIFEST_OUTPUT}" >&2
    if [[ -f "${EVIDENCE_OUTPUT}" ]]; then
      echo "[formal] partial evidence: ${EVIDENCE_OUTPUT}" >&2
    fi
    if [[ -f "${DEVICE_SNAPSHOT_OUTPUT}" ]]; then
      echo "[formal] partial device snapshot: ${DEVICE_SNAPSHOT_OUTPUT}" >&2
    fi
    if [[ -f "${RUNTIME_CONFIG_OUTPUT}" ]]; then
      echo "[formal] partial runtime config: ${RUNTIME_CONFIG_OUTPUT}" >&2
    fi
  fi
  return "${status}"
}
trap formal_exit_notice EXIT

finalize_formal_manifest() {
  local result_csv
  local run_end_utc
  local common_validation_args=(
    --execution-kind "${FORMAL_EXECUTION_KIND}"
    --run-id "${RUN_ID}"
    --shapes "${SHAPES}"
    --tier "${TIER}"
    --variant-mode "${FORMAL_VARIANT_MODE}"
    --expected-softmax "${FORMAL_EXPECT_SOFTMAX}"
    --expected-rmsnorm "${FORMAL_EXPECT_RMSNORM}"
    --device "${ASCIFY_DEVICE}"
    --math-mode "${MATH_MODE}"
    --warmup "${WARMUP}"
    --samples "${SAMPLES}"
    --inner-repeats "${INNER_REPEATS}"
  )
  if [[ "${FORMAL_MANIFEST}" != "1" ]]; then
    return 0
  fi
  if [[ "${FORMAL_EXECUTION_KIND}" == "accuracy" ]]; then
    result_csv="${ACCURACY_CSV}"
  else
    result_csv="${PERF_CSV}"
  fi
  # Capture the device endpoint immediately after the last device binary
  # exits. All remaining runtime/CSV/manifest validation is host-only.
  capture_device_state post >> "${DEVICE_SNAPSHOT_OUTPUT}"
  if [[ "${FORMAL_EXECUTION_KIND}" == "perf" ]]; then
    "${VALIDATOR_SCRIPT}" runtime-config \
      --config "${RUNTIME_CONFIG_OUTPUT}" \
      --run-id "${RUN_ID}" \
      --variant-mode "${FORMAL_VARIANT_MODE}" \
      --device "${ASCIFY_DEVICE}"
  fi
  "${VALIDATOR_SCRIPT}" csv \
    "${common_validation_args[@]}" \
    --snapshot "${EVIDENCE_OUTPUT}" \
    --csv "${result_csv}"
  run_end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "${VALIDATOR_SCRIPT}" device-snapshot \
    --snapshot "${DEVICE_SNAPSHOT_OUTPUT}" \
    --device "${ASCIFY_DEVICE}" \
    --run-id "${RUN_ID}" \
    --hostname "${RUN_HOSTNAME}" \
    --run-start-utc "${RUN_START_UTC}" \
    --run-end-utc "${run_end_utc}"
  {
    manifest_file_as evidence_csv "${EVIDENCE}" "${EVIDENCE_OUTPUT}"
    manifest_file_as device_snapshot "${DEVICE_SNAPSHOT}" "${DEVICE_SNAPSHOT_OUTPUT}"
    if [[ "${FORMAL_EXECUTION_KIND}" == "perf" ]]; then
      manifest_file_as runtime_grid_config "${RUNTIME_CONFIG}" \
        "${RUNTIME_CONFIG_OUTPUT}"
    fi
    manifest_value run_end_utc_time "${run_end_utc}"
    manifest_value run_status "complete"
  } >> "${MANIFEST_OUTPUT}"
  mv -- "${EVIDENCE_OUTPUT}" "${EVIDENCE}"
  mv -- "${DEVICE_SNAPSHOT_OUTPUT}" "${DEVICE_SNAPSHOT}"
  if [[ "${FORMAL_EXECUTION_KIND}" == "perf" ]]; then
    mv -- "${RUNTIME_CONFIG_OUTPUT}" "${RUNTIME_CONFIG}"
  fi
  "${VALIDATOR_SCRIPT}" manifest \
    "${common_validation_args[@]}" \
    --csv "${result_csv}" \
    --manifest "${MANIFEST_OUTPUT}" \
    --final-manifest-path "${MANIFEST}" \
    --evidence "${EVIDENCE}" \
    --formal-set-id "${FORMAL_SET_ID}" \
    --formal-phase "${FORMAL_PHASE}" \
    --ascify-binary-sha256 "${ASCIFY_BINARY_SHA256}"
  mv -- "${MANIFEST_OUTPUT}" "${MANIFEST}"
  FORMAL_COMPLETE=1
  echo "[formal] completed manifest=${MANIFEST}"
  echo "[formal] immutable evidence=${EVIDENCE}"
}

# Runtime paths are scoped to this smoke process only.
set +u
# shellcheck disable=SC1090
source "${CANN_ROOT}/set_env.sh" >/dev/null
set -u

echo "[smoke] run_id=${RUN_ID}"
echo "[smoke] device=${ASCIFY_DEVICE} lock=${ASCIFY_DEVICE_LOCK:-unknown}"
echo "[smoke] math_mode=${MATH_MODE} tier=${TIER}"
echo "[smoke] softmax_variant=${SOFTMAX_VARIANT}"
echo "[smoke] rmsnorm_variant=${RMSNORM_VARIANT}"
if [[ "${RMSNORM_VARIANT}" == "native" ]]; then
  echo "[smoke] rmsnorm_block_row_threads=${RMSNORM_BLOCK_ROW_THREADS}"
  echo "[smoke] rmsnorm_block_row_affine_threads=${RMSNORM_BLOCK_ROW_AFFINE_THREADS}"
fi
echo "[smoke] op_filter=${OP_FILTER} skip_check=${SKIP_CHECK:-0}"
echo "[smoke] shapes=${SHAPES}"
echo "[smoke] manifest=${MANIFEST}"
npu-smi info -t usages -i "${ASCIFY_DEVICE}" 2>/dev/null | sed -n '1,24p' >&2 || true

runtime_config_args=()
if [[ "${FORMAL_MANIFEST}" == "1" && "${FORMAL_EXECUTION_KIND}" == "perf" ]]; then
  runtime_config_args=(--runtime-config-out "${RUNTIME_CONFIG_OUTPUT}")
fi

if [[ "${SKIP_CHECK:-0}" != "1" ]]; then
  if [[ "${RUN_SOFTMAX}" == "1" ]]; then
    echo "[smoke] softmax correctness gate"
    "${CHECK_BINARY}" \
      --device "${ASCIFY_DEVICE}" \
      --shapes "${SHAPES}" \
      --tier "${TIER}" \
      --run-id "${RUN_ID}" \
      --max-elements "${CHECK_MAX_ELEMENTS}" \
      --out "${ACCURACY_CSV}"
  fi

  if [[ "${RUN_RMSNORM}" == "1" ]]; then
    echo "[smoke] rms_norm correctness gate"
    "${RMS_CHECK_BINARY}" \
      --device "${ASCIFY_DEVICE}" \
      --shapes "${SHAPES}" \
      --tier "${TIER}" \
      --run-id "${RUN_ID}" \
      --max-elements "${CHECK_MAX_ELEMENTS}" \
      --out "${ACCURACY_CSV}"
  fi
fi

if [[ "${CHECK_ONLY:-0}" == "1" ]]; then
  finalize_formal_manifest
  echo "[ok] ${SELECTED_OPS} correctness passed; CHECK_ONLY=1"
  echo "[ok] ${ACCURACY_CSV}"
  exit 0
fi

if [[ "${RUN_SOFTMAX}" == "1" ]]; then
  echo "[smoke] softmax performance"
  "${BENCH_BINARY}" \
    --device "${ASCIFY_DEVICE}" \
    --shapes "${SHAPES}" \
    --tier "${TIER}" \
    --run-id "${RUN_ID}" \
    --out "${PERF_CSV}" \
    --warmup "${WARMUP}" \
    --samples "${SAMPLES}" \
    --inner-repeats "${INNER_REPEATS}" \
    "${runtime_config_args[@]}"
fi

if [[ "${RUN_RMSNORM}" == "1" ]]; then
  echo "[smoke] rms_norm performance"
  "${RMS_BENCH_BINARY}" \
    --device "${ASCIFY_DEVICE}" \
    --shapes "${SHAPES}" \
    --tier "${TIER}" \
    --run-id "${RUN_ID}" \
    --out "${PERF_CSV}" \
    --warmup "${WARMUP}" \
    --samples "${SAMPLES}" \
    --inner-repeats "${INNER_REPEATS}" \
    "${runtime_config_args[@]}"
fi

finalize_formal_manifest
echo "[ok] smoke completed under one held device lock"
echo "[ok] ${ACCURACY_CSV}"
echo "[ok] ${PERF_CSV}"
