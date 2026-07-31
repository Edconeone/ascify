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
MATH_MODE="${MATH_MODE:-production-fast}"

FORMAL_TAG="${FORMAL_TAG:-v3}"
FORMAL_STAGE="${ASCIFY_FORMAL_STAGE:-prepare}"
FORMAL_SET_ID="${FORMAL_SET_ID:-ascify_recipe_formal_${FORMAL_TAG}}"
DIRECT_BOUNDARY_RUN_ID="${DIRECT_BOUNDARY_RUN_ID:-ascify_recipe_direct_correctness_${FORMAL_TAG}}"
NATIVE_BOUNDARY_RUN_ID="${NATIVE_BOUNDARY_RUN_ID:-ascify_recipe_native_correctness_${FORMAL_TAG}}"
DIRECT_TUNE_RUN_ID="${DIRECT_TUNE_RUN_ID:-ascify_recipe_direct_tune_gate_${FORMAL_TAG}}"
NATIVE_TUNE_RUN_ID="${NATIVE_TUNE_RUN_ID:-ascify_recipe_native_tune_gate_${FORMAL_TAG}}"
PREHEAT_RUN_ID="${PREHEAT_RUN_ID:-ascify_recipe_direct_preheat_${FORMAL_TAG}}"
DIRECT_PERF_A_RUN_ID="${DIRECT_PERF_A_RUN_ID:-ascify_recipe_direct_perf_${FORMAL_TAG}a}"
NATIVE_PERF_RUN_ID="${NATIVE_PERF_RUN_ID:-ascify_recipe_native_control_${FORMAL_TAG}}"
DIRECT_PERF_B_RUN_ID="${DIRECT_PERF_B_RUN_ID:-ascify_recipe_direct_perf_${FORMAL_TAG}b}"

BOUNDARY_SHAPES="${TEST_ROOT}/shapes/correctness.csv"
TUNE_SHAPES="${TEST_ROOT}/shapes/unified_tune.csv"
BOUNDARY_CHECK_MAX_ELEMENTS=5000000
TUNE_CHECK_MAX_ELEMENTS=10000000
PREHEAT_WARMUP=5
PREHEAT_SAMPLES=5
PREHEAT_INNER_REPEATS=5
FORMAL_WARMUP=20
FORMAL_SAMPLES=50
FORMAL_INNER_REPEATS=20
FORMAL_PROCESS_SETTLE_ATTEMPTS=30

SOFTMAX_BLOCK_THREADS="${SOFTMAX_BLOCK_THREADS:-adaptive}"
SOFTMAX_BLOCK_ROW_MAX="${SOFTMAX_BLOCK_ROW_MAX:-2048}"
RMSNORM_BLOCK_ROW_THREADS="${RMSNORM_BLOCK_ROW_THREADS:-512}"
RMSNORM_BLOCK_ROW_AFFINE_THREADS="${RMSNORM_BLOCK_ROW_AFFINE_THREADS:-256}"

ACCURACY_CSV="${ACCURACY_CSV:-${RESULT_DIR}/accuracy_history_v2.csv}"
PERF_CSV="${PERF_CSV:-${RESULT_DIR}/perf_history.csv}"
MANIFEST_DIR="${RESULT_DIR}/manifests"
SET_MANIFEST="${MANIFEST_DIR}/${FORMAL_SET_ID}.tsv"
BRACKET_SUMMARY="${MANIFEST_DIR}/${FORMAL_SET_ID}.bracket_summary.csv"
FORMAL_METRICS="${MANIFEST_DIR}/${FORMAL_SET_ID}.formal_metrics.csv"
PREHEAT_PERF_CSV="${PREHEAT_PERF_CSV:-${RESULT_DIR}/${FORMAL_SET_ID}.preheat_history.csv}"
VALIDATOR="${SCRIPT_DIR}/validate_formal_run.py"
VALIDATOR_TEST="${SCRIPT_DIR}/test_validate_formal_run.py"
RUN_SMOKE="${SCRIPT_DIR}/run_smoke.sh"
LOCK_HELPER="${SCRIPT_DIR}/safe_lock_exec.py"
METRICS_TOOL="${TEST_ROOT}/tools/derive_work_metrics.py"
METRICS_TEST="${TEST_ROOT}/tools/test_derive_work_metrics.py"
ASCIFY_CONVERSION_MANIFEST="${ASCIFY_CONVERSION_MANIFEST:-${WORK_ROOT}/conversion/evidence_v3/manifest.json}"
BUILD_INPUT_SNAPSHOT="${MANIFEST_DIR}/${FORMAL_SET_ID}.build_inputs.tsv"
FORMAL_BINARY_BUNDLE_DIR="${MANIFEST_DIR}/${FORMAL_SET_ID}.binaries"
FORMAL_BINARY_BUNDLE_MANIFEST="${FORMAL_BINARY_BUNDLE_DIR}/manifest.tsv"

GENERATED_SOFTMAX="${GENERATED_ROOT}/oneflow/core/cuda/softmax.cuh"
GENERATED_LAYER_NORM="${GENERATED_ROOT}/oneflow/core/cuda/layer_norm.cuh"
GENERATED_RMSNORM="${GENERATED_ROOT}/oneflow/core/cuda/rms_norm.cuh"
GENERATED_RMSNORM_ADAPTER="${GENERATED_ROOT}/ascify950/rmsnorm_affine_store.cuh"
TARGET_RECIPE_HEADER="${REPO_ROOT}/include/ascify/target/dav_c310/rowwise_norm_recipes.hpp"
TARGET_SOFTMAX_IMPL="${REPO_ROOT}/include/ascify/target/dav_c310/detail/softmax_fp16_impl.hpp"
TARGET_RMSNORM_IMPL="${REPO_ROOT}/include/ascify/target/dav_c310/detail/rmsnorm_fp16_impl.hpp"
ACLCUB_HEADER="${REPO_ROOT}/acl_cub/aclcub.hpp"
ASCIFY_RECIPE_SOURCE="${REPO_ROOT}/src/DavC310TargetRecipe.cpp"
ASCIFY_RECIPE_HEADER="${REPO_ROOT}/src/DavC310TargetRecipe.h"
SOFTMAX_NATIVE_SOURCE="${TEST_ROOT}/kernels/softmax_native_kernel.cce"
RMSNORM_NATIVE_SOURCE="${TEST_ROOT}/kernels/rmsnorm_native_kernel.cce"
RMSNORM_ADAPTER_INPUT="${TEST_ROOT}/inputs/rmsnorm_affine_store.cuh"
RUNNER_COMMON_HEADER="${TEST_ROOT}/common/runner_common.hpp"
CCEC_BINARY="${CCEC:-${CANN_ROOT}/tools/bisheng_compiler/bin/ccec}"

if [[ "${FORMAL_STAGE}" != "prepare"
      && "${FORMAL_STAGE}" != "resume"
      && "${FORMAL_STAGE}" != "measure" ]]; then
  echo "ASCIFY_FORMAL_STAGE must be prepare, resume, or measure" >&2
  exit 2
fi
if [[ -z "${CANN_ROOT}" ]]; then
  echo "CANN_ROOT must point to a user-owned CANN installation" >&2
  exit 2
fi
if [[ "${FORMAL_STAGE}" != "measure" ]]; then
  # Device state inherited from an outer shell is irrelevant before selection.
  unset ASCIFY_DEVICE ASCIFY_DEVICE_LOCK
fi

if [[ "${FORMAL_STAGE}" == "measure" ]]; then
  if [[ "${UTIL_MAX:-}" != "0" || "${HBM_BW_MAX:-}" != "0" ]]; then
    echo "v3 formal selection requires UTIL_MAX=0 and HBM_BW_MAX=0" >&2
    exit 2
  fi
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required for formal validation" >&2
  exit 1
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "flock is required for formal validation" >&2
  exit 1
fi
if [[ ! "${FORMAL_TAG}" =~ ^[A-Za-z0-9_.-]+$
      || ! "${FORMAL_SET_ID}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "FORMAL_TAG and FORMAL_SET_ID must be safe identifiers" >&2
  exit 2
fi
if [[ ! "${ASCIFY_BINARY_SHA256:-}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "export the exact 910C Ascify binary hash as ASCIFY_BINARY_SHA256" >&2
  exit 2
fi
if [[ "${MATH_MODE}" != "production-fast" ]]; then
  echo "the v3 formal comparison is fixed to MATH_MODE=production-fast" >&2
  exit 2
fi
if [[ "${SOFTMAX_BLOCK_THREADS}" != "adaptive"
      || "${SOFTMAX_BLOCK_ROW_MAX}" != "2048"
      || "${RMSNORM_BLOCK_ROW_THREADS}" != "512"
      || "${RMSNORM_BLOCK_ROW_AFFINE_THREADS}" != "256" ]]; then
  echo "v3 formal tuning is fixed to adaptive/2048/512/256" >&2
  exit 2
fi
for numeric_value in \
  "${BOUNDARY_CHECK_MAX_ELEMENTS}" "${TUNE_CHECK_MAX_ELEMENTS}" \
  "${PREHEAT_WARMUP}" "${PREHEAT_SAMPLES}" "${PREHEAT_INNER_REPEATS}" \
  "${FORMAL_WARMUP}" "${FORMAL_SAMPLES}" "${FORMAL_INNER_REPEATS}" \
  "${FORMAL_PROCESS_SETTLE_ATTEMPTS}"; do
  if [[ ! "${numeric_value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "formal limits and timing parameters must be positive integers" >&2
    exit 2
  fi
done
for required_file in \
  "${BOUNDARY_SHAPES}" "${TUNE_SHAPES}" "${VALIDATOR}" "${VALIDATOR_TEST}" \
  "${RUN_SMOKE}" \
  "${LOCK_HELPER}" \
  "${METRICS_TOOL}" "${METRICS_TEST}" \
  "${SCRIPT_DIR}/build.sh" "${CANN_ROOT}/set_env.sh" \
  "${ASCIFY_CONVERSION_MANIFEST}" \
  "${GENERATED_SOFTMAX}" "${GENERATED_LAYER_NORM}" \
  "${GENERATED_RMSNORM}" "${GENERATED_RMSNORM_ADAPTER}" \
  "${ASCIFY_RECIPE_SOURCE}" "${ASCIFY_RECIPE_HEADER}" \
  "${ACLCUB_HEADER}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "missing formal-run input: ${required_file}" >&2
    exit 1
  fi
done

FORMAL_RESULT_LOCK="${RESULT_DIR}/.formal-results.lock"
mkdir -p -- "${WORK_ROOT}/locks" "${RESULT_DIR}"
if [[ "${FORMAL_STAGE}" != "measure"
      && "${ASCIFY_FORMAL_RESULT_LOCK_READY:-}" != "${FORMAL_RESULT_LOCK}" ]]; then
  if [[ -n "${ASCIFY_FORMAL_RESULT_LOCK_READY:-}" ]]; then
    echo "formal result lock re-entry path mismatch" >&2
    exit 1
  fi
  ASCIFY_FORMAL_RESULT_LOCK_READY="${FORMAL_RESULT_LOCK}" \
    exec python3 -B "${LOCK_HELPER}" \
      --fd 8 --path "${FORMAL_RESULT_LOCK}" --busy-exit 1 -- "$0" "$@"
fi
if [[ "${FORMAL_STAGE}" == "measure"
      && "${ASCIFY_FORMAL_RESULT_LOCK_READY:-}" != "${FORMAL_RESULT_LOCK}" ]]; then
  echo "measure stage requires the inherited prepare-stage result lock" >&2
  exit 1
fi
if [[ -L "${FORMAL_RESULT_LOCK}" ]] \
    || ! { true >&8; } 2>/dev/null || ! flock -n 8; then
  echo "formal result lock is unsafe or not held: ${FORMAL_RESULT_LOCK}" >&2
  exit 1
fi
if [[ "${FORMAL_STAGE}" == "prepare" ]]; then
  python3 -B "${VALIDATOR_TEST}"
  python3 -B "${METRICS_TEST}"
fi

if [[ "${FORMAL_STAGE}" == "measure" ]]; then
  if [[ -z "${ASCIFY_DEVICE:-}" || -z "${ASCIFY_DEVICE_LOCK:-}" ]]; then
    # The result lock survives both selector exec layers. Ignore any stale
    # caller DEVICE so the post-build scan can choose any truly idle card.
    unset ASCIFY_DEVICE ASCIFY_DEVICE_LOCK
    exec env -u DEVICE UTIL_MAX=0 HBM_BW_MAX=0 \
      "${SCRIPT_DIR}/select_device.sh" -- "$0" "$@"
  fi
  if [[ ! "${ASCIFY_DEVICE}" =~ ^[0-9]+$ ]]; then
    echo "ASCIFY_DEVICE must be a non-negative integer" >&2
    exit 2
  fi
  if ! { true >&9; } 2>/dev/null || ! flock -n 9; then
    echo "formal runs require the inherited select_device fd 9 lock" >&2
    exit 1
  fi
  "${VALIDATOR}" lock \
    --fd 9 \
    --declared "${ASCIFY_DEVICE_LOCK}" \
    --expected "${WORK_ROOT}/locks/device-${ASCIFY_DEVICE}.lock"
fi
"${VALIDATOR}" lock \
  --fd 8 \
  --declared "${FORMAL_RESULT_LOCK}" \
  --expected "${RESULT_DIR}/.formal-results.lock"
"${VALIDATOR}" paths \
  --work-root "${WORK_ROOT}" \
  --result-dir "${RESULT_DIR}" \
  --accuracy-csv "${ACCURACY_CSV}" \
  --perf-csv "${PERF_CSV}" \
  --preheat-csv "${PREHEAT_PERF_CSV}" \
  --manifest-dir "${MANIFEST_DIR}" \
  --set-manifest "${SET_MANIFEST}" \
  --bracket-summary "${BRACKET_SUMMARY}" \
  --formal-metrics "${FORMAL_METRICS}" \
  --result-lock "${FORMAL_RESULT_LOCK}" \
  --build-input-snapshot "${BUILD_INPUT_SNAPSHOT}" \
  --binary-bundle-dir "${FORMAL_BINARY_BUNDLE_DIR}"

RUN_IDS=(
  "${DIRECT_BOUNDARY_RUN_ID}"
  "${NATIVE_BOUNDARY_RUN_ID}"
  "${DIRECT_TUNE_RUN_ID}"
  "${NATIVE_TUNE_RUN_ID}"
  "${PREHEAT_RUN_ID}"
  "${DIRECT_PERF_A_RUN_ID}"
  "${NATIVE_PERF_RUN_ID}"
  "${DIRECT_PERF_B_RUN_ID}"
)

preflight_args=(
  preflight
  --accuracy-csv "${ACCURACY_CSV}"
  --perf-csv "${PERF_CSV}"
  --manifest-dir "${MANIFEST_DIR}"
  --set-id "${FORMAL_SET_ID}"
)
if [[ "${FORMAL_STAGE}" != "prepare" ]]; then
  preflight_args+=(--prepared)
fi
for run_id in "${RUN_IDS[@]}"; do
  preflight_args+=(--run-id "${run_id}")
done
"${VALIDATOR}" "${preflight_args[@]}"
if [[ "${PREHEAT_PERF_CSV}" == "${PERF_CSV}" ]]; then
  echo "PREHEAT_PERF_CSV must be isolated from PERF_CSV" >&2
  exit 2
fi
if [[ -e "${PREHEAT_PERF_CSV}" ]]; then
  echo "refusing to reuse preheat history: ${PREHEAT_PERF_CSV}" >&2
  exit 1
fi

conversion_summary="$(
  "${VALIDATOR}" conversion \
    --manifest "${ASCIFY_CONVERSION_MANIFEST}" \
    --ascify-binary-sha256 "${ASCIFY_BINARY_SHA256}" \
    --recipe-source "${ASCIFY_RECIPE_SOURCE}" \
    --recipe-header "${ASCIFY_RECIPE_HEADER}" \
    --staged-softmax "${GENERATED_SOFTMAX}" \
    --staged-layer-norm "${GENERATED_LAYER_NORM}" \
    --staged-rmsnorm "${GENERATED_RMSNORM}" \
    --staged-rmsnorm-adapter "${GENERATED_RMSNORM_ADAPTER}" \
    --emit-env
)"
conversion_value() {
  local name="$1"
  awk -F '\t' -v name="${name}" '$1 == name { print $2; found=1 } END { exit !found }' \
    <<<"${conversion_summary}"
}
ASCIFY_CONVERSION_SET_ID="$(conversion_value ASCIFY_CONVERSION_SET_ID)"
ASCIFY_CONVERSION_MANIFEST_SHA256="$(
  conversion_value ASCIFY_CONVERSION_MANIFEST_SHA256
)"
ASCIFY_INPUT_SOFTMAX_SHA256="$(conversion_value ASCIFY_INPUT_SOFTMAX_SHA256)"
ASCIFY_INPUT_LAYER_NORM_SHA256="$(
  conversion_value ASCIFY_INPUT_LAYER_NORM_SHA256
)"
ASCIFY_INPUT_RMSNORM_SHA256="$(conversion_value ASCIFY_INPUT_RMSNORM_SHA256)"
ASCIFY_INPUT_RMSNORM_ADAPTER_SHA256="$(
  conversion_value ASCIFY_INPUT_RMSNORM_ADAPTER_SHA256
)"

if [[ "${FORMAL_STAGE}" == "prepare" ]]; then
  "${VALIDATOR}" freeze \
    --formal-contract \
    --output "${BUILD_INPUT_SNAPSHOT}" \
    --artifact "conversion_manifest=${ASCIFY_CONVERSION_MANIFEST}" \
    --artifact "generated_softmax=${GENERATED_SOFTMAX}" \
    --artifact "generated_layer_norm=${GENERATED_LAYER_NORM}" \
    --artifact "generated_rmsnorm=${GENERATED_RMSNORM}" \
    --artifact "generated_rmsnorm_adapter=${GENERATED_RMSNORM_ADAPTER}" \
    --artifact "target_recipe_header=${TARGET_RECIPE_HEADER}" \
    --artifact "target_softmax_impl=${TARGET_SOFTMAX_IMPL}" \
    --artifact "target_rmsnorm_impl=${TARGET_RMSNORM_IMPL}" \
    --artifact "ascify_recipe_source=${ASCIFY_RECIPE_SOURCE}" \
    --artifact "ascify_recipe_header=${ASCIFY_RECIPE_HEADER}" \
    --artifact "softmax_native_source=${SOFTMAX_NATIVE_SOURCE}" \
    --artifact "rmsnorm_native_source=${RMSNORM_NATIVE_SOURCE}" \
    --artifact "rmsnorm_adapter_input=${RMSNORM_ADAPTER_INPUT}" \
    --artifact "runner_common_header=${RUNNER_COMMON_HEADER}" \
    --artifact "softmax_check_source=${TEST_ROOT}/softmax_check.cce" \
    --artifact "rmsnorm_check_source=${TEST_ROOT}/rmsnorm_check.cce" \
    --artifact "softmax_bench_source=${TEST_ROOT}/softmax_bench.cce" \
    --artifact "rmsnorm_bench_source=${TEST_ROOT}/rmsnorm_bench.cce" \
    --artifact "build_script=${SCRIPT_DIR}/build.sh" \
    --artifact "run_smoke_script=${RUN_SMOKE}" \
    --artifact "formal_runner_script=$0" \
    --artifact "validator_script=${VALIDATOR}" \
    --artifact "validator_test=${VALIDATOR_TEST}" \
    --artifact "safe_lock_helper=${LOCK_HELPER}" \
    --artifact "derive_work_metrics_tool=${METRICS_TOOL}" \
    --artifact "derive_work_metrics_test=${METRICS_TEST}" \
    --artifact "select_device_script=${SCRIPT_DIR}/select_device.sh" \
    --artifact "ccec_binary=${CCEC_BINARY}" \
    --artifact "cann_set_env=${CANN_ROOT}/set_env.sh" \
    --artifact "aclcub_header=${ACLCUB_HEADER}" \
    --artifact "boundary_shapes=${BOUNDARY_SHAPES}" \
    --artifact "tune_shapes=${TUNE_SHAPES}"
else
  "${VALIDATOR}" freeze \
    --formal-contract \
    --check "${BUILD_INPUT_SNAPSHOT}"
fi

"${VALIDATOR}" source-config \
  --generated-softmax "${GENERATED_SOFTMAX}" \
  --generated-rmsnorm "${GENERATED_RMSNORM}" \
  --target-header "${TARGET_RECIPE_HEADER}" \
  --target-softmax-impl "${TARGET_SOFTMAX_IMPL}" \
  --target-rmsnorm-impl "${TARGET_RMSNORM_IMPL}" \
  --softmax-bench "${TEST_ROOT}/softmax_bench.cce" \
  --softmax-native "${SOFTMAX_NATIVE_SOURCE}" \
  --rmsnorm-native "${RMSNORM_NATIVE_SOURCE}"

common_env=(
  WORK_ROOT="${WORK_ROOT}"
  GENERATED_ROOT="${GENERATED_ROOT}"
  BIN_DIR="${BIN_DIR}"
  RESULT_DIR="${RESULT_DIR}"
  CANN_ROOT="${CANN_ROOT}"
  MATH_MODE="${MATH_MODE}"
  SOFTMAX_BLOCK_THREADS="${SOFTMAX_BLOCK_THREADS}"
  SOFTMAX_BLOCK_ROW_MAX="${SOFTMAX_BLOCK_ROW_MAX}"
  RMSNORM_BLOCK_ROW_THREADS="${RMSNORM_BLOCK_ROW_THREADS}"
  RMSNORM_BLOCK_ROW_AFFINE_THREADS="${RMSNORM_BLOCK_ROW_AFFINE_THREADS}"
  ASCIFY_BINARY_SHA256="${ASCIFY_BINARY_SHA256}"
  ASCIFY_CONVERSION_MANIFEST="${ASCIFY_CONVERSION_MANIFEST}"
  ASCIFY_CONVERSION_SET_ID="${ASCIFY_CONVERSION_SET_ID}"
  ASCIFY_INPUT_SOFTMAX_SHA256="${ASCIFY_INPUT_SOFTMAX_SHA256}"
  ASCIFY_INPUT_LAYER_NORM_SHA256="${ASCIFY_INPUT_LAYER_NORM_SHA256}"
  ASCIFY_INPUT_RMSNORM_SHA256="${ASCIFY_INPUT_RMSNORM_SHA256}"
  ASCIFY_INPUT_RMSNORM_ADAPTER_SHA256="${ASCIFY_INPUT_RMSNORM_ADAPTER_SHA256}"
  ASCIFY_CONVERSION_MANIFEST_SHA256="${ASCIFY_CONVERSION_MANIFEST_SHA256}"
  FORMAL_BUILD_INPUT_SNAPSHOT="${BUILD_INPUT_SNAPSHOT}"
  FORMAL_BINARY_BUNDLE_MANIFEST="${FORMAL_BINARY_BUNDLE_MANIFEST}"
  FORMAL_BOUNDARY_SHAPES="${BOUNDARY_SHAPES}"
  FORMAL_TUNE_SHAPES="${TUNE_SHAPES}"
  FORMAL_DEVICE_UTIL_MAX=0
  FORMAL_DEVICE_HBM_BW_MAX=0
  UTIL_MAX=0
  HBM_BW_MAX=0
  FORMAL_RESULT_LOCK="${FORMAL_RESULT_LOCK}"
)

build_variant() {
  local softmax_variant="$1"
  local rmsnorm_variant="$2"
  echo "[formal] build side-by-side ${softmax_variant}/${rmsnorm_variant}"
  env "${common_env[@]}" \
    SOFTMAX_VARIANT="${softmax_variant}" \
    RMSNORM_VARIANT="${rmsnorm_variant}" \
    "${SCRIPT_DIR}/build.sh" check "${MATH_MODE}"
  env "${common_env[@]}" \
    SOFTMAX_VARIANT="${softmax_variant}" \
    RMSNORM_VARIANT="${rmsnorm_variant}" \
    "${SCRIPT_DIR}/build.sh" bench "${MATH_MODE}"
  "${VALIDATOR}" freeze \
    --formal-contract \
    --check "${BUILD_INPUT_SNAPSHOT}"
}

validate_completed_run() {
  local execution_kind="$1"
  local run_id="$2"
  local phase="$3"
  local variant_mode="$4"
  local shapes="$5"
  local tier="$6"
  local expected_softmax="$7"
  local expected_rmsnorm="$8"
  local result_csv="$9"
  local run_warmup="${10}"
  local run_samples="${11}"
  local run_inner_repeats="${12}"
  "${VALIDATOR}" manifest \
    --execution-kind "${execution_kind}" \
    --run-id "${run_id}" \
    --shapes "${shapes}" \
    --tier "${tier}" \
    --variant-mode "${variant_mode}" \
    --expected-softmax "${expected_softmax}" \
    --expected-rmsnorm "${expected_rmsnorm}" \
    --device "${ASCIFY_DEVICE}" \
    --math-mode "${MATH_MODE}" \
    --warmup "${run_warmup}" \
    --samples "${run_samples}" \
    --inner-repeats "${run_inner_repeats}" \
    --csv "${result_csv}" \
    --manifest "${MANIFEST_DIR}/${run_id}.tsv" \
    --evidence "${MANIFEST_DIR}/${run_id}.csv" \
    --formal-set-id "${FORMAL_SET_ID}" \
    --formal-phase "${phase}" \
    --ascify-binary-sha256 "${ASCIFY_BINARY_SHA256}"
}

wait_for_device_process_cleanup() {
  local phase="$1"
  local attempt=1
  local process_text=""
  while [[ "${attempt}" -le "${FORMAL_PROCESS_SETTLE_ATTEMPTS}" ]]; do
    if process_text="$(
      npu-smi info -t proc-mem -i "${ASCIFY_DEVICE}" 2>/dev/null
    )" && grep -Eiq 'no[[:space:]].*process' <<<"${process_text}"; then
      if [[ "${attempt}" -gt 1 ]]; then
        echo "[formal] ${phase}: device process state settled after $((attempt - 1))s"
      fi
      return 0
    fi
    if [[ "${attempt}" -eq "${FORMAL_PROCESS_SETTLE_ATTEMPTS}" ]]; then
      echo "[formal] ${phase}: device process state did not settle" >&2
      printf '%s\n' "${process_text}" >&2
      return 1
    fi
    sleep 1
    attempt=$((attempt + 1))
  done
}

run_one() {
  local execution_kind="$1"
  local run_id="$2"
  local phase="$3"
  local variant_mode="$4"
  local shapes="$5"
  local tier="$6"
  local expected_softmax="$7"
  local expected_rmsnorm="$8"
  local check_max="$9"
  local perf_csv="${10:-${PERF_CSV}}"
  local run_warmup="${11:-${FORMAL_WARMUP}}"
  local run_samples="${12:-${FORMAL_SAMPLES}}"
  local run_inner_repeats="${13:-${FORMAL_INNER_REPEATS}}"
  local softmax_variant
  local rmsnorm_variant
  local check_only=0
  local skip_check=0
  local result_csv="${perf_csv}"
  case "${variant_mode}" in
    direct)
      softmax_variant="converted"
      rmsnorm_variant="converted"
      ;;
    native)
      softmax_variant="native-half2-hybrid"
      rmsnorm_variant="native"
      ;;
    *)
      echo "internal error: unknown formal variant mode ${variant_mode}" >&2
      return 2
      ;;
  esac
  if [[ "${execution_kind}" == "accuracy" ]]; then
    check_only=1
    result_csv="${ACCURACY_CSV}"
  else
    skip_check=1
  fi
  wait_for_device_process_cleanup "${phase}"
  echo "[formal] ${phase}: run_id=${run_id} device=${ASCIFY_DEVICE}"
  env "${common_env[@]}" \
    BIN_DIR="${FORMAL_BINARY_BUNDLE_DIR}" \
    RUN_ID="${run_id}" \
    FORMAL_MANIFEST=1 \
    FORMAL_SET_ID="${FORMAL_SET_ID}" \
    FORMAL_PHASE="${phase}" \
    FORMAL_EXECUTION_KIND="${execution_kind}" \
    FORMAL_VARIANT_MODE="${variant_mode}" \
    FORMAL_EXPECT_SOFTMAX="${expected_softmax}" \
    FORMAL_EXPECT_RMSNORM="${expected_rmsnorm}" \
    SOFTMAX_VARIANT="${softmax_variant}" \
    RMSNORM_VARIANT="${rmsnorm_variant}" \
    SHAPES="${shapes}" \
    TIER="${tier}" \
    OP_FILTER=both \
    CHECK_MAX_ELEMENTS="${check_max}" \
    CHECK_ONLY="${check_only}" \
    SKIP_CHECK="${skip_check}" \
    SKIP_BUILD=1 \
    WARMUP="${run_warmup}" \
    SAMPLES="${run_samples}" \
    INNER_REPEATS="${run_inner_repeats}" \
    ACCURACY_CSV="${ACCURACY_CSV}" \
    PERF_CSV="${perf_csv}" \
    "${RUN_SMOKE}"
  validate_completed_run \
    "${execution_kind}" "${run_id}" "${phase}" "${variant_mode}" \
    "${shapes}" "${tier}" "${expected_softmax}" "${expected_rmsnorm}" \
    "${result_csv}" "${run_warmup}" "${run_samples}" "${run_inner_repeats}"
}

if [[ "${FORMAL_STAGE}" == "prepare" ]]; then
  build_variant converted converted
  build_variant native-half2-hybrid native

  MODE_SUFFIX="${MATH_MODE//-/_}"
  SIDE_BY_SIDE_BINARIES=(
    "${BIN_DIR}/softmax_check_${MODE_SUFFIX}"
    "${BIN_DIR}/rmsnorm_check_${MODE_SUFFIX}"
    "${BIN_DIR}/softmax_bench_${MODE_SUFFIX}"
    "${BIN_DIR}/rmsnorm_bench_${MODE_SUFFIX}"
    "${BIN_DIR}/softmax_check_${MODE_SUFFIX}_native_half2_hybrid"
    "${BIN_DIR}/rmsnorm_check_${MODE_SUFFIX}_native"
    "${BIN_DIR}/softmax_bench_${MODE_SUFFIX}_native_half2_hybrid"
    "${BIN_DIR}/rmsnorm_bench_${MODE_SUFFIX}_native"
  )
  for binary in "${SIDE_BY_SIDE_BINARIES[@]}"; do
    if [[ ! -x "${binary}" ]]; then
      echo "side-by-side build is incomplete: ${binary}" >&2
      exit 1
    fi
  done

  "${VALIDATOR}" binary-bundle \
    --formal-set-id "${FORMAL_SET_ID}" \
    --output-dir "${FORMAL_BINARY_BUNDLE_DIR}" \
    --build-input-snapshot "${BUILD_INPUT_SNAPSHOT}" \
    --artifact \
      "softmax_check_production_fast=${BIN_DIR}/softmax_check_production_fast" \
    --artifact \
      "rmsnorm_check_production_fast=${BIN_DIR}/rmsnorm_check_production_fast" \
    --artifact \
      "softmax_bench_production_fast=${BIN_DIR}/softmax_bench_production_fast" \
    --artifact \
      "rmsnorm_bench_production_fast=${BIN_DIR}/rmsnorm_bench_production_fast" \
    --artifact \
      "softmax_check_production_fast_native_half2_hybrid=${BIN_DIR}/softmax_check_production_fast_native_half2_hybrid" \
    --artifact \
      "rmsnorm_check_production_fast_native=${BIN_DIR}/rmsnorm_check_production_fast_native" \
    --artifact \
      "softmax_bench_production_fast_native_half2_hybrid=${BIN_DIR}/softmax_bench_production_fast_native_half2_hybrid" \
    --artifact \
      "rmsnorm_bench_production_fast_native=${BIN_DIR}/rmsnorm_bench_production_fast_native"

fi

"${VALIDATOR}" freeze \
  --formal-contract \
  --check "${BUILD_INPUT_SNAPSHOT}"
"${VALIDATOR}" binary-bundle-check \
  --manifest "${FORMAL_BINARY_BUNDLE_MANIFEST}" \
  --formal-set-id "${FORMAL_SET_ID}" \
  --build-input-snapshot "${BUILD_INPUT_SNAPSHOT}"
if [[ "${FORMAL_STAGE}" != "measure" ]]; then
  echo "[formal] immutable binaries verified; selecting a fresh idle device"
  exec env -u DEVICE -u ASCIFY_DEVICE -u ASCIFY_DEVICE_LOCK \
    ASCIFY_FORMAL_STAGE=measure UTIL_MAX=0 HBM_BW_MAX=0 \
    "${SCRIPT_DIR}/select_device.sh" -- "$0" "$@"
fi

# Boundary coverage: exact 42 Softmax + 18 RMSNorm rows for each implementation.
run_one accuracy "${DIRECT_BOUNDARY_RUN_ID}" direct_boundary direct \
  "${BOUNDARY_SHAPES}" smoke 42 18 "${BOUNDARY_CHECK_MAX_ELEMENTS}"
run_one accuracy "${NATIVE_BOUNDARY_RUN_ID}" native_boundary native \
  "${BOUNDARY_SHAPES}" smoke 42 18 "${BOUNDARY_CHECK_MAX_ELEMENTS}"

# Performance-shape safety gate: exact 5 Softmax + 10 RMSNorm rows.
run_one accuracy "${DIRECT_TUNE_RUN_ID}" direct_tune direct \
  "${TUNE_SHAPES}" tune 5 10 "${TUNE_CHECK_MAX_ELEMENTS}"
run_one accuracy "${NATIVE_TUNE_RUN_ID}" native_tune native \
  "${TUNE_SHAPES}" tune 5 10 "${TUNE_CHECK_MAX_ELEMENTS}"

# Explicit short warm-up is isolated from perf_history.csv and formal A/native/B IDs.
run_one perf "${PREHEAT_RUN_ID}" preheat direct \
  "${TUNE_SHAPES}" tune 5 10 "${TUNE_CHECK_MAX_ELEMENTS}" \
  "${PREHEAT_PERF_CSV}" \
  "${PREHEAT_WARMUP}" "${PREHEAT_SAMPLES}" "${PREHEAT_INNER_REPEATS}"

# No build/check work is interleaved with the formal direct-A/native/direct-B bracket.
run_one perf "${DIRECT_PERF_A_RUN_ID}" direct_perf_a direct \
  "${TUNE_SHAPES}" tune 5 10 "${TUNE_CHECK_MAX_ELEMENTS}"
run_one perf "${NATIVE_PERF_RUN_ID}" native_perf native \
  "${TUNE_SHAPES}" tune 5 10 "${TUNE_CHECK_MAX_ELEMENTS}"
run_one perf "${DIRECT_PERF_B_RUN_ID}" direct_perf_b direct \
  "${TUNE_SHAPES}" tune 5 10 "${TUNE_CHECK_MAX_ELEMENTS}"

"${VALIDATOR}" freeze \
  --formal-contract \
  --check "${BUILD_INPUT_SNAPSHOT}"
"${VALIDATOR}" bracket \
  --formal-set-id "${FORMAL_SET_ID}" \
  --manifest-a "${MANIFEST_DIR}/${DIRECT_PERF_A_RUN_ID}.tsv" \
  --manifest-native "${MANIFEST_DIR}/${NATIVE_PERF_RUN_ID}.tsv" \
  --manifest-b "${MANIFEST_DIR}/${DIRECT_PERF_B_RUN_ID}.tsv" \
  --gate-manifest "${MANIFEST_DIR}/${DIRECT_BOUNDARY_RUN_ID}.tsv" \
  --gate-manifest "${MANIFEST_DIR}/${NATIVE_BOUNDARY_RUN_ID}.tsv" \
  --gate-manifest "${MANIFEST_DIR}/${DIRECT_TUNE_RUN_ID}.tsv" \
  --gate-manifest "${MANIFEST_DIR}/${NATIVE_TUNE_RUN_ID}.tsv" \
  --preheat-manifest "${MANIFEST_DIR}/${PREHEAT_RUN_ID}.tsv" \
  --build-input-snapshot "${BUILD_INPUT_SNAPSHOT}" \
  --metrics-tool "${METRICS_TOOL}" \
  --output-summary "${BRACKET_SUMMARY}" \
  --output-metrics "${FORMAL_METRICS}" \
  --output-manifest "${SET_MANIFEST}"

echo "[ok] formal set completed under one device lock"
echo "[ok] device=${ASCIFY_DEVICE} lock=${ASCIFY_DEVICE_LOCK}"
echo "[ok] set_manifest=${SET_MANIFEST}"
echo "[ok] bracket_summary=${BRACKET_SUMMARY}"
echo "[ok] formal_metrics=${FORMAL_METRICS}"
