#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEST_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${TEST_ROOT}/../.." && pwd)"

WORK_ROOT="${WORK_ROOT:-${REPO_ROOT}/.work/softmax_rmsnorm_950}"
BIN_DIR="${BIN_DIR:-${WORK_ROOT}/bin}"
RESULT_DIR="${RESULT_DIR:-${WORK_ROOT}/results}"
CANN_ROOT="${CANN_ROOT:-}"

if [[ -z "${CANN_ROOT}" ]]; then
  echo "CANN_ROOT must point to a user-owned CANN installation" >&2
  exit 2
fi

if [[ -z "${ASCIFY_DEVICE:-}" || -z "${ASCIFY_DEVICE_LOCK:-}" ]]; then
  # select_device keeps the project-local flock open across this exec and every
  # build/run/parser child, closing the npu-smi selection race.
  REQUESTED_DEVICE="${ASCIFY_DEVICE:-${DEVICE:-}}"
  unset ASCIFY_DEVICE ASCIFY_DEVICE_LOCK
  if [[ -n "${REQUESTED_DEVICE}" ]]; then
    DEVICE="${REQUESTED_DEVICE}" exec "${SCRIPT_DIR}/select_device.sh" -- "$0" "$@"
  fi
  exec "${SCRIPT_DIR}/select_device.sh" -- "$0" "$@"
fi
if [[ ! "${ASCIFY_DEVICE}" =~ ^[0-9]+$ ]]; then
  echo "ASCIFY_DEVICE must be a non-negative integer" >&2
  exit 2
fi
if [[ ! -f "${CANN_ROOT}/set_env.sh" ]]; then
  echo "missing user-owned CANN environment: ${CANN_ROOT}/set_env.sh" >&2
  exit 1
fi

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_simt_hw_probes_$$}"
BUILD_LOG="${RESULT_DIR}/${RUN_ID}.build.log"
RUN_LOG="${RESULT_DIR}/${RUN_ID}.run.log"
RAW_CSV="${RESULT_DIR}/${RUN_ID}.csv"
SUMMARY_MD="${RESULT_DIR}/${RUN_ID}.summary.md"
PROBE_BINARY="${BIN_DIR}/simt_hw_probes"
CONTRACT_BINARY="${BIN_DIR}/rowwise_recipe_contract_probe"
PARSER="${TEST_ROOT}/probes/parse_results.py"

mkdir -p "${RESULT_DIR}"

{
  echo "[probe] run_id=${RUN_ID}"
  echo "[probe] physical_device=${ASCIFY_DEVICE}"
  echo "[probe] lock=${ASCIFY_DEVICE_LOCK}"
  echo "[probe] CANN_ROOT=${CANN_ROOT}"
  echo "[probe] source=${TEST_ROOT}/probes/simt_hw_probes.cce"
} | tee "${BUILD_LOG}"

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  "${SCRIPT_DIR}/build.sh" probe 2>&1 | tee -a "${BUILD_LOG}"
else
  echo "[probe] SKIP_BUILD=1" | tee -a "${BUILD_LOG}"
fi
if [[ ! -x "${PROBE_BINARY}" ]]; then
  echo "probe binary is missing: ${PROBE_BINARY}" >&2
  exit 1
fi
if [[ ! -x "${CONTRACT_BINARY}" ]]; then
  echo "recipe contract probe binary is missing: ${CONTRACT_BINARY}" >&2
  exit 1
fi

# Runtime paths are scoped to this process; no system or user environment file
# is modified.
set +u
# shellcheck disable=SC1090
source "${CANN_ROOT}/set_env.sh" >/dev/null
set -u

{
  echo "[probe] started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sha256sum \
    "${TEST_ROOT}/probes/rowwise_recipe_contract_probe.cce" \
    "${CONTRACT_BINARY}" \
    "${TEST_ROOT}/probes/simt_hw_probes.cce" \
    "${PROBE_BINARY}"
  echo "[probe] recipe_contract"
  "${CONTRACT_BINARY}"
  echo "[probe] health_before"
  npu-smi info -t health -i "${ASCIFY_DEVICE}" || true
  echo "[probe] usages_before"
  npu-smi info -t usages -i "${ASCIFY_DEVICE}" || true
  echo "[probe] proc_mem_before"
  npu-smi info -t proc-mem -i "${ASCIFY_DEVICE}" || true
  echo "[probe] command=${PROBE_BINARY} --device ${ASCIFY_DEVICE} $*"
} >"${RUN_LOG}" 2>&1

"${PROBE_BINARY}" --device "${ASCIFY_DEVICE}" "$@" >"${RAW_CSV}" 2>>"${RUN_LOG}"

result_rows="$(awk 'END { if (NR > 0) print NR - 1; else print 0 }' "${RAW_CSV}")"
if [[ "${result_rows}" != "17" ]]; then
  echo "expected 17 CSV result rows, got ${result_rows}" | tee -a "${RUN_LOG}" >&2
  exit 1
fi
python3 "${PARSER}" --strict "${RAW_CSV}" >"${SUMMARY_MD}" 2>>"${RUN_LOG}"

{
  echo "[probe] completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[probe] rows=${result_rows}"
  echo "[probe] health_after"
  npu-smi info -t health -i "${ASCIFY_DEVICE}" || true
  echo "[probe] usages_after"
  npu-smi info -t usages -i "${ASCIFY_DEVICE}" || true
  echo "[probe] raw_csv=${RAW_CSV}"
  echo "[probe] summary=${SUMMARY_MD}"
} >>"${RUN_LOG}" 2>&1

cat "${SUMMARY_MD}"
echo "[ok] complete 17-row SIMT hardware probe under one held device lock"
echo "[ok] build log: ${BUILD_LOG}"
echo "[ok] runtime log: ${RUN_LOG}"
echo "[ok] raw CSV: ${RAW_CSV}"
echo "[ok] summary: ${SUMMARY_MD}"
