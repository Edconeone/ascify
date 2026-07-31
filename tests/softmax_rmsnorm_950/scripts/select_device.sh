#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEST_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${TEST_ROOT}/../.." && pwd)"
WORK_ROOT="${WORK_ROOT:-${REPO_ROOT}/.work/softmax_rmsnorm_950}"
LOCK_DIR="${WORK_ROOT}/locks"
LOCK_HELPER="${SCRIPT_DIR}/safe_lock_exec.py"
UTIL_MAX="${UTIL_MAX:-0}"
HBM_BW_MAX="${HBM_BW_MAX:-${HBM_UTIL_MAX:-0}}"

LOCKED_STAGE=0
LOCKED_DEVICE_ID=""
LOCKED_DEVICE_PATH=""
if [[ "${1:-}" == "--locked-device" ]]; then
  if (( $# < 3 )); then
    echo "--locked-device requires DEVICE_ID and LOCK_PATH" >&2
    exit 2
  fi
  LOCKED_STAGE=1
  LOCKED_DEVICE_ID="$2"
  LOCKED_DEVICE_PATH="$3"
  shift 3
  if [[ "${1:-}" == "--" ]]; then shift; fi
elif [[ "${1:-}" == "--" ]]; then
  shift
fi

if ! command -v npu-smi >/dev/null 2>&1; then
  echo "npu-smi is unavailable; cannot safely auto-select a 950PR device" >&2
  exit 1
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "flock is unavailable; refusing an unlocked device selection" >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1 || [[ ! -f "${LOCK_HELPER}" ]]; then
  echo "python3 and ${LOCK_HELPER} are required for safe device locking" >&2
  exit 1
fi

mkdir -p "${LOCK_DIR}"
LOCK_DIR="$(cd -- "${LOCK_DIR}" && pwd -P)"

metric_value() {
  local text="$1"
  local key="$2"
  awk -F: -v key="${key}" '
    {
      field=$1
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", field)
    }
    index(field, key) == 1 {
      value=substr($0, index($0, ":") + 1)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
      print value
      exit
    }
  ' <<<"${text}"
}

numeric_metric() {
  local text="$1"
  local key="$2"
  local value
  value="$(metric_value "${text}" "${key}")"
  value="$(tr -d '%[:space:]' <<<"${value}")"
  if [[ "${value}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "${value}"
  fi
}

metric_at_most() {
  local value="$1"
  local maximum="$2"
  awk -v value="${value}" -v maximum="${maximum}" \
    'BEGIN { exit !((value + 0.0) <= (maximum + 0.0)) }'
}

if [[ ! "${UTIL_MAX}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "UTIL_MAX must be a non-negative number, got: ${UTIL_MAX}" >&2
  exit 2
fi
if [[ ! "${HBM_BW_MAX}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "HBM_BW_MAX must be a non-negative number, got: ${HBM_BW_MAX}" >&2
  exit 2
fi

DEVICE_REJECTION_REASON=""

device_is_idle_and_healthy() {
  local device_id="$1"
  local health_text usage_text process_text
  local health npu_util aicore_util aiv_util hbm_bw_util
  local compute_util
  local compute_metric_count=0

  if ! health_text="$(npu-smi info -t health -i "${device_id}" 2>/dev/null)"; then
    DEVICE_REJECTION_REASON="health query failed"
    return 1
  fi
  if ! usage_text="$(npu-smi info -t usages -i "${device_id}" 2>/dev/null)"; then
    DEVICE_REJECTION_REASON="usage query failed"
    return 1
  fi
  if ! process_text="$(npu-smi info -t proc-mem -i "${device_id}" 2>/dev/null)"; then
    DEVICE_REJECTION_REASON="process query failed"
    return 1
  fi

  health="$(metric_value "${health_text}" "Health")"
  if [[ "${health}" != "OK" ]]; then
    DEVICE_REJECTION_REASON="health=${health:-unparseable}"
    return 1
  fi
  if ! grep -Eiq 'no[[:space:]].*process' <<<"${process_text}"; then
    DEVICE_REJECTION_REASON="device has a process or process output is unparseable"
    return 1
  fi

  npu_util="$(numeric_metric "${usage_text}" "NPU Utilization")"
  aicore_util="$(numeric_metric "${usage_text}" "Aicore Usage Rate")"
  aiv_util="$(numeric_metric "${usage_text}" "Aivector Usage Rate")"
  hbm_bw_util="$(numeric_metric "${usage_text}" "HBM Bandwidth Usage Rate")"
  if [[ -z "${hbm_bw_util}" ]]; then
    DEVICE_REJECTION_REASON="HBM bandwidth utilization is unparseable"
    return 1
  fi
  if ! metric_at_most "${hbm_bw_util}" "${HBM_BW_MAX}"; then
    DEVICE_REJECTION_REASON="HBM bandwidth utilization ${hbm_bw_util}% exceeds ${HBM_BW_MAX}%"
    return 1
  fi
  for compute_util in "${npu_util}" "${aicore_util}" "${aiv_util}"; do
    [[ -z "${compute_util}" ]] && continue
    compute_metric_count=$((compute_metric_count + 1))
    if ! metric_at_most "${compute_util}" "${UTIL_MAX}"; then
      DEVICE_REJECTION_REASON="compute utilization ${compute_util}% exceeds ${UTIL_MAX}%"
      return 1
    fi
  done
  if (( compute_metric_count == 0 )); then
    DEVICE_REJECTION_REASON="all compute utilization fields are unparseable"
    return 1
  fi
  DEVICE_REJECTION_REASON=""
  return 0
}

if [[ "${LOCKED_STAGE}" == "1" ]]; then
  if [[ ! "${LOCKED_DEVICE_ID}" =~ ^[0-9]+$ ]]; then
    echo "locked device ID is invalid: ${LOCKED_DEVICE_ID}" >&2
    exit 2
  fi
  expected_lock="${LOCK_DIR}/device-${LOCKED_DEVICE_ID}.lock"
  if [[ "${LOCKED_DEVICE_PATH}" != "${expected_lock}"
        || -L "${LOCKED_DEVICE_PATH}" ]] \
      || ! { true >&9; } 2>/dev/null || ! flock -n 9; then
    echo "safe device lock identity is invalid: ${LOCKED_DEVICE_PATH}" >&2
    exit 1
  fi
  if ! device_is_idle_and_healthy "${LOCKED_DEVICE_ID}"; then
    echo "[device] reject id=${LOCKED_DEVICE_ID} after lock: ${DEVICE_REJECTION_REASON}" >&2
    exit 1
  fi
  if (( $# == 0 )); then
    echo "${LOCKED_DEVICE_ID}"
    echo "warning: no command supplied; the device lock is released on exit" >&2
    exit 0
  fi
  export ASCIFY_DEVICE="${LOCKED_DEVICE_ID}"
  export ASCIFY_DEVICE_LOCK="${LOCKED_DEVICE_PATH}"
  echo "[device] selected healthy idle Ascend950PR id=${LOCKED_DEVICE_ID}; lock=${LOCKED_DEVICE_PATH}" >&2
  set +e
  "$@"
  command_status=$?
  set -e
  # Exit 200 is reserved for the parent helper's lock-busy signal.
  if (( command_status == 200 )); then command_status=199; fi
  exit "${command_status}"
fi

candidate_ids=()
if [[ -n "${DEVICE:-}" ]]; then
  if [[ ! "${DEVICE}" =~ ^[0-9]+$ ]]; then
    echo "DEVICE must be a non-negative integer, got: ${DEVICE}" >&2
    exit 2
  fi
  candidate_ids=("${DEVICE}")
else
  while IFS= read -r discovered_id; do
    [[ -n "${discovered_id}" ]] && candidate_ids+=("${discovered_id}")
  done < <(
    npu-smi info -m 2>/dev/null |
      awk '
        NR > 1 && $1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ && $3 != "-" {
          for (field = 4; field <= NF; ++field) {
            if ($field ~ /Ascend/) {
              print $1
              break
            }
          }
        }
      ' |
      sort -n -u
  )
fi

if (( ${#candidate_ids[@]} == 0 )); then
  echo "no Ascend device IDs were discovered" >&2
  exit 1
fi

for device_id in "${candidate_ids[@]}"; do
  if ! device_is_idle_and_healthy "${device_id}"; then
    echo "[device] skip id=${device_id}: ${DEVICE_REJECTION_REASON}" >&2
    continue
  fi
  lock_path="${LOCK_DIR}/device-${device_id}.lock"
  set +e
  python3 -B "${LOCK_HELPER}" \
    --fd 9 --path "${lock_path}" --busy-exit 200 -- \
    "$0" --locked-device "${device_id}" "${lock_path}" -- "$@"
  lock_status=$?
  set -e
  if (( lock_status == 200 )); then
    echo "[device] skip id=${device_id}: project lock is held" >&2
    continue
  fi
  exit "${lock_status}"
done

if [[ -n "${DEVICE:-}" ]]; then
  echo "requested DEVICE=${DEVICE} is unhealthy, busy, utilized, or already locked" >&2
else
  echo "no healthy idle unlocked Ascend950PR device is currently available" >&2
fi
exit 1
