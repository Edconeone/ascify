#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEST_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${TEST_ROOT}/../.." && pwd)"

WORK_ROOT="${WORK_ROOT:-${REPO_ROOT}/.work/softmax_rmsnorm_950}"
BIN_DIR="${BIN_DIR:-${WORK_ROOT}/bin}"
ROWWISE_SIMD_RUNTIME_DIR="${ROWWISE_SIMD_RUNTIME_DIR:-}"
LAYERNORM_CHECK_PATHS="${LAYERNORM_CHECK_PATHS:-both}"
CANN_ROOT="${CANN_ROOT:-}"

if [[ -z "${ASCIFY_DEVICE:-}" || -z "${ASCIFY_DEVICE_LOCK:-}" ]]; then
  requested_device="${ASCIFY_DEVICE:-${DEVICE:-}}"
  unset ASCIFY_DEVICE ASCIFY_DEVICE_LOCK
  if [[ -n "${requested_device}" ]]; then
    DEVICE="${requested_device}" exec env UTIL_MAX=0 HBM_BW_MAX=0 \
      "${SCRIPT_DIR}/select_device.sh" -- "$0" "$@"
  fi
  exec env UTIL_MAX=0 HBM_BW_MAX=0 \
    "${SCRIPT_DIR}/select_device.sh" -- "$0" "$@"
fi

if [[ ! "${ASCIFY_DEVICE}" =~ ^[0-9]+$ ]]; then
  echo "ASCIFY_DEVICE must be a non-negative integer" >&2
  exit 2
fi
if [[ -z "${ROWWISE_SIMD_RUNTIME_DIR}" \
      || ! -d "${ROWWISE_SIMD_RUNTIME_DIR}" ]]; then
  echo "ROWWISE_SIMD_RUNTIME_DIR must name the built runtime lib directory" >&2
  exit 2
fi
case "${LAYERNORM_CHECK_PATHS}" in
  direct|generated|both) ;;
  *)
    echo "LAYERNORM_CHECK_PATHS must be direct, generated, or both" >&2
    exit 2
    ;;
esac
if [[ -n "${CANN_ROOT}" ]]; then
  if [[ ! -f "${CANN_ROOT}/set_env.sh" ]]; then
    echo "missing user-owned CANN environment: ${CANN_ROOT}/set_env.sh" >&2
    exit 2
  fi
  set +u
  # shellcheck disable=SC1090
  source "${CANN_ROOT}/set_env.sh" >/dev/null
  set -u
fi

direct_binary="${BIN_DIR}/layernorm_hybrid_check_production_fast"
generated_binary="${BIN_DIR}/layernorm_generated_check_production_fast"
if [[ "${LAYERNORM_CHECK_PATHS}" != "generated" \
      && ! -x "${direct_binary}" ]]; then
  echo "missing direct-ABI LayerNorm check: ${direct_binary}" >&2
  exit 1
fi
if [[ "${LAYERNORM_CHECK_PATHS}" != "direct" \
      && ! -x "${generated_binary}" ]]; then
  echo "missing generated-wrapper LayerNorm check: ${generated_binary}" >&2
  exit 1
fi

export LD_LIBRARY_PATH="${ROWWISE_SIMD_RUNTIME_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
echo "[layernorm] device=${ASCIFY_DEVICE} paths=${LAYERNORM_CHECK_PATHS}" >&2
if [[ "${LAYERNORM_CHECK_PATHS}" != "generated" ]]; then
  "${direct_binary}" --device "${ASCIFY_DEVICE}" "$@"
fi
if [[ "${LAYERNORM_CHECK_PATHS}" != "direct" ]]; then
  "${generated_binary}" --device "${ASCIFY_DEVICE}" "$@"
fi
echo "[layernorm] all requested device paths passed" >&2
