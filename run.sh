#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ASCIFY_BINARY="${ASCIFY_BINARY:-${SCRIPT_DIR}/build/ascify-clang}"

: "${CUDA_PATH:?set CUDA_PATH to the CUDA parsing root}"
: "${CLANG_RESOURCE_DIRECTORY:?set CLANG_RESOURCE_DIRECTORY to the directory containing include/__clang_cuda_runtime_wrapper.h}"

if [[ "$#" -lt 1 ]]; then
  echo "usage: CUDA_PATH=... CLANG_RESOURCE_DIRECTORY=... $0 INPUT [ASCIFY_OPTIONS] [-- CLANG_OPTIONS]" >&2
  exit 2
fi
if [[ ! -x "${ASCIFY_BINARY}" ]]; then
  echo "Ascify binary is not executable: ${ASCIFY_BINARY}" >&2
  exit 1
fi

input="$1"
shift
exec "${ASCIFY_BINARY}" "${input}" \
  "--cuda-path=${CUDA_PATH}" \
  "--clang-resource-directory=${CLANG_RESOURCE_DIRECTORY}" \
  "$@"
