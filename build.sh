#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

: "${LLVM_PROJECT_PATH:?set LLVM_PROJECT_PATH to an LLVM/Clang source tree with a configured build directory}"

BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build}"
INSTALL_ROOT="${INSTALL_ROOT:-${SCRIPT_DIR}/ascify_install}"
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-${LLVM_PROJECT_PATH}/build}"
ASCIFY_CC="${ASCIFY_CC:-${LLVM_BUILD_DIR}/bin/clang}"
ASCIFY_CXX="${ASCIFY_CXX:-${LLVM_BUILD_DIR}/bin/clang++}"
ASCIFY_BUILD_JOBS="${ASCIFY_BUILD_JOBS:-2}"

for required_path in \
  "${LLVM_BUILD_DIR}" "${ASCIFY_CC}" "${ASCIFY_CXX}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "missing build dependency: ${required_path}" >&2
    exit 1
  fi
done

linker_args=()
if [[ -n "${ASCIFY_LINKER:-}" ]]; then
  if [[ ! -e "${ASCIFY_LINKER}" ]]; then
    echo "ASCIFY_LINKER does not exist: ${ASCIFY_LINKER}" >&2
    exit 1
  fi
  linker_args+=("-DCMAKE_LINKER=${ASCIFY_LINKER}")
elif [[ -e "${LLVM_BUILD_DIR}/bin/lld" ]]; then
  linker_args+=("-DCMAKE_LINKER=${LLVM_BUILD_DIR}/bin/lld")
fi

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -G "${CMAKE_GENERATOR:-Ninja}" \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_ROOT}" \
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}" \
  -DCMAKE_PREFIX_PATH="${LLVM_BUILD_DIR}" \
  -DCMAKE_C_COMPILER="${ASCIFY_CC}" \
  -DCMAKE_CXX_COMPILER="${ASCIFY_CXX}" \
  "${linker_args[@]}" \
  "$@"

cmake --build "${BUILD_DIR}" --parallel "${ASCIFY_BUILD_JOBS}"
