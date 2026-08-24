#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
device_map="$repo_root/src/CUDA2DPP_Device_API.cpp"
runtime_map="$repo_root/src/CUDA2DPP_Runtime_API_functions.cpp"
runtime_type_map="$repo_root/src/CUDA2DPP_Runtime_API_types.cpp"
action_cpp="$repo_root/src/AscifyAction.cpp"
action_header="$repo_root/src/AscifyAction.h"
argparse_cpp="$repo_root/src/ArgParse.cpp"
argparse_header="$repo_root/src/ArgParse.h"
compat_header="$repo_root/include/ascify/ascify_cuda_compat.hpp"
golden_input="$repo_root/tests/rewrite/simt_compat_input.cu"
golden_rules="$repo_root/tests/rewrite/simt_compat.expected"

search_fixed() {
  needle=$1
  file=$2
  if command -v rg >/dev/null 2>&1; then
    rg -F -- "$needle" "$file"
  else
    grep -F -- "$needle" "$file"
  fi
}

require_fixed() {
  needle=$1
  file=$2
  if ! search_fixed "$needle" "$file" >/dev/null; then
    echo "missing '$needle' in $file" >&2
    exit 1
  fi
}

forbid_fixed() {
  needle=$1
  file=$2
  if search_fixed "$needle" "$file" >/dev/null; then
    echo "forbidden '$needle' found in $file" >&2
    exit 1
  fi
}

check_static_contract() {
  require_fixed 'm["__shfl_xor_sync"]   = {"ascify::shfl_xor_sync"' "$device_map"
  require_fixed 'm["__expf"]            = {"expf"' "$device_map"
  require_fixed 'm["__frsqrt_rn"]        = {"rsqrtf"' "$device_map"
  require_fixed 'm["__align__"]          = {"ASCIFY_ALIGN"' "$device_map"
  require_fixed 'm["__syncwarp"]        = {"ascify::syncwarp"' "$device_map"
  require_fixed 'm["cudaDeviceGetAttribute"]' "$runtime_map"
  require_fixed '{"ascify::cudaDeviceGetAttribute"' "$runtime_map"
  require_fixed '{"ascify::cudaPeekAtLastError"' "$runtime_map"
  require_fixed '{"ascify::cudaOccupancyMaxActiveBlocksPerMultiprocessor"' "$runtime_map"
  require_fixed '{"ascify::cudaFuncGetAttributes"' "$runtime_map"
  require_fixed '{"ascify::cudaFuncSetAttribute"' "$runtime_map"
  require_fixed 'SEC::OCCUPANCY, FULL' "$runtime_map"
  require_fixed 'm["cudaDevAttrMaxSharedMemoryPerBlockOptin"]' "$runtime_type_map"
  require_fixed '{"ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE"' "$runtime_type_map"
  require_fixed 'm["cudaFuncAttributes"]' "$runtime_type_map"
  require_fixed '#include <ascify/ascify_cuda_compat.hpp>' "$action_cpp"
  require_fixed 'NoLowerDeviceDoubleParams' "$argparse_header"
  require_fixed '"no-lower-device-double-params"' "$argparse_cpp"
  require_fixed 'cl::init(false)' "$argparse_cpp"
  require_fixed 'std::string(NoLowerDeviceDoubleParams.ArgStr)' "$argparse_cpp"
  require_fixed '"target-policy"' "$argparse_cpp"
  require_fixed 'cl::init("portable")' "$argparse_cpp"
  require_fixed '"simt-math"' "$argparse_cpp"
  require_fixed 'cl::init("precise")' "$argparse_cpp"
  require_fixed 'extern cl::opt<std::string> TargetRecipe' "$argparse_header"
  require_fixed '"target-recipe"' "$argparse_cpp"
  require_fixed 'cl::init("none")' "$argparse_cpp"
  require_fixed 'std::string(TargetRecipe.ArgStr)' "$argparse_cpp"
  require_fixed 'sCudaGlobalScalarDoubleParam' "$action_cpp"
  require_fixed 'lowerCudaGlobalScalarDoubleParam' "$action_cpp"
  require_fixed 'clang::CUDAGlobalAttr' "$action_cpp"
  require_fixed 'clang::CUDADeviceAttr' "$action_cpp"
  require_fixed 'clang::BuiltinTypeLoc' "$action_cpp"
  require_fixed 'getNodeAs<clang::ParmVarDecl>' "$action_cpp"
  require_fixed 'loweredDeviceDoubleParamOffsets' "$action_header"
  require_fixed 'CONV_DEVICE_TYPE' "$action_cpp"
  require_fixed 'lowering by-value scalar double parameter' "$action_cpp"
  forbid_fixed 'if (cudaLaunchKernel(Result)) return;' "$action_cpp"
  require_fixed '__global__ void NarrowScalarDoubleParams' "$golden_input"
  require_fixed 'const double eps, double scale' "$golden_input"
  require_fixed '+__global__ void NarrowScalarDoubleParams(float* output, const float eps, float scale,' "$golden_rules"
  require_fixed '+const double* pointer, double values[])' "$golden_rules"
  require_fixed '+__aicore__ double KeepDeviceDoubleParam(double value)' "$golden_rules"
  require_fixed '+double KeepHostDoublePrecision(double value)' "$golden_rules"
  require_fixed '+std::is_same<double, double>' "$golden_rules"
  require_fixed '#include <algorithm>' "$compat_header"
  require_fixed '#include <type_traits>' "$compat_header"
  require_fixed 'inline aclError cudaDeviceGetAttribute' "$compat_header"
  require_fixed 'inline aclError cudaPeekAtLastError' "$compat_header"
  require_fixed 'cudaOccupancyMaxActiveBlocksPerMultiprocessor' "$compat_header"
  require_fixed 'inline aclError cudaFuncGetAttributes' "$compat_header"
  require_fixed 'inline aclError cudaFuncSetAttribute' "$compat_header"
  require_fixed 'float warp_reduce_add(float value)' "$compat_header"
  require_fixed 'int32_t warp_reduce_add(int32_t value)' "$compat_header"
  require_fixed 'uint32_t warp_reduce_add(uint32_t value)' "$compat_header"
  require_fixed 'return ::fdividef(numerator, denominator);' "$compat_header"
  require_fixed '#define ASCIFY_ALIGN' "$compat_header"
  forbid_fixed 'm["double"]' "$device_map"
  forbid_fixed '{"aclrtGetDeviceInfo"' "$runtime_map"
  forbid_fixed '{"aclrtPeekAtLastError"' "$runtime_map"
}

check_translated_output() {
  output=$1
  expected="$repo_root/tests/rewrite/simt_compat.expected"
  while IFS= read -r rule || [ -n "$rule" ]; do
    case "$rule" in
      ''|'#'*) continue ;;
      '+'*) require_fixed "${rule#?}" "$output" ;;
      '-'*) forbid_fixed "${rule#?}" "$output" ;;
      *)
        echo "invalid golden rule: $rule" >&2
        exit 1
        ;;
    esac
  done < "$expected"
}

check_static_contract

cxx=${CXX:-c++}
if command -v "$cxx" >/dev/null 2>&1; then
  "$cxx" -std=c++17 -fsyntax-only \
    -I"$repo_root/tests/rewrite/stubs" \
    -I"$repo_root/include" \
    "$repo_root/tests/rewrite/compat_header_syntax.cpp"
fi

if [ "$#" -eq 2 ] && [ "$1" = "--translated" ]; then
  check_translated_output "$2"
elif [ "$#" -ne 0 ]; then
  echo "usage: sh tests/rewrite/check_simt_compat.sh [--translated OUTPUT]" >&2
  exit 2
fi

echo "SIMT compatibility rewrite checks passed"
