#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
compat_header="$repo_root/include/ascify/ascify_cuda_compat.hpp"
action_cpp="$repo_root/src/AscifyAction.cpp"
action_header="$repo_root/src/AscifyAction.h"
runtime_map="$repo_root/src/CUDA2DPP_Runtime_API_functions.cpp"
runtime_type_map="$repo_root/src/CUDA2DPP_Runtime_API_types.cpp"

require_fixed() {
  if command -v rg >/dev/null 2>&1; then
    rg -F -- "$1" "$2" >/dev/null
  else
    grep -F -- "$1" "$2" >/dev/null
  fi
}

forbid_fixed() {
  if command -v rg >/dev/null 2>&1; then
    ! rg -F -- "$1" "$2" >/dev/null
  else
    ! grep -F -- "$1" "$2" >/dev/null
  fi
}

require_fixed '#include <simt_api/vector_functions.h>' "$compat_header"
require_fixed '#define ASCIFY_SIMT_LEGACY_HAS_VECTOR_CONSTRUCTORS 1' "$compat_header"
forbid_fixed 'struct float4' "$compat_header"
forbid_fixed 'make_float4(' "$compat_header"
require_fixed 'cudaStreamCreateWithFlags' "$runtime_map"
require_fixed 'ascify::cudaStreamDefault' "$runtime_type_map"
require_fixed 'ascify::cudaStreamNonBlocking' "$runtime_type_map"
require_fixed 'return ACL_ERROR_FEATURE_UNSUPPORTED;' "$compat_header"
require_fixed 'aclrtCtxGetCurrentDefaultStream(&stream)' "$compat_header"
require_fixed 'sCudaDefaultDim3' "$action_cpp"
require_fixed 'rewriteCudaDefaultDim3' "$action_cpp"
require_fixed '!variable->isLocalVarDecl()' "$action_cpp"
require_fixed 'variable->hasExternalStorage()' "$action_cpp"
require_fixed 'clang::CUDADeviceAttr' "$action_cpp"
require_fixed 'clang::CUDAGlobalAttr' "$action_cpp"
require_fixed 'rewrittenCudaDefaultDim3Offsets' "$action_header"
require_fixed 'sourceName.str() + " = dim3(1U, 1U, 1U)"' "$action_cpp"

if [ "$#" -eq 2 ] && [ "$1" = "--translated" ]; then
  output=$2
  require_fixed 'dim3 threads = dim3(1U, 1U, 1U), blocks = dim3(1U, 1U, 1U);' "$output"
  require_fixed 'static dim3 static_shape = dim3(1U, 1U, 1U);' "$output"
  require_fixed 'LaunchShape alias_shape = dim3(1U, 1U, 1U);' "$output"
  require_fixed 'dim3 explicit_shape(8U, 2U, 1U);' "$output"
  require_fixed 'dim3 brace_shape{4U, 2U, 1U};' "$output"
  require_fixed 'dim3 assignment_shape = dim3(2U, 1U, 1U);' "$output"
  require_fixed 'dim3 shape_array[2];' "$output"
  require_fixed 'dim3 attributed_shape __attribute__((unused));' "$output"
  require_fixed 'ASCIFY_TEST_DECLARE_DIM3(macro_shape);' "$output"
  require_fixed 'extern dim3 external_shape;' "$output"
  require_fixed 'dim3 global_shape;' "$output"
  require_fixed 'dim3 device_shape;' "$output"
  require_fixed 'dim3 shared_shapes[];' "$output"
  require_fixed 'dim3 local_device_shape;' "$output"
  require_fixed 'namespace user_types' "$output"
  require_fixed 'dim3 local;' "$output"
  forbid_fixed 'external_shape = dim3(1U, 1U, 1U)' "$output"
  forbid_fixed 'global_shape = dim3(1U, 1U, 1U)' "$output"
  forbid_fixed 'device_shape = dim3(1U, 1U, 1U)' "$output"
  forbid_fixed 'shared_shapes = dim3(1U, 1U, 1U)' "$output"
  forbid_fixed 'local_device_shape = dim3(1U, 1U, 1U)' "$output"
  forbid_fixed 'dim3 local = dim3(1U, 1U, 1U);' "$output"
  forbid_fixed 'shape_array = dim3(1U, 1U, 1U)' "$output"
  forbid_fixed 'attributed_shape = dim3(1U, 1U, 1U)' "$output"
  forbid_fixed 'macro_shape = dim3(1U, 1U, 1U)' "$output"
  require_fixed 'void BuildLaunchShapes(dim3* output)' "$output"
elif [ "$#" -ne 0 ]; then
  echo "usage: $0 [--translated OUTPUT]" >&2
  exit 2
fi

echo "target ABI compatibility checks passed"
