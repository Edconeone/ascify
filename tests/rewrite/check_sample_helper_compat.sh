#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
action_cpp="$repo_root/src/AscifyAction.cpp"
compat_header="$repo_root/include/ascify/ascify_cuda_compat.hpp"
host_test="$repo_root/tests/rewrite/sample_helper_compat_test.cpp"
fixtures="$repo_root/tests/rewrite/fixtures"

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

require_regex() {
  pattern=$1
  file=$2
  if ! grep -E -- "$pattern" "$file" >/dev/null; then
    echo "missing regex '$pattern' in $file" >&2
    exit 1
  fi
}

require_fixed 'MacroExpands(const clang::Token &MacroNameTok' "$action_cpp"
require_fixed 'isRecognizedNvidiaSampleHelperCuda' "$action_cpp"
require_fixed 'locationComesFromRecognizedNvidiaSampleHelper' "$action_cpp"
require_fixed 'macroNameLocation.isMacroID()' "$action_cpp"
require_fixed 'ct::Replacements staged(*replacements);' "$action_cpp"
require_fixed '*replacements = std::move(staged);' "$action_cpp"
require_fixed 'nvidiaSampleHelperRawAuditCompleted = true;' "$action_cpp"
require_fixed 'nvidiaSampleHelperRawAuditCompleted &&' "$action_cpp"
require_fixed 'sampleCheckCudaErrors' "$compat_header"
require_fixed 'const aclError status = cudaGetLastError();' "$compat_header"
require_fixed '#expression, __FILE__, __LINE__' "$compat_header"
forbid_fixed 'findCudaDevice(' "$compat_header"

test_tmp=$(mktemp -d "${TMPDIR:-/tmp}/ascify-sample-helper.XXXXXX")
cleanup() {
  rm -rf -- "$test_tmp"
}
trap cleanup 0 1 2 15

cxx=${CXX:-c++}
if command -v "$cxx" >/dev/null 2>&1; then
  "$cxx" -std=c++17 \
    -I"$repo_root/tests/rewrite/stubs" \
    -I"$repo_root/include" \
    "$host_test" -o "$test_tmp/sample-helper-host"
  "$test_tmp/sample-helper-host" check-success
  "$test_tmp/sample-helper-host" get-success-twice

  set +e
  "$test_tmp/sample-helper-host" check-failure \
    >"$test_tmp/check-failure.stdout" \
    2>"$test_tmp/check-failure.stderr"
  check_failure_rc=$?
  "$test_tmp/sample-helper-host" get-failure \
    >"$test_tmp/get-failure.stdout" \
    2>"$test_tmp/get-failure.stderr"
  get_failure_rc=$?
  set -e
  [ "$check_failure_rc" -eq 1 ]
  [ "$get_failure_rc" -eq 1 ]
  require_fixed 'Ascify CUDA sample helper error' \
    "$test_tmp/check-failure.stderr"
  require_fixed 'statusOnce(ACL_ERROR_RT_PARAM_INVALID)' \
    "$test_tmp/check-failure.stderr"
  require_regex 'sample_helper_compat_test\.cpp\([1-9][0-9]*\)' \
    "$test_tmp/check-failure.stderr"
  require_fixed 'Ascify CUDA sample helper last error' \
    "$test_tmp/get-failure.stderr"
  require_fixed 'expected failure' "$test_tmp/get-failure.stderr"
  require_regex 'sample_helper_compat_test\.cpp\([1-9][0-9]*\)' \
    "$test_tmp/get-failure.stderr"
fi

if [ "$#" -eq 0 ]; then
  echo "sample-helper host contracts passed"
  exit 0
fi
if [ "$#" -ne 6 ] || [ "$1" != "--binary" ] || \
   [ "$3" != "--cuda-path" ] || [ "$5" != "--resource-dir" ]; then
  echo "usage: $0 [--binary ASCIFY --cuda-path CUDA --resource-dir RESOURCE]" >&2
  exit 2
fi

binary=$2
cuda_path=$4
resource_dir=$6
[ -x "$binary" ]
[ -d "$cuda_path" ]
[ -d "$resource_dir" ]

translate() {
  input=$1
  include_dir=$2
  output=$3
  log=$4
  "$binary" "$input" \
    --target-policy=dav-c310-vec \
    --simt-math=fast \
    --default-preprocessor \
    --cuda-path="$cuda_path" \
    --clang-resource-directory="$resource_dir" \
    -o "$output" -- -x cuda -std=c++17 -I"$include_dir" -I"$fixtures" \
    >"$log.stdout" 2>"$log.stderr"
}

translate_with_command_line_collision() {
  input=$1
  include_dir=$2
  output=$3
  log=$4
  "$binary" "$input" \
    --target-policy=dav-c310-vec \
    --simt-math=fast \
    --default-preprocessor \
    --cuda-path="$cuda_path" \
    --clang-resource-directory="$resource_dir" \
    -o "$output" -- -x cuda -std=c++17 -I"$include_dir" -I"$fixtures" \
    '-DASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS(expression)=ascifyCommandLineCheck(expression)' \
    '-DASCIFY_NVIDIA_SAMPLE_GET_LAST_CUDA_ERROR(message)=ascifyCommandLineLastError(message)' \
    >"$log.stdout" 2>"$log.stderr"
}

translate "$fixtures/sample_helper_supported.cu" \
  "$fixtures/nvidia_samples/Common" \
  "$test_tmp/supported.cpp" "$test_tmp/supported"
require_fixed '#include <ascify/ascify_cuda_compat.hpp>' \
  "$test_tmp/supported.cpp"
require_fixed 'ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS(ascify::cudaMalloc' \
  "$test_tmp/supported.cpp"
require_fixed 'ASCIFY_NVIDIA_SAMPLE_GET_LAST_CUDA_ERROR("kernel launch")' \
  "$test_tmp/supported.cpp"
forbid_fixed 'helper_cuda.h' "$test_tmp/supported.cpp"
require_fixed 'include removed' "$test_tmp/supported.stderr"

translate "$fixtures/sample_helper_residual.cu" \
  "$fixtures/nvidia_samples/Common" \
  "$test_tmp/residual.cpp" "$test_tmp/residual"
require_fixed '#include <helper_cuda.h>' "$test_tmp/residual.cpp"
require_fixed 'findCudaDevice(argc, argv)' "$test_tmp/residual.cpp"
require_fixed 'include kept' "$test_tmp/residual.stderr"

translate "$fixtures/sample_helper_unsupported_macro.cu" \
  "$fixtures/nvidia_samples/Common" \
  "$test_tmp/unsupported-macro.cpp" "$test_tmp/unsupported-macro"
require_fixed '#include <helper_cuda.h>' "$test_tmp/unsupported-macro.cpp"
require_fixed 'NVIDIA_SAMPLE_HELPER_ONLY_MACRO(value)' \
  "$test_tmp/unsupported-macro.cpp"
require_fixed 'unsupported_macro=1' \
  "$test_tmp/unsupported-macro.stderr"

translate "$fixtures/sample_helper_user_same_names.cu" \
  "$fixtures/user_helpers" \
  "$test_tmp/user.cpp" "$test_tmp/user"
require_fixed '#include <helper_cuda.h>' "$test_tmp/user.cpp"
require_fixed 'checkCudaErrors(0)' "$test_tmp/user.cpp"
require_fixed 'getLastCudaError("user macro")' "$test_tmp/user.cpp"
forbid_fixed 'ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS' "$test_tmp/user.cpp"
forbid_fixed 'ASCIFY_NVIDIA_SAMPLE_GET_LAST_CUDA_ERROR' "$test_tmp/user.cpp"

translate "$fixtures/sample_helper_duplicate_include.cu" \
  "$fixtures/nvidia_samples/Common" \
  "$test_tmp/duplicate.cpp" "$test_tmp/duplicate"
[ "$(grep -F -c '#include <helper_cuda.h>' "$test_tmp/duplicate.cpp")" -eq 2 ]
require_fixed 'checkCudaErrors(ascify::cudaMalloc' "$test_tmp/duplicate.cpp"
forbid_fixed 'ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS' \
  "$test_tmp/duplicate.cpp"
require_fixed 'expected exactly one recognized helper_cuda.h include' \
  "$test_tmp/duplicate.stderr"
require_fixed 'include kept' "$test_tmp/duplicate.stderr"

translate "$fixtures/sample_helper_altered_active_macro.cu" \
  "$fixtures/altered_nvidia_samples/Common" \
  "$test_tmp/altered-active.cpp" "$test_tmp/altered-active"
require_fixed '#include <helper_cuda.h>' "$test_tmp/altered-active.cpp"
require_fixed 'checkCudaErrors(0)' "$test_tmp/altered-active.cpp"
require_fixed 'getLastCudaError("altered")' "$test_tmp/altered-active.cpp"
forbid_fixed 'ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS' \
  "$test_tmp/altered-active.cpp"
forbid_fixed 'ASCIFY_NVIDIA_SAMPLE_GET_LAST_CUDA_ERROR' \
  "$test_tmp/altered-active.cpp"
require_fixed "active macro body for 'checkCudaErrors' is not admitted" \
  "$test_tmp/altered-active.stderr"
require_fixed "active macro body for 'getLastCudaError' is not admitted" \
  "$test_tmp/altered-active.stderr"
require_fixed 'include kept' "$test_tmp/altered-active.stderr"

translate "$fixtures/sample_helper_mixed_residual.cu" \
  "$fixtures/nvidia_samples/Common" \
  "$test_tmp/mixed-residual.cpp" "$test_tmp/mixed-residual"
require_fixed '#include <helper_cuda.h>' "$test_tmp/mixed-residual.cpp"
require_fixed 'checkCudaErrors(ascify::cudaMalloc' \
  "$test_tmp/mixed-residual.cpp"
require_fixed 'findCudaDevice(argc' "$test_tmp/mixed-residual.cpp"
forbid_fixed 'ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS' \
  "$test_tmp/mixed-residual.cpp"
require_fixed 'global proof failed before replacement' \
  "$test_tmp/mixed-residual.stderr"
require_fixed 'include kept' "$test_tmp/mixed-residual.stderr"

translate "$fixtures/sample_helper_mixed_custom_int.cu" \
  "$fixtures/nvidia_samples/Common" \
  "$test_tmp/mixed-custom.cpp" "$test_tmp/mixed-custom"
require_fixed '#include <helper_cuda.h>' "$test_tmp/mixed-custom.cpp"
[ "$(grep -F -c 'checkCudaErrors(' "$test_tmp/mixed-custom.cpp")" -eq 2 ]
require_fixed 'checkCudaErrors(ascify::cudaMalloc' "$test_tmp/mixed-custom.cpp"
require_fixed 'checkCudaErrors(customStatus())' "$test_tmp/mixed-custom.cpp"
forbid_fixed 'ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS' \
  "$test_tmp/mixed-custom.cpp"
require_fixed 'status domain not proven' "$test_tmp/mixed-custom.stderr"
require_fixed 'global proof failed before replacement' \
  "$test_tmp/mixed-custom.stderr"
require_fixed 'include kept' "$test_tmp/mixed-custom.stderr"

translate "$fixtures/sample_helper_mixed_pp_raw.cu" \
  "$fixtures/nvidia_samples/Common" \
  "$test_tmp/mixed-pp-raw.cpp" "$test_tmp/mixed-pp-raw"
require_fixed '#include <helper_cuda.h>' "$test_tmp/mixed-pp-raw.cpp"
require_fixed 'checkCudaErrors(ascify::cudaMalloc' \
  "$test_tmp/mixed-pp-raw.cpp"
require_fixed '#ifdef getLastCudaError' "$test_tmp/mixed-pp-raw.cpp"
require_fixed 'ASCIFY_TEST_STRINGIZE(checkCudaErrors)' \
  "$test_tmp/mixed-pp-raw.cpp"
forbid_fixed 'ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS' \
  "$test_tmp/mixed-pp-raw.cpp"
require_fixed 'residual raw PP use' "$test_tmp/mixed-pp-raw.stderr"
require_fixed 'include kept' "$test_tmp/mixed-pp-raw.stderr"

translate "$fixtures/sample_helper_pp_alias_invocation.cu" \
  "$fixtures/nvidia_samples/Common" \
  "$test_tmp/pp-alias.cpp" "$test_tmp/pp-alias"
require_fixed '#include <helper_cuda.h>' "$test_tmp/pp-alias.cpp"
require_fixed '#define ASCIFY_TEST_CHECK_ALIAS checkCudaErrors' \
  "$test_tmp/pp-alias.cpp"
require_fixed 'ASCIFY_TEST_CHECK_ALIAS(ascify::cudaMalloc' \
  "$test_tmp/pp-alias.cpp"
forbid_fixed 'ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS' \
  "$test_tmp/pp-alias.cpp"
require_fixed 'not a direct source-token invocation' \
  "$test_tmp/pp-alias.stderr"
require_fixed 'global proof failed before replacement' \
  "$test_tmp/pp-alias.stderr"
require_fixed 'include kept' "$test_tmp/pp-alias.stderr"

translate "$fixtures/sample_helper_preexisting_output_macro.cu" \
  "$fixtures/nvidia_samples/Common" \
  "$test_tmp/output-collision.cpp" "$test_tmp/output-collision"
require_fixed '#include <helper_cuda.h>' "$test_tmp/output-collision.cpp"
require_fixed 'checkCudaErrors(ascify::cudaMalloc' \
  "$test_tmp/output-collision.cpp"
require_fixed 'getLastCudaError("user output macro collision")' \
  "$test_tmp/output-collision.cpp"
[ "$(grep -F -c 'ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS' \
       "$test_tmp/output-collision.cpp")" -eq 1 ]
[ "$(grep -F -c 'ASCIFY_NVIDIA_SAMPLE_GET_LAST_CUDA_ERROR' \
       "$test_tmp/output-collision.cpp")" -eq 1 ]
require_fixed 'reserved output macro' "$test_tmp/output-collision.stderr"
require_fixed 'global proof failed before replacement' \
  "$test_tmp/output-collision.stderr"
require_fixed 'include kept' "$test_tmp/output-collision.stderr"

translate "$fixtures/sample_helper_user_header_output_macro.cu" \
  "$fixtures/nvidia_samples/Common" \
  "$test_tmp/user-header-collision.cpp" \
  "$test_tmp/user-header-collision"
require_fixed '#include "sample_helper_user_output_macros.h"' \
  "$test_tmp/user-header-collision.cpp"
require_fixed '#include <helper_cuda.h>' \
  "$test_tmp/user-header-collision.cpp"
require_fixed 'checkCudaErrors(ascify::cudaMalloc' \
  "$test_tmp/user-header-collision.cpp"
require_fixed 'getLastCudaError("user header output macro collision")' \
  "$test_tmp/user-header-collision.cpp"
forbid_fixed 'ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS(' \
  "$test_tmp/user-header-collision.cpp"
forbid_fixed 'ASCIFY_NVIDIA_SAMPLE_GET_LAST_CUDA_ERROR(' \
  "$test_tmp/user-header-collision.cpp"
require_fixed 'reserved output macro' \
  "$test_tmp/user-header-collision.stderr"
require_fixed 'include kept' "$test_tmp/user-header-collision.stderr"

translate_with_command_line_collision \
  "$fixtures/sample_helper_supported.cu" \
  "$fixtures/nvidia_samples/Common" \
  "$test_tmp/command-line-collision.cpp" \
  "$test_tmp/command-line-collision"
require_fixed '#include <helper_cuda.h>' \
  "$test_tmp/command-line-collision.cpp"
require_fixed 'checkCudaErrors(ascify::cudaMalloc' \
  "$test_tmp/command-line-collision.cpp"
require_fixed 'getLastCudaError("kernel launch")' \
  "$test_tmp/command-line-collision.cpp"
forbid_fixed 'ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS(' \
  "$test_tmp/command-line-collision.cpp"
forbid_fixed 'ASCIFY_NVIDIA_SAMPLE_GET_LAST_CUDA_ERROR(' \
  "$test_tmp/command-line-collision.cpp"
require_fixed 'reserved output macro' \
  "$test_tmp/command-line-collision.stderr"
require_fixed 'include kept' "$test_tmp/command-line-collision.stderr"

translate "$fixtures/sample_helper_macro_include.cu" \
  "$fixtures/nvidia_samples/Common" \
  "$test_tmp/macro-include.cpp" "$test_tmp/macro-include"
require_fixed '#define ASCIFY_TEST_HELPER_HEADER <helper_cuda.h>' \
  "$test_tmp/macro-include.cpp"
require_fixed '#include ASCIFY_TEST_HELPER_HEADER' \
  "$test_tmp/macro-include.cpp"
require_fixed 'checkCudaErrors(ascify::cudaMalloc' \
  "$test_tmp/macro-include.cpp"
forbid_fixed 'ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS' \
  "$test_tmp/macro-include.cpp"
require_fixed 'not a direct main-file literal' \
  "$test_tmp/macro-include.stderr"

translate "$fixtures/sample_helper_mixed_direct_macro_include.cu" \
  "$fixtures/nvidia_samples/Common" \
  "$test_tmp/mixed-direct-macro-include.cpp" \
  "$test_tmp/mixed-direct-macro-include"
require_fixed '#include <helper_cuda.h>' \
  "$test_tmp/mixed-direct-macro-include.cpp"
require_fixed '#include ASCIFY_TEST_SECOND_HELPER_HEADER' \
  "$test_tmp/mixed-direct-macro-include.cpp"
require_fixed 'checkCudaErrors(ascify::cudaMalloc' \
  "$test_tmp/mixed-direct-macro-include.cpp"
forbid_fixed 'ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS' \
  "$test_tmp/mixed-direct-macro-include.cpp"
require_fixed 'not a direct main-file literal' \
  "$test_tmp/mixed-direct-macro-include.stderr"
require_fixed 'global proof failed before replacement' \
  "$test_tmp/mixed-direct-macro-include.stderr"
require_fixed 'include kept' \
  "$test_tmp/mixed-direct-macro-include.stderr"

translate "$fixtures/sample_helper_mixed_direct_relative_include.cu" \
  "$fixtures/nvidia_samples/Common" \
  "$test_tmp/mixed-direct-relative-include.cpp" \
  "$test_tmp/mixed-direct-relative-include"
require_fixed '#include <helper_cuda.h>' \
  "$test_tmp/mixed-direct-relative-include.cpp"
require_fixed '#include "nvidia_samples/Common/helper_cuda.h"' \
  "$test_tmp/mixed-direct-relative-include.cpp"
require_fixed 'checkCudaErrors(ascify::cudaMalloc' \
  "$test_tmp/mixed-direct-relative-include.cpp"
forbid_fixed 'ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS' \
  "$test_tmp/mixed-direct-relative-include.cpp"
require_fixed 'not a direct main-file literal' \
  "$test_tmp/mixed-direct-relative-include.stderr"
require_fixed 'include kept' \
  "$test_tmp/mixed-direct-relative-include.stderr"

translate "$fixtures/sample_helper_mixed_direct_transitive_include.cu" \
  "$fixtures/nvidia_samples/Common" \
  "$test_tmp/mixed-direct-transitive-include.cpp" \
  "$test_tmp/mixed-direct-transitive-include"
require_fixed '#include "sample_helper_wrapper.h"' \
  "$test_tmp/mixed-direct-transitive-include.cpp"
require_fixed '#include <helper_cuda.h>' \
  "$test_tmp/mixed-direct-transitive-include.cpp"
require_fixed 'checkCudaErrors(ascify::cudaMalloc' \
  "$test_tmp/mixed-direct-transitive-include.cpp"
forbid_fixed 'ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS' \
  "$test_tmp/mixed-direct-transitive-include.cpp"
require_fixed 'included transitively' \
  "$test_tmp/mixed-direct-transitive-include.stderr"
require_fixed 'include kept' \
  "$test_tmp/mixed-direct-transitive-include.stderr"

translate "$fixtures/sample_helper_external_use.cu" \
  "$fixtures/nvidia_samples/Common" \
  "$test_tmp/external-use.cpp" "$test_tmp/external-use"
require_fixed '#include <helper_cuda.h>' "$test_tmp/external-use.cpp"
require_fixed '#include "sample_helper_external_use.h"' \
  "$test_tmp/external-use.cpp"
forbid_fixed 'ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS' \
  "$test_tmp/external-use.cpp"
require_fixed 'helper macro expansion outside the main file' \
  "$test_tmp/external-use.stderr"
require_fixed "residual helper declaration 'findCudaDevice'" \
  "$test_tmp/external-use.stderr"
require_fixed 'include kept' "$test_tmp/external-use.stderr"

translate "$fixtures/sample_helper_external_pp.cu" \
  "$fixtures/nvidia_samples/Common" \
  "$test_tmp/external-pp.cpp" "$test_tmp/external-pp"
require_fixed '#include <helper_cuda.h>' "$test_tmp/external-pp.cpp"
require_fixed '#include "sample_helper_external_pp.h"' \
  "$test_tmp/external-pp.cpp"
forbid_fixed 'ASCIFY_NVIDIA_SAMPLE_GET_LAST_CUDA_ERROR' \
  "$test_tmp/external-pp.cpp"
require_fixed "external #ifdef use of 'getLastCudaError'" \
  "$test_tmp/external-pp.stderr"
require_fixed "external defined use of 'checkCudaErrors'" \
  "$test_tmp/external-pp.stderr"
require_fixed "external #undef use of 'checkCudaErrors'" \
  "$test_tmp/external-pp.stderr"
require_fixed "external #ifndef use of 'checkCudaErrors'" \
  "$test_tmp/external-pp.stderr"
require_fixed 'include kept' "$test_tmp/external-pp.stderr"

for status_case in custom_int driver_status library_status; do
  translate "$fixtures/sample_helper_${status_case}.cu" \
    "$fixtures/nvidia_samples/Common" \
    "$test_tmp/${status_case}.cpp" "$test_tmp/${status_case}"
  require_fixed '#include <helper_cuda.h>' "$test_tmp/${status_case}.cpp"
  require_fixed 'checkCudaErrors(' "$test_tmp/${status_case}.cpp"
  forbid_fixed 'ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS' \
    "$test_tmp/${status_case}.cpp"
  require_fixed 'status domain not proven' \
    "$test_tmp/${status_case}.stderr"
done

for pp_case in ifdef defined undef stringize forward; do
  translate "$fixtures/sample_helper_pp_${pp_case}.cu" \
    "$fixtures/nvidia_samples/Common" \
    "$test_tmp/pp-${pp_case}.cpp" "$test_tmp/pp-${pp_case}"
  require_fixed '#include <helper_cuda.h>' "$test_tmp/pp-${pp_case}.cpp"
  require_fixed 'residual raw PP use' "$test_tmp/pp-${pp_case}.stderr"
  require_fixed 'include kept' "$test_tmp/pp-${pp_case}.stderr"
done

echo "sample-helper frontend provenance and closure contracts passed"
