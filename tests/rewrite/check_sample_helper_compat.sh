#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
action_cpp="$repo_root/src/AscifyAction.cpp"
compat_header="$repo_root/include/ascify/ascify_cuda_compat.hpp"
host_test="$repo_root/tests/rewrite/sample_helper_compat_test.cpp"
find_device_host_test="$repo_root/tests/rewrite/sample_find_device_compat_test.cpp"
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
require_fixed 'hasProvenNvidiaFindCudaDeviceDefinition' "$action_cpp"
require_fixed 'nameLocation.isInvalid() || nameLocation.isMacroID()' "$action_cpp"
require_fixed 'reference == nullptr || reference->hasQualifier()' "$action_cpp"
require_fixed 'FrozenOfficialNvidiaSampleHelperCuda' "$action_cpp"
require_fixed '#if LLVM_VERSION_MAJOR >= 13' "$action_cpp"
require_fixed 'Support/SHA256.h is unavailable before LLVM 13' "$action_cpp"
require_fixed 'sourceManager.getBufferData(file, &invalidBuffer)' "$action_cpp"
require_fixed 'auditFrozenNvidiaSampleHelperMacroDependency' "$action_cpp"
require_fixed 'macroInfo->isBuiltinMacro()' "$action_cpp"
require_fixed 'getPredefinesFileID()' "$action_cpp"
require_fixed '!sourceManager.isWrittenInCommandLineFile(definitionLocation)' "$action_cpp"
require_fixed '#if LLVM_VERSION_MAJOR >= 8' "$action_cpp"
require_fixed 'const bool compilerPredefinition = false;' "$action_cpp"
require_fixed 'reason != clang::PPCallbacks::EnterFile' "$action_cpp"
require_fixed 'trustedSystemFileIds.insert(file.getHashValue())' "$action_cpp"
require_fixed 'initiallyUntrustedSystemFileIdentities.count(identity)' "$action_cpp"
require_fixed 'sourceManager.isFileOverridden(entry)' "$action_cpp"
require_fixed 'locationComesFromInitiallyTrustedSystemFile' "$action_cpp"
require_fixed 'FrozenNvidiaSampleHelperCallClosure' "$action_cpp"
require_fixed 'allFunctionRedeclarationsComeFromInitiallyTrustedSystemFiles' "$action_cpp"
require_fixed 'attribute->getLocation()' "$action_cpp"
require_fixed 'attribute->isImplicit()' "$action_cpp"
require_fixed 'callee, sourceManager, trustedSystemFileIds' "$action_cpp"
require_fixed 'hasSystemFunctionDeclarationProvenance' "$action_cpp"
require_fixed 'directUnqualifiedSourceCallNamed' "$action_cpp"
require_fixed '997f9ac1f8e5f8e5f45f8b11eebab5b89305dee7430b90654bafe62283cffee1' "$action_cpp"
require_fixed '26e988c97fb3d77d498e384c685177ed7966e41d5d58ebc9b7d3d696859f5e57' "$action_cpp"
require_fixed 'FrozenOfficialNvidiaSampleHelperFunctions' "$action_cpp"
require_fixed 'FrozenOfficialNvidiaSampleHelperImage' "$action_cpp"
require_fixed '3fdcd18e41ffc2a9c88ade3595384e9cd05a2d84f80b86a2d5982035ca79c426' "$action_cpp"
require_fixed 'bc1fe7921bafad278ffa2e4bc8a99c18825208b9f5f47842a7cf7e86cae8b3f1' "$action_cpp"
require_fixed 'activeFrozenHelperFunctionsProviderMacroBodyMatches' "$action_cpp"
require_fixed 'name == "EXIT_WAIVED" || name == "MAX"' "$action_cpp"
require_fixed 'admittedConsumer && useKind == "#ifndef"' "$action_cpp"
require_fixed 'frozenNvidiaSampleMacroProviderIdentities.count' "$action_cpp"
require_fixed 'providerRole == FrozenOfficialNvidiaSampleHelperImage' "$action_cpp"
require_fixed '!isFrozenHelperFunctionsProviderPolicyMacro(macroName)' "$action_cpp"
require_fixed '::ascify::sampleFindCudaDevice' "$action_cpp"
require_fixed 'ct::Replacements staged(*replacements);' "$action_cpp"
require_fixed '*replacements = std::move(staged);' "$action_cpp"
require_fixed 'nvidiaSampleHelperRawAuditCompleted = true;' "$action_cpp"
require_fixed 'nvidiaSampleHelperRawAuditCompleted &&' "$action_cpp"
require_fixed 'sampleCheckCudaErrors' "$compat_header"
require_fixed 'sampleFindCudaDevice(int argc, const char** argv)' "$compat_header"
require_fixed 'exactly one visible logical device is required' "$compat_header"
require_fixed 'const aclError status = cudaGetLastError();' "$compat_header"
require_fixed '#expression, __FILE__, __LINE__' "$compat_header"
forbid_fixed 'gpuGetMaxGflopsDeviceId' "$compat_header"

official_helper_functions="$fixtures/nvidia_samples/Common/helper_functions.h"
[ "$(wc -c <"$official_helper_functions" | tr -d ' ')" -eq 2358 ]
if command -v sha256sum >/dev/null 2>&1; then
  helper_functions_sha=$(sha256sum "$official_helper_functions" | awk '{print $1}')
else
  helper_functions_sha=$(shasum -a 256 "$official_helper_functions" | awk '{print $1}')
fi
[ "$helper_functions_sha" = \
  3fdcd18e41ffc2a9c88ade3595384e9cd05a2d84f80b86a2d5982035ca79c426 ]
for frozen_provider in \
  'helper_image.h:28739:bc1fe7921bafad278ffa2e4bc8a99c18825208b9f5f47842a7cf7e86cae8b3f1' \
  'helper_timer.h:16060:c48552a7c7b7a5840fcfbc176bfb5a19b501fdc56796b64a63e78e39ab547078'; do
  provider_name=${frozen_provider%%:*}
  provider_tail=${frozen_provider#*:}
  provider_size=${provider_tail%%:*}
  provider_sha=${provider_tail#*:}
  provider_path="$fixtures/nvidia_samples/Common/$provider_name"
  [ "$(wc -c <"$provider_path" | tr -d ' ')" -eq "$provider_size" ]
  if command -v sha256sum >/dev/null 2>&1; then
    actual_provider_sha=$(sha256sum "$provider_path" | awk '{print $1}')
  else
    actual_provider_sha=$(shasum -a 256 "$provider_path" | awk '{print $1}')
  fi
  [ "$actual_provider_sha" = "$provider_sha" ]
done

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
  "$cxx" -std=c++17 \
    -I"$repo_root/tests/rewrite/stubs" \
    -I"$repo_root/include" \
    "$find_device_host_test" -o "$test_tmp/sample-find-device-host"
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

  "$test_tmp/sample-find-device-host" success-default
  "$test_tmp/sample-find-device-host" success-explicit-zero
  for find_failure in \
    zero-devices multiple-devices other-device legacy-device-spelling \
    duplicate-device negative-argc null-argv null-argument \
    query-failure bind-failure; do
    set +e
    "$test_tmp/sample-find-device-host" "$find_failure" \
      >"$test_tmp/find-$find_failure.stdout" \
      2>"$test_tmp/find-$find_failure.stderr"
    find_failure_rc=$?
    set -e
    [ "$find_failure_rc" -eq 1 ]
    require_fixed 'Ascify CUDA sample device selection failed' \
      "$test_tmp/find-$find_failure.stderr"
  done
  require_fixed 'exactly one visible logical device is required' \
    "$test_tmp/find-zero-devices.stderr"
  require_fixed 'exactly one visible logical device is required' \
    "$test_tmp/find-multiple-devices.stderr"
  require_fixed 'only one explicit --device=0 selector is supported' \
    "$test_tmp/find-other-device.stderr"
  require_fixed 'invalid argc/argv' \
    "$test_tmp/find-negative-argc.stderr"
  require_fixed 'invalid argc/argv' \
    "$test_tmp/find-null-argv.stderr"
  require_fixed 'null command-line argument' \
    "$test_tmp/find-null-argument.stderr"
  require_fixed 'device-count query failed' \
    "$test_tmp/find-query-failure.stderr"
  require_fixed 'binding logical device 0 failed' \
    "$test_tmp/find-bind-failure.stderr"
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
  shift 4
  "$binary" "$input" \
    --target-policy=dav-c310-vec \
    --simt-math=fast \
    --default-preprocessor \
    --cuda-path="$cuda_path" \
    --clang-resource-directory="$resource_dir" \
    -o "$output" -- -x cuda -std=c++17 -I"$include_dir" -I"$fixtures" "$@" \
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
    '-DsampleFindCudaDevice(argc,argv)=ascifyCommandLineDevice((argc),(argv))' \
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
require_fixed '#include <ascify/ascify_cuda_compat.hpp>' \
  "$test_tmp/residual.cpp"
require_fixed '::ascify::sampleFindCudaDevice(argc, argv)' \
  "$test_tmp/residual.cpp"
forbid_fixed 'helper_cuda.h' "$test_tmp/residual.cpp"
require_fixed 'find_device_rewrites=1' "$test_tmp/residual.stderr"
require_fixed 'include removed' "$test_tmp/residual.stderr"

translate "$fixtures/sample_helper_find_device_direct.cu" \
  "$fixtures/nvidia_samples/Common" \
  "$test_tmp/find-direct.cpp" "$test_tmp/find-direct"
require_fixed '#include <ascify/ascify_cuda_compat.hpp>' \
  "$test_tmp/find-direct.cpp"
require_fixed 'return ::ascify::sampleFindCudaDevice(argc, argv);' \
  "$test_tmp/find-direct.cpp"
forbid_fixed 'helper_cuda.h' "$test_tmp/find-direct.cpp"
require_fixed 'find_device_rewrites=1' "$test_tmp/find-direct.stderr"
require_fixed 'include removed' "$test_tmp/find-direct.stderr"

# Reproduce simpleAtomicIntrinsics' unedited Common-header graph. The exact
# helper_functions root enters exact helper_image first, which establishes
# EXIT_WAIVED=2 before exact helper_string and helper_cuda consume it.
translate "$fixtures/sample_helper_find_device_official_exit_waived.cu" \
  "$fixtures/nvidia_samples/Common" \
  "$test_tmp/find-official-exit-waived.cpp" \
  "$test_tmp/find-official-exit-waived"
require_fixed '#include <helper_functions.h>' \
  "$test_tmp/find-official-exit-waived.cpp"
require_fixed 'return ::ascify::sampleFindCudaDevice(argc, argv);' \
  "$test_tmp/find-official-exit-waived.cpp"
forbid_fixed 'helper_cuda.h' \
  "$test_tmp/find-official-exit-waived.cpp"
require_fixed 'find_device_rewrites=1' \
  "$test_tmp/find-official-exit-waived.stderr"
require_fixed 'include removed' \
  "$test_tmp/find-official-exit-waived.stderr"

assert_exit_waived_rejected() {
  label=$1
  output=$2
  log=$3
  require_fixed '#include <helper_cuda.h>' "$output"
  require_fixed 'findCudaDevice(argc, argv)' "$output"
  forbid_fixed 'sampleFindCudaDevice' "$output"
  require_fixed "macro dependency 'EXIT_WAIVED'" "$log.stderr"
  require_fixed 'include kept' "$log.stderr"
}

for origin in user pragma reentry line_spoof; do
  translate \
    "$fixtures/sample_helper_find_device_${origin}_exit_waived.cu" \
    "$fixtures/nvidia_samples/Common" \
    "$test_tmp/find-${origin}-exit-waived.cpp" \
    "$test_tmp/find-${origin}-exit-waived"
  assert_exit_waived_rejected "$origin" \
    "$test_tmp/find-${origin}-exit-waived.cpp" \
    "$test_tmp/find-${origin}-exit-waived"
done

translate "$fixtures/sample_helper_find_device_official_exit_waived.cu" \
  "$fixtures/nvidia_samples/Common" \
  "$test_tmp/find-command-line-exit-waived.cpp" \
  "$test_tmp/find-command-line-exit-waived" \
  '-DEXIT_WAIVED=2'
assert_exit_waived_rejected command-line \
  "$test_tmp/find-command-line-exit-waived.cpp" \
  "$test_tmp/find-command-line-exit-waived"

for mutation in value body; do
  mutation_dir="$test_tmp/helper-image-exit-$mutation"
  mkdir -p "$mutation_dir"
  cp "$fixtures/nvidia_samples/Common/helper_functions.h" \
    "$mutation_dir/helper_functions.h"
  if [ "$mutation" = value ]; then
    sed 's/^#define EXIT_WAIVED 2$/#define EXIT_WAIVED 3/' \
      "$fixtures/nvidia_samples/Common/helper_image.h" \
      >"$mutation_dir/helper_image.h"
  else
    sed 's/^#define EXIT_WAIVED 2$/#define EXIT_WAIVED (1 + 1)/' \
      "$fixtures/nvidia_samples/Common/helper_image.h" \
      >"$mutation_dir/helper_image.h"
  fi
  translate "$fixtures/sample_helper_find_device_official_exit_waived.cu" \
    "$mutation_dir" \
    "$test_tmp/find-mutated-$mutation-exit-waived.cpp" \
    "$test_tmp/find-mutated-$mutation-exit-waived" \
    "-I$fixtures/nvidia_samples/Common"
  assert_exit_waived_rejected "mutated-$mutation" \
    "$test_tmp/find-mutated-$mutation-exit-waived.cpp" \
    "$test_tmp/find-mutated-$mutation-exit-waived"
done

shadow_dir="$test_tmp/helper-functions-shadow"
mkdir -p "$shadow_dir"
printf '%s\n' \
  '#ifndef COMMON_HELPER_FUNCTIONS_H_' \
  '#define COMMON_HELPER_FUNCTIONS_H_' \
  '#define EXIT_WAIVED 2' \
  '#endif' >"$shadow_dir/helper_functions.h"
translate "$fixtures/sample_helper_find_device_official_exit_waived.cu" \
  "$shadow_dir" \
  "$test_tmp/find-shadow-exit-waived.cpp" \
  "$test_tmp/find-shadow-exit-waived" \
  "-I$fixtures/nvidia_samples/Common"
assert_exit_waived_rejected shadow \
  "$test_tmp/find-shadow-exit-waived.cpp" \
  "$test_tmp/find-shadow-exit-waived"

assert_max_rejected() {
  output=$1
  log=$2
  require_fixed '#include <helper_cuda.h>' "$output"
  require_fixed 'findCudaDevice(argc, argv)' "$output"
  forbid_fixed 'sampleFindCudaDevice' "$output"
  require_fixed "macro dependency 'MAX'" "$log.stderr"
  require_fixed 'include kept' "$log.stderr"
}

for max_origin in user redefined; do
  translate "$fixtures/sample_helper_find_device_${max_origin}_max.cu" \
    "$fixtures/nvidia_samples/Common" \
    "$test_tmp/find-${max_origin}-max.cpp" \
    "$test_tmp/find-${max_origin}-max"
  assert_max_rejected \
    "$test_tmp/find-${max_origin}-max.cpp" \
    "$test_tmp/find-${max_origin}-max"
done

translate "$fixtures/sample_helper_find_device_official_exit_waived.cu" \
  "$fixtures/nvidia_samples/Common" \
  "$test_tmp/find-command-line-max.cpp" \
  "$test_tmp/find-command-line-max" \
  '-DMAX(a,b)=((a>b)?a:b)'
assert_max_rejected \
  "$test_tmp/find-command-line-max.cpp" \
  "$test_tmp/find-command-line-max"

mutated_image_dir="$test_tmp/helper-image-same-size-mutation"
mkdir -p "$mutated_image_dir"
cp "$fixtures/nvidia_samples/Common/helper_functions.h" \
  "$mutated_image_dir/helper_functions.h"
sed 's/^#define MAX(a, b) ((a > b) ? a : b)$/#define MAX(a, b) ((a < b) ? a : b)/' \
  "$fixtures/nvidia_samples/Common/helper_image.h" \
  >"$mutated_image_dir/helper_image.h"
[ "$(wc -c <"$mutated_image_dir/helper_image.h" | tr -d ' ')" -eq 28739 ]
translate "$fixtures/sample_helper_find_device_mutated_image_max.cu" \
  "$mutated_image_dir" \
  "$test_tmp/find-mutated-image-max.cpp" \
  "$test_tmp/find-mutated-image-max" \
  "-I$fixtures/nvidia_samples/Common"
assert_max_rejected \
  "$test_tmp/find-mutated-image-max.cpp" \
  "$test_tmp/find-mutated-image-max"

translate "$fixtures/sample_helper_find_device_discarded_call.cu" \
  "$fixtures/nvidia_samples/Common" \
  "$test_tmp/find-discarded-call.cpp" \
  "$test_tmp/find-discarded-call"
require_fixed '#include <ascify/ascify_cuda_compat.hpp>' \
  "$test_tmp/find-discarded-call.cpp"
require_fixed '::ascify::sampleFindCudaDevice(argc, argv);' \
  "$test_tmp/find-discarded-call.cpp"
forbid_fixed 'helper_cuda.h' "$test_tmp/find-discarded-call.cpp"
require_fixed 'find_device_rewrites=1' \
  "$test_tmp/find-discarded-call.stderr"
require_fixed 'include removed' "$test_tmp/find-discarded-call.stderr"

# The exact admitted helper_cuda fixture is paired with a helper_string whose
# active checkCmdLineFlag body gained telemetry. A same path/name plus a proven
# findCudaDevice skeleton must not bypass the frozen dependency-body profile.
translate "$fixtures/sample_helper_find_device_direct.cu" \
  "$fixtures/mutated_dependency_nvidia_samples/Common" \
  "$test_tmp/find-mutated-dependency.cpp" \
  "$test_tmp/find-mutated-dependency"
require_fixed '#include <helper_cuda.h>' \
  "$test_tmp/find-mutated-dependency.cpp"
require_fixed 'findCudaDevice(argc, argv)' \
  "$test_tmp/find-mutated-dependency.cpp"
forbid_fixed 'sampleFindCudaDevice' \
  "$test_tmp/find-mutated-dependency.cpp"
require_fixed "residual helper declaration 'findCudaDevice'" \
  "$test_tmp/find-mutated-dependency.stderr"
require_fixed 'include kept' \
  "$test_tmp/find-mutated-dependency.stderr"

translate "$fixtures/sample_helper_unsupported_macro.cu" \
  "$fixtures/nvidia_samples/Common" \
  "$test_tmp/unsupported-macro.cpp" "$test_tmp/unsupported-macro"
require_fixed '#include <helper_cuda.h>' "$test_tmp/unsupported-macro.cpp"
require_fixed 'MAX(value, 1)' \
  "$test_tmp/unsupported-macro.cpp"
require_fixed 'unsupported_macro=1' \
  "$test_tmp/unsupported-macro.stderr"

translate "$fixtures/sample_helper_user_same_names.cu" \
  "$fixtures/user_helpers" \
  "$test_tmp/user.cpp" "$test_tmp/user"
require_fixed '#include <helper_cuda.h>' "$test_tmp/user.cpp"
require_fixed 'checkCudaErrors(0)' "$test_tmp/user.cpp"
require_fixed 'getLastCudaError("user macro")' "$test_tmp/user.cpp"
require_fixed 'findCudaDevice(argc, argv)' "$test_tmp/user.cpp"
forbid_fixed 'ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS' "$test_tmp/user.cpp"
forbid_fixed 'ASCIFY_NVIDIA_SAMPLE_GET_LAST_CUDA_ERROR' "$test_tmp/user.cpp"
forbid_fixed 'sampleFindCudaDevice' "$test_tmp/user.cpp"

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
require_fixed '#include <ascify/ascify_cuda_compat.hpp>' \
  "$test_tmp/mixed-residual.cpp"
require_fixed 'ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS(ascify::cudaMalloc' \
  "$test_tmp/mixed-residual.cpp"
require_fixed '::ascify::sampleFindCudaDevice(argc' \
  "$test_tmp/mixed-residual.cpp"
forbid_fixed 'helper_cuda.h' "$test_tmp/mixed-residual.cpp"
require_fixed 'check_rewrites=1' "$test_tmp/mixed-residual.stderr"
require_fixed 'find_device_rewrites=1' "$test_tmp/mixed-residual.stderr"
require_fixed 'include removed' "$test_tmp/mixed-residual.stderr"

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

translate "$fixtures/sample_helper_check_preinclude_pragma_runtime_redirect.cu" \
  "$fixtures/nvidia_samples/Common" \
  "$test_tmp/pragma-runtime-status.cpp" \
  "$test_tmp/pragma-runtime-status"
require_fixed '#include <helper_cuda.h>' \
  "$test_tmp/pragma-runtime-status.cpp"
require_fixed 'checkCudaErrors(ascify::cudaSetDevice(0L))' \
  "$test_tmp/pragma-runtime-status.cpp"
forbid_fixed 'ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS' \
  "$test_tmp/pragma-runtime-status.cpp"
require_fixed 'status domain not proven' \
  "$test_tmp/pragma-runtime-status.stderr"
require_fixed 'include kept' \
  "$test_tmp/pragma-runtime-status.stderr"

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
require_fixed 'findCudaDevice(argc, argv)' \
  "$test_tmp/output-collision.cpp"
[ "$(grep -F -c 'ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS' \
       "$test_tmp/output-collision.cpp")" -eq 1 ]
[ "$(grep -F -c 'ASCIFY_NVIDIA_SAMPLE_GET_LAST_CUDA_ERROR' \
       "$test_tmp/output-collision.cpp")" -eq 1 ]
[ "$(grep -F -c 'sampleFindCudaDevice' \
       "$test_tmp/output-collision.cpp")" -eq 1 ]
require_fixed 'reserved output macro' "$test_tmp/output-collision.stderr"
require_fixed 'global proof failed before replacement' \
  "$test_tmp/output-collision.stderr"
require_fixed 'include kept' "$test_tmp/output-collision.stderr"

translate "$fixtures/sample_helper_find_device_altered_definition.cu" \
  "$fixtures/altered_find_nvidia_samples/Common" \
  "$test_tmp/altered-find.cpp" "$test_tmp/altered-find"
require_fixed '#include <helper_cuda.h>' "$test_tmp/altered-find.cpp"
require_fixed 'findCudaDevice(argc, argv)' "$test_tmp/altered-find.cpp"
forbid_fixed 'sampleFindCudaDevice' "$test_tmp/altered-find.cpp"
require_fixed "residual helper declaration 'findCudaDevice'" \
  "$test_tmp/altered-find.stderr"
require_fixed 'include kept' "$test_tmp/altered-find.stderr"

for find_negative in \
  macro alias indirect qualified preinclude_stdio_redirect \
  preinclude_strncasecmp_redirect preinclude_builtin_spoof \
  preinclude_pragma_clang_system_redirect \
  preinclude_pragma_gcc_system_redirect \
  preinclude_pragma_system_reentry_redirect \
  preinclude_redefine_extname_strlen \
  preinclude_user_libc_definition \
  preinclude_user_strlen_definition \
  preinclude_user_atoi_definition; do
  translate "$fixtures/sample_helper_find_device_${find_negative}.cu" \
    "$fixtures/nvidia_samples/Common" \
    "$test_tmp/find-${find_negative}.cpp" \
    "$test_tmp/find-${find_negative}"
  require_fixed '#include <helper_cuda.h>' \
    "$test_tmp/find-${find_negative}.cpp"
  require_fixed 'findCudaDevice' "$test_tmp/find-${find_negative}.cpp"
  forbid_fixed 'sampleFindCudaDevice' "$test_tmp/find-${find_negative}.cpp"
  require_fixed "residual helper declaration 'findCudaDevice'" \
    "$test_tmp/find-${find_negative}.stderr"
  require_fixed 'include kept' \
    "$test_tmp/find-${find_negative}.stderr"
done
require_fixed 'untrusted ' \
  "$test_tmp/find-preinclude_strncasecmp_redirect.stderr"
require_fixed "macro dependency 'STRNCASECMP'" \
  "$test_tmp/find-preinclude_strncasecmp_redirect.stderr"
require_fixed "macro dependency 'STRNCASECMP'" \
  "$test_tmp/find-preinclude_builtin_spoof.stderr"
require_fixed "macro dependency 'STRNCASECMP'" \
  "$test_tmp/find-preinclude_pragma_clang_system_redirect.stderr"
require_fixed "macro dependency 'STRNCASECMP'" \
  "$test_tmp/find-preinclude_pragma_gcc_system_redirect.stderr"
require_fixed "macro dependency 'STRNCASECMP'" \
  "$test_tmp/find-preinclude_pragma_system_reentry_redirect.stderr"
require_fixed 'strlen(const char* text)' \
  "$test_tmp/find-preinclude_user_libc_definition.cpp"
require_fixed 'atoi(const char* text)' \
  "$test_tmp/find-preinclude_user_libc_definition.cpp"
require_fixed 'strlen(const char* text)' \
  "$test_tmp/find-preinclude_user_strlen_definition.cpp"
require_fixed 'atoi(const char* text)' \
  "$test_tmp/find-preinclude_user_atoi_definition.cpp"
require_fixed '#pragma redefine_extname strlen ascifyTestEvilStrlen' \
  "$test_tmp/find-preinclude_redefine_extname_strlen.cpp"

# These source/header mutations are rejected by the frozen production profile
# before an unsafe variant can reach the strict AST branch matcher. They are
# profile-gate negatives, not claims of per-branch matcher-unit coverage.
for find_mutation in \
  EXTRA_SIDE_EFFECT DISCARDED_RESULT WRONG_ASSIGNMENT CHANGED_SIGNATURE \
  UNCHECKED_SET_DEVICE WRONG_SET_DEVICE; do
  translate "$fixtures/sample_helper_find_device_mutated_definition.cu" \
    "$fixtures/mutated_find_nvidia_samples/Common" \
    "$test_tmp/find-mutation-${find_mutation}.cpp" \
    "$test_tmp/find-mutation-${find_mutation}" \
    "-DASCIFY_TEST_FIND_${find_mutation}=1"
  require_fixed '#include <helper_cuda.h>' \
    "$test_tmp/find-mutation-${find_mutation}.cpp"
  require_fixed 'findCudaDevice(argc, argv)' \
    "$test_tmp/find-mutation-${find_mutation}.cpp"
  forbid_fixed 'sampleFindCudaDevice' \
    "$test_tmp/find-mutation-${find_mutation}.cpp"
  require_fixed "residual helper declaration 'findCudaDevice'" \
    "$test_tmp/find-mutation-${find_mutation}.stderr"
  require_fixed 'include kept' \
    "$test_tmp/find-mutation-${find_mutation}.stderr"
done

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
