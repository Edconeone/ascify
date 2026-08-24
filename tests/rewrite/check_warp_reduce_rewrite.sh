#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
action_cpp="$repo_root/src/AscifyAction.cpp"
action_header="$repo_root/src/AscifyAction.h"
argparse_cpp="$repo_root/src/ArgParse.cpp"
argparse_header="$repo_root/src/ArgParse.h"
compat_header="$repo_root/include/ascify/ascify_cuda_compat.hpp"
llvm_compat_cpp="$repo_root/src/LLVMCompat.cpp"
llvm_compat_header="$repo_root/src/LLVMCompat.h"
main_cpp="$repo_root/src/main.cpp"

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

count_fixed() {
  needle=$1
  file=$2
  expected=$3
  if command -v rg >/dev/null 2>&1; then
    actual=$( (rg -F -o -- "$needle" "$file" || true) | wc -l | tr -d ' ' )
  else
    actual=$( (grep -F -o -- "$needle" "$file" || true) | wc -l | tr -d ' ' )
  fi
  if [ "$actual" -ne "$expected" ]; then
    echo "expected $expected occurrences of '$needle' in $file, found $actual" >&2
    exit 1
  fi
}

check_rules() {
  output=$1
  rules=$2
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
  done < "$rules"
}

check_static_contract() {
  require_fixed 'SemanticRewriteRanges' "$action_header"
  require_fixed 'isInSemanticRewriteRange(tok.getLocation())' "$action_cpp"
  require_fixed 'insertSemanticReplacement' "$action_cpp"
  require_fixed 'include rewrite the sole owner of that span' "$action_cpp"
  require_fixed 'failed to add source replacement' "$llvm_compat_cpp"
  require_fixed 'bool insertReplacement' "$llvm_compat_header"

  require_fixed 'extern cl::opt<std::string> TargetPolicy' "$argparse_header"
  require_fixed 'extern cl::opt<std::string> SimtMathMode' "$argparse_header"
  require_fixed 'extern cl::opt<std::string> TargetRecipe' "$argparse_header"
  require_fixed '"target-policy"' "$argparse_cpp"
  require_fixed 'cl::init("portable")' "$argparse_cpp"
  require_fixed '"simt-math"' "$argparse_cpp"
  require_fixed 'cl::init("precise")' "$argparse_cpp"
  require_fixed 'std::string(TargetPolicy.ArgStr)' "$argparse_cpp"
  require_fixed 'std::string(SimtMathMode.ArgStr)' "$argparse_cpp"
  require_fixed 'std::string(TargetRecipe.ArgStr)' "$argparse_cpp"
  require_fixed 'current == longOption || current == shortOption' "$main_cpp"

  require_fixed 'sCanonicalWarpAddReduction' "$action_cpp"
  require_fixed 'matchCanonicalWarpAddReduction' "$action_cpp"
  require_fixed 'accumulator == offset' "$action_cpp"
  require_fixed 'callee->getQualifiedNameAsString() != "__shfl_xor_sync"' "$action_cpp"
  require_fixed '!sourceManager.isInSystemHeader(callee->getLocation())' "$action_cpp"
  require_fixed 'function->getTemplatedKind() != clang::FunctionDecl::TK_NonTemplate' "$action_cpp"
  require_fixed 'evaluatesToFullWarpMask' "$action_cpp"
  require_fixed 'evaluatesToNonNegative(offset->getInit(), 16' "$action_cpp"
  require_fixed 'TargetPolicy == "dav-c310-vec" && SimtMathMode == "fast"' "$action_cpp"
  require_fixed 'ascify::warp_reduce_add' "$action_cpp"
  require_fixed 'hasCudaCompatHeader = true' "$action_cpp"
  forbid_fixed 'source.contains("#include <ascify/ascify_cuda_compat.hpp>")' "$action_cpp"
  require_fixed 'float warp_reduce_add(float value)' "$compat_header"
  require_fixed 'int32_t warp_reduce_add(int32_t value)' "$compat_header"
  require_fixed 'uint32_t warp_reduce_add(uint32_t value)' "$compat_header"
  forbid_fixed 'oneflow' "$action_cpp"
}

fast_output=
default_output=
while [ "$#" -gt 0 ]; do
  case "$1" in
    --fast)
      [ "$#" -ge 2 ] || { echo "--fast requires an output file" >&2; exit 2; }
      fast_output=$2
      shift 2
      ;;
    --default)
      [ "$#" -ge 2 ] || { echo "--default requires an output file" >&2; exit 2; }
      default_output=$2
      shift 2
      ;;
    *)
      echo "usage: sh tests/rewrite/check_warp_reduce_rewrite.sh [--fast OUTPUT] [--default OUTPUT]" >&2
      exit 2
      ;;
  esac
done

check_static_contract

if [ -n "$fast_output" ]; then
  check_rules "$fast_output" "$repo_root/tests/rewrite/warp_reduce_fast.expected"
  count_fixed 'value = ascify::warp_reduce_add(value);' "$fast_output" 3
  count_fixed '#include <ascify/ascify_cuda_compat.hpp>' "$fast_output" 2
fi

if [ -n "$default_output" ]; then
  check_rules "$default_output" "$repo_root/tests/rewrite/warp_reduce_default.expected"
  count_fixed 'value = ascify::warp_reduce_add(value);' "$default_output" 0
  count_fixed '#include <ascify/ascify_cuda_compat.hpp>' "$default_output" 2
fi

echo "canonical warp-add rewrite checks passed"
