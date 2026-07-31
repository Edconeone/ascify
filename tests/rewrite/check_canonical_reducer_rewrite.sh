#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
action_cpp="$repo_root/src/AscifyAction.cpp"
action_header="$repo_root/src/AscifyAction.h"
aclcub_header="$repo_root/acl_cub/aclcub.hpp"

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

require_fixed 'sCanonicalBinaryReducer' "$action_cpp"
require_fixed 'classifyCanonicalBinaryReducer' "$action_cpp"
require_fixed 'body->size() != 1' "$action_cpp"
require_fixed 'callOperator->isConst()' "$action_cpp"
require_fixed 'callOperator->getNumParams() != 2' "$action_cpp"
require_fixed 'clang::BO_Add' "$action_cpp"
require_fixed 'clang::BO_GT' "$action_cpp"
require_fixed 'clang::BO_LT' "$action_cpp"
require_fixed 'using ascify_reduction_tag = ::aclcub::' "$action_cpp"
require_fixed 'using ascify_reduction_value_type = ' "$action_cpp"
require_fixed 'using ascify_reduction_owner_type = ' "$action_cpp"
require_fixed 'hasCubCompatHeader' "$action_header"
require_fixed 'firstCubCompatHeaderLoc' "$action_header"
require_fixed 'DeclaredOpTag' "$aclcub_header"
require_fixed 'DispatchTagOf' "$aclcub_header"
require_fixed 'IsAscReduceType' "$aclcub_header"
require_fixed 'kMarkerMatches' "$aclcub_header"
require_fixed 'kMarkerOwned' "$aclcub_header"
require_fixed 'DispatchTagOf<T, ::aclcub::Sum>' "$aclcub_header"
forbid_fixed 'oneflow' "$aclcub_header"
forbid_fixed 'SumOp' "$aclcub_header"
forbid_fixed 'MaxOp' "$aclcub_header"

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
      echo "usage: sh tests/rewrite/check_canonical_reducer_rewrite.sh [--fast OUTPUT] [--default OUTPUT]" >&2
      exit 2
      ;;
  esac
done

if [ -n "$fast_output" ]; then
  check_rules "$fast_output" "$repo_root/tests/rewrite/canonical_reducer_fast.expected"
  count_fixed 'using ascify_reduction_tag = ::aclcub::' "$fast_output" 3
  count_fixed 'using ascify_reduction_value_type = T;' "$fast_output" 3
  count_fixed 'using ascify_reduction_owner_type = ' "$fast_output" 3
fi

if [ -n "$default_output" ]; then
  check_rules "$default_output" "$repo_root/tests/rewrite/canonical_reducer_default.expected"
  count_fixed 'using ascify_reduction_tag = ::aclcub::' "$default_output" 0
  count_fixed 'using ascify_reduction_value_type = T;' "$default_output" 0
  count_fixed 'using ascify_reduction_owner_type = ' "$default_output" 0
fi

cxx=${CXX:-c++}
"$cxx" -std=c++17 -fsyntax-only \
  -I"$repo_root" \
  -I"$repo_root/tests/rewrite/stubs" \
  "$repo_root/tests/rewrite/aclcub_header_syntax.cpp"

echo "canonical binary reducer rewrite checks passed"
