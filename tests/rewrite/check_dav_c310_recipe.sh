#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
recipe_cpp="$repo_root/src/DavC310TargetRecipe.cpp"
recipe_header="$repo_root/src/DavC310TargetRecipe.h"
action_cpp="$repo_root/src/AscifyAction.cpp"
argparse_cpp="$repo_root/src/ArgParse.cpp"
argparse_header="$repo_root/src/ArgParse.h"

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
    actual=$( (rg -F -o -- "$needle" "$file" || true) |
      wc -l | tr -d ' ' )
  else
    actual=$( (grep -F -o -- "$needle" "$file" || true) |
      wc -l | tr -d ' ' )
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
  require_fixed 'class DavC310TargetRecipe' "$recipe_header"
  require_fixed 'void registerMatchers(' "$recipe_header"
  require_fixed 'bool collect(' "$recipe_header"
  require_fixed 'FinalizedEdits finalize(' "$recipe_header"
  require_fixed 'rowwise_norm_recipes.hpp' "$recipe_header"
  require_fixed 'rowwise_simd_recipes.hpp' "$recipe_header"
  require_fixed 'dav-3510-rowwise-simd-v1' "$recipe_header"
  require_fixed 'proveDirectAdapter' "$recipe_cpp"
  require_fixed 'matchesPackedOffset' "$recipe_cpp"
  require_fixed 'constructorInitializesFieldFromParameter' "$recipe_cpp"
  require_fixed 'PackedMemoryTransferVisitor' "$recipe_cpp"
  require_fixed 'SoftmaxKernelVisitor' "$recipe_cpp"
  require_fixed 'proveRmsAlgebra' "$recipe_cpp"
  require_fixed 'proveLayerNormAlgebra' "$recipe_cpp"
  require_fixed 'hasRowWidthRoutingShape' "$recipe_cpp"
  require_fixed 'getDescribedFunctionTemplate' "$recipe_cpp"
  require_fixed 'methodTemplate->getTemplatedDecl()' "$recipe_cpp"
  require_fixed 'TrySoftmax(' "$recipe_cpp"
  require_fixed 'TryRmsNorm(' "$recipe_cpp"
  require_fixed 'TrySoftmaxHybrid(' "$recipe_cpp"
  require_fixed 'TryRmsNormHybrid(' "$recipe_cpp"
  require_fixed 'TryLayerNormHybrid(' "$recipe_cpp"
  require_fixed 'bool useSimdEntryPoints' "$recipe_cpp"
  require_fixed 'bool allowAnnotatedStatus' "$recipe_cpp"
  require_fixed 'bool completeSourcePrimitiveBody(' "$recipe_cpp"
  require_fixed '!allowAnnotatedPrimitives &&' "$recipe_cpp"
  require_fixed '!completeSourcePrimitiveBody(' "$recipe_cpp"
  count_fixed 'completeSourcePrimitiveBody(' "$recipe_cpp" 2
  require_fixed '"TrySoftmaxHybrid",' "$action_cpp"
  require_fixed '"TryRmsNormHybrid",' "$action_cpp"
  require_fixed '"TryLayerNormHybrid",' "$action_cpp"
  require_fixed '"RowwiseHybridFacadeV1",' "$action_cpp"
  require_fixed '"ascify_target_data"' "$action_cpp"
  require_fixed '"ascify_target_data"' "$recipe_cpp"
  require_fixed 'protect(proof.data);' "$recipe_cpp"
  require_fixed 'protect(proof.stride);' "$recipe_cpp"
  require_fixed 'protect(proof.stream);' "$recipe_cpp"
  require_fixed 'protect(proof.load);' "$recipe_cpp"
  require_fixed 'protect(proof.store);' "$recipe_cpp"
  require_fixed 'protect(proof.rows);' "$recipe_cpp"
  require_fixed 'protect(proof.columns);' "$recipe_cpp"
  require_fixed 'RowwiseSimdReservedMacros' "$action_cpp"
  require_fixed 'isRowwiseSimdPublishedHeaderMacro' "$action_cpp"
  require_fixed 'headerShieldNames' "$action_cpp"
  require_fixed 'rejects reserved' "$action_cpp"
  require_fixed 'findRowwiseSimdConflictingDeclaration' "$action_cpp"
  require_fixed 'getAttr<clang::AsmLabelAttr>()' "$action_cpp"
  require_fixed 'reserved by the row-wise SIMD recipe' "$action_cpp"
  require_fixed 'struct RowwiseHybridFacadeV1 final' \
    "$repo_root/include/ascify/target/dav_c310/rowwise_simd_recipes.hpp"
  require_fixed '#pragma push_macro' "$recipe_cpp"
  require_fixed '!useSimdEntryPoints);' "$recipe_cpp"
  require_fixed 'const StaticKernelProofRegistry &kernelProofs' "$recipe_cpp"
  require_fixed 'kernelProofs.familyCount(kernel) != 1U' "$recipe_cpp"
  require_fixed 'DispatchRegisteredHybrid' \
    "$repo_root/include/ascify/target/dav_c310/rowwise_hybrid_registry_v1.hpp"
  require_fixed 'extern cl::opt<std::string> TargetRecipe' "$argparse_header"
  require_fixed '"target-recipe"' "$argparse_cpp"
  require_fixed 'cl::init("none")' "$argparse_cpp"
  require_fixed 'std::string(TargetRecipe.ArgStr)' "$argparse_cpp"
  require_fixed 'unsupported --target-recipe value' "$action_cpp"
  require_fixed '--target-recipe=dav-3510-rowwise-simd-v1 requires' "$action_cpp"
  require_fixed 'DavC310TargetRecipe::SimdTargetHeader' "$action_cpp"
  require_fixed 'DavC310TargetRecipe::SimdRecipeName' "$action_cpp"

  # Recipe decisions may use declarations, types, control/data flow and
  # attributes, but never source-project spelling.
  forbid_fixed 'oneflow' "$recipe_cpp"
  forbid_fixed 'ONEFLOW' "$recipe_cpp"
  forbid_fixed 'softmax.cuh' "$recipe_cpp"
  forbid_fixed 'rms_norm.cuh' "$recipe_cpp"
  forbid_fixed 'layer_norm.cuh' "$recipe_cpp"
  forbid_fixed '"src"' "$recipe_cpp"
  forbid_fixed '"dst"' "$recipe_cpp"
  forbid_fixed '"row_size"' "$recipe_cpp"
}

fast_output=
simd_output=
portable_precise_output=
portable_fast_output=
dav_precise_output=

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dav-fast)
      [ "$#" -ge 2 ] ||
        { echo "--dav-fast requires an output file" >&2; exit 2; }
      fast_output=$2
      shift 2
      ;;
    --dav-fast-simd)
      [ "$#" -ge 2 ] ||
        { echo "--dav-fast-simd requires an output file" >&2; exit 2; }
      simd_output=$2
      shift 2
      ;;
    --portable-precise)
      [ "$#" -ge 2 ] ||
        { echo "--portable-precise requires an output file" >&2; exit 2; }
      portable_precise_output=$2
      shift 2
      ;;
    --portable-fast)
      [ "$#" -ge 2 ] ||
        { echo "--portable-fast requires an output file" >&2; exit 2; }
      portable_fast_output=$2
      shift 2
      ;;
    --dav-precise)
      [ "$#" -ge 2 ] ||
        { echo "--dav-precise requires an output file" >&2; exit 2; }
      dav_precise_output=$2
      shift 2
      ;;
    *)
      echo "usage: sh tests/rewrite/check_dav_c310_recipe.sh [--dav-fast OUTPUT] [--dav-fast-simd OUTPUT] [--portable-precise OUTPUT] [--portable-fast OUTPUT] [--dav-precise OUTPUT]" >&2
      exit 2
      ;;
  esac
done

check_static_contract

if [ -n "$fast_output" ]; then
  check_rules \
    "$fast_output" \
    "$repo_root/tests/rewrite/dav_c310_recipe_fast.expected"
  count_fixed '#include <ascify/target/dav_c310/rowwise_norm_recipes.hpp>' \
    "$fast_output" 0
  count_fixed 'using ascify_target_direct_load_tag = void;' \
    "$fast_output" 1
  count_fixed 'using ascify_target_direct_store_tag = void;' \
    "$fast_output" 2
  count_fixed 'using ascify_target_storage_type = ' "$fast_output" 3
  count_fixed 'using ascify_target_compute_type = ' "$fast_output" 3
  count_fixed '::ascify::target::dav_c310::TrySoftmax(' "$fast_output" 0
  count_fixed '::ascify::target::dav_c310::TryRmsNorm(' "$fast_output" 0
fi

if [ -n "$simd_output" ]; then
  check_rules \
    "$simd_output" \
    "$repo_root/tests/rewrite/dav_c310_recipe_fast.expected"
  count_fixed '#include <ascify/target/dav_c310/rowwise_norm_recipes.hpp>' \
    "$simd_output" 0
  count_fixed '#include <ascify/target/dav_c310/rowwise_simd_recipes.hpp>' \
    "$simd_output" 0
  count_fixed 'using ascify_target_direct_load_tag = void;' \
    "$simd_output" 1
  count_fixed 'using ascify_target_direct_store_tag = void;' \
    "$simd_output" 2
  count_fixed '::ascify::target::dav_c310::TrySoftmax(' "$simd_output" 0
  count_fixed '::ascify::target::dav_c310::TryRmsNorm(' "$simd_output" 0
  count_fixed '::ascify::target::dav_c310::rowwise_simd_v1::RowwiseHybridFacadeV1::TrySoftmaxHybrid(' "$simd_output" 0
  count_fixed '::ascify::target::dav_c310::rowwise_simd_v1::RowwiseHybridFacadeV1::TryRmsNormHybrid(' "$simd_output" 0
fi

for fallback_output in \
  "$portable_precise_output" \
  "$portable_fast_output" \
  "$dav_precise_output"
do
  if [ -n "$fallback_output" ]; then
    check_rules \
      "$fallback_output" \
      "$repo_root/tests/rewrite/dav_c310_recipe_fallback.expected"
  fi
done

echo "dav-c310 rowwise recipe rewrite checks passed"
