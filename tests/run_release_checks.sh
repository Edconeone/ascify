#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
rewrite_dir="$repo_root/tests/rewrite"
python=${PYTHON:-python3}

run_check() {
  printf '[release-check] %s\n' "$1"
  shift
  "$@"
}

if ! command -v "$python" >/dev/null 2>&1; then
  echo "Python 3 executable not found: $python" >&2
  exit 1
fi

cd -- "$repo_root"

run_check "dav-c310 recipe static contract" \
  sh "$rewrite_dir/check_dav_c310_recipe.sh"
run_check "canonical warp reduction static contract" \
  sh "$rewrite_dir/check_warp_reduce_rewrite.sh"
run_check "canonical reducer static contract" \
  sh "$rewrite_dir/check_canonical_reducer_rewrite.sh"
run_check "SIMT compatibility static contract" \
  sh "$rewrite_dir/check_simt_compat.sh"
run_check "formal-evidence and work-metric Python unit tests" \
  "$python" -B -m unittest -q \
    tests/softmax_rmsnorm_950/scripts/test_validate_formal_run.py \
    tests/softmax_rmsnorm_950/tools/test_derive_work_metrics.py

if [ -z "${ASCIFY_BINARY:-}" ]; then
  echo "[release-check] translated fixtures skipped (ASCIFY_BINARY is unset)"
  echo "[release-check] all host-only checks passed"
  exit 0
fi

if [ ! -x "$ASCIFY_BINARY" ]; then
  echo "ASCIFY_BINARY is not executable: $ASCIFY_BINARY" >&2
  exit 1
fi
if [ -z "${ASCIFY_CUDA_PATH:-}" ]; then
  echo "ASCIFY_CUDA_PATH is required when ASCIFY_BINARY is set" >&2
  exit 1
fi
if [ ! -d "$ASCIFY_CUDA_PATH" ]; then
  echo "ASCIFY_CUDA_PATH is not a directory: $ASCIFY_CUDA_PATH" >&2
  exit 1
fi
if [ -z "${ASCIFY_CLANG_RESOURCE_DIRECTORY:-}" ]; then
  echo "ASCIFY_CLANG_RESOURCE_DIRECTORY is required when ASCIFY_BINARY is set" >&2
  exit 1
fi
if [ ! -d "$ASCIFY_CLANG_RESOURCE_DIRECTORY/include" ]; then
  echo "Clang resource include directory not found: $ASCIFY_CLANG_RESOURCE_DIRECTORY/include" >&2
  exit 1
fi

work_dir=$(mktemp -d "${TMPDIR:-/tmp}/ascify-release-checks.XXXXXX")
cleanup() {
  case "$work_dir" in
    "${TMPDIR:-/tmp}"/ascify-release-checks.*)
      rm -rf -- "$work_dir"
      ;;
    *)
      echo "refusing to remove unexpected work directory: $work_dir" >&2
      ;;
  esac
}
trap cleanup 0 1 2 15

translate_fixture() {
  input=$1
  output=$2
  target_policy=$3
  math_mode=$4
  run_check "translate $(basename "$input") ($target_policy/$math_mode)" \
    "$ASCIFY_BINARY" "$input" \
      "--target-policy=$target_policy" \
      "--simt-math=$math_mode" \
      "--cuda-path=$ASCIFY_CUDA_PATH" \
      "--clang-resource-directory=$ASCIFY_CLANG_RESOURCE_DIRECTORY" \
      -o "$output" \
      -- "-I$repo_root/tests/support/cuda" -std=c++17
}

warp_fast="$work_dir/warp_reduce.fast.cce"
warp_default="$work_dir/warp_reduce.default.cce"
reducer_fast="$work_dir/canonical_reducer.fast.cce"
reducer_default="$work_dir/canonical_reducer.default.cce"
simt_translated="$work_dir/simt_compat.cce"
recipe_fast="$work_dir/dav_c310_recipe.fast.cce"
recipe_portable_precise="$work_dir/dav_c310_recipe.portable_precise.cce"
recipe_portable_fast="$work_dir/dav_c310_recipe.portable_fast.cce"
recipe_dav_precise="$work_dir/dav_c310_recipe.dav_precise.cce"

translate_fixture \
  "$rewrite_dir/warp_reduce_input.cu" "$warp_fast" dav-c310-vec fast
translate_fixture \
  "$rewrite_dir/warp_reduce_input.cu" "$warp_default" portable precise
translate_fixture \
  "$rewrite_dir/canonical_reducer_input.cu" "$reducer_fast" dav-c310-vec fast
translate_fixture \
  "$rewrite_dir/canonical_reducer_input.cu" "$reducer_default" portable precise
translate_fixture \
  "$rewrite_dir/simt_compat_input.cu" "$simt_translated" portable precise
translate_fixture \
  "$rewrite_dir/dav_c310_recipe_input.cu" "$recipe_fast" dav-c310-vec fast
translate_fixture \
  "$rewrite_dir/dav_c310_recipe_input.cu" \
  "$recipe_portable_precise" portable precise
translate_fixture \
  "$rewrite_dir/dav_c310_recipe_input.cu" \
  "$recipe_portable_fast" portable fast
translate_fixture \
  "$rewrite_dir/dav_c310_recipe_input.cu" \
  "$recipe_dav_precise" dav-c310-vec precise

run_check "canonical warp reduction translated goldens" \
  sh "$rewrite_dir/check_warp_reduce_rewrite.sh" \
    --fast "$warp_fast" --default "$warp_default"
run_check "canonical reducer translated goldens" \
  sh "$rewrite_dir/check_canonical_reducer_rewrite.sh" \
    --fast "$reducer_fast" --default "$reducer_default"
run_check "SIMT compatibility translated golden" \
  sh "$rewrite_dir/check_simt_compat.sh" \
    --translated "$simt_translated"
run_check "dav-c310 recipe translated goldens" \
  sh "$rewrite_dir/check_dav_c310_recipe.sh" \
    --dav-fast "$recipe_fast" \
    --portable-precise "$recipe_portable_precise" \
    --portable-fast "$recipe_portable_fast" \
    --dav-precise "$recipe_dav_precise"

echo "[release-check] all host-only and translated-fixture checks passed"
