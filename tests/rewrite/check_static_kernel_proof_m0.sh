#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
test_build_dir="$(mktemp -d "${TMPDIR:-/tmp}/ascify-static-kernel-proof.XXXXXX")"
trap 'rm -rf "${test_build_dir}"' EXIT

compiler="${CXX:-c++}"
"${compiler}" -std=c++14 -Wall -Wextra -Werror \
  "${repo_root}/tests/rewrite/static_kernel_proof_test.cpp" \
  -o "${test_build_dir}/static_kernel_proof_test"
"${test_build_dir}/static_kernel_proof_test"

recipe_source="${repo_root}/src/DavC310TargetRecipe.cpp"
proof_header="${repo_root}/src/StaticKernelProof.h"

grep -F 'class StaticKernelProofRegistry' "${recipe_source}" >/dev/null
grep -F 'const StaticKernelProofRegistry &kernelProofs' \
  "${recipe_source}" >/dev/null
grep -F 'kernelProofs.find(' "${recipe_source}" >/dev/null
grep -F 'softmaxProof->semanticBindings' "${recipe_source}" >/dev/null
grep -F 'StaticKernelSemanticFamily::Softmax' "${recipe_source}" >/dev/null
grep -F 'StaticKernelSemanticFamily::RmsNorm' "${recipe_source}" >/dev/null
grep -F 'StaticKernelSemanticFamily::LayerNorm' "${recipe_source}" >/dev/null
grep -F 'StaticKernelSourceProvenance::MainFile' "${recipe_source}" >/dev/null
grep -F 'StaticKernelSourceProvenance::UserHeader' "${recipe_source}" >/dev/null
grep -F 'StaticKernelSourceProvenance::SystemHeader' "${recipe_source}" >/dev/null
grep -F 'This is an analyzer-to-emitter contract, not a lowering IR.' \
  "${proof_header}" >/dev/null

if grep -F 'StaticKernelProof' \
    "${repo_root}/include/ascify/target/dav_c310/rowwise_hybrid_registry_v1.hpp" \
    "${repo_root}/include/ascify/target/dav_c310/rowwise_simd_recipes.hpp" \
    >/dev/null; then
  echo "static kernel proof leaked into the runtime recipe/ABI surface" >&2
  exit 1
fi

echo "static kernel proof M0 unit and ownership checks passed"
