#include "../../src/StaticKernelProof.h"

#include <cassert>
#include <cstdint>

namespace {

const clang::FunctionDecl *opaqueFunction(std::uintptr_t value) {
  return reinterpret_cast<const clang::FunctionDecl *>(value);
}

ascify::detail::StaticKernelProof completeProof(
    ascify::detail::StaticKernelSemanticFamily family,
    ascify::detail::StaticKernelSourceProvenance provenance) {
  ascify::detail::StaticKernelProof proof;
  proof.canonicalDecl = opaqueFunction(1);
  proof.definitionDecl = opaqueFunction(2);
  proof.definitionRange = {3, 4};
  proof.provenance = provenance;
  proof.family = family;
  return proof;
}

} // namespace

int main() {
  using ascify::detail::StaticKernelSemanticFamily;
  using ascify::detail::StaticKernelSourceProvenance;

  const StaticKernelSemanticFamily families[] = {
      StaticKernelSemanticFamily::Softmax,
      StaticKernelSemanticFamily::RmsNorm,
      StaticKernelSemanticFamily::LayerNorm,
  };
  const StaticKernelSourceProvenance provenances[] = {
      StaticKernelSourceProvenance::MainFile,
      StaticKernelSourceProvenance::UserHeader,
      StaticKernelSourceProvenance::SystemHeader,
  };
  for (const StaticKernelSemanticFamily family : families) {
    for (const StaticKernelSourceProvenance provenance : provenances)
      assert(completeProof(family, provenance).proven());
  }

  ascify::detail::StaticKernelProof incomplete;
  assert(!incomplete.proven());
  incomplete = completeProof(
      StaticKernelSemanticFamily::Softmax,
      StaticKernelSourceProvenance::MainFile);
  incomplete.canonicalDecl = nullptr;
  assert(!incomplete.proven());
  incomplete = completeProof(
      StaticKernelSemanticFamily::Softmax,
      StaticKernelSourceProvenance::MainFile);
  incomplete.definitionRange.beginRawEncoding = 0;
  assert(!incomplete.proven());
  incomplete = completeProof(
      StaticKernelSemanticFamily::Softmax,
      StaticKernelSourceProvenance::Unknown);
  assert(!incomplete.proven());
  incomplete = completeProof(
      StaticKernelSemanticFamily::Unknown,
      StaticKernelSourceProvenance::MainFile);
  assert(!incomplete.proven());
}
