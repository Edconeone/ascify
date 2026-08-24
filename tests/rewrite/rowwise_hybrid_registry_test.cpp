#include <ascify/target/dav_c310/rowwise_simd_recipes.hpp>

#include <cassert>
#include <type_traits>

namespace hybrid = ascify::target::dav_c310::rowwise_simd_v1;

namespace {

struct SyntheticLayerNormRecipe final {
  inline static constexpr bool kRegistered = true;
  inline static constexpr hybrid::HybridStageDescriptor kStages[] = {
      {hybrid::HybridEngine::kMte, hybrid::HybridStageKind::kLoad},
      {hybrid::HybridEngine::kSimd, hybrid::HybridStageKind::kMean},
      {hybrid::HybridEngine::kSimd, hybrid::HybridStageKind::kSquare},
      {hybrid::HybridEngine::kSimd, hybrid::HybridStageKind::kReduceSum},
      {hybrid::HybridEngine::kSimd,
       hybrid::HybridStageKind::kReciprocalSqrt},
      {hybrid::HybridEngine::kSimt, hybrid::HybridStageKind::kNormalize},
      {hybrid::HybridEngine::kSimt, hybrid::HybridStageKind::kOutputAdapt},
      {hybrid::HybridEngine::kMte, hybrid::HybridStageKind::kStore},
  };

  static constexpr hybrid::HybridRecipeDescriptor Descriptor() {
    return {"synthetic.layernorm.hybrid.v1", kStages,
            sizeof(kStages) / sizeof(kStages[0])};
  }

  static hybrid::HybridTryResult Try(int status) {
    return {true, status};
  }
};

struct InvalidSimtOnlyRecipe final {
  inline static constexpr bool kRegistered = true;
  inline static constexpr hybrid::HybridStageDescriptor kStages[] = {
      {hybrid::HybridEngine::kMte, hybrid::HybridStageKind::kLoad},
      {hybrid::HybridEngine::kSimt, hybrid::HybridStageKind::kOutputAdapt},
      {hybrid::HybridEngine::kSimt, hybrid::HybridStageKind::kOutputAdapt},
      {hybrid::HybridEngine::kMte, hybrid::HybridStageKind::kStore},
  };

  static constexpr hybrid::HybridRecipeDescriptor Descriptor() {
    return {"invalid.simt-only", kStages,
            sizeof(kStages) / sizeof(kStages[0])};
  }
};

struct InvalidInteriorMteRecipe final {
  inline static constexpr bool kRegistered = true;
  inline static constexpr hybrid::HybridStageDescriptor kStages[] = {
      {hybrid::HybridEngine::kMte, hybrid::HybridStageKind::kLoad},
      {hybrid::HybridEngine::kSimd, hybrid::HybridStageKind::kMean},
      {hybrid::HybridEngine::kMte, hybrid::HybridStageKind::kStore},
      {hybrid::HybridEngine::kSimt, hybrid::HybridStageKind::kOutputAdapt},
      {hybrid::HybridEngine::kMte, hybrid::HybridStageKind::kStore},
  };

  static constexpr hybrid::HybridRecipeDescriptor Descriptor() {
    return {"invalid.interior-mte", kStages,
            sizeof(kStages) / sizeof(kStages[0])};
  }
};

struct InvalidSecondHandoffRecipe final {
  inline static constexpr bool kRegistered = true;
  inline static constexpr hybrid::HybridStageDescriptor kStages[] = {
      {hybrid::HybridEngine::kMte, hybrid::HybridStageKind::kLoad},
      {hybrid::HybridEngine::kSimd, hybrid::HybridStageKind::kMean},
      {hybrid::HybridEngine::kSimt, hybrid::HybridStageKind::kOutputAdapt},
      {hybrid::HybridEngine::kSimd, hybrid::HybridStageKind::kNormalize},
      {hybrid::HybridEngine::kMte, hybrid::HybridStageKind::kStore},
  };

  static constexpr hybrid::HybridRecipeDescriptor Descriptor() {
    return {"invalid.second-handoff", kStages,
            sizeof(kStages) / sizeof(kStages[0])};
  }
};

struct InvalidPostOutputRecipe final {
  inline static constexpr bool kRegistered = true;
  inline static constexpr hybrid::HybridStageDescriptor kStages[] = {
      {hybrid::HybridEngine::kMte, hybrid::HybridStageKind::kLoad},
      {hybrid::HybridEngine::kSimd, hybrid::HybridStageKind::kMean},
      {hybrid::HybridEngine::kSimt, hybrid::HybridStageKind::kOutputAdapt},
      {hybrid::HybridEngine::kSimt, hybrid::HybridStageKind::kNormalize},
      {hybrid::HybridEngine::kMte, hybrid::HybridStageKind::kStore},
  };

  static constexpr hybrid::HybridRecipeDescriptor Descriptor() {
    return {"invalid.post-output-stage", kStages,
            sizeof(kStages) / sizeof(kStages[0])};
  }
};

struct UnregisteredRecipe final {
  inline static constexpr bool kRegistered = false;
  inline static constexpr hybrid::HybridStageDescriptor kStages[] = {
      {hybrid::HybridEngine::kMte, hybrid::HybridStageKind::kLoad},
      {hybrid::HybridEngine::kSimd, hybrid::HybridStageKind::kNormalize},
      {hybrid::HybridEngine::kSimt, hybrid::HybridStageKind::kOutputAdapt},
      {hybrid::HybridEngine::kMte, hybrid::HybridStageKind::kStore},
  };

  static constexpr hybrid::HybridRecipeDescriptor Descriptor() {
    return {"unregistered.valid-shape", kStages,
            sizeof(kStages) / sizeof(kStages[0])};
  }
};

}  // namespace

static_assert(
    hybrid::IsRegisteredHybridRecipe<hybrid::SoftmaxHybridRecipeV1>::value,
    "Softmax must be a registered mixed recipe");
static_assert(
    hybrid::IsRegisteredHybridRecipe<hybrid::RmsNormHybridRecipeV1>::value,
    "RMSNorm must be a registered mixed recipe");
static_assert(
    hybrid::IsRegisteredHybridRecipe<hybrid::LayerNormHybridRecipeV1>::value,
    "LayerNorm must be a registered mixed recipe");
static_assert(
    hybrid::IsRegisteredHybridRecipe<SyntheticLayerNormRecipe>::value,
    "a third recipe must register without changing the facade");
static_assert(
    !hybrid::IsRegisteredHybridRecipe<InvalidSimtOnlyRecipe>::value,
    "the registry must reject recipes without a SIMD-to-SIMT handoff");
static_assert(
    !hybrid::IsRegisteredHybridRecipe<InvalidInteriorMteRecipe>::value,
    "the registry must reject an interior MTE stage");
static_assert(
    !hybrid::IsRegisteredHybridRecipe<InvalidSecondHandoffRecipe>::value,
    "the registry must reject SIMD after the SIMT handoff");
static_assert(
    !hybrid::IsRegisteredHybridRecipe<InvalidPostOutputRecipe>::value,
    "the registry must reject work after output adaptation");
static_assert(
    !hybrid::IsRegisteredHybridRecipe<UnregisteredRecipe>::value,
    "the registry must require explicit recipe registration");

int main() {
  const hybrid::HybridTryResult result =
      hybrid::RowwiseHybridFacadeV1::Try<SyntheticLayerNormRecipe>(37);
  assert(result.handled);
  assert(result.status == 37);
  return 0;
}
