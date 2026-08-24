// Operator-independent registry contract for dav-3510 row-wise hybrid
// recipes.  A recipe describes stage ownership and supplies a typed Try
// entry; the registry validates the mixed-stage shape at compile time.
#ifndef ASCIFY_TARGET_DAV_C310_ROWWISE_HYBRID_REGISTRY_V1_HPP_
#define ASCIFY_TARGET_DAV_C310_ROWWISE_HYBRID_REGISTRY_V1_HPP_

#include <cstddef>
#include <type_traits>
#include <utility>

namespace ascify {
namespace target {
namespace dav_c310 {
namespace rowwise_simd_v1 {

enum class HybridEngine : unsigned char {
  kMte,
  kSimd,
  kSimt,
};

enum class HybridStageKind : unsigned char {
  kLoad,
  kReduceMaximum,
  kExponentiate,
  kReduceSum,
  kSquare,
  kMean,
  kReciprocalSqrt,
  kNormalize,
  kAffine,
  kOutputAdapt,
  kStore,
};

struct HybridStageDescriptor {
  HybridEngine engine;
  HybridStageKind kind;
};

struct HybridRecipeDescriptor {
  const char* id;
  const HybridStageDescriptor* stages;
  std::size_t stage_count;
};

constexpr bool HasText(const char* text) {
  return text != nullptr && text[0] != '\0';
}

// The first registry version intentionally validates only the composition
// contract shared by all supported operators.  Per-operator semantics,
// layouts, aliases, shapes, and ABI checks remain in their proof/selector.
constexpr bool IsValidHybridRecipe(const HybridRecipeDescriptor& recipe) {
  if (!HasText(recipe.id) || recipe.stages == nullptr
      || recipe.stage_count < 4U
      || recipe.stages[0].engine != HybridEngine::kMte
      || recipe.stages[0].kind != HybridStageKind::kLoad
      || recipe.stages[recipe.stage_count - 1U].engine
             != HybridEngine::kMte
      || recipe.stages[recipe.stage_count - 1U].kind
             != HybridStageKind::kStore) {
    return false;
  }

  bool saw_simd = false;
  bool saw_simt = false;
  bool saw_output_adapt = false;
  for (std::size_t index = 1U; index + 1U < recipe.stage_count; ++index) {
    const HybridStageDescriptor stage = recipe.stages[index];
    if (stage.engine == HybridEngine::kMte) {
      return false;
    }
    if (stage.engine == HybridEngine::kSimd) {
      // Version 1 has one SIMD-to-SIMT handoff.  A SIMD stage after SIMT
      // would need another synchronization and ownership contract.
      if (saw_simt) {
        return false;
      }
      saw_simd = true;
      continue;
    }
    if (stage.engine != HybridEngine::kSimt || !saw_simd
        || saw_output_adapt) {
      return false;
    }
    saw_simt = true;
    if (stage.kind == HybridStageKind::kOutputAdapt) {
      // Output adaptation terminates the SIMT region.  Additional SIMT work
      // before it is valid; a stage after it would violate output ownership.
      saw_output_adapt = true;
    }
  }
  return saw_simd && saw_simt && saw_output_adapt;
}

template<typename Recipe, typename = void>
struct IsRegisteredHybridRecipe : std::false_type {};

template<typename Recipe>
struct IsRegisteredHybridRecipe<
    Recipe,
    typename std::enable_if<
        Recipe::kRegistered &&
        IsValidHybridRecipe(Recipe::Descriptor())>::type>
    : std::true_type {};

template<typename Recipe, typename... Arguments>
inline auto DispatchRegisteredHybrid(Arguments&&... arguments)
    -> decltype(Recipe::Try(std::forward<Arguments>(arguments)...)) {
  static_assert(
      IsRegisteredHybridRecipe<Recipe>::value,
      "hybrid recipe must publish a valid registered stage descriptor");
  return Recipe::Try(std::forward<Arguments>(arguments)...);
}

}  // namespace rowwise_simd_v1
}  // namespace dav_c310
}  // namespace target
}  // namespace ascify

#endif  // ASCIFY_TARGET_DAV_C310_ROWWISE_HYBRID_REGISTRY_V1_HPP_
