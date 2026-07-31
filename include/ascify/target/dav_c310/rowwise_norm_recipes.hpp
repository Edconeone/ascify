// Header-only dav-c310 SIMT recipes for fp16 row-wise normalization.
// v1 deliberately exposes only raw row-major fp16 entry points plus adapters
// carrying Ascify's exact-owner, semantically proven row-access contract.
#ifndef ASCIFY_TARGET_DAV_C310_ROWWISE_NORM_RECIPES_HPP_
#define ASCIFY_TARGET_DAV_C310_ROWWISE_NORM_RECIPES_HPP_

// Directly converted code gets one fixed production configuration regardless
// of unrelated TU macros. The native tuning wrapper explicitly opts into the
// caller configuration; all caller definitions are restored after inclusion.
#pragma push_macro("ACLCUB_WARP_SIZE")
#undef ACLCUB_WARP_SIZE

#ifndef ASCIFY_DAV_C310_USE_CALLER_SOFTMAX_FEATURE_FLAGS
#ifdef ASCIFY_TARGET_DAV_C310_DETAIL_SOFTMAX_FP16_IMPL_HPP_
#error "include rowwise_norm_recipes.hpp before the softmax detail header"
#endif
#pragma push_macro("SMN_BLOCK_ROW_MAX")
#pragma push_macro("SMN_FALLBACK_AIV_COUNT")
#pragma push_macro("SMN_USE_ASC_REDUCE")
#pragma push_macro("SMN_FAST_EXP")
#pragma push_macro("SMN_USE_HALF2_EXP")
#pragma push_macro("SMN_HALF2_REFINE")
#pragma push_macro("SMN_HALF2_REFINE_COMPACT")
#pragma push_macro("SMN_STREAM_THREE_PASS")
#pragma push_macro("SMN_USE_BLOCK_PER_ROW")
#pragma push_macro("SMN_FORCE_BLOCK_THREADS")
#undef SMN_BLOCK_ROW_MAX
#undef SMN_FALLBACK_AIV_COUNT
#undef SMN_USE_ASC_REDUCE
#undef SMN_FAST_EXP
#undef SMN_USE_HALF2_EXP
#undef SMN_HALF2_REFINE
#undef SMN_HALF2_REFINE_COMPACT
#undef SMN_STREAM_THREE_PASS
#undef SMN_USE_BLOCK_PER_ROW
#undef SMN_FORCE_BLOCK_THREADS
#define SMN_BLOCK_ROW_MAX 2048
#define SMN_FALLBACK_AIV_COUNT 56
#define SMN_USE_ASC_REDUCE
#define SMN_USE_HALF2_EXP
#define SMN_HALF2_REFINE_COMPACT
#define SMN_USE_BLOCK_PER_ROW
#include <ascify/target/dav_c310/detail/softmax_fp16_impl.hpp>
#undef SMN_BLOCK_ROW_MAX
#undef SMN_FALLBACK_AIV_COUNT
#undef SMN_USE_ASC_REDUCE
#undef SMN_USE_HALF2_EXP
#undef SMN_HALF2_REFINE_COMPACT
#undef SMN_USE_BLOCK_PER_ROW
#pragma pop_macro("SMN_FORCE_BLOCK_THREADS")
#pragma pop_macro("SMN_USE_BLOCK_PER_ROW")
#pragma pop_macro("SMN_STREAM_THREE_PASS")
#pragma pop_macro("SMN_HALF2_REFINE_COMPACT")
#pragma pop_macro("SMN_HALF2_REFINE")
#pragma pop_macro("SMN_USE_HALF2_EXP")
#pragma pop_macro("SMN_FAST_EXP")
#pragma pop_macro("SMN_USE_ASC_REDUCE")
#pragma pop_macro("SMN_FALLBACK_AIV_COUNT")
#pragma pop_macro("SMN_BLOCK_ROW_MAX")
#else
#if defined(SMN_USE_BLOCK_PER_ROW) && !defined(SMN_USE_HALF2_EXP)
#error "SMN_USE_BLOCK_PER_ROW requires SMN_USE_HALF2_EXP"
#endif
// REFINE+COMPACT is a valid hybrid: warp paths keep float2 full refinement
// while the block-per-row path stores compact refined half2 exponentials.
#if defined(SMN_HALF2_REFINE_COMPACT) && !defined(SMN_USE_HALF2_EXP) \
    && !defined(SMN_HALF2_REFINE)
#error "SMN_HALF2_REFINE_COMPACT requires a half2 cache mode"
#endif
#pragma push_macro("SMN_BLOCK_ROW_MAX")
#pragma push_macro("SMN_FALLBACK_AIV_COUNT")
#include <ascify/target/dav_c310/detail/softmax_fp16_impl.hpp>
#pragma pop_macro("SMN_FALLBACK_AIV_COUNT")
#pragma pop_macro("SMN_BLOCK_ROW_MAX")
#endif

#ifndef ASCIFY_DAV_C310_USE_CALLER_RMSNORM_FEATURE_FLAGS
#ifdef ASCIFY_TARGET_DAV_C310_DETAIL_RMSNORM_FP16_IMPL_HPP_
#error "include rowwise_norm_recipes.hpp before the RMSNorm detail header"
#endif
#pragma push_macro("ASCIFY950_RMN_BLOCK_ROW_THREADS")
#pragma push_macro("ASCIFY950_RMN_BLOCK_ROW_AFFINE_THREADS")
#undef ASCIFY950_RMN_BLOCK_ROW_THREADS
#undef ASCIFY950_RMN_BLOCK_ROW_AFFINE_THREADS
#define ASCIFY950_RMN_BLOCK_ROW_THREADS 512
#define ASCIFY950_RMN_BLOCK_ROW_AFFINE_THREADS 256
#include <ascify/target/dav_c310/detail/rmsnorm_fp16_impl.hpp>
#undef ASCIFY950_RMN_BLOCK_ROW_THREADS
#undef ASCIFY950_RMN_BLOCK_ROW_AFFINE_THREADS
#pragma pop_macro("ASCIFY950_RMN_BLOCK_ROW_AFFINE_THREADS")
#pragma pop_macro("ASCIFY950_RMN_BLOCK_ROW_THREADS")
#else
#pragma push_macro("ASCIFY950_RMN_BLOCK_ROW_THREADS")
#pragma push_macro("ASCIFY950_RMN_BLOCK_ROW_AFFINE_THREADS")
#include <ascify/target/dav_c310/detail/rmsnorm_fp16_impl.hpp>
#pragma pop_macro("ASCIFY950_RMN_BLOCK_ROW_AFFINE_THREADS")
#pragma pop_macro("ASCIFY950_RMN_BLOCK_ROW_THREADS")
#endif

#pragma pop_macro("ACLCUB_WARP_SIZE")

#include <cstdint>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

namespace ascify {
namespace target {
namespace dav_c310 {
namespace v1 {

struct TryResult {
  // handled=true means the caller must return status, including error values.
  // handled=false is the only state in which the converted fallback may run.
  bool handled;
  aclError status;
};

constexpr int64_t kMaxValidatedRecipeColumns = 32768;

static inline TryResult NotHandled() { return {false, ACL_SUCCESS}; }

namespace contract_detail {

constexpr bool SoftmaxColumnsFitTargetKernel(int64_t columns) {
  return columns > 0 && columns <= std::numeric_limits<int>::max();
}

// The converted RMSNorm CUDA kernels receive nrow/ncol as int even though
// their host launch wrappers accept int64_t. Only intercept the range in
// which those wrapper-to-kernel conversions preserve the original values.
constexpr bool RmsNormDimensionsFitOriginalKernel(
    int64_t rows, int64_t columns) {
  return rows > 0 && rows <= std::numeric_limits<int>::max()
      && columns > 0 && columns <= std::numeric_limits<int>::max()
      && columns <= kMaxValidatedRecipeColumns
      && rows <= std::numeric_limits<int64_t>::max() / columns;
}

}  // namespace contract_detail

// Raw-pointer APIs are intentionally limited to contiguous row-major fp16.
// block_threads/grid_cap are tuning controls; zero asks the implementation to
// select the validated target defaults.
static inline aclError SoftmaxFp16(
    aclrtStream stream, const half* input, half* output, int64_t rows,
    int64_t columns, int block_threads = 0, int grid_cap = 0) {
  // Validate both bounds before narrowing. In particular, a large negative
  // int64_t must never wrap to a positive kernel column count.
  if (!contract_detail::SoftmaxColumnsFitTargetKernel(columns)) {
    return ACL_ERROR_RT_PARAM_INVALID;
  }
  return detail::softmax_fp16::Launch(
      stream, input, output, rows, static_cast<int>(columns), block_threads,
      grid_cap);
}

static inline aclError RmsNormFp16(
    aclrtStream stream, const half* input, half* output, const half* weight,
    int64_t rows, int64_t columns, double epsilon, float* inverse_rms,
    int block_threads = 0, int grid_cap = 0) {
  return detail::rmsnorm_fp16::Launch(
      stream, input, output, weight, inverse_rms, rows, columns, epsilon,
      block_threads, grid_cap);
}

static inline TryResult TrySoftmax(
    aclrtStream stream, const half* input, half* output, int64_t rows,
    int64_t columns) {
  if (stream == nullptr || input == nullptr || output == nullptr || rows <= 0
      || !contract_detail::SoftmaxColumnsFitTargetKernel(columns)
      || columns > kMaxValidatedRecipeColumns
      || rows > std::numeric_limits<int64_t>::max() / columns) {
    return NotHandled();
  }
  return {true, SoftmaxFp16(stream, input, output, rows, columns)};
}

static inline TryResult TryRmsNorm(
    aclrtStream stream, const half* input, half* output, const half* weight,
    int64_t rows, int64_t columns, double epsilon, float* inverse_rms) {
  if (stream == nullptr || input == nullptr || output == nullptr
      || inverse_rms == nullptr
      || !contract_detail::RmsNormDimensionsFitOriginalKernel(
             rows, columns)
      || !(epsilon > 0.0) || !std::isfinite(epsilon)) {
    return NotHandled();
  }
  return {true, RmsNormFp16(stream, input, output, weight, rows, columns,
                           epsilon, inverse_rms)};
}

namespace adapter_detail {

template<typename...>
using VoidT = void;

template<typename Pointer>
struct IsHalfInputPointer
    : std::integral_constant<
          bool,
          std::is_convertible<
              typename std::remove_reference<Pointer>::type,
              const half*>::value> {};

template<typename Pointer>
struct IsHalfOutputPointer
    : std::integral_constant<
          bool,
          std::is_convertible<
              typename std::remove_reference<Pointer>::type,
              half*>::value> {};

template<typename T, typename = void>
struct HasMarkedLoadAccess : std::false_type {};

template<typename T>
struct HasMarkedLoadAccess<
    T, VoidT<typename T::ascify_target_direct_load_tag,
             typename T::ascify_target_adapter_owner_type,
             typename T::ascify_target_storage_type,
             typename T::ascify_target_compute_type,
             decltype(std::declval<const T&>().ascify_target_data()),
             decltype(std::declval<const T&>().ascify_target_row_stride())>>
    : std::integral_constant<
          bool,
          std::is_same<T, typename T::ascify_target_adapter_owner_type>::value
              && std::is_same<
              typename std::remove_cv<
                  typename T::ascify_target_storage_type>::type,
              half>::value
              && std::is_same<
                     typename std::remove_cv<
                         typename T::ascify_target_compute_type>::type,
                     float>::value
              && IsHalfInputPointer<decltype(
                     std::declval<const T&>().ascify_target_data())>::value> {};

template<typename T, typename = void>
struct HasMarkedStoreAccess : std::false_type {};

template<typename T>
struct HasMarkedStoreAccess<
    T, VoidT<typename T::ascify_target_direct_store_tag,
             typename T::ascify_target_adapter_owner_type,
             typename T::ascify_target_storage_type,
             typename T::ascify_target_compute_type,
             typename std::integral_constant<
                 bool, static_cast<bool>(
                           T::ascify_target_store_is_affine)>::type,
             decltype(std::declval<const T&>().ascify_target_data()),
             decltype(std::declval<const T&>().ascify_target_row_stride())>>
    : std::integral_constant<
          bool,
          std::is_same<T, typename T::ascify_target_adapter_owner_type>::value
              && std::is_same<
              typename std::remove_cv<
                  typename T::ascify_target_storage_type>::type,
              half>::value
              && std::is_same<
                     typename std::remove_cv<
                         typename T::ascify_target_compute_type>::type,
                     float>::value
              && IsHalfOutputPointer<decltype(
                     std::declval<const T&>().ascify_target_data())>::value> {};

template<typename T, typename = void>
struct HasMarkedAffineStoreAccess : std::false_type {};

template<typename T>
struct HasMarkedAffineStoreAccess<
    T, VoidT<typename T::ascify_target_direct_store_tag,
             typename T::ascify_target_adapter_owner_type,
             decltype(std::declval<const T&>().ascify_target_weight())>>
    : std::integral_constant<
          bool,
          std::is_same<T, typename T::ascify_target_adapter_owner_type>::value
              && IsHalfInputPointer<decltype(
                     std::declval<const T&>().ascify_target_weight())>::value> {};

template<typename T, typename Enable = void>
struct LoadAccess {
  static constexpr bool kValid = false;
};

template<typename T>
struct LoadAccess<
    T, typename std::enable_if<HasMarkedLoadAccess<T>::value>::type> {
  static constexpr bool kValid = true;
  static const half* Data(const T& value) {
    return value.ascify_target_data();
  }
  static int64_t Stride(const T& value) {
    return static_cast<int64_t>(value.ascify_target_row_stride());
  }
};

template<typename T, typename Enable = void>
struct StoreAccess {
  static constexpr bool kValid = false;
  static constexpr bool kAffine = false;
};

template<typename T>
struct StoreAccess<
    T, typename std::enable_if<HasMarkedStoreAccess<T>::value>::type> {
  static constexpr bool kAffine =
      static_cast<bool>(T::ascify_target_store_is_affine);
  static constexpr bool kValid =
      !kAffine || HasMarkedAffineStoreAccess<T>::value;
  static half* Data(const T& value) { return value.ascify_target_data(); }
  static int64_t Stride(const T& value) {
    return static_cast<int64_t>(value.ascify_target_row_stride());
  }
  static const half* Weight(const T& value) {
    return WeightImpl(
        value,
        std::integral_constant<
            bool, kAffine && HasMarkedAffineStoreAccess<T>::value>{});
  }

 private:
  static const half* WeightImpl(const T& value, std::true_type) {
    return value.ascify_target_weight();
  }
  static const half* WeightImpl(const T&, std::false_type) { return nullptr; }
};

template<typename T>
using Bare = typename std::remove_cv<
    typename std::remove_reference<T>::type>::type;

template<typename T, typename = void>
struct MarkedComputeType {
  using type = void;
};

template<typename T>
struct MarkedComputeType<
    T, VoidT<typename T::ascify_target_compute_type>> {
  using type = typename T::ascify_target_compute_type;
};

template<typename Compute, typename Load, typename Store>
struct HasSupportedComputeContract
    : std::integral_constant<
          bool,
          std::is_same<Compute, float>::value
              && std::is_same<
                     Compute,
                     typename MarkedComputeType<Bare<Load>>::type>::value
              && std::is_same<
                     Compute,
                     typename MarkedComputeType<Bare<Store>>::type>::value> {};

}  // namespace adapter_detail

template<typename Load, typename Store, typename Compute>
static inline TryResult TrySoftmax(
    aclrtStream stream, const Load& load, const Store& store, int64_t rows,
    int64_t columns, Compute*) {
  using LoadAccess = adapter_detail::LoadAccess<
      adapter_detail::Bare<Load>>;
  using StoreAccess = adapter_detail::StoreAccess<
      adapter_detail::Bare<Store>>;
  if constexpr (!LoadAccess::kValid || !StoreAccess::kValid
                || StoreAccess::kAffine
                || !adapter_detail::HasSupportedComputeContract<
                       Compute, Load, Store>::value) {
    return NotHandled();
  } else {
    if (LoadAccess::Stride(load) != columns
        || StoreAccess::Stride(store) != columns) {
      return NotHandled();
    }
    return TrySoftmax(stream, LoadAccess::Data(load),
                      StoreAccess::Data(store), rows, columns);
  }
}

template<typename Load, typename Store, typename Inverse, typename Compute>
static inline TryResult TryRmsNorm(
    aclrtStream stream, const Load& load, const Store& store, int64_t rows,
    int64_t columns, double epsilon, Inverse* inverse_rms, Compute*) {
  using LoadAccess = adapter_detail::LoadAccess<
      adapter_detail::Bare<Load>>;
  using StoreAccess = adapter_detail::StoreAccess<
      adapter_detail::Bare<Store>>;
  if constexpr (!LoadAccess::kValid || !StoreAccess::kValid
                || !std::is_same<Inverse, float>::value
                || !adapter_detail::HasSupportedComputeContract<
                       Compute, Load, Store>::value) {
    return NotHandled();
  } else {
    if (LoadAccess::Stride(load) != columns
        || StoreAccess::Stride(store) != columns) {
      return NotHandled();
    }
    const half* weight = StoreAccess::Weight(store);
    if constexpr (StoreAccess::kAffine) {
      if (weight == nullptr) {
        // The affine contract requires a live per-column weight. Preserve the
        // converted implementation when a caller violates that precondition.
        return NotHandled();
      }
    }
    return TryRmsNorm(stream, LoadAccess::Data(load),
                      StoreAccess::Data(store), weight, rows, columns, epsilon,
                      inverse_rms);
  }
}

}  // namespace v1

// Recipe-generated source uses the unversioned aliases. Explicit v1 remains
// available so generated artifacts can pin their ABI/semantic contract.
using v1::RmsNormFp16;
using v1::SoftmaxFp16;
using v1::TryResult;
using v1::TryRmsNorm;
using v1::TrySoftmax;

}  // namespace dav_c310
}  // namespace target
}  // namespace ascify

#endif  // ASCIFY_TARGET_DAV_C310_ROWWISE_NORM_RECIPES_HPP_
