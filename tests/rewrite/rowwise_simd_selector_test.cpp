#include <ascify/target/dav_c310/rowwise_simd_selectors_v1.hpp>

#include <cassert>
#include <cstdint>
#include <limits>

namespace simd = ascify::target::dav_c310::rowwise_simd_v1;

template<typename T>
T* Address(uintptr_t value) {
  return reinterpret_cast<T*>(value);
}

int main() {
  static_assert(simd::kHalfStorageBytes == 2U,
                "selector v1 requires two-byte fp16 storage");
  void* stream = Address<void>(0x1000U);
  const void* input = Address<void>(0x100000U);
  void* output = Address<void>(0x200000U);
  void* inverse_rms = Address<void>(0x300000U);
  const void* weight = Address<void>(0x400000U);
  void* mean = Address<void>(0x500000U);
  void* inverse_variance = Address<void>(0x600000U);

  assert(simd::IsSoftmaxSimdDomain(
      stream, input, output, 8, 4096));
  assert(simd::IsSoftmaxSimdDomain(
      stream, input, input, 8, 4096));
  assert(!simd::IsSoftmaxSimdDomain(
      stream, input, Address<void>(0x100002U), 8, 4096));
  assert(!simd::IsSoftmaxSimdDomain(
      stream, input, output, 8, 2048));
  assert(!simd::IsSoftmaxSimdDomain(
      stream, input, output, 8, 4096, 256, 0));

  assert(simd::IsRmsNormPlainRowBatchSimdDomain(
      stream, input, output, nullptr, inverse_rms,
      8, 1536, 1.0e-5));
  assert(simd::IsRmsNormCachedSimdDomain(
      stream, input, output, nullptr, inverse_rms,
      8, 1536, 1.0e-5));
  assert(simd::SelectRmsNormRoute(
             stream, input, output, nullptr, inverse_rms,
             8, 1536, 1.0e-5)
         == simd::RmsNormRoute::kPlainRowBatch);
  assert(simd::SelectRmsNormRoute(
             stream, input, output, nullptr, inverse_rms,
             8, 1544, 1.0e-5)
         == simd::RmsNormRoute::kCached);
  assert(simd::SelectRmsNormRoute(
             stream, input, output, weight, inverse_rms,
             8, 1536, 1.0e-5)
         == simd::RmsNormRoute::kCached);
  assert(simd::SelectRmsNormRoute(
             stream, input, input, nullptr, inverse_rms,
             8, 1536, 1.0e-5)
         == simd::RmsNormRoute::kPlainRowBatch);

  assert(simd::SelectRmsNormRoute(
             stream, input, Address<void>(0x100010U), nullptr, inverse_rms,
             8, 1536, 1.0e-5)
         == simd::RmsNormRoute::kSimt);

  assert(simd::IsLayerNormCachedHybridDomain(
      stream, input, output, mean, inverse_variance,
      8, 1536, 1.0e-5));
  assert(simd::IsLayerNormCachedHybridDomain(
      stream, input, input, mean, inverse_variance,
      8, 1536, 1.0e-5));
  assert(!simd::IsLayerNormCachedHybridDomain(
      stream, input, Address<void>(0x100010U), mean, inverse_variance,
      8, 1536, 1.0e-5));
  assert(!simd::IsLayerNormCachedHybridDomain(
      stream, input, output, Address<void>(0x100000U), inverse_variance,
      8, 1536, 1.0e-5));
  assert(!simd::IsLayerNormCachedHybridDomain(
      stream, input, output, mean, mean, 8, 1536, 1.0e-5));
  assert(!simd::IsLayerNormCachedHybridDomain(
      stream, input, output, mean, inverse_variance,
      8, 1537, 1.0e-5));
  assert(!simd::IsLayerNormCachedHybridDomain(
      stream, input, output, mean, inverse_variance,
      8, 1536, -1.0));
  assert(simd::SelectRmsNormRoute(
             stream, Address<void>(0x100002U), output, nullptr, inverse_rms,
             8, 1536, 1.0e-5)
         == simd::RmsNormRoute::kSimt);
  assert(simd::SelectRmsNormRoute(
             stream, input, output, nullptr, Address<void>(0x100000U),
             8, 1536, 1.0e-5)
         == simd::RmsNormRoute::kSimt);
  assert(simd::SelectRmsNormRoute(
             stream, input, output, Address<void>(0x100000U), inverse_rms,
             8, 1536, 1.0e-5)
         == simd::RmsNormRoute::kSimt);
  assert(simd::SelectRmsNormRoute(
             stream, input, output, nullptr, inverse_rms,
             8, 1536, -1.0)
         == simd::RmsNormRoute::kSimt);
  assert(simd::SelectRmsNormRoute(
             stream, input, output, nullptr, inverse_rms,
             8, 1536, 1.0e-5, 256, 0)
         == simd::RmsNormRoute::kSimt);
  assert(simd::SelectRmsNormRoute(
             stream, input, output, nullptr, inverse_rms,
             std::numeric_limits<int64_t>::max(), 1536, 1.0e-5)
         == simd::RmsNormRoute::kSimt);

  simd::ByteSpan overflow_span{};
  assert(!simd::TryMakeMatrixByteSpan(
      Address<void>(0x1000U), std::numeric_limits<int64_t>::max(), 1,
      simd::kHalfStorageBytes, &overflow_span));

  return 0;
}
