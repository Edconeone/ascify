#define __aicore__
#define __global__

#include <ascify/ascify_cuda_compat.hpp>

#include <cassert>
#include <cstdint>

#if !defined(ASCIFY_SIMT_HEADER_FAMILY_LEGACY_BETA3) || \
    !defined(ASCIFY_SIMT_LEGACY_HAS_GM_ATOMICS)
#error "the positive atomic contract requires verified legacy GM atomics"
#endif

int main() {
  int32_t signed_value = 10;
  assert(ascify::atomic_add_global(&signed_value, int32_t{5}) == 10);
  assert(signed_value == 15);
  assert(ascify::atomic_sub_global(&signed_value, int32_t{3}) == 15);
  assert(signed_value == 12);
  assert(ascify::atomic_exch_global(&signed_value, int32_t{7}) == 12);
  assert(signed_value == 7);
  assert(ascify::atomic_max_global(&signed_value, int32_t{9}) == 7);
  assert(signed_value == 9);
  assert(ascify::atomic_min_global(&signed_value, int32_t{4}) == 9);
  assert(signed_value == 4);
  assert(ascify::atomic_cas_global(
             &signed_value, int32_t{4}, int32_t{11}) == 4);
  assert(signed_value == 11);
  assert(ascify::atomic_and_global(&signed_value, int32_t{6}) == 11);
  assert(signed_value == 2);
  assert(ascify::atomic_or_global(&signed_value, int32_t{8}) == 2);
  assert(signed_value == 10);
  assert(ascify::atomic_xor_global(&signed_value, int32_t{3}) == 10);
  assert(signed_value == 9);

  uint32_t unsigned_value = 10U;
  assert(ascify::atomic_add_global(&unsigned_value, uint32_t{5}) == 10U);
  assert(unsigned_value == 15U);
  assert(ascify::atomic_sub_global(&unsigned_value, uint32_t{3}) == 15U);
  assert(unsigned_value == 12U);
  assert(ascify::atomic_exch_global(&unsigned_value, uint32_t{7}) == 12U);
  assert(unsigned_value == 7U);
  assert(ascify::atomic_max_global(&unsigned_value, uint32_t{9}) == 7U);
  assert(unsigned_value == 9U);
  assert(ascify::atomic_min_global(&unsigned_value, uint32_t{4}) == 9U);
  assert(unsigned_value == 4U);
  assert(ascify::atomic_cas_global(
             &unsigned_value, uint32_t{4}, uint32_t{11}) == 4U);
  assert(unsigned_value == 11U);
  assert(ascify::atomic_and_global(&unsigned_value, uint32_t{6}) == 11U);
  assert(unsigned_value == 2U);
  assert(ascify::atomic_or_global(&unsigned_value, uint32_t{8}) == 2U);
  assert(unsigned_value == 10U);
  assert(ascify::atomic_xor_global(&unsigned_value, uint32_t{3}) == 10U);
  assert(unsigned_value == 9U);
  assert(ascify::atomic_inc_global(&unsigned_value, uint32_t{9}) == 9U);
  assert(unsigned_value == 0U);
  assert(ascify::atomic_dec_global(&unsigned_value, uint32_t{9}) == 0U);
  assert(unsigned_value == 9U);

  assert(ascify_test_legacy_global_atomic_calls == 20);
  return 0;
}
