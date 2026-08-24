#ifndef ASCIFY_TEST_STUB_DEVICE_ATOMIC_FUNCTIONS_H
#define ASCIFY_TEST_STUB_DEVICE_ATOMIC_FUNCTIONS_H

#include <stdint.h>

#ifndef __gm__
#define __gm__
#endif

inline int ascify_test_legacy_global_atomic_calls = 0;

#define ASCIFY_TEST_DEFINE_ATOMIC_BINARY(name, expression)                  \
  inline int32_t name(int32_t* address, int32_t value) {                    \
    ++ascify_test_legacy_global_atomic_calls;                               \
    const int32_t old = *address;                                           \
    *address = (expression);                                                \
    return old;                                                             \
  }                                                                         \
  inline uint32_t name(uint32_t* address, uint32_t value) {                 \
    ++ascify_test_legacy_global_atomic_calls;                               \
    const uint32_t old = *address;                                          \
    *address = (expression);                                                \
    return old;                                                             \
  }

ASCIFY_TEST_DEFINE_ATOMIC_BINARY(asc_atomic_add, old + value)
ASCIFY_TEST_DEFINE_ATOMIC_BINARY(asc_atomic_sub, old - value)
ASCIFY_TEST_DEFINE_ATOMIC_BINARY(asc_atomic_exch, value)
ASCIFY_TEST_DEFINE_ATOMIC_BINARY(asc_atomic_max, old > value ? old : value)
ASCIFY_TEST_DEFINE_ATOMIC_BINARY(asc_atomic_min, old < value ? old : value)
ASCIFY_TEST_DEFINE_ATOMIC_BINARY(asc_atomic_and, old & value)
ASCIFY_TEST_DEFINE_ATOMIC_BINARY(asc_atomic_or, old | value)
ASCIFY_TEST_DEFINE_ATOMIC_BINARY(asc_atomic_xor, old ^ value)

#undef ASCIFY_TEST_DEFINE_ATOMIC_BINARY

inline uint32_t asc_atomic_inc(uint32_t* address, uint32_t limit) {
  ++ascify_test_legacy_global_atomic_calls;
  const uint32_t old = *address;
  *address = old >= limit ? 0U : old + 1U;
  return old;
}

inline uint32_t asc_atomic_dec(uint32_t* address, uint32_t limit) {
  ++ascify_test_legacy_global_atomic_calls;
  const uint32_t old = *address;
  *address = (old == 0U || old > limit) ? limit : old - 1U;
  return old;
}

inline int32_t asc_atomic_cas(
    int32_t* address, int32_t compare, int32_t value) {
  ++ascify_test_legacy_global_atomic_calls;
  const int32_t old = *address;
  if (old == compare) { *address = value; }
  return old;
}

inline uint32_t asc_atomic_cas(
    uint32_t* address, uint32_t compare, uint32_t value) {
  ++ascify_test_legacy_global_atomic_calls;
  const uint32_t old = *address;
  if (old == compare) { *address = value; }
  return old;
}

#endif
