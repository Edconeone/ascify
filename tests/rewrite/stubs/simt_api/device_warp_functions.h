#ifndef ASCIFY_TEST_STUB_DEVICE_WARP_FUNCTIONS_H
#define ASCIFY_TEST_STUB_DEVICE_WARP_FUNCTIONS_H

#include <stdint.h>

inline float asc_reduce_add(float value) {
  return value;
}

inline int32_t asc_reduce_add(int32_t value) {
  return value;
}

inline uint32_t asc_reduce_add(uint32_t value) {
  return value;
}

inline float asc_reduce_max(float value) {
  return value;
}

inline int32_t asc_reduce_max(int32_t value) {
  return value;
}

inline uint32_t asc_reduce_max(uint32_t value) {
  return value;
}

inline float asc_reduce_min(float value) {
  return value;
}

inline int32_t asc_reduce_min(int32_t value) {
  return value;
}

inline uint32_t asc_reduce_min(uint32_t value) {
  return value;
}

template<typename T>
inline T asc_shfl(T value, int, int) {
  return value;
}

template<typename T>
inline T asc_shfl_up(T value, unsigned int, int) {
  return value;
}

template<typename T>
inline T asc_shfl_down(T value, unsigned int, int) {
  return value;
}

template<typename T>
inline T asc_shfl_xor(T value, int, int) {
  return value;
}

#endif
