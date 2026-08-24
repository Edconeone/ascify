#ifndef ASCIFY_TEST_STUB_VECTOR_FUNCTIONS_H
#define ASCIFY_TEST_STUB_VECTOR_FUNCTIONS_H

struct float4 {
  float x;
  float y;
  float z;
  float w;
};

inline float4 make_float4(float x, float y, float z, float w) {
  return {x, y, z, w};
}

#endif
