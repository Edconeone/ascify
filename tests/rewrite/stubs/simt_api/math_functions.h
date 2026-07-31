#ifndef ASCIFY_TEST_STUB_MATH_FUNCTIONS_H
#define ASCIFY_TEST_STUB_MATH_FUNCTIONS_H

float expf(float);
float logf(float);
float log2f(float);
float log10f(float);
float sqrtf(float);
float rsqrtf(float);
inline float fdividef(float numerator, float denominator) {
  return numerator / denominator;
}

#endif
