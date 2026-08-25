#include <helper_cuda.h>

int unsupported_sample_macro(int value) {
  return MAX(value, 1);
}
