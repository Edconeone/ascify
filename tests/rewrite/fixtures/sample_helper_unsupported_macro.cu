#include <helper_cuda.h>

int unsupported_sample_macro(int value) {
  return NVIDIA_SAMPLE_HELPER_ONLY_MACRO(value);
}
