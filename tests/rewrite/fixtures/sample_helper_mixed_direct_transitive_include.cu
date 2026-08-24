#include "sample_helper_wrapper.h"
#include <helper_cuda.h>

void mixedDirectAndTransitiveInclude(void **pointer) {
  checkCudaErrors(cudaMalloc(pointer, 64));
}
