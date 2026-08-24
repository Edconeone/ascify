#include <helper_cuda.h>
#include "nvidia_samples/Common/helper_cuda.h"

void mixedDirectAndRelativeInclude(void **pointer) {
  checkCudaErrors(cudaMalloc(pointer, 64));
}
