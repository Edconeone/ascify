#include "sample_helper_user_output_macros.h"
#include <helper_cuda.h>

void userHeaderOutputMacros(void **pointer) {
  checkCudaErrors(cudaMalloc(pointer, 64));
  getLastCudaError("user header output macro collision");
}
