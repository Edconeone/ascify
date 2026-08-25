#include <helper_cuda.h>

cudaError_t customStatus();

void mixedCustomStatus(void **pointer) {
  checkCudaErrors(cudaMalloc(pointer, 64));
  checkCudaErrors(customStatus());
}
