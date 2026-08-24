#include <helper_cuda.h>

int customStatus();

void mixedCustomStatus(void **pointer) {
  checkCudaErrors(cudaMalloc(pointer, 64));
  checkCudaErrors(customStatus());
}
