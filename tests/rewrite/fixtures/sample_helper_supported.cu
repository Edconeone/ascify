#include <helper_cuda.h>

void admitted_sample_helpers(void** pointer) {
  checkCudaErrors(cudaMalloc(pointer, 16));
  getLastCudaError("kernel launch");
}
