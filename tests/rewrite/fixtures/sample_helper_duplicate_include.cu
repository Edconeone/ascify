#include <helper_cuda.h>
#include <helper_cuda.h>

void duplicate_helper_include(void** pointer) {
  checkCudaErrors(cudaMalloc(pointer, 16));
}
