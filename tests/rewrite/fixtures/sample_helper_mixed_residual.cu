#include <helper_cuda.h>

void mixedResidual(int argc, char **argv, void **pointer) {
  checkCudaErrors(cudaMalloc(pointer, 64));
  findCudaDevice(argc, (const char **)argv);
}
