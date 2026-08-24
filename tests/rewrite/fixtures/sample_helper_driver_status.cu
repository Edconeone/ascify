#include <cuda.h>
#include <helper_cuda.h>

void reject_driver_status() {
  checkCudaErrors(cuInit(0));
}
