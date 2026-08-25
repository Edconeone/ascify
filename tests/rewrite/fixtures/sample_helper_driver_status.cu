#include <cuda.h>
static const char* _cudaGetErrorEnum(CUresult) { return "driver status"; }

#include <helper_cuda.h>

void reject_driver_status() {
  checkCudaErrors(cuInit(0));
}
