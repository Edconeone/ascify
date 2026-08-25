#include <cublas_v2.h>
static const char* _cudaGetErrorEnum(cublasStatus_t) {
  return "library status";
}

#include <helper_cuda.h>

void reject_library_status(cublasHandle_t* handle) {
  checkCudaErrors(cublasCreate(handle));
}
