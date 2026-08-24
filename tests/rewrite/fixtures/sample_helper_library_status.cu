#include <cublas_v2.h>
#include <helper_cuda.h>

void reject_library_status(cublasHandle_t* handle) {
  checkCudaErrors(cublasCreate(handle));
}
