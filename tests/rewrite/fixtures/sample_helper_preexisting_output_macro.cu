#define ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS(expression) \
  ascifyUserCheck(expression)
#define ASCIFY_NVIDIA_SAMPLE_GET_LAST_CUDA_ERROR(message) \
  ascifyUserLastError(message)

#include <helper_cuda.h>

void preexistingOutputMacros(void **pointer) {
  checkCudaErrors(cudaMalloc(pointer, 64));
  getLastCudaError("user output macro collision");
}
