#define ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS(expression) \
  ascifyUserCheck(expression)
#define ASCIFY_NVIDIA_SAMPLE_GET_LAST_CUDA_ERROR(message) \
  ascifyUserLastError(message)
#define sampleFindCudaDevice(argc, argv) ascifyUserDevice((argc), (argv))

#include <helper_cuda.h>

int preexistingOutputMacros(int argc, const char** argv, void **pointer) {
  checkCudaErrors(cudaMalloc(pointer, 64));
  getLastCudaError("user output macro collision");
  return findCudaDevice(argc, argv);
}
