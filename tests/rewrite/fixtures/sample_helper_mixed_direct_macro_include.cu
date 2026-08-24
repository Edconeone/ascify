#include <helper_cuda.h>
#define ASCIFY_TEST_SECOND_HELPER_HEADER <helper_cuda.h>
#include ASCIFY_TEST_SECOND_HELPER_HEADER

void mixedDirectAndMacroInclude(void **pointer) {
  checkCudaErrors(cudaMalloc(pointer, 64));
}
