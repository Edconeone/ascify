#define ASCIFY_TEST_HELPER_HEADER <helper_cuda.h>
#include ASCIFY_TEST_HELPER_HEADER

void macroInclude(void **pointer) {
  checkCudaErrors(cudaMalloc(pointer, 64));
}
