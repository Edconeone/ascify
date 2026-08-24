#include <helper_cuda.h>

#define ASCIFY_TEST_CHECK_ALIAS checkCudaErrors

void aliasInvocation(void **pointer) {
  ASCIFY_TEST_CHECK_ALIAS(cudaMalloc(pointer, 64));
}
