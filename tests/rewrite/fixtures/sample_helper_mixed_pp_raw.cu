#include <helper_cuda.h>

void mixedRawPreprocessor(void **pointer) {
  checkCudaErrors(cudaMalloc(pointer, 64));
}

#ifdef getLastCudaError
int helper_last_error_macro_is_defined = 1;
#endif

#define ASCIFY_TEST_STRINGIZE_INNER(token) #token
#define ASCIFY_TEST_STRINGIZE(token) ASCIFY_TEST_STRINGIZE_INNER(token)
const char *mixed_raw_helper_name = ASCIFY_TEST_STRINGIZE(checkCudaErrors);
