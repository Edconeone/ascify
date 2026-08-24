#include <helper_cuda.h>

#define ASCIFY_TEST_STRINGIZE_INNER(token) #token
#define ASCIFY_TEST_STRINGIZE(token) ASCIFY_TEST_STRINGIZE_INNER(token)
const char* helper_macro_name = ASCIFY_TEST_STRINGIZE(checkCudaErrors);
