#include <helper_cuda.h>

#define ASCIFY_TEST_FORWARD_SINK(token) #token
#define ASCIFY_TEST_FORWARD(token) ASCIFY_TEST_FORWARD_SINK(token)
const char* forwarded_helper_name = ASCIFY_TEST_FORWARD(getLastCudaError);
