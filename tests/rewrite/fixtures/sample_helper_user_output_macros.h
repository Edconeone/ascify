#ifndef ASCIFY_TEST_USER_OUTPUT_MACROS_H
#define ASCIFY_TEST_USER_OUTPUT_MACROS_H

#define ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS(expression) \
  ascifyHeaderCheck(expression)
#define ASCIFY_NVIDIA_SAMPLE_GET_LAST_CUDA_ERROR(message) \
  ascifyHeaderLastError(message)

#endif
