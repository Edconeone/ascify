#ifndef ASCIFY_TEST_CUBLAS_V2_H_
#define ASCIFY_TEST_CUBLAS_V2_H_

typedef enum cublasStatus_t {
  CUBLAS_STATUS_SUCCESS = 0,
  CUBLAS_STATUS_INTERNAL_ERROR = 14,
} cublasStatus_t;

struct ascify_test_cublas_context;
typedef ascify_test_cublas_context* cublasHandle_t;

inline cublasStatus_t cublasCreate(cublasHandle_t*) {
  return CUBLAS_STATUS_SUCCESS;
}

#endif
