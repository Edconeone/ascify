#define __aicore__
#define ASCIFY_TEST_ACL_CONTROLLABLE_RUNTIME_MEMORY
#define ASCIFY_TEST_ACL_CONSUMING_LAST_ERROR

#include <ascify/ascify_cuda_compat.hpp>

#include <cassert>
#include <cstdlib>
#include <cstring>

aclError ascify_test_runtime_status = ACL_SUCCESS;
void* ascify_test_allocation_result = nullptr;
const char* ascify_test_recent_error_message = "recorded ACL failure";
aclError ascify_test_last_error = ACL_SUCCESS;
int ascify_test_get_last_error_calls = 0;
int ascify_test_malloc_calls = 0;
int ascify_test_malloc_host_calls = 0;
int ascify_test_free_calls = 0;
int ascify_test_free_host_calls = 0;
int ascify_test_memcpy_calls = 0;
int ascify_test_memcpy_async_calls = 0;
int ascify_test_memset_calls = 0;
int ascify_test_memset_async_calls = 0;
void** ascify_test_output_pointer = nullptr;
void* ascify_test_destination = nullptr;
const void* ascify_test_source = nullptr;
size_t ascify_test_destination_max = 0;
size_t ascify_test_count = 0;
int ascify_test_value = 0;
aclrtMemcpyKind ascify_test_memcpy_kind = ACL_MEMCPY_HOST_TO_HOST;
aclrtStream ascify_test_stream = nullptr;
aclrtMemMallocPolicy ascify_test_malloc_policy = ACL_MEM_MALLOC_HUGE_FIRST;

namespace {

int status_calls = 0;

aclError statusOnce(aclError status) {
  ++status_calls;
  return status;
}

void assertSingleEvaluationAtExit() {
  assert(status_calls == 1);
}

void assertLastErrorConsumedAtExit() {
  assert(ascify_test_get_last_error_calls == 1);
  assert(ascify_test_last_error == ACL_SUCCESS);
}

}  // namespace

int main(int argc, char** argv) {
  assert(argc == 2);
  if (std::strcmp(argv[1], "check-success") == 0) {
    ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS(statusOnce(ACL_SUCCESS));
    assert(status_calls == 1);
    return 0;
  }
  if (std::strcmp(argv[1], "check-failure") == 0) {
    std::atexit(assertSingleEvaluationAtExit);
    ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS(
        statusOnce(ACL_ERROR_RT_PARAM_INVALID));
    return 90;
  }
  if (std::strcmp(argv[1], "get-success-twice") == 0) {
    ascify_test_last_error = ACL_SUCCESS;
    ASCIFY_NVIDIA_SAMPLE_GET_LAST_CUDA_ERROR("first");
    ASCIFY_NVIDIA_SAMPLE_GET_LAST_CUDA_ERROR("second");
    assert(ascify_test_get_last_error_calls == 2);
    assert(ascify_test_last_error == ACL_SUCCESS);
    return 0;
  }
  if (std::strcmp(argv[1], "get-failure") == 0) {
    ascify_test_last_error = ACL_ERROR_RT_PARAM_INVALID;
    std::atexit(assertLastErrorConsumedAtExit);
    ASCIFY_NVIDIA_SAMPLE_GET_LAST_CUDA_ERROR("expected failure");
    return 91;
  }
  return 2;
}
