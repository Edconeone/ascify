#define ASCIFY_TEST_ACL_CONTROLLABLE_DEVICE_QUERY
#include "vector_core_query_v1.hpp"

#include <cassert>

aclError ascify_test_get_device_status = ACL_SUCCESS;
aclError ascify_test_vector_core_info_status = ACL_SUCCESS;
aclError ascify_test_max_thread_info_status = ACL_SUCCESS;
int32_t ascify_test_device = 0;
int64_t ascify_test_vector_core_count = 56;
int64_t ascify_test_max_threads_per_core = 256;
int ascify_test_get_device_calls = 0;
int ascify_test_get_device_info_calls = 0;

namespace query = ascify950_rowwise_runtime_v1;

int main() {
  int count = -1;

  ascify_test_get_device_status = 73;
  assert(query::QueryVectorCoreCount(&count) == 73);
  assert(count == -1);
  assert(ascify_test_get_device_calls == 1);
  assert(ascify_test_get_device_info_calls == 0);

  ascify_test_get_device_status = ACL_SUCCESS;
  ascify_test_device = -1;
  assert(query::QueryVectorCoreCount(&count) == ACL_ERROR_RT_PARAM_INVALID);
  assert(count == -1);
  assert(ascify_test_get_device_calls == 2);
  assert(ascify_test_get_device_info_calls == 0);

  ascify_test_device = 3;
  ascify_test_vector_core_info_status = 74;
  assert(query::QueryVectorCoreCount(&count) == 74);
  assert(count == -1);
  assert(ascify_test_get_device_calls == 3);
  assert(ascify_test_get_device_info_calls == 1);

  ascify_test_vector_core_info_status = ACL_SUCCESS;
  ascify_test_max_thread_info_status = 75;
  ascify_test_device = 3;
  assert(query::QueryVectorCoreCount(&count) == 75);
  assert(count == -1);
  assert(ascify_test_get_device_calls == 4);
  assert(ascify_test_get_device_info_calls == 3);

  ascify_test_max_thread_info_status = ACL_SUCCESS;
  ascify_test_vector_core_count = 0;
  assert(query::QueryVectorCoreCount(&count) == ACL_ERROR_RT_PARAM_INVALID);
  assert(count == -1);
  assert(ascify_test_get_device_info_calls == 5);

  ascify_test_vector_core_count = 48;
  ascify_test_max_threads_per_core = 0;
  assert(query::QueryVectorCoreCount(&count) == ACL_ERROR_RT_PARAM_INVALID);
  assert(count == -1);
  assert(ascify_test_get_device_info_calls == 7);

  ascify_test_max_threads_per_core = 256;
  assert(query::QueryVectorCoreCount(&count) == ACL_SUCCESS);
  assert(count == 48);

  // A second successful query observes the current device value rather than
  // a process-global fallback/cache from an earlier device.
  ascify_test_device = 7;
  ascify_test_vector_core_count = 64;
  assert(query::QueryVectorCoreCount(&count) == ACL_SUCCESS);
  assert(count == 64);
  assert(ascify_test_get_device_calls == 8);
  assert(ascify_test_get_device_info_calls == 11);

  assert(query::QueryVectorCoreCount(nullptr) == ACL_ERROR_RT_PARAM_INVALID);
  assert(ascify_test_get_device_calls == 8);
  assert(ascify_test_get_device_info_calls == 11);
  return 0;
}
