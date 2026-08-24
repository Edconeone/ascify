#ifndef ASCIFY_TEST_STUB_ACL_H
#define ASCIFY_TEST_STUB_ACL_H

#include <stdint.h>

typedef int aclError;
typedef void* aclrtStream;

enum aclrtDevAttr {
  ACL_DEV_ATTR_VECTOR_CORE_NUM,
  ACL_DEV_ATTR_MAX_THREAD_PER_VECTOR_CORE,
  ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE
};

enum aclrtLastErrLevel {
  ACL_RT_THREAD_LEVEL
};

static const aclError ACL_SUCCESS = 0;
static const aclError ACL_ERROR_RT_PARAM_INVALID = 1;

#ifdef ASCIFY_TEST_ACL_CONTROLLABLE_DEVICE_QUERY
extern aclError ascify_test_get_device_status;
extern aclError ascify_test_vector_core_info_status;
extern aclError ascify_test_max_thread_info_status;
extern int32_t ascify_test_device;
extern int64_t ascify_test_vector_core_count;
extern int64_t ascify_test_max_threads_per_core;
extern int ascify_test_get_device_calls;
extern int ascify_test_get_device_info_calls;

inline aclError aclrtGetDevice(int32_t* device) {
  ++ascify_test_get_device_calls;
  if (ascify_test_get_device_status == ACL_SUCCESS) {
    *device = ascify_test_device;
  }
  return ascify_test_get_device_status;
}

inline aclError aclrtGetDeviceInfo(
    uint32_t device, aclrtDevAttr attribute, int64_t* value) {
  (void)device;
  ++ascify_test_get_device_info_calls;
  if (attribute == ACL_DEV_ATTR_VECTOR_CORE_NUM) {
    if (ascify_test_vector_core_info_status == ACL_SUCCESS) {
      *value = ascify_test_vector_core_count;
    }
    return ascify_test_vector_core_info_status;
  }
  if (attribute == ACL_DEV_ATTR_MAX_THREAD_PER_VECTOR_CORE) {
    if (ascify_test_max_thread_info_status == ACL_SUCCESS) {
      *value = ascify_test_max_threads_per_core;
    }
    return ascify_test_max_thread_info_status;
  }
  return ACL_ERROR_RT_PARAM_INVALID;
}
#else
inline aclError aclrtGetDevice(int32_t* device) {
  *device = 0;
  return ACL_SUCCESS;
}

inline aclError aclrtGetDeviceInfo(uint32_t, aclrtDevAttr, int64_t* value) {
  *value = 1024;
  return ACL_SUCCESS;
}
#endif

inline aclError aclrtPeekAtLastError(aclrtLastErrLevel) {
  return ACL_SUCCESS;
}

#endif
