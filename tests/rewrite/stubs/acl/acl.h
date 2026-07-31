#ifndef ASCIFY_TEST_STUB_ACL_H
#define ASCIFY_TEST_STUB_ACL_H

#include <stdint.h>

typedef int aclError;

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

inline aclError aclrtGetDevice(int32_t* device) {
  *device = 0;
  return ACL_SUCCESS;
}

inline aclError aclrtGetDeviceInfo(uint32_t, aclrtDevAttr, int64_t* value) {
  *value = 1024;
  return ACL_SUCCESS;
}

inline aclError aclrtPeekAtLastError(aclrtLastErrLevel) {
  return ACL_SUCCESS;
}

#endif
