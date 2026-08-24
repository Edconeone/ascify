#ifndef ASCIFY950_ROWWISE_VECTOR_CORE_QUERY_V1_HPP_
#define ASCIFY950_ROWWISE_VECTOR_CORE_QUERY_V1_HPP_

#include <acl/acl.h>

#include <cstdint>
#include <limits>

namespace ascify950_rowwise_runtime_v1 {

inline aclError QueryVectorCoreCount(int* vectorCoreCount) {
  if (vectorCoreCount == nullptr) {
    return ACL_ERROR_RT_PARAM_INVALID;
  }

  int32_t device = 0;
  aclError status = aclrtGetDevice(&device);
  if (status != ACL_SUCCESS) {
    return status;
  }
  if (device < 0) {
    return ACL_ERROR_RT_PARAM_INVALID;
  }

  int64_t count = 0;
  status = aclrtGetDeviceInfo(
      static_cast<uint32_t>(device), ACL_DEV_ATTR_VECTOR_CORE_NUM, &count);
  if (status != ACL_SUCCESS) {
    return status;
  }

  int64_t maxThreadsPerCore = 0;
  status = aclrtGetDeviceInfo(
      static_cast<uint32_t>(device),
      ACL_DEV_ATTR_MAX_THREAD_PER_VECTOR_CORE, &maxThreadsPerCore);
  if (status != ACL_SUCCESS) {
    return status;
  }

  if (count <= 0 || count > 1024 || maxThreadsPerCore <= 0 ||
      maxThreadsPerCore > std::numeric_limits<int>::max()) {
    return ACL_ERROR_RT_PARAM_INVALID;
  }

  *vectorCoreCount = static_cast<int>(count);
  return ACL_SUCCESS;
}

}  // namespace ascify950_rowwise_runtime_v1

#endif  // ASCIFY950_ROWWISE_VECTOR_CORE_QUERY_V1_HPP_
