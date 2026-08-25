#ifndef ASCIFY_TEST_SAMPLE_HELPER_PRAGMA_RUNTIME_STATUS_REDIRECT_H_
#define ASCIFY_TEST_SAMPLE_HELPER_PRAGMA_RUNTIME_STATUS_REDIRECT_H_

#pragma clang system_header

inline int ascifyTestRuntimeStatusTelemetry = 0;
inline cudaError_t cudaSetDevice(long device) {
  ++ascifyTestRuntimeStatusTelemetry;
  return static_cast<cudaError_t>(device);
}

#endif
