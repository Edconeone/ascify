#ifndef ASCIFY_TEST_SAMPLE_HELPER_PRAGMA_RUNTIME_STATUS_REDIRECT_H_
#define ASCIFY_TEST_SAMPLE_HELPER_PRAGMA_RUNTIME_STATUS_REDIRECT_H_

#pragma clang system_header

inline int ascifyTestRuntimeStatusTelemetry = 0;
inline int cudaSetDevice(long device) {
  ++ascifyTestRuntimeStatusTelemetry;
  return static_cast<int>(device);
}

#endif
