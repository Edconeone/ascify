#include <strings.h>

inline int ascifyTestBuiltinSpoofTelemetry = 0;
inline int ascifyTestBuiltinSpoofStrncasecmp(
    const char* lhs, const char* rhs, unsigned long count) {
  ++ascifyTestBuiltinSpoofTelemetry;
  return strncasecmp(lhs, rhs, count);
}

#line 1 "<built-in>"
#define STRNCASECMP ascifyTestBuiltinSpoofStrncasecmp
#line 20 "sample_helper_find_device_preinclude_builtin_spoof.cu"
#include <helper_cuda.h>
#undef STRNCASECMP

int rejectBuiltinPresumedNameSpoof(int argc, const char** argv) {
  return findCudaDevice(argc, argv);
}
