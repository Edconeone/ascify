#include <strings.h>

inline int ascifyTestStrncasecmpTelemetry = 0;
inline int ascifyTestRedirectStrncasecmp(
    const char* lhs, const char* rhs, unsigned long count) {
  ++ascifyTestStrncasecmpTelemetry;
  return strncasecmp(lhs, rhs, count);
}

#define STRNCASECMP ascifyTestRedirectStrncasecmp
#include <helper_cuda.h>
#undef STRNCASECMP

int rejectPreincludeTransitiveHelperRedirect(int argc, const char** argv) {
  return findCudaDevice(argc, argv);
}
