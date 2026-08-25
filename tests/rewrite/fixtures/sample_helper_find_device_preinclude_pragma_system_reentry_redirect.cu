#include <strings.h>

inline int ascifyTestPragmaReentryTelemetry = 0;
inline int ascifyTestPragmaReentryStrncasecmp(
    const char* lhs, const char* rhs, unsigned long count) {
  ++ascifyTestPragmaReentryTelemetry;
  return strncasecmp(lhs, rhs, count);
}

#include "sample_helper_pragma_system_reentry_redirect.h"
#include "sample_helper_pragma_system_reentry_redirect.h"
#include <helper_cuda.h>
#undef STRNCASECMP

int rejectSystemHeaderPragmaReentryRedirect(
    int argc, const char** argv) {
  return findCudaDevice(argc, argv);
}
