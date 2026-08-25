#ifndef ASCIFY_TEST_SAMPLE_HELPER_PRAGMA_GCC_SYSTEM_REDIRECT_H_
#define ASCIFY_TEST_SAMPLE_HELPER_PRAGMA_GCC_SYSTEM_REDIRECT_H_

#include <strings.h>

#pragma GCC system_header

inline int ascifyTestPragmaGccTelemetry = 0;
inline int ascifyTestPragmaGccStrncasecmp(
    const char* lhs, const char* rhs, unsigned long count) {
  ++ascifyTestPragmaGccTelemetry;
  return strncasecmp(lhs, rhs, count);
}

#define STRNCASECMP ascifyTestPragmaGccStrncasecmp

#endif
