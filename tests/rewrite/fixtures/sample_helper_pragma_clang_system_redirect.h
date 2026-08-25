#ifndef ASCIFY_TEST_SAMPLE_HELPER_PRAGMA_CLANG_SYSTEM_REDIRECT_H_
#define ASCIFY_TEST_SAMPLE_HELPER_PRAGMA_CLANG_SYSTEM_REDIRECT_H_

#include <strings.h>

#pragma clang system_header

inline int ascifyTestPragmaClangTelemetry = 0;
inline int ascifyTestPragmaClangStrncasecmp(
    const char* lhs, const char* rhs, unsigned long count) {
  ++ascifyTestPragmaClangTelemetry;
  return strncasecmp(lhs, rhs, count);
}

#define STRNCASECMP ascifyTestPragmaClangStrncasecmp

#endif
