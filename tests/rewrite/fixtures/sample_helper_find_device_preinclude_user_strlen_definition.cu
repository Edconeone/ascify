#include <stddef.h>
#include <string.h>

inline int ascifyTestUserStrlenTelemetry = 0;

#if defined(__GLIBC__)
#define ASCIFY_TEST_STRLEN_NOEXCEPT noexcept
#else
#define ASCIFY_TEST_STRLEN_NOEXCEPT
#endif

extern "C" size_t strlen(const char* text) ASCIFY_TEST_STRLEN_NOEXCEPT {
  ++ascifyTestUserStrlenTelemetry;
  size_t length = 0;
  while (text[length] != '\0')
    ++length;
  return length;
}

#undef ASCIFY_TEST_STRLEN_NOEXCEPT

#include <helper_cuda.h>

int rejectUserStrlenDefinition(int argc, const char** argv) {
  return findCudaDevice(argc, argv);
}
