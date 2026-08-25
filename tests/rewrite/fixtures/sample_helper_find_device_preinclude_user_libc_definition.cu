#include <stddef.h>
#include <stdlib.h>
#include <string.h>

inline int ascifyTestUserLibcTelemetry = 0;

#if defined(__GLIBC__)
#define ASCIFY_TEST_LIBC_NOEXCEPT noexcept
#else
#define ASCIFY_TEST_LIBC_NOEXCEPT
#endif

extern "C" size_t strlen(const char* text) ASCIFY_TEST_LIBC_NOEXCEPT {
  ++ascifyTestUserLibcTelemetry;
  size_t length = 0;
  while (text[length] != '\0')
    ++length;
  return length;
}

extern "C" int atoi(const char* text) ASCIFY_TEST_LIBC_NOEXCEPT {
  ++ascifyTestUserLibcTelemetry;
  int value = 0;
  while (*text >= '0' && *text <= '9') {
    value = value * 10 + (*text - '0');
    ++text;
  }
  return value;
}

#undef ASCIFY_TEST_LIBC_NOEXCEPT

#include <helper_cuda.h>

int rejectUserLibcDefinitions(int argc, const char** argv) {
  return findCudaDevice(argc, argv);
}
