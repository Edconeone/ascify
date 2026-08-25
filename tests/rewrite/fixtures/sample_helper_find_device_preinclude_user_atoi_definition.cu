#include <stdlib.h>

inline int ascifyTestUserAtoiTelemetry = 0;

#if defined(__GLIBC__)
#define ASCIFY_TEST_ATOI_NOEXCEPT noexcept
#else
#define ASCIFY_TEST_ATOI_NOEXCEPT
#endif

extern "C" int atoi(const char* text) ASCIFY_TEST_ATOI_NOEXCEPT {
  ++ascifyTestUserAtoiTelemetry;
  int value = 0;
  while (*text >= '0' && *text <= '9') {
    value = value * 10 + (*text - '0');
    ++text;
  }
  return value;
}

#undef ASCIFY_TEST_ATOI_NOEXCEPT

#include <helper_cuda.h>

int rejectUserAtoiDefinition(int argc, const char** argv) {
  return findCudaDevice(argc, argv);
}
