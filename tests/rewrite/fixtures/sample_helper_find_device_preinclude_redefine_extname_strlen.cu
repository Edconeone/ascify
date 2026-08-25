#pragma redefine_extname strlen ascifyTestEvilStrlen
#include <helper_cuda.h>

inline int ascifyTestRedefineExtnameTelemetry = 0;
extern "C" size_t ascifyTestEvilStrlen(const char* text) {
  ++ascifyTestRedefineExtnameTelemetry;
  size_t length = 0;
  while (text[length] != '\0')
    ++length;
  return length;
}

int rejectRedefineExtnameStrlen(int argc, const char** argv) {
  return findCudaDevice(argc, argv);
}
