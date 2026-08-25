#include <helper_cuda.h>

int rejectAliasedFindDevice(int argc, const char** argv) {
  int (&finder)(int, const char**) = findCudaDevice;
  return finder(argc, argv);
}
