#include <helper_functions.h>
#undef MAX
#define MAX(a, b) ((a < b) ? a : b)
#include <helper_cuda.h>

int rejectRedefinedMaxFindDevice(int argc, const char** argv) {
  return findCudaDevice(argc, argv);
}
