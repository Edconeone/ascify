#include <helper_cuda.h>

#define ASCIFY_TEST_FIND_DEVICE(argc, argv) findCudaDevice((argc), (argv))

int rejectMacroFindDevice(int argc, const char** argv) {
  return ASCIFY_TEST_FIND_DEVICE(argc, argv);
}
