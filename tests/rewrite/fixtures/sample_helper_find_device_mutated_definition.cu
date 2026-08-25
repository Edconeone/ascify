#include <helper_cuda.h>

int rejectMutatedFindDeviceDefinition(int argc, const char** argv) {
  return findCudaDevice(argc, argv);
}
