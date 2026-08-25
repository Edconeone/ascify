#include <helper_cuda.h>

int rejectAlteredFindDeviceDefinition(int argc, const char** argv) {
  return findCudaDevice(argc, argv);
}
