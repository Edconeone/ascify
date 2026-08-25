#include <helper_cuda.h>

int rejectQualifiedFindDevice(int argc, const char** argv) {
  return ::findCudaDevice(argc, argv);
}
