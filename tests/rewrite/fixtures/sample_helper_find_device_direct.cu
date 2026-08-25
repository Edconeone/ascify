#include <helper_cuda.h>

int acceptDirectFindDevice(int argc, const char** argv) {
  return findCudaDevice(argc, argv);
}
