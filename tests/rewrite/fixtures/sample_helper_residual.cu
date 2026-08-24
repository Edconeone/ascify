#include <helper_cuda.h>

int unsupported_sample_helper(int argc, const char** argv) {
  return findCudaDevice(argc, argv);
}
