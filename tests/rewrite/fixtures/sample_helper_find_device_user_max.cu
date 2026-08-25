#include "sample_helper_max_user.h"
#include <helper_functions.h>
#include <helper_cuda.h>

int rejectUserMaxFindDevice(int argc, const char** argv) {
  return findCudaDevice(argc, argv);
}
