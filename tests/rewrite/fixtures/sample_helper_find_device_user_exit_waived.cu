#include "sample_helper_exit_waived_user.h"
#include <helper_functions.h>
#include <helper_cuda.h>

int rejectUserExitWaivedFindDevice(int argc, const char** argv) {
  return findCudaDevice(argc, argv);
}
