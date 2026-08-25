#include "sample_helper_exit_waived_reentry.h"
#include "sample_helper_exit_waived_reentry.h"
#include <helper_functions.h>
#include <helper_cuda.h>

int rejectReentryExitWaivedFindDevice(int argc, const char** argv) {
  return findCudaDevice(argc, argv);
}
