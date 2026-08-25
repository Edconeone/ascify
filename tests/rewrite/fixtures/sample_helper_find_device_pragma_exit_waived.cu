#include "sample_helper_exit_waived_pragma.h"
#include <helper_functions.h>
#include <helper_cuda.h>

int rejectPragmaExitWaivedFindDevice(int argc, const char** argv) {
  return findCudaDevice(argc, argv);
}
