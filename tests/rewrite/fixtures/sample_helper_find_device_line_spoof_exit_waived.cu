#include "sample_helper_exit_waived_line_spoof.h"
#include <helper_functions.h>
#include <helper_cuda.h>

int rejectLineSpoofExitWaivedFindDevice(int argc, const char** argv) {
  return findCudaDevice(argc, argv);
}
