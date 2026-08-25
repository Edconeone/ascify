#include <helper_functions.h>
#include <helper_cuda.h>

int acceptOfficialExitWaivedFindDevice(int argc, const char** argv) {
  return findCudaDevice(argc, argv);
}
