#include <helper_string.h>
#include <helper_functions.h>
#include <helper_cuda.h>

int rejectMutatedImageMaxFindDevice(int argc, const char** argv) {
  return findCudaDevice(argc, argv);
}
