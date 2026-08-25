#include <helper_cuda.h>

int user_same_names(int argc, const char** argv) {
  checkCudaErrors(0);
  getLastCudaError("user macro");
  return findCudaDevice(argc, argv);
}
