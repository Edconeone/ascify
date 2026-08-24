#include <helper_cuda.h>

int user_same_names() {
  checkCudaErrors(0);
  return getLastCudaError("user macro");
}
