#include <helper_cuda.h>

void reject_altered_active_macros() {
  checkCudaErrors(0);
  getLastCudaError("altered");
}
