static const char* _cudaGetErrorEnum(int) { return "custom status"; }

#include <helper_cuda.h>

int customStatus() { return 0; }

void reject_custom_integer_status() {
  checkCudaErrors(customStatus());
}
