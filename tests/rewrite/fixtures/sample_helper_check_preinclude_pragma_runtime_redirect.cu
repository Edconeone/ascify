#include "sample_helper_pragma_runtime_status_redirect.h"
#include <helper_cuda.h>

void rejectPragmaRuntimeStatusRedirect() {
  checkCudaErrors(cudaSetDevice(0L));
}
