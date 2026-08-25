#include "sample_helper_pragma_gcc_system_redirect.h"
#include <helper_cuda.h>
#undef STRNCASECMP

int rejectGccSystemHeaderPragmaRedirect(
    int argc, const char** argv) {
  return findCudaDevice(argc, argv);
}
