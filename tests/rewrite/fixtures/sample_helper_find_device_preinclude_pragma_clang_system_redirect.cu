#include "sample_helper_pragma_clang_system_redirect.h"
#include <helper_cuda.h>
#undef STRNCASECMP

int rejectClangSystemHeaderPragmaRedirect(
    int argc, const char** argv) {
  return findCudaDevice(argc, argv);
}
