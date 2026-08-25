#include <stdio.h>
#include <stdlib.h>

#define printf ::printf
#define exit ::exit
#include <helper_cuda.h>
#undef exit
#undef printf

int rejectPreincludeStdioRedirect(int argc, const char** argv) {
  return findCudaDevice(argc, argv);
}
