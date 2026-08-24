#include <helper_cuda.h>
#include "sample_helper_external_use.h"

int invokeSampleHelperExternalUse(int argc, char **argv, void **pointer) {
  sampleHelperExternalUse(pointer);
  return sampleHelperExternalDevice(argc, argv);
}
