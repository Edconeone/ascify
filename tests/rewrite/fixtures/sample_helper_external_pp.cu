#include <helper_cuda.h>
#include "sample_helper_external_pp.h"

int invokeSampleHelperExternalPpUse() {
  return sampleHelperExternalPpUse() + sampleHelperExternalDefinedUse() +
         sampleHelperExternalUndefUse();
}
