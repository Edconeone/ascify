#include "./cooperative_groups.h"

__global__ void LocalAliasSpelling(float* output) {
  output[threadIdx.x] = 1.0f;
}
