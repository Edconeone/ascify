#include <cooperative_groups/scan.h>

__global__ void UnknownCooperativeGroupsScan(float* output) {
  output[threadIdx.x] = 1.0f;
}
