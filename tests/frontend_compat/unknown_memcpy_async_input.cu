#include <cooperative_groups/memcpy_async.h>

__global__ void UnknownCooperativeGroupsSurface(float* output) {
  output[threadIdx.x] = 1.0f;
}
