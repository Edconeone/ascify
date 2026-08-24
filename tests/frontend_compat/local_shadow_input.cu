#include "cooperative_groups.h"

namespace cg = cooperative_groups;

__global__ void LocalShadow(float* output) {
  const cg::thread_block block = cg::this_thread_block();
  output[threadIdx.x] = 1.0f;
  block.sync();
}
