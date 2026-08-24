#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void CooperativeBlock(float* output) {
  const cg::thread_block block = cg::this_thread_block();
  output[threadIdx.x] = static_cast<float>(threadIdx.x);
  cg::sync(block);
}
