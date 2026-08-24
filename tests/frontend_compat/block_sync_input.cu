#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void BlockSync(float* output) {
  const cg::thread_block block = cg::this_thread_block();
  output[threadIdx.x] = static_cast<float>(threadIdx.x);
  block.sync();
  cg::sync(block);
}
