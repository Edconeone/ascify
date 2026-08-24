#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

__global__ void UnsupportedReduce(int* output) {
  output[threadIdx.x] = threadIdx.x;
}
