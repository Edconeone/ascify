#define __aicore__

#include <ascify/ascify_cuda_compat.hpp>

void SyntaxKernel(float*) {}

int main() {
  ASCIFY_ALIGN(16) unsigned char shared[16] = {};
  const float shuffled =
      ascify::shfl_xor_sync(0xffffffffu, ascify::fdividef(4.0f, 2.0f), 1, 32);
  const float reduced_float = ascify::warp_reduce_add(shuffled);
  const int32_t reduced_int = ascify::warp_reduce_add(int32_t{2});
  const uint32_t reduced_uint = ascify::warp_reduce_add(uint32_t{3});
  ascify::syncwarp();
  int value = 0;
  if (ascify::cudaDeviceGetAttribute(
          &value, ACL_DEV_ATTR_VECTOR_CORE_NUM, 0) != ACL_SUCCESS) {
    return 1;
  }
  if (ascify::cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &value, SyntaxKernel, 128, 0) != ACL_SUCCESS) {
    return 2;
  }
  ascify::cudaFuncAttributes attributes{};
  if (ascify::cudaFuncGetAttributes(
          &attributes, SyntaxKernel) != ACL_SUCCESS) {
    return 3;
  }
  if (ascify::cudaFuncSetAttribute(
          SyntaxKernel, ascify::cudaFuncAttributeMaxDynamicSharedMemorySize,
          64 * 1024) != ACL_SUCCESS) {
    return 4;
  }
  if (ascify::cudaPeekAtLastError() != ACL_SUCCESS) { return 5; }
  return reduced_float == 2.0f && reduced_int == 2 && reduced_uint == 3
             ? shared[0]
             : 6;
}
