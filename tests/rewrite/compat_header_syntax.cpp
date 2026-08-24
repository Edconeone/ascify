#define __aicore__
#define __global__

#include <ascify/ascify_cuda_compat.hpp>

ASCIFY_GLOBAL void SyntaxKernel(float*) {}

int main() {
  if (static_cast<unsigned int>(ascify::cudaDevAttrWarpSize) != 202U ||
      static_cast<unsigned int>(
          ascify::cudaDevAttrMaxThreadsPerVectorCore) != 203U ||
      static_cast<unsigned int>(
          ascify::cudaDevAttrLocalMemoryPerVectorCore) != 204U ||
      static_cast<unsigned int>(
          ascify::cudaDevAttrMaxThreadsPerBlock) != 209U) {
    return 1;
  }
  // Real CUDA samples often obtain these C declarations transitively from
  // cuda_runtime.h. The injected compatibility header must remain sufficient.
  const int random_value = rand();
  const double absolute_value = fabs(-1.0);
  if (RAND_MAX < 0 || EXIT_FAILURE == EXIT_SUCCESS ||
      random_value < 0 || absolute_value != 1.0) {
    return 2;
  }
  ASCIFY_ALIGN(16) unsigned char shared[16] = {};
  void* device = nullptr;
  void* host = nullptr;
  if (ascify::cudaMalloc(&device, sizeof(shared)) != ACL_SUCCESS) { return 1; }
  if (ascify::cudaMallocHost(&host, sizeof(shared)) != ACL_SUCCESS) { return 2; }
  if (ascify::cudaMemcpy(device, host, sizeof(shared),
                         ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
    return 3;
  }
  if (ascify::cudaMemcpyAsync(host, device, sizeof(shared),
                              ACL_MEMCPY_DEVICE_TO_HOST,
                              nullptr) != ACL_SUCCESS) {
    return 4;
  }
  if (ascify::cudaMemset(device, 0, sizeof(shared)) != ACL_SUCCESS) { return 5; }
  if (ascify::cudaMemsetAsync(device, 0, sizeof(shared),
                              nullptr) != ACL_SUCCESS) {
    return 6;
  }
  if (ascify::cudaGetLastError() != ACL_SUCCESS) { return 7; }
  if (ascify::cudaGetErrorString(ACL_SUCCESS) == nullptr) { return 8; }
  if (ascify::cudaFree(device) != ACL_SUCCESS) { return 9; }
  if (ascify::cudaFreeHost(host) != ACL_SUCCESS) { return 10; }
  const float shuffled =
      ascify::shfl_xor_sync(0xffffffffu, ascify::fdividef(4.0f, 2.0f), 1, 32);
  const float reduced_float = ascify::warp_reduce_add(shuffled);
  const int32_t reduced_int = ascify::warp_reduce_add(int32_t{2});
  const uint32_t reduced_uint = ascify::warp_reduce_add(uint32_t{3});
  const float maximum = ascify::warp_reduce_max(reduced_float);
  const float minimum = ascify::warp_reduce_min(reduced_float);
  ascify::syncwarp();
  ascify::syncthreads();
  ascify::threadfence_block();
  ascify::threadfence();
  int value = 0;
  if (ascify::cudaDeviceGetAttribute(
          &value, ACL_DEV_ATTR_VECTOR_CORE_NUM, 0) != ACL_SUCCESS) {
    return 11;
  }
  if (ascify::cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &value, SyntaxKernel, 128, 0) != ACL_SUCCESS) {
    return 12;
  }
  ascify::cudaFuncAttributes attributes{};
  if (ascify::cudaFuncGetAttributes(
          &attributes, SyntaxKernel) != ACL_SUCCESS) {
    return 13;
  }
  if (ascify::cudaFuncSetAttribute(
          SyntaxKernel, ascify::cudaFuncAttributeMaxDynamicSharedMemorySize,
          64 * 1024) != ACL_SUCCESS) {
    return 14;
  }
  if (ascify::cudaPeekAtLastError() != ACL_SUCCESS) { return 15; }
  return reduced_float == 2.0f && reduced_int == 2 && reduced_uint == 3 &&
                 maximum == 2.0f && minimum == 2.0f
             ? shared[0]
             : 16;
}
