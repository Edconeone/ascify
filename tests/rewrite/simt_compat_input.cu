#include <cuda.h>
#include <cuda_bf16.h>
#include <type_traits>

template<typename T>
__inline__ __device__ T FullMaskShuffle(T value) {
  return __shfl_xor_sync(0xffffffffu, value, 1, 32);
}

__global__ void SimtCompatKernel(float* output) {
  extern __shared__ __align__(16) unsigned char shared[];
  __syncwarp();
  float value = output[threadIdx.x];
  value = exp(value) + __expf(value);
  value = log(value) + __logf(value);
  value = sqrt(value) + __fsqrt_rn(value) + __frsqrt_rn(value);
  value = __fdividef(value, 2.0f);
  output[threadIdx.x] = FullMaskShuffle(value) + shared[0];
}

__global__ void NarrowScalarDoubleParams(float* output, const double eps, double scale,
                                         const double* pointer, double values[]) {
  if (pointer != nullptr && values != nullptr) { output[0] = eps + scale; }
}

__device__ double KeepDeviceDoubleParam(double value) { return value; }

cudaError_t QueryCompatApis(int* value) {
  int device = 0;
  cudaError_t status =
      cudaDeviceGetAttribute(value, cudaDevAttrMultiProcessorCount, device);
  if (status != cudaSuccess) { return status; }
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      value, SimtCompatKernel, 128, 0);
  if (status != cudaSuccess) { return status; }
  cudaFuncAttributes attributes{};
  status = cudaFuncGetAttributes(&attributes, SimtCompatKernel);
  if (status != cudaSuccess) { return status; }
  status = cudaFuncSetAttribute(
      SimtCompatKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024);
  if (status != cudaSuccess) { return status; }
  status = cudaDeviceGetAttribute(
      value, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  if (status != cudaSuccess) { return status; }
  return cudaPeekAtLastError();
}

double KeepHostDoublePrecision(double value) { return value + 1.0; }

static_assert(std::is_same<double, double>::value, "double traits must be preserved");
