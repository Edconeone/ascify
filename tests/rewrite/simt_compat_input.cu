#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdlib>
#include <stdlib.h>
#include <type_traits>

template<typename T>
__inline__ __device__ T FullMaskShuffle(T value) {
  return __shfl_xor_sync(0xffffffffu, value, 1, 32);
}

__global__ void SimtCompatKernel(float* output) {
  extern __shared__ __align__(16) unsigned char shared[];
  __syncwarp();
  __syncthreads();
  float value = output[threadIdx.x];
  value = exp(value) + __expf(value);
  value = log(value) + __logf(value);
  value = sqrt(value) + __fsqrt_rn(value) + __frsqrt_rn(value);
  value = __fdividef(value, 2.0f);
  __threadfence_block();
  __threadfence();
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
  status = cudaDeviceGetAttribute(value, cudaDevAttrWarpSize, device);
  if (status != cudaSuccess) { return status; }
  status = cudaDeviceGetAttribute(
      value, cudaDevAttrMaxThreadsPerMultiProcessor, device);
  if (status != cudaSuccess) { return status; }
  status = cudaDeviceGetAttribute(
      value, cudaDevAttrMaxThreadsPerBlock, device);
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

cudaError_t MemoryCompatApis(void** device, void** pinned, void* destination,
                             const void* source, size_t bytes,
                             cudaMemcpyKind kind, cudaStream_t stream) {
  cudaError_t status = cudaMalloc(device, bytes);
  if (status != cudaSuccess) { return status; }
  status = cudaMallocHost(pinned, bytes);
  if (status != cudaSuccess) { return status; }
  status = cudaMemcpy(destination, source, bytes, kind);
  if (status != cudaSuccess) { return status; }
  status = cudaMemcpyAsync(destination, source, bytes, kind, stream);
  if (status != cudaSuccess) { return status; }
  status = cudaMemset(destination, 0, bytes);
  if (status != cudaSuccess) { return status; }
  status = cudaMemsetAsync(destination, 0, bytes, stream);
  if (status != cudaSuccess) { return status; }
  if (cudaGetErrorString(status) == nullptr) { return cudaErrorInvalidValue; }
  status = cudaFree(*device);
  if (status != cudaSuccess) { return status; }
  status = cudaFreeHost(*pinned);
  if (status != cudaSuccess) { return status; }
  return cudaGetLastError();
}

cudaError_t RuntimeLifecycleCompatApis() {
  int count = 0;
  cudaError_t status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess || count == 0) { return status; }
  int device = -1;
  status = cudaGetDevice(&device);
  if (status != cudaSuccess) { return status; }
  status = cudaSetDevice(device);
  if (status != cudaSuccess) { return status; }

  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t end = nullptr;
  status = cudaStreamCreate(&stream);
  if (status != cudaSuccess) { return status; }
  status = cudaEventCreate(&start);
  if (status != cudaSuccess) { return status; }
  status = cudaEventCreate(&end);
  if (status != cudaSuccess) { return status; }
  status = cudaEventRecord(start, stream);
  if (status != cudaSuccess) { return status; }
  status = cudaEventSynchronize(start);
  if (status != cudaSuccess) { return status; }
  float elapsed_ms = 0.0f;
  status = cudaEventElapsedTime(&elapsed_ms, start, end);
  if (status != cudaSuccess) { return status; }
  status = cudaEventDestroy(start);
  if (status != cudaSuccess) { return status; }
  status = cudaEventDestroy(end);
  if (status != cudaSuccess) { return status; }
  status = cudaStreamSynchronize(stream);
  if (status != cudaSuccess) { return status; }
  status = cudaStreamDestroy(stream);
  if (status != cudaSuccess) { return status; }
  status = cudaDeviceSynchronize();
  if (status != cudaSuccess) { return status; }
  return cudaDeviceReset();
}

void CanonicalHostAllocation(size_t bytes) {
  float* host = (float*)malloc(bytes);
  free(host);
}

void KeepQualifiedHostFree(void* pointer) { std::free(pointer); }

double KeepHostDoublePrecision(double value) { return value + 1.0; }

static_assert(std::is_same<double, double>::value, "double traits must be preserved");
