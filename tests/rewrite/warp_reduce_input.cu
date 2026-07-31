#include <cuda.h>
#include <stdint.h>

// A comment is not an include directive:
// #include <ascify/ascify_cuda_compat.hpp>

__device__ float CanonicalFloatAdd(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ int CanonicalIntAdd(int value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(UINT32_MAX, value, offset, 32);
  }
  return value;
}

__device__ unsigned int CanonicalUintAdd(unsigned int value) {
  for (int offset = 16; offset > 0; offset /= 2) {
    value = __shfl_xor_sync(0xffffffffu, value, offset) + value;
  }
  return value;
}

__device__ float KeepSubwarpAdd(float value) {
  for (int offset = 8; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset, 16);
  }
  return value;
}

__device__ float KeepPartialMaskAdd(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0x0000ffffu, value, offset, 32);
  }
  return value;
}

__device__ float KeepNonCanonicalStepAdd(float value) {
  for (int offset = 16; offset > 0; --offset) {
    value += __shfl_xor_sync(0xffffffffu, value, offset, 32);
  }
  return value;
}

__device__ float KeepVolatileOffsetAdd(float value) {
  for (volatile int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset, 32);
  }
  return value;
}

__device__ int KeepAccumulatorIsOffset() {
  for (int offset = 16; offset > 0; offset >>= 1) {
    offset += __shfl_xor_sync(0xffffffffu, offset, offset, 32);
  }
  return 0;
}

#define ASCIFY_JOIN_IMPL(lhs, rhs) lhs##rhs
#define ASCIFY_JOIN(lhs, rhs) ASCIFY_JOIN_IMPL(lhs, rhs)
#define ASCIFY_FAKE_SHFL ASCIFY_JOIN(__shfl_xor_, sync)

namespace fake {
__device__ float ASCIFY_FAKE_SHFL(
    unsigned int, float value, int, int = 32) {
  return value;
}
}  // namespace fake

__device__ float KeepQualifiedSameNameWrapper(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += fake::ASCIFY_FAKE_SHFL(
        0xffffffffu, value, offset, 32);
  }
  return value;
}

struct AddFunctor {
  __device__ float operator()(float lhs, float rhs) const {
    return lhs + rhs;
  }
};

__device__ float KeepUnknownFunctor(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = AddFunctor()(value,
                         __shfl_xor_sync(0xffffffffu, value, offset, 32));
  }
  return value;
}

__device__ float KeepNestedCanonicalAdd(float value, bool enabled) {
  if (enabled) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_xor_sync(0xffffffffu, value, offset, 32);
    }
  }
  return value;
}

__device__ double KeepUnsupportedDoubleAdd(double value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset, 32);
  }
  return value;
}

template<typename T>
__device__ T KeepTemplateAdd(T value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset, 32);
  }
  return value;
}

__global__ void InstantiateTemplate(float* output) {
  output[0] = KeepTemplateAdd(output[0]);
}
