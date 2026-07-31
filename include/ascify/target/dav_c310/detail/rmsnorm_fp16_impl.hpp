// Hand-written Ascend950PR AIV/SIMT fp16 RMSNorm candidate.
//
// The fast path assigns one warp to one row and uses:
//   * 128-bit int4 transactions (eight fp16 values) for aligned rows;
//   * asc_reduce_add for the warp-wide FP32 square sum;
//   * one rsqrtf per row;
//   * a raw-fp16 register cache through 8192 columns, avoiding a second x read.
//
// Rows are grid-strided, so odd/tail row counts require no padding. Arbitrary
// column counts use a scalar, tail-safe path. Rows wider than the register
// cache use a two-pass streaming path. The affine specialization intentionally
// preserves OneFlow's fp16-before-weight-multiply store semantics.
#ifndef ASCIFY_TARGET_DAV_C310_DETAIL_RMSNORM_FP16_IMPL_HPP_
#define ASCIFY_TARGET_DAV_C310_DETAIL_RMSNORM_FP16_IMPL_HPP_

#include <acl/acl.h>
#include <simt_api/asc_fp16.h>
#include <simt_api/device_types.h>
#include <simt_api/device_warp_functions.h>
#include <simt_api/math_functions.h>
#include <acl_cub/aclcub.hpp>

#include <cmath>
#include <cstdint>
#include <limits>

#ifdef ASCIFY950_RMSNORM_NATIVE_STANDALONE
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#endif

namespace ascify {
namespace target {
namespace dav_c310 {
namespace v1 {
namespace detail {
namespace rmsnorm_fp16 {
namespace {

constexpr int kWarpSize = 32;
constexpr int kMaxThreadsPerBlock = 1024;
constexpr int kFallbackAivCount = 56;
constexpr int kTargetBlocksPerAiv = 4;
constexpr int kVectorElements = 8;
constexpr int kCacheColumns = 8192;
constexpr int kBlockRowMinColumns = 4096;
constexpr int kBlockRowMaxRows = 256;
#ifndef ASCIFY950_RMN_BLOCK_ROW_THREADS
#define ASCIFY950_RMN_BLOCK_ROW_THREADS 512
#endif
#ifndef ASCIFY950_RMN_BLOCK_ROW_AFFINE_THREADS
#define ASCIFY950_RMN_BLOCK_ROW_AFFINE_THREADS 256
#endif
constexpr int kDefaultBlockRowThreads = ASCIFY950_RMN_BLOCK_ROW_THREADS;
constexpr int kDefaultBlockRowAffineThreads =
    ASCIFY950_RMN_BLOCK_ROW_AFFINE_THREADS;
constexpr int kCachedVectorsPerLane =
    (kCacheColumns / kVectorElements + kWarpSize - 1) / kWarpSize;
constexpr int kCachedScalarsPerLane = (kCacheColumns + kWarpSize - 1) / kWarpSize;

static_assert(sizeof(int4) == 16, "int4 must remain a 128-bit GM transaction");
static_assert(kCacheColumns % (kWarpSize * kVectorElements) == 0,
              "vector cache geometry must be exact");
static_assert(kDefaultBlockRowThreads == 128
                  || kDefaultBlockRowThreads == 256
                  || kDefaultBlockRowThreads == 512
                  || kDefaultBlockRowThreads == 1024,
              "plain block-row threads must be 128, 256, 512, or 1024");
static_assert(kDefaultBlockRowAffineThreads == 128
                  || kDefaultBlockRowAffineThreads == 256
                  || kDefaultBlockRowAffineThreads == 512
                  || kDefaultBlockRowAffineThreads == 1024,
              "affine block-row threads must be 128, 256, 512, or 1024");

__aicore__ inline float PackedSquareSum(const int4& packed) {
  const half* values = reinterpret_cast<const half*>(&packed);
  float sum = 0.0f;
#pragma unroll
  for (int index = 0; index < kVectorElements; ++index) {
    const float value = static_cast<float>(values[index]);
    sum += value * value;
  }
  return sum;
}

template<bool kAffine>
__aicore__ inline int4 NormalizePacked(const int4& packed, const int4* weight,
                                       int vector_index, float inverse_rms) {
  int4 result;
  const half* input = reinterpret_cast<const half*>(&packed);
  half* output = reinterpret_cast<half*>(&result);
  if (kAffine) {
    const int4 packed_weight = weight[vector_index];
    const half* weight_values =
        reinterpret_cast<const half*>(&packed_weight);
#pragma unroll
    for (int index = 0; index < kVectorElements; ++index) {
      // Match OneFlow exactly: first round normalized values to FP16, then
      // perform the FP16 weight multiply and round the FP16 result.
      const half normalized =
          static_cast<half>(static_cast<float>(input[index]) * inverse_rms);
      output[index] = normalized * weight_values[index];
    }
  } else {
#pragma unroll
    for (int index = 0; index < kVectorElements; ++index) {
      output[index] =
          static_cast<half>(static_cast<float>(input[index]) * inverse_rms);
    }
  }
  return result;
}

// The 950PR compiler maps this half2 form efficiently for the one-row-per-block
// kernel. Keep it scoped to that path: small warp kernels need the scalar form
// above to preserve the validated OneFlow result.
__aicore__ inline int4 NormalizePackedBlockAffine(
    const int4& packed, const int4* weight, int vector_index,
    float inverse_rms) {
  int4 result;
  const int4 packed_weight = weight[vector_index];
  const half2* input_pairs = reinterpret_cast<const half2*>(&packed);
  const half2* weight_pairs =
      reinterpret_cast<const half2*>(&packed_weight);
  half2* output_pairs = reinterpret_cast<half2*>(&result);
#pragma unroll
  for (int pair = 0; pair < kVectorElements / 2; ++pair) {
    const half2 normalized = __floats2half2_rn(
        __low2float(input_pairs[pair]) * inverse_rms,
        __high2float(input_pairs[pair]) * inverse_rms);
    output_pairs[pair] = __hmulx2(normalized, weight_pairs[pair]);
  }
  return result;
}

template<bool kAffine>
__aicore__ inline half NormalizeScalar(half input, const half* weight, int64_t column,
                                       float inverse_rms) {
  const float normalized = static_cast<float>(input) * inverse_rms;
  if (kAffine) {
    const half rounded_normalized = static_cast<half>(normalized);
    return rounded_normalized * weight[column];
  }
  return static_cast<half>(normalized);
}

template<bool kAffine, int kCacheVectors>
__global__ void RmsNormCachedVectorKernel(const half* __restrict__ input,
                                          half* __restrict__ output,
                                          const half* __restrict__ weight,
                                          float* __restrict__ inverse_rms, int64_t rows,
                                          int columns, float inverse_columns, float epsilon) {
  static_assert(kCacheVectors > 0
                    && kCacheVectors <= kCachedVectorsPerLane,
                "invalid vector cache specialization");
  const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
  const int warp_in_block = static_cast<int>(threadIdx.x) / kWarpSize;
  const int warps_per_block = static_cast<int>(blockDim.x) / kWarpSize;
  const int64_t global_warp =
      static_cast<int64_t>(blockIdx.x) * warps_per_block + warp_in_block;
  const int64_t warp_stride = static_cast<int64_t>(gridDim.x) * warps_per_block;
  const int vector_count = columns / kVectorElements;
  int4 cache[kCacheVectors];

  for (int64_t row = global_warp; row < rows; row += warp_stride) {
    const int64_t row_offset = row * static_cast<int64_t>(columns);
    const int4* row_input =
        reinterpret_cast<const int4*>(input + row_offset);
    int4* row_output = reinterpret_cast<int4*>(output + row_offset);
    const int4* packed_weight = reinterpret_cast<const int4*>(weight);

    float square_sum = 0.0f;
    int cached = 0;
    for (int vector_index = lane; vector_index < vector_count;
         vector_index += kWarpSize) {
      const int4 packed = row_input[vector_index];
      cache[cached++] = packed;
      square_sum += PackedSquareSum(packed);
    }

    square_sum = asc_reduce_add(square_sum);
    const float row_inverse_rms =
        rsqrtf(square_sum * inverse_columns + epsilon);
    if (lane == 0) { inverse_rms[row] = row_inverse_rms; }

    cached = 0;
    for (int vector_index = lane; vector_index < vector_count;
         vector_index += kWarpSize) {
      row_output[vector_index] =
          NormalizePacked<kAffine>(cache[cached++], packed_weight, vector_index,
                                   row_inverse_rms);
    }
  }
}

template<bool kAffine>
__global__ void RmsNormStreamingVectorKernel(const half* __restrict__ input,
                                             half* __restrict__ output,
                                             const half* __restrict__ weight,
                                             float* __restrict__ inverse_rms, int64_t rows,
                                             int columns, float inverse_columns,
                                             float epsilon) {
  const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
  const int warp_in_block = static_cast<int>(threadIdx.x) / kWarpSize;
  const int warps_per_block = static_cast<int>(blockDim.x) / kWarpSize;
  const int64_t global_warp =
      static_cast<int64_t>(blockIdx.x) * warps_per_block + warp_in_block;
  const int64_t warp_stride = static_cast<int64_t>(gridDim.x) * warps_per_block;
  const int vector_count = columns / kVectorElements;

  for (int64_t row = global_warp; row < rows; row += warp_stride) {
    const int64_t row_offset = row * static_cast<int64_t>(columns);
    const int4* row_input =
        reinterpret_cast<const int4*>(input + row_offset);
    int4* row_output = reinterpret_cast<int4*>(output + row_offset);
    const int4* packed_weight = reinterpret_cast<const int4*>(weight);

    float square_sum = 0.0f;
    for (int vector_index = lane; vector_index < vector_count;
         vector_index += kWarpSize) {
      square_sum += PackedSquareSum(row_input[vector_index]);
    }

    square_sum = asc_reduce_add(square_sum);
    const float row_inverse_rms =
        rsqrtf(square_sum * inverse_columns + epsilon);
    if (lane == 0) { inverse_rms[row] = row_inverse_rms; }

    for (int vector_index = lane; vector_index < vector_count;
         vector_index += kWarpSize) {
      row_output[vector_index] =
          NormalizePacked<kAffine>(row_input[vector_index], packed_weight,
                                   vector_index, row_inverse_rms);
    }
  }
}

template<bool kAffine, int kCacheScalars>
__global__ void RmsNormCachedScalarKernel(const half* __restrict__ input,
                                          half* __restrict__ output,
                                          const half* __restrict__ weight,
                                          float* __restrict__ inverse_rms, int64_t rows,
                                          int columns, float inverse_columns, float epsilon) {
  static_assert(kCacheScalars > 0
                    && kCacheScalars <= kCachedScalarsPerLane,
                "invalid scalar cache specialization");
  const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
  const int warp_in_block = static_cast<int>(threadIdx.x) / kWarpSize;
  const int warps_per_block = static_cast<int>(blockDim.x) / kWarpSize;
  const int64_t global_warp =
      static_cast<int64_t>(blockIdx.x) * warps_per_block + warp_in_block;
  const int64_t warp_stride = static_cast<int64_t>(gridDim.x) * warps_per_block;
  half cache[kCacheScalars];

  for (int64_t row = global_warp; row < rows; row += warp_stride) {
    const int64_t row_offset = row * static_cast<int64_t>(columns);
    const half* row_input = input + row_offset;
    half* row_output = output + row_offset;

    float square_sum = 0.0f;
    int cached = 0;
    for (int column = lane; column < columns; column += kWarpSize) {
      const half value = row_input[column];
      cache[cached++] = value;
      const float as_float = static_cast<float>(value);
      square_sum += as_float * as_float;
    }

    square_sum = asc_reduce_add(square_sum);
    const float row_inverse_rms =
        rsqrtf(square_sum * inverse_columns + epsilon);
    if (lane == 0) { inverse_rms[row] = row_inverse_rms; }

    cached = 0;
    for (int column = lane; column < columns; column += kWarpSize) {
      row_output[column] =
          NormalizeScalar<kAffine>(cache[cached++], weight, column,
                                   row_inverse_rms);
    }
  }
}

template<bool kAffine>
__global__ void RmsNormStreamingScalarKernel(const half* __restrict__ input,
                                             half* __restrict__ output,
                                             const half* __restrict__ weight,
                                             float* __restrict__ inverse_rms, int64_t rows,
                                             int columns, float inverse_columns,
                                             float epsilon) {
  const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
  const int warp_in_block = static_cast<int>(threadIdx.x) / kWarpSize;
  const int warps_per_block = static_cast<int>(blockDim.x) / kWarpSize;
  const int64_t global_warp =
      static_cast<int64_t>(blockIdx.x) * warps_per_block + warp_in_block;
  const int64_t warp_stride = static_cast<int64_t>(gridDim.x) * warps_per_block;

  for (int64_t row = global_warp; row < rows; row += warp_stride) {
    const int64_t row_offset = row * static_cast<int64_t>(columns);
    const half* row_input = input + row_offset;
    half* row_output = output + row_offset;

    float square_sum = 0.0f;
    for (int64_t column = lane; column < columns; column += kWarpSize) {
      const float value = static_cast<float>(row_input[column]);
      square_sum += value * value;
    }

    square_sum = asc_reduce_add(square_sum);
    const float row_inverse_rms =
        rsqrtf(square_sum * inverse_columns + epsilon);
    if (lane == 0) { inverse_rms[row] = row_inverse_rms; }

    for (int64_t column = lane; column < columns; column += kWarpSize) {
      row_output[column] =
          NormalizeScalar<kAffine>(row_input[column], weight, column,
                                   row_inverse_rms);
    }
  }
}

// Low-row-count long rows cannot expose enough independent warps with the
// warp-per-row kernels. One whole block owns a row here. Raw FP16 values are
// cached once in UB, and aclcub's native Sum tag performs a two-level
// asc_reduce_add reduction with one block barrier.
template<bool kAffine, int kBlockThreads, int kBlockCacheColumns>
__global__ void RmsNormBlockCachedVectorKernel(
    const half* __restrict__ input, half* __restrict__ output,
    const half* __restrict__ weight, float* __restrict__ inverse_rms,
    int64_t rows, int columns, float inverse_columns, float epsilon) {
  static_assert(kBlockThreads == 128 || kBlockThreads == 256
                    || kBlockThreads == 512
                    || kBlockThreads == 1024,
                "unsupported block-row width");
  static_assert(kBlockCacheColumns == 4096
                    || kBlockCacheColumns == kCacheColumns,
                "unsupported block-row UB capacity");
  static_assert(kBlockCacheColumns % kVectorElements == 0,
                "block-row UB cache must hold whole int4 values");
  using BlockReduce = aclcub::BlockReduce<float, kBlockThreads>;
  __ubuf__ typename BlockReduce::TempStorage reduce_storage;
  __ubuf__ __attribute__((aligned(16)))
      int4 cache[kBlockCacheColumns / kVectorElements];

  const int thread = static_cast<int>(threadIdx.x);
  const int vector_count = columns / kVectorElements;
  int reduction_iteration = 0;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const int64_t row_offset = row * static_cast<int64_t>(columns);
    const int4* row_input =
        reinterpret_cast<const int4*>(input + row_offset);
    int4* row_output = reinterpret_cast<int4*>(output + row_offset);
    const int4* packed_weight = reinterpret_cast<const int4*>(weight);

    float local_square_sum = 0.0f;
    for (int vector_index = thread; vector_index < vector_count;
         vector_index += kBlockThreads) {
      const int4 packed = row_input[vector_index];
      cache[vector_index] = packed;
      local_square_sum += PackedSquareSum(packed);
    }

    const float square_sum =
        BlockReduce(reduce_storage)
            .AllReduce(local_square_sum, aclcub::Sum{},
                       reduction_iteration++);
    const float row_inverse_rms =
        rsqrtf(square_sum * inverse_columns + epsilon);
    if (thread == 0) { inverse_rms[row] = row_inverse_rms; }

    for (int vector_index = thread; vector_index < vector_count;
         vector_index += kBlockThreads) {
      if (kAffine) {
        row_output[vector_index] = NormalizePackedBlockAffine(
            cache[vector_index], packed_weight, vector_index,
            row_inverse_rms);
      } else {
        row_output[vector_index] =
            NormalizePacked<false>(cache[vector_index], nullptr,
                                   vector_index, row_inverse_rms);
      }
    }
  }
}

inline int VectorCoreCount() {
  // The process pins one device for its lifetime. Query once so the launch
  // geometry follows the selected target without adding work to timed calls.
  static const int cached_count = []() {
    int32_t device = 0;
    int64_t count = 0;
    if (aclrtGetDevice(&device) == ACL_SUCCESS
        && aclrtGetDeviceInfo(static_cast<uint32_t>(device),
                              ACL_DEV_ATTR_VECTOR_CORE_NUM, &count)
               == ACL_SUCCESS
        && count > 0 && count <= 1024) {
      return static_cast<int>(count);
    }
    return kFallbackAivCount;
  }();
  return cached_count;
}

inline int ChooseBlockThreads(int64_t rows) {
  const int target_blocks = VectorCoreCount() * kTargetBlocksPerAiv;
  int64_t required_warps =
      1 + (rows - 1) / static_cast<int64_t>(target_blocks);
  int warps_per_block = 1;
  while (warps_per_block < kMaxThreadsPerBlock / kWarpSize
         && static_cast<int64_t>(warps_per_block) < required_warps) {
    warps_per_block <<= 1;
  }
  return warps_per_block * kWarpSize;
}

inline bool IsAligned16(const void* pointer) {
  return (reinterpret_cast<uintptr_t>(pointer) & 15U) == 0;
}

inline bool IsVectorAligned(const half* input, const half* output,
                            const half* weight, int columns) {
  return columns % kVectorElements == 0 && IsAligned16(input)
         && IsAligned16(output)
         && (weight == nullptr || IsAligned16(weight));
}

inline bool ShouldUseBlockRow(int64_t rows, int columns,
                              bool vector_aligned) {
  return vector_aligned && rows <= kBlockRowMaxRows
         && columns >= kBlockRowMinColumns && columns <= kCacheColumns;
}

template<bool kAffine, int kBlockThreads, int kBlockCacheColumns>
inline void LaunchBlockCachedVectorCapacity(
    int grid_blocks, aclrtStream stream, const half* input, half* output,
    const half* weight, float* inverse_rms, int64_t rows, int columns,
    float inverse_columns, float epsilon) {
  RmsNormBlockCachedVectorKernel<kAffine, kBlockThreads, kBlockCacheColumns>
      <<<grid_blocks, kBlockThreads, 0, stream>>>(
          input, output, weight, inverse_rms, rows, columns, inverse_columns,
          epsilon);
}

template<bool kAffine, int kBlockThreads>
inline void DispatchBlockCachedVectorCapacity(
    int grid_blocks, aclrtStream stream, const half* input, half* output,
    const half* weight, float* inverse_rms, int64_t rows, int columns,
    float inverse_columns, float epsilon) {
  if (columns <= 4096) {
    LaunchBlockCachedVectorCapacity<kAffine, kBlockThreads, 4096>(
        grid_blocks, stream, input, output, weight, inverse_rms, rows, columns,
        inverse_columns, epsilon);
  } else {
    LaunchBlockCachedVectorCapacity<kAffine, kBlockThreads, kCacheColumns>(
        grid_blocks, stream, input, output, weight, inverse_rms, rows, columns,
        inverse_columns, epsilon);
  }
}

template<bool kAffine>
inline aclError LaunchBlockCachedVector(
    aclrtStream stream, const half* input, half* output, const half* weight,
    float* inverse_rms, int64_t rows, int columns, float inverse_columns,
    float epsilon, int block_threads, int grid_cap) {
  int64_t grid_blocks = rows;
  if (grid_blocks > grid_cap) { grid_blocks = grid_cap; }
  if (block_threads == 128) {
    DispatchBlockCachedVectorCapacity<kAffine, 128>(
        static_cast<int>(grid_blocks), stream, input, output, weight,
        inverse_rms, rows, columns, inverse_columns, epsilon);
  } else if (block_threads == 256) {
    DispatchBlockCachedVectorCapacity<kAffine, 256>(
        static_cast<int>(grid_blocks), stream, input, output, weight,
        inverse_rms, rows, columns, inverse_columns, epsilon);
  } else if (block_threads == 512) {
    DispatchBlockCachedVectorCapacity<kAffine, 512>(
        static_cast<int>(grid_blocks), stream, input, output, weight,
        inverse_rms, rows, columns, inverse_columns, epsilon);
  } else {
    DispatchBlockCachedVectorCapacity<kAffine, 1024>(
        static_cast<int>(grid_blocks), stream, input, output, weight,
        inverse_rms, rows, columns, inverse_columns, epsilon);
  }
  return aclrtPeekAtLastError(static_cast<aclrtLastErrLevel>(0));
}

template<bool kAffine, int kCacheVectors>
inline void LaunchCachedVector(int grid_blocks, int block_threads,
                               aclrtStream stream, const half* input,
                               half* output, const half* weight,
                               float* inverse_rms, int64_t rows, int columns,
                               float inverse_columns, float epsilon) {
  RmsNormCachedVectorKernel<kAffine, kCacheVectors>
      <<<grid_blocks, block_threads, 0, stream>>>(
          input, output, weight, inverse_rms, rows, columns, inverse_columns,
          epsilon);
}

template<bool kAffine>
inline void DispatchCachedVector(int grid_blocks, int block_threads,
                                 aclrtStream stream, const half* input,
                                 half* output, const half* weight,
                                 float* inverse_rms, int64_t rows, int columns,
                                 float inverse_columns, float epsilon) {
  // Each warp iteration covers 32 lanes * 8 fp16 values = 256 columns.
  if (columns <= 256) {
    LaunchCachedVector<kAffine, 1>(
        grid_blocks, block_threads, stream, input, output, weight, inverse_rms,
        rows, columns, inverse_columns, epsilon);
  } else if (columns <= 512) {
    LaunchCachedVector<kAffine, 2>(
        grid_blocks, block_threads, stream, input, output, weight, inverse_rms,
        rows, columns, inverse_columns, epsilon);
  } else if (columns <= 1024) {
    LaunchCachedVector<kAffine, 4>(
        grid_blocks, block_threads, stream, input, output, weight, inverse_rms,
        rows, columns, inverse_columns, epsilon);
  } else if (columns <= 2048) {
    LaunchCachedVector<kAffine, 8>(
        grid_blocks, block_threads, stream, input, output, weight, inverse_rms,
        rows, columns, inverse_columns, epsilon);
  } else if (columns <= 4096) {
    LaunchCachedVector<kAffine, 16>(
        grid_blocks, block_threads, stream, input, output, weight, inverse_rms,
        rows, columns, inverse_columns, epsilon);
  } else {
    LaunchCachedVector<kAffine, 32>(
        grid_blocks, block_threads, stream, input, output, weight, inverse_rms,
        rows, columns, inverse_columns, epsilon);
  }
}

template<bool kAffine, int kCacheScalars>
inline void LaunchCachedScalar(int grid_blocks, int block_threads,
                               aclrtStream stream, const half* input,
                               half* output, const half* weight,
                               float* inverse_rms, int64_t rows, int columns,
                               float inverse_columns, float epsilon) {
  RmsNormCachedScalarKernel<kAffine, kCacheScalars>
      <<<grid_blocks, block_threads, 0, stream>>>(
          input, output, weight, inverse_rms, rows, columns, inverse_columns,
          epsilon);
}

template<bool kAffine>
inline void DispatchCachedScalar(int grid_blocks, int block_threads,
                                 aclrtStream stream, const half* input,
                                 half* output, const half* weight,
                                 float* inverse_rms, int64_t rows, int columns,
                                 float inverse_columns, float epsilon) {
  // One scalar iteration covers 32 columns; power-of-two capacities keep the
  // generated local allocation close to the actual row width.
  if (columns <= 32) {
    LaunchCachedScalar<kAffine, 1>(
        grid_blocks, block_threads, stream, input, output, weight, inverse_rms,
        rows, columns, inverse_columns, epsilon);
  } else if (columns <= 64) {
    LaunchCachedScalar<kAffine, 2>(
        grid_blocks, block_threads, stream, input, output, weight, inverse_rms,
        rows, columns, inverse_columns, epsilon);
  } else if (columns <= 128) {
    LaunchCachedScalar<kAffine, 4>(
        grid_blocks, block_threads, stream, input, output, weight, inverse_rms,
        rows, columns, inverse_columns, epsilon);
  } else if (columns <= 256) {
    LaunchCachedScalar<kAffine, 8>(
        grid_blocks, block_threads, stream, input, output, weight, inverse_rms,
        rows, columns, inverse_columns, epsilon);
  } else if (columns <= 512) {
    LaunchCachedScalar<kAffine, 16>(
        grid_blocks, block_threads, stream, input, output, weight, inverse_rms,
        rows, columns, inverse_columns, epsilon);
  } else if (columns <= 1024) {
    LaunchCachedScalar<kAffine, 32>(
        grid_blocks, block_threads, stream, input, output, weight, inverse_rms,
        rows, columns, inverse_columns, epsilon);
  } else if (columns <= 2048) {
    LaunchCachedScalar<kAffine, 64>(
        grid_blocks, block_threads, stream, input, output, weight, inverse_rms,
        rows, columns, inverse_columns, epsilon);
  } else if (columns <= 4096) {
    LaunchCachedScalar<kAffine, 128>(
        grid_blocks, block_threads, stream, input, output, weight, inverse_rms,
        rows, columns, inverse_columns, epsilon);
  } else {
    LaunchCachedScalar<kAffine, 256>(
        grid_blocks, block_threads, stream, input, output, weight, inverse_rms,
        rows, columns, inverse_columns, epsilon);
  }
}

template<bool kAffine>
inline aclError LaunchSpecialized(aclrtStream stream, const half* input, half* output,
                                  const half* weight, float* inverse_rms, int64_t rows,
                                  int columns, float inverse_columns, float epsilon,
                                  int block_threads, int grid_cap) {
  const int warps_per_block = block_threads / kWarpSize;
  int64_t grid_blocks = 1 + (rows - 1) / warps_per_block;
  if (grid_blocks > grid_cap) { grid_blocks = grid_cap; }

  const bool vector_aligned =
      IsVectorAligned(input, output, kAffine ? weight : nullptr, columns);
  if (columns <= kCacheColumns) {
    if (vector_aligned) {
      DispatchCachedVector<kAffine>(
          static_cast<int>(grid_blocks), block_threads, stream, input, output,
          weight, inverse_rms, rows, columns, inverse_columns, epsilon);
    } else {
      DispatchCachedScalar<kAffine>(
          static_cast<int>(grid_blocks), block_threads, stream, input, output,
          weight, inverse_rms, rows, columns, inverse_columns, epsilon);
    }
  } else {
    if (vector_aligned) {
      RmsNormStreamingVectorKernel<kAffine>
          <<<static_cast<int>(grid_blocks), block_threads, 0, stream>>>(
              input, output, weight, inverse_rms, rows, columns, inverse_columns,
              epsilon);
    } else {
      RmsNormStreamingScalarKernel<kAffine>
          <<<static_cast<int>(grid_blocks), block_threads, 0, stream>>>(
              input, output, weight, inverse_rms, rows, columns, inverse_columns,
              epsilon);
    }
  }
  return aclrtPeekAtLastError(static_cast<aclrtLastErrLevel>(0));
}

// A null weight selects plain RMSNorm. A non-null weight selects the affine
// specialization. block_threads==0 and grid_cap==0 enable the 950PR defaults.
inline aclError Launch(aclrtStream stream, const half* input, half* output,
                       const half* weight, float* inverse_rms, int64_t rows,
                       int64_t columns, double epsilon, int block_threads = 0,
                       int grid_cap = 0) {
  if (stream == nullptr || input == nullptr || output == nullptr
      || inverse_rms == nullptr || rows <= 0 || columns <= 0
      || columns > std::numeric_limits<int>::max()
      || rows > std::numeric_limits<int64_t>::max() / columns
      || !(epsilon > 0.0) || !std::isfinite(epsilon)) {
    return ACL_ERROR_RT_PARAM_INVALID;
  }
  if ((block_threads != 0
       && (block_threads < kWarpSize
           || block_threads > kMaxThreadsPerBlock
           || block_threads % kWarpSize != 0))
      || grid_cap < 0) {
    return ACL_ERROR_RT_PARAM_INVALID;
  }
  if (grid_cap == 0) { grid_cap = VectorCoreCount() * 32; }

  const int narrow_columns = static_cast<int>(columns);
  const float inverse_columns =
      1.0f / static_cast<float>(narrow_columns);
  const float narrow_epsilon = static_cast<float>(epsilon);
  const bool vector_aligned =
      IsVectorAligned(input, output, weight, narrow_columns);
  if (ShouldUseBlockRow(rows, narrow_columns, vector_aligned)
      && (block_threads == 0 || block_threads == 128
          || block_threads == 256
          || block_threads == 512 || block_threads == 1024)) {
    const int block_row_threads = block_threads == 0
                                      ? (weight == nullptr
                                             ? kDefaultBlockRowThreads
                                             : kDefaultBlockRowAffineThreads)
                                      : block_threads;
    if (weight != nullptr) {
      return LaunchBlockCachedVector<true>(
          stream, input, output, weight, inverse_rms, rows, narrow_columns,
          inverse_columns, narrow_epsilon, block_row_threads, grid_cap);
    }
    return LaunchBlockCachedVector<false>(
        stream, input, output, nullptr, inverse_rms, rows, narrow_columns,
        inverse_columns, narrow_epsilon, block_row_threads, grid_cap);
  }

  if (block_threads == 0) { block_threads = ChooseBlockThreads(rows); }
  if (weight != nullptr) {
    return LaunchSpecialized<true>(
        stream, input, output, weight, inverse_rms, rows, narrow_columns,
        inverse_columns, narrow_epsilon, block_threads, grid_cap);
  }
  return LaunchSpecialized<false>(
      stream, input, output, nullptr, inverse_rms, rows, narrow_columns,
      inverse_columns, narrow_epsilon, block_threads, grid_cap);
}

}  // namespace
}  // namespace rmsnorm_fp16
}  // namespace detail
}  // namespace v1
}  // namespace dav_c310
}  // namespace target
}  // namespace ascify

#ifdef ASCIFY950_RMSNORM_NATIVE_STANDALONE
namespace {

namespace native_rmsnorm =
    ::ascify::target::dav_c310::v1::detail::rmsnorm_fp16;

constexpr size_t kSelfTestGuardBytes = 256;
constexpr uint8_t kSelfTestGuardValue = 0xd3;

void SelfTestAclCheck(aclError error, const char* operation, int line) {
  if (error == ACL_SUCCESS) { return; }
  std::fprintf(stderr, "ACL error %d at line %d: %s\n", static_cast<int>(error),
               line, operation);
  std::exit(1);
}

#define ASCIFY950_RMN_SELF_TEST_ACL(expression) \
  SelfTestAclCheck((expression), #expression, __LINE__)

class SelfTestBuffer {
 public:
  SelfTestBuffer() = default;
  ~SelfTestBuffer() {
    if (storage_ != nullptr) { aclrtFree(storage_); }
  }
  SelfTestBuffer(const SelfTestBuffer&) = delete;
  SelfTestBuffer& operator=(const SelfTestBuffer&) = delete;

  void Allocate(size_t payload_bytes) {
    payload_bytes_ = payload_bytes;
    total_bytes_ = payload_bytes_ + 2 * kSelfTestGuardBytes;
    ASCIFY950_RMN_SELF_TEST_ACL(
        aclrtMalloc(&storage_, total_bytes_, ACL_MEM_MALLOC_HUGE_FIRST));
    ASCIFY950_RMN_SELF_TEST_ACL(
        aclrtMemset(storage_, total_bytes_, kSelfTestGuardValue, total_bytes_));
  }

  void* Payload() {
    return static_cast<void*>(
        static_cast<uint8_t*>(storage_) + kSelfTestGuardBytes);
  }

  template<typename T>
  T* As() {
    return static_cast<T*>(Payload());
  }

  void CopyPayloadFromHost(const void* source) {
    ASCIFY950_RMN_SELF_TEST_ACL(
        aclrtMemcpy(Payload(), payload_bytes_, source, payload_bytes_,
                    ACL_MEMCPY_HOST_TO_DEVICE));
  }

  std::vector<uint8_t> Snapshot() const {
    std::vector<uint8_t> result(total_bytes_);
    ASCIFY950_RMN_SELF_TEST_ACL(
        aclrtMemcpy(result.data(), result.size(), storage_, result.size(),
                    ACL_MEMCPY_DEVICE_TO_HOST));
    return result;
  }

  size_t PayloadBytes() const { return payload_bytes_; }

 private:
  void* storage_ = nullptr;
  size_t payload_bytes_ = 0;
  size_t total_bytes_ = 0;
};

bool GuardsMatch(const std::vector<uint8_t>& snapshot, size_t payload_bytes) {
  for (size_t index = 0; index < kSelfTestGuardBytes; ++index) {
    if (snapshot[index] != kSelfTestGuardValue) { return false; }
  }
  const size_t suffix = kSelfTestGuardBytes + payload_bytes;
  for (size_t index = 0; index < kSelfTestGuardBytes; ++index) {
    if (snapshot[suffix + index] != kSelfTestGuardValue) { return false; }
  }
  return true;
}

bool RunSelfTestCase(aclrtStream stream, int64_t rows, int64_t columns,
                     bool affine, int block_threads = 0) {
  const size_t elements =
      static_cast<size_t>(rows) * static_cast<size_t>(columns);
  const size_t tensor_bytes = elements * sizeof(uint16_t);
  const size_t weight_bytes =
      static_cast<size_t>(columns) * sizeof(uint16_t);
  const size_t inverse_bytes = static_cast<size_t>(rows) * sizeof(float);
  std::vector<uint16_t> host_input(elements, 0x3c00U);   // fp16 1.0
  std::vector<uint16_t> host_weight(static_cast<size_t>(columns),
                                    0x3800U);            // fp16 0.5

  SelfTestBuffer device_input;
  SelfTestBuffer device_output;
  SelfTestBuffer device_inverse;
  SelfTestBuffer device_weight;
  device_input.Allocate(tensor_bytes);
  device_output.Allocate(tensor_bytes);
  device_inverse.Allocate(inverse_bytes);
  if (affine) { device_weight.Allocate(weight_bytes); }
  device_input.CopyPayloadFromHost(host_input.data());
  if (affine) { device_weight.CopyPayloadFromHost(host_weight.data()); }

  const aclError launch_error = native_rmsnorm::Launch(
      stream, device_input.As<const half>(), device_output.As<half>(),
      affine ? device_weight.As<const half>() : nullptr,
      device_inverse.As<float>(), rows, columns, 1.0e-5, block_threads);
  if (launch_error != ACL_SUCCESS) {
    std::fprintf(stderr,
                 "launch failed: rows=%lld cols=%lld affine=%d ACL=%d\n",
                 static_cast<long long>(rows),
                 static_cast<long long>(columns), affine ? 1 : 0,
                 static_cast<int>(launch_error));
    return false;
  }
  const aclError synchronize_error = aclrtSynchronizeStream(stream);
  if (synchronize_error != ACL_SUCCESS) {
    std::fprintf(stderr,
                 "synchronize failed: rows=%lld cols=%lld affine=%d ACL=%d\n",
                 static_cast<long long>(rows),
                 static_cast<long long>(columns), affine ? 1 : 0,
                 static_cast<int>(synchronize_error));
    return false;
  }

  const std::vector<uint8_t> input_snapshot = device_input.Snapshot();
  const std::vector<uint8_t> output_snapshot = device_output.Snapshot();
  const std::vector<uint8_t> inverse_snapshot = device_inverse.Snapshot();
  const std::vector<uint8_t> weight_snapshot =
      affine ? device_weight.Snapshot() : std::vector<uint8_t>();
  bool ok = GuardsMatch(input_snapshot, tensor_bytes)
            && GuardsMatch(output_snapshot, tensor_bytes)
            && GuardsMatch(inverse_snapshot, inverse_bytes)
            && (!affine || GuardsMatch(weight_snapshot, weight_bytes));
  ok = ok
       && std::memcmp(input_snapshot.data() + kSelfTestGuardBytes,
                      host_input.data(), tensor_bytes)
              == 0;
  if (affine) {
    ok = ok
         && std::memcmp(weight_snapshot.data() + kSelfTestGuardBytes,
                        host_weight.data(), weight_bytes)
                == 0;
  }

  const uint16_t expected_output = affine ? 0x3800U : 0x3c00U;
  for (size_t index = 0; index < elements; ++index) {
    uint16_t actual = 0;
    std::memcpy(
        &actual,
        output_snapshot.data() + kSelfTestGuardBytes
            + index * sizeof(uint16_t),
        sizeof(actual));
    if (actual != expected_output) {
      ok = false;
      break;
    }
  }
  const float reference_inverse = 1.0f / std::sqrt(1.0f + 1.0e-5f);
  for (int64_t row = 0; row < rows; ++row) {
    float actual = 0.0f;
    std::memcpy(
        &actual,
        inverse_snapshot.data() + kSelfTestGuardBytes
            + static_cast<size_t>(row) * sizeof(float),
        sizeof(actual));
    if (!std::isfinite(actual)
        || std::fabs(actual - reference_inverse) > 1.0e-4f) {
      ok = false;
      break;
    }
  }

  int launched_block_threads = block_threads;
  if (launched_block_threads == 0) {
    launched_block_threads =
        native_rmsnorm::ShouldUseBlockRow(
            rows, static_cast<int>(columns),
            columns % native_rmsnorm::kVectorElements == 0)
            ? (affine
                   ? native_rmsnorm::kDefaultBlockRowAffineThreads
                   : native_rmsnorm::kDefaultBlockRowThreads)
            : native_rmsnorm::ChooseBlockThreads(rows);
  }
  std::fprintf(stderr, "[%s] rows=%lld cols=%lld affine=%d block=%d\n",
               ok ? "pass" : "fail", static_cast<long long>(rows),
               static_cast<long long>(columns), affine ? 1 : 0,
               launched_block_threads);
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  int device = 0;
  const char* selected_device = std::getenv("ASCIFY_DEVICE");
  if (selected_device != nullptr) { device = std::atoi(selected_device); }
  if (argc > 1) { device = std::atoi(argv[1]); }

  ASCIFY950_RMN_SELF_TEST_ACL(aclInit(nullptr));
  ASCIFY950_RMN_SELF_TEST_ACL(aclrtSetDevice(device));
  aclrtContext context = nullptr;
  aclrtStream stream = nullptr;
  ASCIFY950_RMN_SELF_TEST_ACL(aclrtCreateContext(&context, device));
  ASCIFY950_RMN_SELF_TEST_ACL(aclrtCreateStream(&stream));

  const int64_t columns[] = {
      1, 31, 32, 64, 127, 128, 513, 1024, 4097, 8191, 8192, 8193, 8200,
  };
  int failures = 0;
  size_t cases = 0;
  for (int64_t column : columns) {
    if (!RunSelfTestCase(stream, 3, column, false)) { ++failures; }
    if (!RunSelfTestCase(stream, 3, column, true)) { ++failures; }
    cases += 2;
  }

  // Opt-in coverage for the auto block-size choices used by the tune matrix.
  // Kept out of the default smoke because it performs larger D2H guard checks.
  if (std::getenv("ASCIFY950_RMSNORM_NATIVE_SATURATION_TEST") != nullptr) {
    struct Shape {
      int64_t rows;
      int64_t columns;
      int block_threads;
    };
    const Shape shapes[] = {
        {8192, 128, 0},
        {8192, 512, 0},
        {8192, 1024, 0},
        {1024, 4096, 0},
        {128, 8192, 0},
        // Exercise the maximum block size with the maximum cached row width,
        // without allocating the full 8192x8192 tensor.
        {128, 8192, 1024},
        // An explicit non-block-row width must retain the warp fallback.
        {128, 8192, 32},
    };
    for (const Shape& shape : shapes) {
      if (!RunSelfTestCase(stream, shape.rows, shape.columns, false,
                           shape.block_threads)) {
        ++failures;
      }
      if (!RunSelfTestCase(stream, shape.rows, shape.columns, true,
                           shape.block_threads)) {
        ++failures;
      }
      cases += 2;
    }
  }

  ASCIFY950_RMN_SELF_TEST_ACL(aclrtDestroyStream(stream));
  ASCIFY950_RMN_SELF_TEST_ACL(aclrtDestroyContext(context));
  ASCIFY950_RMN_SELF_TEST_ACL(aclrtResetDevice(device));
  ASCIFY950_RMN_SELF_TEST_ACL(aclFinalize());
  std::fprintf(stderr, "native RMSNorm self-test: cases=%zu failures=%d\n",
               cases, failures);
  return failures == 0 ? 0 : 2;
}
#endif  // ASCIFY950_RMSNORM_NATIVE_STANDALONE

#endif  // ASCIFY_TARGET_DAV_C310_DETAIL_RMSNORM_FP16_IMPL_HPP_
