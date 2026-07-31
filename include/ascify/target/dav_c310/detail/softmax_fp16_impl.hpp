// Hand-written Ascend-native fp16 softmax for dav-c310-vec (-x dpp SIMT).
// Design (from the 0628 perf probe): warp-per-row + 128-bit (int4 = 8 half) vectorized
// load/store + native warp reductions. Strategies are dispatched by cols and rows:
//   * cols <= CACHE_MAX : single-pass, row cached in per-lane registers (2x traffic).
//   * cols >  CACHE_MAX : online (running max/sum) streaming, 2 reads + 1 write.
//   * optional low-row-count block path: row cached in UB as FP16, with one block
//     per row. This restores AIV occupancy when warp-per-row exposes too few warps.
// Alignment: 128-bit int4 loads require each row to start on a 16-byte boundary, which
// holds iff cols%8==0. For cols%8!=0 the row stride is unaligned, so we dispatch a
// scalar fallback (correct, no vectorization). One warp (32 lanes) owns one row;
// rows are grid-strided across warps.
#ifndef ASCIFY_TARGET_DAV_C310_DETAIL_SOFTMAX_FP16_IMPL_HPP_
#define ASCIFY_TARGET_DAV_C310_DETAIL_SOFTMAX_FP16_IMPL_HPP_
#include <acl/acl.h>
#include "simt_api/asc_fp16.h"
#include "simt_api/math_functions.h"
#include "simt_api/device_warp_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_types.h"
#include <acl_cub/aclcub.hpp>
#include <cstdint>
#include <limits>

namespace ascify {
namespace target {
namespace dav_c310 {
namespace v1 {
namespace detail {
namespace softmax_fp16 {
namespace {

constexpr int kWarpSize = 32;
constexpr int kCacheColumns = 4096;  // per-lane float buf <= 128 (cols/32)
constexpr int kLaneBufferElements = 128;  // ceil(kCacheColumns/32)

#ifndef SMN_BLOCK_ROW_MAX
#define SMN_BLOCK_ROW_MAX 2048
#endif
#ifndef SMN_FALLBACK_AIV_COUNT
#define SMN_FALLBACK_AIV_COUNT 56
#endif

// ---- warp-wide butterfly reductions (broadcast result to all lanes) ----
__aicore__ inline float smn_warp_max(float v) {
#ifdef SMN_USE_ASC_REDUCE
  return asc_reduce_max(v);
#else
#pragma unroll
  for (int o = kWarpSize / 2; o > 0; o >>= 1) {
    float t = asc_shfl_xor(v, o);
    v = t > v ? t : v;
  }
  return v;
#endif
}
__aicore__ inline float smn_warp_sum(float v) {
#ifdef SMN_USE_ASC_REDUCE
  return asc_reduce_add(v);
#else
#pragma unroll
  for (int o = kWarpSize / 2; o > 0; o >>= 1) v += asc_shfl_xor(v, o);
  return v;
#endif
}

// ---- exp: libm by default; -DSMN_FAST_EXP enables a branchless polynomial exp ----
// Range-reduce x = i*ln2 + f (|f|<=ln2/2), exp(f) via degree-7 poly, scale by 2^i.
// Softmax args are <= 0 (x - rowmax); err ~1e-6 (<< 0.2% requirement).
#ifdef SMN_FAST_EXP
__aicore__ inline float smn_expf(float x) {
  const float LOG2E = 1.4426950408889634f;
  const float C1 = 0.693145751953125f;       // ln2 hi
  const float C2 = 1.42860682030941723e-6f;  // ln2 lo
  float t = x * LOG2E;
  float r = t + 12582912.0f;                  // round-to-nearest (1.5*2^23 magic)
  r = r - 12582912.0f;
  int i = (int)r;
  float f = x - r * C1 - r * C2;
  float p = 0.00019875691f;
  p = p * f + 0.0013981999f;
  p = p * f + 0.0083333310f;
  p = p * f + 0.0416666620f;
  p = p * f + 0.1666666660f;
  p = p * f + 0.5000000000f;
  p = p * f + 1.0000000000f;
  p = p * f + 1.0000000000f;                  // exp(f)
  if (i < -126) return 0.0f;                  // underflow -> 0 (negligible term)
  union { float fv; int iv; } u;
  u.iv = (i + 127) << 23;                     // 2^i
  return p * u.fv;
}
#else
__aicore__ inline float smn_expf(float x) { return expf(x); }
#endif

#ifdef SMN_USE_HALF2_EXP
__aicore__ inline half2 smn_h2exp_centered(half2 x, float center) {
  // Subtract in FP32 before packing the two exponent arguments. Performing
  // __hsubx2 directly adds an avoidable FP16 subtraction rounding step.
  const float lo = __low2float(x) - center;
  const float hi = __high2float(x) - center;
  return h2exp(__floats2half2_rn(lo, hi));
}
#endif

#if defined(SMN_HALF2_REFINE) || defined(SMN_HALF2_REFINE_COMPACT)
__aicore__ inline float2 smn_half2_to_float2(half2 x) {
  float2 result;
  result.x = __low2float(x);
  result.y = __high2float(x);
  return result;
}

__aicore__ inline float2 smn_h2exp_refine_finite_args(float arg0, float arg1) {
  // Clamp finite centered differences to the fp16 floor before packing. A
  // smaller value would convert to -inf and make the residual refinement
  // evaluate 0 * inf. Conditional selects preserve NaN lanes.
  float2 args;
  args.x = arg0 < -65504.0f ? -65504.0f : arg0;
  args.y = arg1 < -65504.0f ? -65504.0f : arg1;
  const half2 packed_arg = __float22half2_rn(args);
  const half2 packed_exp = h2exp(packed_arg);
  const float residual0 = arg0 - __low2float(packed_arg);
  const float residual1 = arg1 - __high2float(packed_arg);
  float2 result;
  result.x = __low2float(packed_exp)
             * (1.0f + residual0 + 0.5f * residual0 * residual0);
  result.y = __high2float(packed_exp)
             * (1.0f + residual1 + 0.5f * residual1 * residual1);
  return result;
}

__aicore__ inline float2 smn_h2exp_centered_refined(float2 x, float center) {
  const float arg0 = x.x - center;
  const float arg1 = x.y - center;
  // A difference between two finite fp16 inputs can reach -131008. Packing a
  // centered argument below the finite fp16 floor would produce -inf; the
  // residual correction would then evaluate 0 * inf and turn a mathematically
  // underflowed exponential into NaN. The refinement helper conditionally
  // clamps each FP32 argument to the finite fp16 floor before packing.
  return smn_h2exp_refine_finite_args(arg0, arg1);
}

#ifdef SMN_HALF2_REFINE_COMPACT
__aicore__ inline half2 smn_h2exp_centered_refined_compact(half2 x, float center) {
  const float2 result = smn_h2exp_centered_refined(smn_half2_to_float2(x), center);
  return __floats2half2_rn(result.x, result.y);
}
#endif
#endif

// ================= VECTORIZED (cols % 8 == 0) =================
// cached single-pass: cols <= kCacheColumns
__global__ void smn_softmax_cached(const half* x, half* y, int64_t rows, int cols) {
  int lane = threadIdx.x & (kWarpSize - 1);
  int64_t gwarp = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  int64_t total = (int64_t)gridDim.x * (blockDim.x >> 5);
  int nv = cols >> 3;
#if defined(SMN_HALF2_REFINE)
  float2 buf[kLaneBufferElements / 2];
#elif defined(SMN_USE_HALF2_EXP)
  half2 buf[kLaneBufferElements / 2];
#else
  float buf[kLaneBufferElements];
#endif
  for (int64_t row = gwarp; row < rows; row += total) {
    const int4* xr = (const int4*)(x + row * (int64_t)cols);
    int4* yr = (int4*)(y + row * (int64_t)cols);
    float m = -3.0e38f;
    int cnt = 0;
    for (int j = lane; j < nv; j += kWarpSize) {
      int4 v = xr[j];
#if defined(SMN_HALF2_REFINE)
      half2* h2 = (half2*)&v;
#pragma unroll
      for (int k = 0; k < 4; k++) {
        const float2 pair = smn_half2_to_float2(h2[k]);
        buf[(cnt << 2) + k] = pair;
        if (pair.x > m) m = pair.x;
        if (pair.y > m) m = pair.y;
      }
#elif defined(SMN_USE_HALF2_EXP)
      half2* h2 = (half2*)&v;
#pragma unroll
      for (int k = 0; k < 4; k++) {
        const half2 pair = h2[k];
        buf[(cnt << 2) + k] = pair;
        const float lo = __low2float(pair);
        const float hi = __high2float(pair);
        if (lo > m) m = lo;
        if (hi > m) m = hi;
      }
#else
      half* h = (half*)&v;
#pragma unroll
      for (int k = 0; k < 8; k++) {
        float f = (float)h[k];
        buf[(cnt << 3) + k] = f;
        if (f > m) m = f;
      }
#endif
      cnt++;
    }
    m = smn_warp_max(m);
    float s = 0.f;
#if defined(SMN_HALF2_REFINE)
    for (int c = 0; c < (cnt << 2); c++) {
      const float2 e = smn_h2exp_centered_refined(buf[c], m);
      buf[c] = e;
      s += e.x + e.y;
    }
#elif defined(SMN_HALF2_REFINE_COMPACT)
    for (int c = 0; c < (cnt << 2); c++) {
      const half2 e = smn_h2exp_centered_refined_compact(buf[c], m);
      buf[c] = e;
      s += __low2float(e) + __high2float(e);
    }
#elif defined(SMN_USE_HALF2_EXP)
    for (int c = 0; c < (cnt << 2); c++) {
      const half2 e = smn_h2exp_centered(buf[c], m);
      buf[c] = e;
      s += __low2float(e) + __high2float(e);
    }
#else
    for (int c = 0; c < cnt; c++) {
#pragma unroll
      for (int k = 0; k < 8; k++) {
        float e = smn_expf(buf[(c << 3) + k] - m);
        buf[(c << 3) + k] = e;
        s += e;
      }
    }
#endif
    s = smn_warp_sum(s);
    float inv = 1.0f / s;
    cnt = 0;
    for (int j = lane; j < nv; j += kWarpSize) {
      int4 v;
#if defined(SMN_HALF2_REFINE)
      half2* h2 = (half2*)&v;
#pragma unroll
      for (int k = 0; k < 4; k++) {
        const float2 e = buf[(cnt << 2) + k];
        h2[k] = __floats2half2_rn(e.x * inv, e.y * inv);
      }
#elif defined(SMN_USE_HALF2_EXP)
      half2* h2 = (half2*)&v;
#pragma unroll
      for (int k = 0; k < 4; k++) {
        const half2 e = buf[(cnt << 2) + k];
        h2[k] = __floats2half2_rn(__low2float(e) * inv, __high2float(e) * inv);
      }
#else
      half* h = (half*)&v;
#pragma unroll
      for (int k = 0; k < 8; k++) h[k] = (half)(buf[(cnt << 3) + k] * inv);
#endif
      yr[j] = v;
      cnt++;
    }
  }
}

// NOTE: a cached path for 4097..8192 (buf[256]) was tried and rejected — buf[256] is
// 1KB/thread, over the per-thread local-mem limit (507035), at any block size. Cached is
// hard-capped at buf[128] = cols<=4096; larger cols must use the streaming path below.

// online streaming: cols > kCacheColumns (2 reads + 1 write, no big buffer)
__global__ void smn_softmax_stream(const half* x, half* y, int64_t rows, int cols) {
  int lane = threadIdx.x & (kWarpSize - 1);
  int64_t gwarp = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  int64_t total = (int64_t)gridDim.x * (blockDim.x >> 5);
  int nv = cols >> 3;
  for (int64_t row = gwarp; row < rows; row += total) {
    const int4* xr = (const int4*)(x + row * (int64_t)cols);
    int4* yr = (int4*)(y + row * (int64_t)cols);
#if defined(SMN_HALF2_REFINE)
    float m = -3.0e38f;
    for (int j = lane; j < nv; j += kWarpSize) {
      int4 v = xr[j];
      half2* h2 = (half2*)&v;
#pragma unroll
      for (int k = 0; k < 4; k++) {
        const float2 pair = smn_half2_to_float2(h2[k]);
        if (pair.x > m) m = pair.x;
        if (pair.y > m) m = pair.y;
      }
    }
    m = smn_warp_max(m);
    float s = 0.f;
    for (int j = lane; j < nv; j += kWarpSize) {
      int4 v = xr[j];
      half2* h2 = (half2*)&v;
#pragma unroll
      for (int k = 0; k < 4; k++) {
        const float2 e = smn_h2exp_centered_refined(smn_half2_to_float2(h2[k]), m);
        s += e.x + e.y;
      }
    }
    s = smn_warp_sum(s);
    float inv = 1.0f / s;
    for (int j = lane; j < nv; j += kWarpSize) {
      int4 v = xr[j];
      half2* h2 = (half2*)&v;
      int4 o;
      half2* oh2 = (half2*)&o;
#pragma unroll
      for (int k = 0; k < 4; k++) {
        const float2 e = smn_h2exp_centered_refined(smn_half2_to_float2(h2[k]), m);
        oh2[k] = __floats2half2_rn(e.x * inv, e.y * inv);
      }
      yr[j] = o;
    }
#elif defined(SMN_USE_HALF2_EXP)
    float m = -3.0e38f;
    for (int j = lane; j < nv; j += kWarpSize) {
      int4 v = xr[j];
      half2* h2 = (half2*)&v;
#pragma unroll
      for (int k = 0; k < 4; k++) {
        const float lo = __low2float(h2[k]);
        const float hi = __high2float(h2[k]);
        if (lo > m) m = lo;
        if (hi > m) m = hi;
      }
    }
    m = smn_warp_max(m);
    float s = 0.f;
    for (int j = lane; j < nv; j += kWarpSize) {
      int4 v = xr[j];
      half2* h2 = (half2*)&v;
#pragma unroll
      for (int k = 0; k < 4; k++) {
#ifdef SMN_HALF2_REFINE_COMPACT
        const half2 e = smn_h2exp_centered_refined_compact(h2[k], m);
#else
        const half2 e = smn_h2exp_centered(h2[k], m);
#endif
        s += __low2float(e) + __high2float(e);
      }
    }
    s = smn_warp_sum(s);
    float inv = 1.0f / s;
    for (int j = lane; j < nv; j += kWarpSize) {
      int4 v = xr[j];
      half2* h2 = (half2*)&v;
      int4 o;
      half2* oh2 = (half2*)&o;
#pragma unroll
      for (int k = 0; k < 4; k++) {
#ifdef SMN_HALF2_REFINE_COMPACT
        const half2 e = smn_h2exp_centered_refined_compact(h2[k], m);
#else
        const half2 e = smn_h2exp_centered(h2[k], m);
#endif
        oh2[k] = __floats2half2_rn(__low2float(e) * inv, __high2float(e) * inv);
      }
      yr[j] = o;
    }
#elif defined(SMN_STREAM_THREE_PASS)
    float m = -3.0e38f;
    for (int j = lane; j < nv; j += kWarpSize) {
      int4 v = xr[j];
      half* h = (half*)&v;
#pragma unroll
      for (int k = 0; k < 8; k++) {
        float f = (float)h[k];
        m = f > m ? f : m;
      }
    }
    m = smn_warp_max(m);
    float s = 0.f;
    for (int j = lane; j < nv; j += kWarpSize) {
      int4 v = xr[j];
      half* h = (half*)&v;
#pragma unroll
      for (int k = 0; k < 8; k++) s += smn_expf((float)h[k] - m);
    }
    s = smn_warp_sum(s);
    float inv = 1.0f / s;
    for (int j = lane; j < nv; j += kWarpSize) {
      int4 v = xr[j];
      half* h = (half*)&v;
      int4 o;
      half* oh = (half*)&o;
#pragma unroll
      for (int k = 0; k < 8; k++) oh[k] = (half)(smn_expf((float)h[k] - m) * inv);
      yr[j] = o;
    }
#else
    float m = -3.0e38f, s = 0.f;
    for (int j = lane; j < nv; j += kWarpSize) {
      int4 v = xr[j];
      half* h = (half*)&v;
#pragma unroll
      for (int k = 0; k < 8; k++) {
        float f = (float)h[k];
        float mn = f > m ? f : m;
        s = s * smn_expf(m - mn) + smn_expf(f - mn);
        m = mn;
      }
    }
    float gm = smn_warp_max(m);
    s = s * smn_expf(m - gm);
    float gs = smn_warp_sum(s);
    float inv = 1.0f / gs;
    for (int j = lane; j < nv; j += kWarpSize) {
      int4 v = xr[j];
      half* h = (half*)&v;
      int4 o;
      half* oh = (half*)&o;
#pragma unroll
      for (int k = 0; k < 8; k++) oh[k] = (half)(smn_expf((float)h[k] - gm) * inv);
      yr[j] = o;
    }
#endif
  }
}

#ifdef SMN_USE_BLOCK_PER_ROW
// Low-row-count long softmax. The warp-per-row path cannot expose enough warps
// when rows is small; one block per row restores parallelism. Keeping compact
// FP16 exponentials in UB makes HBM traffic one input read plus one output
// write, while the FP32 residual correction preserves the production error
// gate used by the warp path.
template<int block_size, int cache_cols>
__global__ void smn_softmax_block_cached(const half* x, half* y, int64_t rows, int cols) {
  static_assert(block_size % kWarpSize == 0, "");
  static_assert(block_size <= 1024, "");
  static_assert(cache_cols % 8 == 0, "");
  using BlockReduce = aclcub::BlockReduce<float, block_size>;
  __ubuf__ typename BlockReduce::TempStorage reduce_storage;
  __ubuf__ __attribute__((aligned(16))) int4 cached[cache_cols / 8];
  const int tid = threadIdx.x;
  const int nv = cols >> 3;
  int reduce_iteration = 0;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const int4* xr = reinterpret_cast<const int4*>(x + row * static_cast<int64_t>(cols));
    int4* yr = reinterpret_cast<int4*>(y + row * static_cast<int64_t>(cols));
    float local_max = -3.0e38f;
    for (int j = tid; j < nv; j += block_size) {
      const int4 packed = xr[j];
      cached[j] = packed;
      const half2* pairs = reinterpret_cast<const half2*>(&packed);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const float lo = __low2float(pairs[k]);
        const float hi = __high2float(pairs[k]);
        if (lo > local_max) local_max = lo;
        if (hi > local_max) local_max = hi;
      }
    }
    const float row_max =
        BlockReduce(reduce_storage).AllReduce(local_max, aclcub::Max{}, reduce_iteration++);

    float local_sum = 0.0f;
    for (int j = tid; j < nv; j += block_size) {
      int4 packed = cached[j];
      half2* pairs = reinterpret_cast<half2*>(&packed);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
#ifdef SMN_HALF2_REFINE_COMPACT
        const half2 exponential = smn_h2exp_centered_refined_compact(pairs[k], row_max);
#else
        const half2 exponential = smn_h2exp_centered(pairs[k], row_max);
#endif
        pairs[k] = exponential;
        local_sum += __low2float(exponential) + __high2float(exponential);
      }
      cached[j] = packed;
    }
    const float row_sum =
        BlockReduce(reduce_storage).AllReduce(local_sum, aclcub::Sum{}, reduce_iteration++);
    const float inv_sum = 1.0f / row_sum;

    for (int j = tid; j < nv; j += block_size) {
      int4 packed = cached[j];
      half2* pairs = reinterpret_cast<half2*>(&packed);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        pairs[k] = __floats2half2_rn(__low2float(pairs[k]) * inv_sum,
                                     __high2float(pairs[k]) * inv_sum);
      }
      yr[j] = packed;
    }
  }
}
#endif

// ================= SCALAR FALLBACK (cols % 8 != 0, unaligned rows) =================
__global__ void smn_softmax_cached_s(const half* x, half* y, int64_t rows, int cols) {
  int lane = threadIdx.x & (kWarpSize - 1);
  int64_t gwarp = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  int64_t total = (int64_t)gridDim.x * (blockDim.x >> 5);
  float buf[kLaneBufferElements];
  for (int64_t row = gwarp; row < rows; row += total) {
    const half* xr = x + row * (int64_t)cols;
    half* yr = y + row * (int64_t)cols;
    float m = -3.0e38f;
    int cnt = 0;
    for (int64_t c = lane; c < cols; c += kWarpSize) {
      float f = (float)xr[c];
      buf[cnt++] = f;
      if (f > m) m = f;
    }
    m = smn_warp_max(m);
    float s = 0.f;
    for (int i = 0; i < cnt; i++) {
      float e = smn_expf(buf[i] - m);
      buf[i] = e;
      s += e;
    }
    s = smn_warp_sum(s);
    float inv = 1.0f / s;
    cnt = 0;
    for (int64_t c = lane; c < cols; c += kWarpSize) {
      yr[c] = (half)(buf[cnt++] * inv);
    }
  }
}

__global__ void smn_softmax_stream_s(const half* x, half* y, int64_t rows, int cols) {
  int lane = threadIdx.x & (kWarpSize - 1);
  int64_t gwarp = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  int64_t total = (int64_t)gridDim.x * (blockDim.x >> 5);
  for (int64_t row = gwarp; row < rows; row += total) {
    const half* xr = x + row * (int64_t)cols;
    half* yr = y + row * (int64_t)cols;
#ifdef SMN_STREAM_THREE_PASS
    float m = -3.0e38f;
    for (int64_t c = lane; c < cols; c += kWarpSize) {
      float f = (float)xr[c];
      m = f > m ? f : m;
    }
    m = smn_warp_max(m);
    float s = 0.f;
    for (int64_t c = lane; c < cols; c += kWarpSize) {
      s += smn_expf((float)xr[c] - m);
    }
    s = smn_warp_sum(s);
    float inv = 1.0f / s;
    for (int64_t c = lane; c < cols; c += kWarpSize) {
      yr[c] = (half)(smn_expf((float)xr[c] - m) * inv);
    }
#else
    float m = -3.0e38f, s = 0.f;
    for (int64_t c = lane; c < cols; c += kWarpSize) {
      float f = (float)xr[c];
      float mn = f > m ? f : m;
      s = s * smn_expf(m - mn) + smn_expf(f - mn);
      m = mn;
    }
    float gm = smn_warp_max(m);
    s = s * smn_expf(m - gm);
    float gs = smn_warp_sum(s);
    float inv = 1.0f / gs;
    for (int64_t c = lane; c < cols; c += kWarpSize) {
      yr[c] = (half)(smn_expf((float)xr[c] - gm) * inv);
    }
#endif
  }
}

// ---- host dispatch: 1 warp per row, grid-strided, capped grid ----
static inline int smn_vector_core_count() {
  // The benchmark process pins one device for its lifetime. Cache the
  // attribute after the first launch so launch latency does not include a
  // runtime device query.
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
    return SMN_FALLBACK_AIV_COUNT;
  }();
  return cached_count;
}

static inline bool smn_is_aligned16(const void* pointer) {
  return (reinterpret_cast<uintptr_t>(pointer) & 15U) == 0;
}

inline aclError Launch(
    aclrtStream s, const half* x, half* y, int64_t rows, int cols, int blk,
    int grid_cap) {
  if (s == nullptr || x == nullptr || y == nullptr || rows <= 0 || cols <= 0
      || rows > std::numeric_limits<int64_t>::max() / cols
      || grid_cap < 0) {
    return ACL_ERROR_RT_PARAM_INVALID;
  }
  if (blk != 0 && (blk < kWarpSize || blk > 1024 || blk % kWarpSize != 0)) {
    return ACL_ERROR_RT_PARAM_INVALID;
  }
  if (grid_cap == 0) grid_cap = smn_vector_core_count() * 32;
  const bool vector_aligned =
      cols % 8 == 0 && smn_is_aligned16(x) && smn_is_aligned16(y);
#ifdef SMN_USE_BLOCK_PER_ROW
  const int vector_packets = cols > 0 ? cols >> 3 : 0;
  const int packet_loops_at_1024 = (vector_packets + 1023) / 1024;
  // A warp-per-row launch needs enough independent rows to amortize its
  // per-row streaming work without creating a lightly populated final AIV
  // wave.  The crossover grows with the number of 1024-thread packet cohorts:
  // q=1 needs about 20 warps/AIV, q=2 about 24 warps/AIV on the probed target.
  // Derive the row boundary from the queried AIV count rather than baking in a
  // device-specific row count (1120/1344 on a 56-AIV 950PR).
  const int warp_rows_per_aiv =
      16 + 4 * (packet_loops_at_1024 > 0 ? packet_loops_at_1024 : 1);
  const int64_t warp_ready_rows =
      static_cast<int64_t>(smn_vector_core_count()) * warp_rows_per_aiv;
  // The B1024 cached kernel starts winning once each AIV sees roughly 96
  // 16-byte packet-loop cohorts. This captures both row count and row width:
  // q=4 crosses near 1344 rows; q=3 crosses near 1792 rows on 56 AIV.
  const int64_t block_1024_work_threshold =
      static_cast<int64_t>(smn_vector_core_count()) * 96;
  const bool block_shape =
#ifdef SMN_FORCE_BLOCK_THREADS
      rows <= SMN_BLOCK_ROW_MAX;
#else
      packet_loops_at_1024 >= 3 || rows < warp_ready_rows;
#endif
  if (cols > kCacheColumns && cols <= 32768 && vector_aligned && block_shape) {
    int64_t blocks = rows < grid_cap ? rows : grid_cap;
    if (blocks < 1) blocks = 1;
#ifdef SMN_FORCE_BLOCK_THREADS
    if (cols <= 8192) {
      smn_softmax_block_cached<SMN_FORCE_BLOCK_THREADS, 8192>
          <<<(int)blocks, SMN_FORCE_BLOCK_THREADS, 0, s>>>(x, y, rows, cols);
    } else if (cols <= 16384) {
      smn_softmax_block_cached<SMN_FORCE_BLOCK_THREADS, 16384>
          <<<(int)blocks, SMN_FORCE_BLOCK_THREADS, 0, s>>>(x, y, rows, cols);
    } else {
      smn_softmax_block_cached<SMN_FORCE_BLOCK_THREADS, 32768>
          <<<(int)blocks, SMN_FORCE_BLOCK_THREADS, 0, s>>>(x, y, rows, cols);
    }
#else
    const int64_t rows_for_1024 =
        (block_1024_work_threshold + packet_loops_at_1024 - 1)
        / packet_loops_at_1024;
    const bool use_512_threads = rows < rows_for_1024;
    if (cols <= 8192) {
      smn_softmax_block_cached<512, 8192><<<(int)blocks, 512, 0, s>>>(x, y, rows,
                                                                     cols);
    } else if (cols <= 16384) {
      smn_softmax_block_cached<512, 16384><<<(int)blocks, 512, 0, s>>>(x, y, rows,
                                                                      cols);
    } else if (use_512_threads) {
      smn_softmax_block_cached<512, 32768><<<(int)blocks, 512, 0, s>>>(x, y, rows,
                                                                      cols);
    } else {
      smn_softmax_block_cached<1024, 32768>
          <<<(int)blocks, 1024, 0, s>>>(x, y, rows, cols);
    }
#endif
    return aclrtPeekAtLastError(static_cast<aclrtLastErrLevel>(0));
  }
#endif
  if (blk <= 0) {                                  // auto: fill vector unit vs keep enough blocks
    blk = (cols <= kCacheColumns) ? 1024 : 512;    // small/mid cols: max warps/block;
    int wpb = blk >> 5;                            // big cols (few rows): 512 keeps more blocks
    while (blk > 256 && rows < (int64_t)wpb * 16) { blk >>= 1; wpb = blk >> 5; }  // tiny rows -> spread
  }
  int warps_per_blk = blk >> 5;
  int64_t blocks = 1 + (rows - 1) / warps_per_blk;
  if (blocks > grid_cap) blocks = grid_cap;
  if (blocks < 1) blocks = 1;
  if (cols <= kCacheColumns) {
    if (vector_aligned) smn_softmax_cached<<<(int)blocks, blk, 0, s>>>(x, y, rows, cols);
    else         smn_softmax_cached_s<<<(int)blocks, blk, 0, s>>>(x, y, rows, cols);
  } else {
    if (vector_aligned) smn_softmax_stream<<<(int)blocks, blk, 0, s>>>(x, y, rows, cols);
    else         smn_softmax_stream_s<<<(int)blocks, blk, 0, s>>>(x, y, rows, cols);
  }
  return aclrtPeekAtLastError(static_cast<aclrtLastErrLevel>(0));
}

}  // namespace
}  // namespace softmax_fp16
}  // namespace detail
}  // namespace v1
}  // namespace dav_c310
}  // namespace target
}  // namespace ascify

#endif  // ASCIFY_TARGET_DAV_C310_DETAIL_SOFTMAX_FP16_IMPL_HPP_
