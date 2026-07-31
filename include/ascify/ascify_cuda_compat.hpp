#ifndef ASCIFY_ASCIFY_CUDA_COMPAT_HPP
#define ASCIFY_ASCIFY_CUDA_COMPAT_HPP

#include <algorithm>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <type_traits>

#include <acl/acl.h>
#include <simt_api/asc_bf16.h>
#include <simt_api/device_functions.h>
#include <simt_api/device_sync_functions.h>
#include <simt_api/device_warp_functions.h>
#include <simt_api/math_functions.h>

// CUDA spells alignment as `__align__(N)`. Keeping the argument in a macro
// preserves both dependent expressions such as sizeof(T) and the declaration
// position selected by the source.
#ifndef ASCIFY_ALIGN
#define ASCIFY_ALIGN(bytes) __attribute__((aligned(bytes)))
#endif

#ifndef ASCIFY_FORCEINLINE
#define ASCIFY_FORCEINLINE inline __attribute__((always_inline))
#endif

namespace ascify {

// dav-c310-vec exposes a full 32-lane add reduction for these scalar types.
// Ascify only emits this helper after proving the source is the equivalent
// full-mask, full-width XOR butterfly; all other cases retain the shuffle loop.
__aicore__ ASCIFY_FORCEINLINE float warp_reduce_add(float value) {
  return asc_reduce_add(value);
}

__aicore__ ASCIFY_FORCEINLINE int32_t warp_reduce_add(int32_t value) {
  return asc_reduce_add(value);
}

__aicore__ ASCIFY_FORCEINLINE uint32_t warp_reduce_add(uint32_t value) {
  return asc_reduce_add(value);
}

struct cudaFuncAttributes {
  size_t sharedSizeBytes = 0;
};

enum cudaFuncAttribute {
  cudaFuncAttributeMaxDynamicSharedMemorySize = 0
};

// Keep the CUDA argument order and result width at the translated call site.
// aclrtGetDeviceInfo instead takes device first and writes an int64_t.
inline aclError cudaDeviceGetAttribute(int* value, aclrtDevAttr attribute, int device) {
  if (value == nullptr || device < 0) { return ACL_ERROR_RT_PARAM_INVALID; }

  int64_t raw_value = 0;
  const aclError status =
      aclrtGetDeviceInfo(static_cast<uint32_t>(device), attribute, &raw_value);
  if (status == ACL_SUCCESS) { *value = static_cast<int>(raw_value); }
  return status;
}

// CUDA's API has no arguments. CANN requires the error scope explicitly.
inline aclError cudaPeekAtLastError() {
  return aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
}

// CANN has no direct CUDA occupancy counterpart. This approximate estimate
// reports active blocks per vector core, bounded by threads and dynamic UB.
// Static UB, register pressure, and a hardware resident-block cap are not
// exposed by the beta3 runtime and therefore are not represented here.
// The kernel parameter is intentionally retained so existing CUDA call sites,
// including specialized kernel templates, remain source compatible.
template<typename Kernel>
inline aclError cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    int* active_blocks, Kernel, int block_size, size_t dynamic_ub_bytes) {
  if (active_blocks == nullptr || block_size <= 0) {
    return ACL_ERROR_RT_PARAM_INVALID;
  }

  int32_t device = 0;
  aclError status = aclrtGetDevice(&device);
  if (status != ACL_SUCCESS) { return status; }

  int64_t max_threads = 0;
  status = aclrtGetDeviceInfo(static_cast<uint32_t>(device),
                              ACL_DEV_ATTR_MAX_THREAD_PER_VECTOR_CORE, &max_threads);
  if (status != ACL_SUCCESS) { return status; }

  int64_t blocks = max_threads / block_size;
  if (dynamic_ub_bytes != 0) {
    int64_t ub_bytes = 0;
    status = aclrtGetDeviceInfo(static_cast<uint32_t>(device),
                                ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE, &ub_bytes);
    if (status != ACL_SUCCESS) { return status; }
    const int64_t ub_blocks =
        ub_bytes > 0 ? ub_bytes / static_cast<int64_t>(dynamic_ub_bytes) : 0;
    if (ub_blocks < blocks) { blocks = ub_blocks; }
  }

  if (blocks < 0) { blocks = 0; }
  if (blocks > INT_MAX) { blocks = INT_MAX; }
  *active_blocks = static_cast<int>(blocks);
  return ACL_SUCCESS;
}

// beta3 does not expose CUDA-style per-kernel static/dynamic UB metadata.
// Static shared usage is conservatively reported as zero; setting the dynamic
// limit is a successful no-op because launch-time UB size remains authoritative.
template<typename Kernel>
inline aclError cudaFuncGetAttributes(cudaFuncAttributes* attributes, Kernel) {
  if (attributes == nullptr) { return ACL_ERROR_RT_PARAM_INVALID; }
  attributes->sharedSizeBytes = 0;
  return ACL_SUCCESS;
}

template<typename Kernel>
inline aclError cudaFuncSetAttribute(Kernel, cudaFuncAttribute, int) {
  return ACL_SUCCESS;
}

namespace detail {

__aicore__ ASCIFY_FORCEINLINE void requireFullWarpMask(uint32_t mask) {
  // Ascend beta3 exposes unmasked asc_shfl* operations. Trap instead of
  // silently miscompiling a CUDA partial-mask shuffle.
  if (mask != UINT32_MAX) { __builtin_trap(); }
}

}  // namespace detail

__aicore__ ASCIFY_FORCEINLINE void syncwarp(uint32_t mask = UINT32_MAX) {
  // beta3 warps execute in lockstep and expose no warp-only barrier. A full
  // warp synchronization is therefore a no-op; a partial mask is unsupported.
  detail::requireFullWarpMask(mask);
}

template<typename T>
__aicore__ ASCIFY_FORCEINLINE T shfl_sync(uint32_t mask, T value, int source_lane,
                                          int width = 32) {
  detail::requireFullWarpMask(mask);
  return asc_shfl(value, source_lane, width);
}

template<typename T>
__aicore__ ASCIFY_FORCEINLINE T shfl_up_sync(uint32_t mask, T value, unsigned int delta,
                                             int width = 32) {
  detail::requireFullWarpMask(mask);
  return asc_shfl_up(value, delta, width);
}

template<typename T>
__aicore__ ASCIFY_FORCEINLINE T shfl_down_sync(uint32_t mask, T value, unsigned int delta,
                                               int width = 32) {
  detail::requireFullWarpMask(mask);
  return asc_shfl_down(value, delta, width);
}

template<typename T>
__aicore__ ASCIFY_FORCEINLINE T shfl_xor_sync(uint32_t mask, T value, int lane_mask,
                                              int width = 32) {
  detail::requireFullWarpMask(mask);
  return asc_shfl_xor(value, lane_mask, width);
}

__aicore__ ASCIFY_FORCEINLINE float fdividef(float numerator, float denominator) {
  return ::fdividef(numerator, denominator);
}

}  // namespace ascify

#endif  // ASCIFY_ASCIFY_CUDA_COMPAT_HPP
