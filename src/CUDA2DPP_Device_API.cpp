/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "CUDA2DPP.h"

// Map of CUDA Device type names to DPP type names
const std::map<llvm::StringRef, dppCounter> CUDA_DEVICE_TYPE_NAME_MAP = [] {
  std::map<llvm::StringRef, dppCounter> m;
  // Bfloat16 type (verified against CANN 9.0.0: Ascend uses bfloat16_t, not acl_bfloat16)
  m["nv_bfloat16"]    = {"bfloat16_t",       CONV_DEVICE_TYPE, API_RUNTIME, 0};
  m["nv_bfloat16_2"]  = {"bfloat16x2_t",     CONV_DEVICE_TYPE, API_RUNTIME, 0};
  return m;
}();

// Map of CUDA Device intrinsics/functions/macros to DPP equivalents
const std::map<llvm::StringRef, dppCounter> CUDA_DEVICE_FUNCTION_MAP = [] {
  std::map<llvm::StringRef, dppCounter> m;
  // Math constants (verified: ASCRT_* macros not available in CANN 9.0.0; use standard C limits)
  m["CUDART_INF_F"]      = {"ASCRT_INF_F",     CONV_NUMERIC_LITERAL, API_RUNTIME, 0};
  m["CUDART_INF"]        = {"",                CONV_NUMERIC_LITERAL, API_RUNTIME, UNSUPPORTED};
  m["CUDART_NAN_F"]      = {"ASCRT_NAN_F",    CONV_NUMERIC_LITERAL, API_RUNTIME, 0};
  m["CUDART_NAN"]        = {"",                CONV_NUMERIC_LITERAL, API_RUNTIME, UNSUPPORTED};
  m["CUDART_MAX_NORMAL_F"] = {"__FLT_MAX__", CONV_NUMERIC_LITERAL, API_RUNTIME, 0};
  m["CUDART_MIN_NORMAL_F"] = {"__FLT_MIN__", CONV_NUMERIC_LITERAL, API_RUNTIME, 0};
  m["CUDART_NEG_ZERO_F"] = {"(-0.0f)",    CONV_NUMERIC_LITERAL, API_RUNTIME, 0};
  m["CUDART_ZERO_F"]     = {"(0.0f)",     CONV_NUMERIC_LITERAL, API_RUNTIME, 0};
  // Warp shuffle functions -> official Ascend SIMT API (CANN 9.0.0, simt_api/device_warp_functions.h)
  // Signature change: __shfl_xxx_sync(mask, val, offset) -> asc_shfl_xxx(val, offset, width)
  m["__shfl_down_sync"]  = {"asc_shfl_down",    CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__shfl_up_sync"]    = {"asc_shfl_up",      CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__shfl_xor_sync"]   = {"asc_shfl_xor",     CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__shfl_sync"]       = {"asc_shfl",         CONV_DEVICE_FUNC, API_RUNTIME, 0};
  // Sync intrinsics -> official Ascend SIMT API (CANN 9.0.0, simt_api/device_sync_functions.h)
  // Note: asc_syncthreads currently only supported in SIMD+SIMT mixed mode (not pure SIMT)
  m["__syncthreads"]     = {"asc_syncthreads",     CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__syncthreads_and"] = {"asc_syncthreads_and", CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__syncthreads_or"]  = {"asc_syncthreads_or",  CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__syncthreads_count"] = {"asc_syncthreads_count", CONV_DEVICE_FUNC, API_RUNTIME, 0};
  // Memory fence -> official Ascend SIMT API
  m["__threadfence_block"] = {"asc_threadfence_block", CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__threadfence"]       = {"asc_threadfence",       CONV_DEVICE_FUNC, API_RUNTIME, 0};
  // Fast math intrinsics (keep same on SIMT)
  m["__expf"]            = {"__expf",            CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__logf"]            = {"__logf",            CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__log2f"]           = {"__log2f",           CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__log10f"]          = {"__log10f",          CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__fdividef"]        = {"__fdividef",        CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__fmul_rn"]         = {"__fmul_rn",         CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__fmaf_rn"]          = {"__fmaf_rn",         CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__fadd_rn"]          = {"__fadd_rn",         CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__fsub_rn"]          = {"__fsub_rn",         CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__frsqrt_rn"]        = {"__frsqrt_rn",       CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__fsqrt_rn"]         = {"__fsqrt_rn",        CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__frcp_rn"]          = {"__frcp_rn",         CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__float2half_rn"]    = {"__float2half_rn",    CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__half2float"]       = {"__half2float",       CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__float2half2_rn"]   = {"__float2half2_rn",   CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__half22float2"]     = {"__half22float2",     CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__int_as_float"]     = {"__int_as_float",     CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__float_as_int"]     = {"__float_as_int",     CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__longlong_as_double"] = {"__longlong_as_double", CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__double_as_longlong"] = {"__double_as_longlong", CONV_DEVICE_FUNC, API_RUNTIME, 0};
  // Trap -> __trap (CCE directly supports __trap)
  m["__trap"]             = {"__trap",          CONV_DEVICE_FUNC, API_RUNTIME, 0};
  // Math function type conversion: double-precision -> single-precision (A5 vector core does not support double)
  m["exp"]                = {"expf",              CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["log"]                = {"logf",              CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["sqrt"]               = {"sqrtf",             CONV_DEVICE_FUNC, API_RUNTIME, 0};
  // Type conversion: double -> float (A5 vector core does not support double)
  m["double"]             = {"float",             CONV_DEVICE_TYPE, API_RUNTIME, 0};
  // std::max/min -> Ascend SIMT equivalents
  m["std::max"]           = {"llmax",             CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["std::min"]           = {"llmin",             CONV_DEVICE_FUNC, API_RUNTIME, 0};
  // blockIdx/threadIdx dim3 fields (keep same on SIMT)
  m["blockIdx"]           = {"blockIdx",          CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["threadIdx"]          = {"threadIdx",         CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["blockDim"]           = {"blockDim",          CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["gridDim"]            = {"gridDim",           CONV_DEVICE_FUNC, API_RUNTIME, 0};
  // CUB namespace -> aclcub (note: acl_cub library not yet available in CANN 9.0.0)
  m["cub"]                = {"aclcub",            CONV_DEVICE_FUNC, API_CUB, 0};

  // ---------- CUDA qualifier replacements (note: may need Clang-level Attr rewrite) ----------
  // __device__ / __forceinline__ have no Ascend C equivalent; kernel functions use __global__ only
  m["__device__"]         = {"__aicore__",          CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__forceinline__"]    = {"inline",           CONV_DEVICE_FUNC, API_RUNTIME, 0};
  // __shared__ -> __ubuf__ (Ascend C Unified Buffer qualifier, defined as empty macro in cpu_stub.hpp)
  m["__shared__"]         = {"__ubuf__",         CONV_DEVICE_FUNC, API_RUNTIME, 0};
  return m;
}();