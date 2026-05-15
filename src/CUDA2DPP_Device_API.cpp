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
  // Bfloat16 type
  m["nv_bfloat16"]    = {"acl_bfloat16",    CONV_DEVICE_TYPE, API_RUNTIME, 0};
  m["nv_bfloat16_2"]  = {"acl_bfloat162",   CONV_DEVICE_TYPE, API_RUNTIME, 0};
  return m;
}();

// Map of CUDA Device intrinsics/functions/macros to DPP equivalents
const std::map<llvm::StringRef, dppCounter> CUDA_DEVICE_FUNCTION_MAP = [] {
  std::map<llvm::StringRef, dppCounter> m;
  // Math constants
  m["CUDART_INF_F"]      = {"ASCRT_INF_F",     CONV_NUMERIC_LITERAL, API_RUNTIME, 0};
  m["CUDART_INF"]        = {"ASCRT_INF",       CONV_NUMERIC_LITERAL, API_RUNTIME, 0};
  m["CUDART_NAN_F"]      = {"ASCRT_NAN_F",     CONV_NUMERIC_LITERAL, API_RUNTIME, 0};
  m["CUDART_NAN"]        = {"ASCRT_NAN",       CONV_NUMERIC_LITERAL, API_RUNTIME, 0};
  m["CUDART_MAX_NORMAL_F"] = {"ASCRT_MAX_NORMAL_F", CONV_NUMERIC_LITERAL, API_RUNTIME, 0};
  m["CUDART_MIN_NORMAL_F"] = {"ASCRT_MIN_NORMAL_F", CONV_NUMERIC_LITERAL, API_RUNTIME, 0};
  m["CUDART_NEG_ZERO_F"] = {"ASCRT_NEG_ZERO_F", CONV_NUMERIC_LITERAL, API_RUNTIME, 0};
  m["CUDART_ZERO_F"]     = {"ASCRT_ZERO_F",    CONV_NUMERIC_LITERAL, API_RUNTIME, 0};
  // Warp functions (keep same on SIMT架构, but track)
  m["__shfl_down_sync"]  = {"__shfl_down_sync",  CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__shfl_up_sync"]    = {"__shfl_up_sync",    CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__shfl_xor_sync"]   = {"__shfl_xor_sync",   CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__shfl_sync"]       = {"__shfl_sync",       CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__syncthreads"]     = {"__syncthreads",     CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__syncthreads_and"] = {"__syncthreads_and", CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__syncthreads_or"]  = {"__syncthreads_or",  CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["__syncthreads_count"] = {"__syncthreads_count", CONV_DEVICE_FUNC, API_RUNTIME, 0};
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
  // Trap (keep, no DPP equivalent)
  m["__trap"]             = {"__trap",            CONV_DEVICE_FUNC, API_RUNTIME, UNSUPPORTED};
  // blockIdx/threadIdx dim3 fields (keep same on SIMT)
  m["blockIdx"]           = {"blockIdx",          CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["threadIdx"]          = {"threadIdx",         CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["blockDim"]           = {"blockDim",          CONV_DEVICE_FUNC, API_RUNTIME, 0};
  m["gridDim"]            = {"gridDim",           CONV_DEVICE_FUNC, API_RUNTIME, 0};
  // CUB namespace
  m["cub"]                = {"aclcub",            CONV_DEVICE_FUNC, API_CUB, 0};
  return m;
}();