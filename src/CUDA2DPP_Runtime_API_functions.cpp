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

using SEC = runtime::CUDA_RUNTIME_API_SECTIONS;

// Map of all CUDA Runtime API functions
const std::map<llvm::StringRef, dppCounter> CUDA_RUNTIME_FUNCTION_MAP = [] {
  std::map<llvm::StringRef,  dppCounter> m;
  // Memory management
  m["cudaMalloc"]                                              = {"aclrtMalloc",                                            CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["cudaFree"]                                                = {"aclrtFree",                                              CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["cudaMemcpy"]                                              = {"aclrtMemcpy",                                            CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["cudaMemcpyAsync"]                                         = {"aclrtMemcpyAsync",                                       CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["cudaMemset"]                                              = {"aclrtMemset",                                            CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["cudaMemsetAsync"]                                         = {"aclrtMemsetAsync",                                       CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["cudaMallocHost"]                                         = {"aclrtMallocHost",                                        CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["cudaFreeHost"]                                            = {"aclrtFreeHost",                                          CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  m["free"]                                                    = {"aclrtFreeHost",                                          CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // Device management
  m["cudaGetDevice"]                                           = {"aclrtGetDevice",                                         CONV_DEVICE, API_RUNTIME, SEC::DEVICE_MGMT};
  m["cudaSetDevice"]                                           = {"aclrtSetDevice",                                         CONV_DEVICE, API_RUNTIME, SEC::DEVICE_MGMT};
  m["cudaGetDeviceCount"]                                      = {"aclrtGetDeviceCount",                                    CONV_DEVICE, API_RUNTIME, SEC::DEVICE_MGMT};
  // Device attribute query (CANN 9.0.0 uses aclrtGetDeviceInfo, not aclDeviceGetAttribute)
// cudaDeviceGetAttribute -> aclrtGetDeviceInfo (CANN 9.0.0: aclrtDevAttr + aclrtGetDeviceInfo)
  m["cudaDeviceGetAttribute"]                                  = {"aclrtGetDeviceInfo",                                    CONV_DEVICE, API_RUNTIME, SEC::DEVICE_MGMT};
  m["cudaDeviceSynchronize"]                                   = {"aclrtSynchronizeDevice",                                 CONV_DEVICE, API_RUNTIME, SEC::DEVICE_MGMT};
  m["cudaDeviceReset"]                                         = {"aclrtResetDevice",                                       CONV_DEVICE, API_RUNTIME, SEC::DEVICE_MGMT};
  // Stream management
  m["cudaStreamCreate"]                                        = {"aclrtCreateStream",                                      CONV_STREAM, API_RUNTIME, SEC::STREAM};
  m["cudaStreamDestroy"]                                       = {"aclrtDestroyStream",                                     CONV_STREAM, API_RUNTIME, SEC::STREAM};
  m["cudaStreamSynchronize"]                                   = {"aclrtSynchronizeStream",                                 CONV_STREAM, API_RUNTIME, SEC::STREAM};
  // Event management
  m["cudaEventCreate"]                                         = {"aclrtCreateEvent",                                       CONV_EVENT, API_RUNTIME, SEC::EVENT};
  m["cudaEventDestroy"]                                        = {"aclrtDestroyEvent",                                      CONV_EVENT, API_RUNTIME, SEC::EVENT};
  m["cudaEventRecord"]                                         = {"aclrtRecordEvent",                                       CONV_EVENT, API_RUNTIME, SEC::EVENT};
  m["cudaEventSynchronize"]                                    = {"aclrtSynchronizeEvent",                                  CONV_EVENT, API_RUNTIME, SEC::EVENT};
  m["cudaEventElapsedTime"]                                    = {"aclrtEventElapsedTime",                                  CONV_EVENT, API_RUNTIME, SEC::EVENT};
  // Error handling
  m["cudaGetLastError"]                                        = {"aclrtGetLastError",                                      CONV_ERROR_LOG, API_RUNTIME, SEC::ERROR_HANDLING};
  m["cudaPeekAtLastError"]                                     = {"aclrtPeekAtLastError",                                   CONV_ERROR_LOG, API_RUNTIME, SEC::ERROR_HANDLING};
  m["cudaGetErrorString"]                                      = {"aclGetRecentErrMsg",                                     CONV_ERROR_LOG, API_RUNTIME, SEC::ERROR_HANDLING};
  // Occupancy API: no Ascend equivalent (verified CANN 9.0.0). Manual rewrite needed:
  // Replace with: int sm_count; aclrtGetDeviceCount(&sm_count);
  //               *num_blocks = max(1, min(max_blocks, sm_count * tpm / block_size * waves));
  m["cudaOccupancyMaxActiveBlocksPerMultiprocessor"]            = {"/* [ASCIFY: no Ascend Occupancy API - see GetNumBlocks rewrite] */ cudaOccupancyMaxActiveBlocksPerMultiprocessor", CONV_OCCUPANCY, API_RUNTIME, SEC::OCCUPANCY | UNSUPPORTED};
  return m;
}();