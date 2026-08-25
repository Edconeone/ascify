#ifndef COMMON_HELPER_CUDA_H_
#define COMMON_HELPER_CUDA_H_

#include <stdlib.h>

template <typename T>
void check(T result, const char*, const char*, int) { (void)result; }

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char*, const char*, const int) {
  (void)cudaGetLastError();
}

inline bool checkCmdLineFlag(int, const char**, const char*) { return false; }
inline int getCmdLineArgumentInt(int, const char**, const char*) { return 0; }
inline int gpuDeviceInit(int device) { return device; }
inline int gpuGetMaxGflopsDeviceId() { return 0; }

// Same public spelling and signature, but not NVIDIA's admitted selection AST.
inline int findCudaDevice(int argc, const char **argv) {
  int devID = 0;
  if (checkCmdLineFlag(argc, argv, "device")) {
    devID = getCmdLineArgumentInt(argc, argv, "device=");
    devID = gpuDeviceInit(devID);
  } else {
    checkCudaErrors(cudaSetDevice(devID));
  }
  return devID;
}

#endif
