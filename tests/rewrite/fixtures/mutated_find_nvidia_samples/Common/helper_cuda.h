#ifndef COMMON_HELPER_CUDA_H_
#define COMMON_HELPER_CUDA_H_

#include <stdio.h>
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
inline const char* _ConvertSMVer2ArchName(int, int) { return "test"; }
inline void ascifyTestUnknownSideEffect() {}

#if defined(ASCIFY_TEST_FIND_CHANGED_SIGNATURE)
inline long findCudaDevice(int argc, const char **argv) {
#else
inline int findCudaDevice(int argc, const char **argv) {
#endif
  int devID = 0;
  if (checkCmdLineFlag(argc, argv, "device")) {
#if defined(ASCIFY_TEST_FIND_DISCARDED_RESULT)
    getCmdLineArgumentInt(argc, argv, "device=");
    devID = 0;
#else
    devID = getCmdLineArgumentInt(argc, argv, "device=");
#endif
#if defined(ASCIFY_TEST_FIND_EXTRA_SIDE_EFFECT)
    ascifyTestUnknownSideEffect();
#endif
    if (devID < 0) {
      printf("Invalid command line parameter\n ");
      exit(EXIT_FAILURE);
    } else {
      devID = gpuDeviceInit(devID);
      if (devID < 0) {
        printf("exiting...\n");
        exit(EXIT_FAILURE);
      }
    }
  } else {
#if defined(ASCIFY_TEST_FIND_WRONG_ASSIGNMENT)
    int other = 0;
    other = gpuGetMaxGflopsDeviceId();
#else
    devID = gpuGetMaxGflopsDeviceId();
#endif
#if defined(ASCIFY_TEST_FIND_UNCHECKED_SET_DEVICE)
    cudaSetDevice(devID);
#elif defined(ASCIFY_TEST_FIND_WRONG_SET_DEVICE)
    checkCudaErrors(cudaSetDevice(1));
#else
    checkCudaErrors(cudaSetDevice(devID));
#endif
    int major = 0, minor = 0;
    checkCudaErrors(cudaDeviceGetAttribute(
        &major, cudaDevAttrComputeCapabilityMajor, devID));
    checkCudaErrors(cudaDeviceGetAttribute(
        &minor, cudaDevAttrComputeCapabilityMinor, devID));
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
           devID, _ConvertSMVer2ArchName(major, minor), major, minor);
  }
  return devID;
}

#endif
