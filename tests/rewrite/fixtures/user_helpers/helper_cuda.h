#ifndef USER_HELPER_CUDA_H_
#define USER_HELPER_CUDA_H_

inline int userCheck(int value) { return value; }
inline int findCudaDevice(int, const char**) { return 7; }
#define checkCudaErrors(value) userCheck(value)
#define getLastCudaError(message) userCheck((message) == nullptr)

#endif
