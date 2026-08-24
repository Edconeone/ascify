#ifndef ASCIFY_TEST_SAMPLE_HELPER_EXTERNAL_USE_H
#define ASCIFY_TEST_SAMPLE_HELPER_EXTERNAL_USE_H

inline void sampleHelperExternalUse(void **pointer) {
  checkCudaErrors(cudaMalloc(pointer, 64));
}

inline int sampleHelperExternalDevice(int argc, char **argv) {
  return findCudaDevice(argc, (const char **)argv);
}

#endif
