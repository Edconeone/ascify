#ifndef ASCIFY_TEST_SAMPLE_HELPER_EXTERNAL_PP_H
#define ASCIFY_TEST_SAMPLE_HELPER_EXTERNAL_PP_H

#ifdef getLastCudaError
inline int sampleHelperExternalPpUse() { return 1; }
#else
inline int sampleHelperExternalPpUse() { return 0; }
#endif

#if defined(checkCudaErrors)
inline int sampleHelperExternalDefinedUse() { return 1; }
#endif

#undef checkCudaErrors
#ifndef checkCudaErrors
inline int sampleHelperExternalUndefUse() { return 1; }
#endif

#endif
