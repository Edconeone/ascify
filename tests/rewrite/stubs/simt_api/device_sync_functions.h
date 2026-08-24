#ifndef ASCIFY_TEST_STUB_DEVICE_SYNC_FUNCTIONS_H
#define ASCIFY_TEST_STUB_DEVICE_SYNC_FUNCTIONS_H

inline int ascify_test_legacy_thread_barrier_calls = 0;
inline int ascify_test_legacy_thread_fence_block_calls = 0;
inline int ascify_test_legacy_thread_fence_calls = 0;

inline void asc_syncthreads() {
  ++ascify_test_legacy_thread_barrier_calls;
}

inline void asc_threadfence_block() {
  ++ascify_test_legacy_thread_fence_block_calls;
}

inline void asc_threadfence() {
  ++ascify_test_legacy_thread_fence_calls;
}

#endif
