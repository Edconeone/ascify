#ifndef ASCIFY_FRONTEND_COMPAT_ADMITTED_V1_COOPERATIVE_GROUPS_H_
#define ASCIFY_FRONTEND_COMPAT_ADMITTED_V1_COOPERATIVE_GROUPS_H_

#if defined(__CUDACC__) || defined(__CUDA__)
#define ASCIFY_FRONTEND_COMPAT_DEVICE_ __device__
#else
#define ASCIFY_FRONTEND_COMPAT_DEVICE_
#endif

namespace cooperative_groups {

class thread_block {
 public:
  ASCIFY_FRONTEND_COMPAT_DEVICE_ void sync() const;
};

ASCIFY_FRONTEND_COMPAT_DEVICE_ thread_block this_thread_block();

ASCIFY_FRONTEND_COMPAT_DEVICE_ inline void sync(const thread_block& group) {
  group.sync();
}

}  // namespace cooperative_groups

#undef ASCIFY_FRONTEND_COMPAT_DEVICE_

#endif  // ASCIFY_FRONTEND_COMPAT_ADMITTED_V1_COOPERATIVE_GROUPS_H_
