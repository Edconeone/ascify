#ifndef ASCIFY_FRONTEND_COMPAT_LOCAL_SHADOW_H_
#define ASCIFY_FRONTEND_COMPAT_LOCAL_SHADOW_H_

namespace cooperative_groups {

class thread_block {
 public:
  __device__ void sync() const;
};

__device__ thread_block this_thread_block();

}  // namespace cooperative_groups

#endif  // ASCIFY_FRONTEND_COMPAT_LOCAL_SHADOW_H_
