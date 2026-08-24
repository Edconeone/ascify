#ifndef ASCIFY_TEST_NATIVE_COOPERATIVE_GROUPS_H
#define ASCIFY_TEST_NATIVE_COOPERATIVE_GROUPS_H

#include <simt_api/device_types.h>

namespace cooperative_groups {

struct thread_block {
  void sync() const {}
};

struct coalesced_group {
  void sync() const {}
};

template <unsigned int Size>
struct thread_block_tile {
  void sync() const {}
};

inline thread_block this_thread_block() { return {}; }

}  // namespace cooperative_groups

#endif
