#include <ascify/cooperative_groups_compat.hpp>

namespace cg = cooperative_groups;

#if ASCIFY_REJECT_SYNC_CASE == 1
void RejectCoalescedGroup() {
  const cg::coalesced_group group{};
  cg::sync(group);
}
#elif ASCIFY_REJECT_SYNC_CASE == 2
void RejectThreadBlockTile() {
  const cg::thread_block_tile<32> tile{};
  cg::sync(tile);
}
#else
#error "ASCIFY_REJECT_SYNC_CASE must select a negative test"
#endif
