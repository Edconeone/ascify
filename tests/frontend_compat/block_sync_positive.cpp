#include <cooperative_groups.h>

namespace cg = cooperative_groups;

void AdmitOnlyBlockSynchronization() {
  const cg::thread_block block = cg::this_thread_block();
  block.sync();
  cg::sync(block);
}
