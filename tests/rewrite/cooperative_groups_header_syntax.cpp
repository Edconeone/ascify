#include <ascify/cooperative_groups_compat.hpp>

namespace cg = cooperative_groups;

int main() {
  const cg::thread_block block = cg::this_thread_block();
  cg::sync(block);
  return 0;
}
