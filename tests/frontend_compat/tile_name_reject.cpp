#include <cooperative_groups.h>

void RejectUnprovenTileSurface() {
  cooperative_groups::thread_block_tile<32> tile;
  (void)tile;
}
