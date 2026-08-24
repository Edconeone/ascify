#include <cooperative_groups.h>

struct UnprovenGroup {};

void RejectGenericSynchronization() {
  cooperative_groups::sync(UnprovenGroup{});
}
