#define __aicore__
#define __global__

#include <ascify/ascify_cuda_compat.hpp>

#ifndef ASCIFY_FAIL_CLOSED_CASE
#error "select one negative contract with ASCIFY_FAIL_CLOSED_CASE"
#elif ASCIFY_FAIL_CLOSED_CASE == 1
int main() { return ascify::syncthreads_and(1); }
#elif ASCIFY_FAIL_CLOSED_CASE == 2
int main() { return ascify::syncthreads_or(1); }
#elif ASCIFY_FAIL_CLOSED_CASE == 3
int main() { return ascify::syncthreads_count(1); }
#elif ASCIFY_FAIL_CLOSED_CASE == 4
int main() {
  ascify::threadfence_system();
  return 0;
}
#else
#error "unknown negative contract fixture"
#endif
