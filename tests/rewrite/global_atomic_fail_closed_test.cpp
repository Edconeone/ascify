#define __aicore__
#define __global__

#include <ascify/ascify_cuda_compat.hpp>

#if !defined(ASCIFY_SIMT_HEADER_FAMILY_PUBLIC_85)
#error "this negative atomic fixture requires the public 8.5 header family"
#endif

#ifndef ASCIFY_ATOMIC_FAIL_CLOSED_CASE
#error "select an atomic negative contract"
#elif ASCIFY_ATOMIC_FAIL_CLOSED_CASE == 1
int main() { int32_t value = 0; return ascify::atomic_add_global(&value, 1); }
#elif ASCIFY_ATOMIC_FAIL_CLOSED_CASE == 2
int main() { int32_t value = 0; return ascify::atomic_sub_global(&value, 1); }
#elif ASCIFY_ATOMIC_FAIL_CLOSED_CASE == 3
int main() { int32_t value = 0; return ascify::atomic_exch_global(&value, 1); }
#elif ASCIFY_ATOMIC_FAIL_CLOSED_CASE == 4
int main() { int32_t value = 0; return ascify::atomic_max_global(&value, 1); }
#elif ASCIFY_ATOMIC_FAIL_CLOSED_CASE == 5
int main() { int32_t value = 0; return ascify::atomic_min_global(&value, 1); }
#elif ASCIFY_ATOMIC_FAIL_CLOSED_CASE == 6
int main() { uint32_t value = 0; return ascify::atomic_inc_global(&value, 1U); }
#elif ASCIFY_ATOMIC_FAIL_CLOSED_CASE == 7
int main() { uint32_t value = 0; return ascify::atomic_dec_global(&value, 1U); }
#elif ASCIFY_ATOMIC_FAIL_CLOSED_CASE == 8
int main() { int32_t value = 0; return ascify::atomic_cas_global(&value, 0, 1); }
#elif ASCIFY_ATOMIC_FAIL_CLOSED_CASE == 9
int main() { int32_t value = 0; return ascify::atomic_and_global(&value, 1); }
#elif ASCIFY_ATOMIC_FAIL_CLOSED_CASE == 10
int main() { int32_t value = 0; return ascify::atomic_or_global(&value, 1); }
#elif ASCIFY_ATOMIC_FAIL_CLOSED_CASE == 11
int main() { int32_t value = 0; return ascify::atomic_xor_global(&value, 1); }
#else
#error "unknown atomic negative contract"
#endif
