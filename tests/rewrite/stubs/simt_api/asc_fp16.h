#ifndef ASCIFY_TEST_STUB_ASC_FP16_H
#define ASCIFY_TEST_STUB_ASC_FP16_H

#include <stdint.h>

struct half {
  uint16_t bits;
};

static_assert(sizeof(half) == 2, "test half must occupy two bytes");

#endif
