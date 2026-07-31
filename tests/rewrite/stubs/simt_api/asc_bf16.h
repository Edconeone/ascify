#ifndef ASCIFY_TEST_STUB_ASC_BF16_H
#define ASCIFY_TEST_STUB_ASC_BF16_H

struct bfloat16_t {
  unsigned short value;
};

struct bfloat16x2_t {
  bfloat16_t x;
  bfloat16_t y;
};

#endif
