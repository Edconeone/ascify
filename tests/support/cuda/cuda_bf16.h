#pragma once

#include <cuda_fp16.h>

// Parsing-only CUDA BF16 surface. The converted source replaces these names
// with the target SIMT BF16 types before it is compiled by ccec.
struct nv_bfloat16 {
  unsigned short storage;
};

struct nv_bfloat162 {
  unsigned short storage[2];
};
