#pragma once

// Parsing-only CUDA FP16 surface. DPP/ccec supplies the real half type.
struct half {
  unsigned short storage;
};

using __half = half;

struct half2 {
  unsigned short storage[2];
};

using __half2 = half2;
