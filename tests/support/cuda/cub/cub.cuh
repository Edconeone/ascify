#pragma once

// Ascify only needs the CUB surface used by the source file while building its
// AST. The converted program links against acl_cub instead of this parsing stub.
namespace cub {

template<typename T, int BlockThreads>
class BlockReduce {
 public:
  struct TempStorage {};

  __device__ explicit BlockReduce(TempStorage&) {}

  template<typename ReductionOp>
  __device__ T Reduce(T value, ReductionOp) {
    return value;
  }
};

}  // namespace cub
