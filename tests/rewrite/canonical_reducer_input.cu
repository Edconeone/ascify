#include <cuda.h>

template<typename T>
struct KeepReducerDeclaredBeforeCub {
  __device__ T operator()(const T& lhs, const T& rhs) const {
    return lhs + rhs;
  }
};

#include <cub/cub.cuh>

template<typename T>
struct CanonicalSumReducer {
  __device__ T operator()(const T& lhs, const T& rhs) const {
    return lhs + rhs;
  }
};

template<typename T>
struct CanonicalMaxReducer {
  __device__ T operator()(const T& lhs, const T& rhs) const {
    return lhs > rhs ? lhs : rhs;
  }
};

template<typename T>
struct CanonicalMinReducer {
  __device__ T operator()(const T& lhs, const T& rhs) const {
    return rhs > lhs ? lhs : rhs;
  }
};

template<typename T>
struct KeepCallBasedMaxReducer {
  __device__ T operator()(const T& lhs, const T& rhs) const {
    return max(lhs, rhs);
  }
};

template<typename T>
struct KeepSideEffectReducer {
  __device__ T operator()(T lhs, T rhs) const {
    return ++lhs + rhs;
  }
};

template<typename T>
struct KeepExtraStatementReducer {
  __device__ T operator()(const T& lhs, const T& rhs) const {
    T result = lhs + rhs;
    return result;
  }
};

template<typename T>
struct KeepNonConstReducer {
  __device__ T operator()(const T& lhs, const T& rhs) {
    return lhs + rhs;
  }
};

template<typename T>
struct KeepOverloadedReducer {
  __device__ T operator()(const T& lhs, const T& rhs) const {
    return lhs + rhs;
  }
  __device__ T operator()(const T& lhs, const T& rhs, int) const {
    return lhs + rhs;
  }
};

template<template<typename> class ReductionOp, typename T>
__device__ T RunBlockReduce(T value) {
  using BlockReduce = cub::BlockReduce<T, 32>;
  __shared__ typename BlockReduce::TempStorage storage;
  return BlockReduce(storage).Reduce(value, ReductionOp<T>{});
}

__global__ void InstantiateCanonicalReducers(float* output) {
  output[0] = RunBlockReduce<CanonicalSumReducer>(output[0]);
  output[1] = RunBlockReduce<CanonicalMaxReducer>(output[1]);
  output[2] = RunBlockReduce<CanonicalMinReducer>(output[2]);
  output[3] = RunBlockReduce<KeepCallBasedMaxReducer>(output[3]);
}
