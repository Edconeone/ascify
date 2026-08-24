#include <cuda_runtime.h>

dim3 global_shape;
__device__ dim3 device_shape;

__global__ void KeepDeviceDim3() {
  extern __shared__ dim3 shared_shapes[];
  dim3 local_device_shape;
  (void)shared_shapes;
  (void)local_device_shape;
}

void BuildLaunchShapes(dim3* output) {
  extern dim3 external_shape;
  dim3 threads, blocks;
  static dim3 static_shape;
  using LaunchShape = dim3;
  LaunchShape alias_shape;
  dim3 explicit_shape(8U, 2U, 1U);
  dim3 brace_shape{4U, 2U, 1U};
  dim3 assignment_shape = dim3(2U, 1U, 1U);
  dim3 shape_array[2];
  dim3 attributed_shape __attribute__((unused));
#define ASCIFY_TEST_DECLARE_DIM3(name) dim3 name
  ASCIFY_TEST_DECLARE_DIM3(macro_shape);
#undef ASCIFY_TEST_DECLARE_DIM3
  output[0] = threads;
  output[1] = blocks;
  output[2] = static_shape;
  output[3] = alias_shape;
  output[4] = explicit_shape;
  output[5] = brace_shape;
  output[6] = assignment_shape;
  output[7] = shape_array[0];
  output[8] = macro_shape;
  output[9] = attributed_shape;
  output[10] = external_shape;
}

namespace user_types {
struct dim3 {
  dim3() : x(7U), y(8U), z(9U) {}
  unsigned x;
  unsigned y;
  unsigned z;
};

void KeepUserDim3() {
  dim3 local;
  (void)local;
}
}  // namespace user_types
