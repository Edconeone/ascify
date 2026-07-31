# Third-party notices

## OneFlow CUDA conversion fixtures

The following files are derived from OneFlow
(`https://github.com/Oneflow-Inc/oneflow`) at commit
`25c8978c1c8b1371ef6aa4187dae4495bd233c35`:

- `tests/fixtures/oneflow/oneflow/core/cuda/softmax.cuh`
- `tests/fixtures/oneflow/oneflow/core/cuda/layer_norm.cuh`
- `tests/fixtures/oneflow/oneflow/core/cuda/rms_norm.cuh`
- `tests/softmax_rmsnorm_950/inputs/rmsnorm_affine_store.cuh`

Copyright 2020 The OneFlow Authors. All rights reserved.

These files are licensed under the Apache License, Version 2.0. A copy of the
license is provided in this repository's `LICENSE` file. The original OneFlow
copyright and Apache-2.0 license headers are retained in each fixture.

`rms_norm.cuh` contains the local `rows_per_access` tail-handling patch
documented in `tests/fixtures/oneflow/README.md`.

`rmsnorm_affine_store.cuh` is a reduced conversion fixture derived from the
forward adapter in `oneflow/user/kernels/rms_norm_gpu_kernel.cu`; it contains
only the load/store contract needed by the Ascify semantic matcher.
