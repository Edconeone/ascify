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

## NVIDIA CUDA Samples helper fixtures

The following reduced test fixtures are derived from NVIDIA CUDA Samples
`Common/helper_cuda.h` at commit
`b7c5481c556c3fe98db060207ecaa41a4b9a9abc`:

- `tests/rewrite/fixtures/nvidia_samples/Common/helper_cuda.h`
- `tests/rewrite/fixtures/altered_nvidia_samples/Common/helper_cuda.h`

Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of NVIDIA CORPORATION nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
