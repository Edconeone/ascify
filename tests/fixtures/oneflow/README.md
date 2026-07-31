# OneFlow CUDA conversion fixtures

Source repository: `https://github.com/Oneflow-Inc/oneflow.git`

Pinned upstream commit: `25c8978c1c8b1371ef6aa4187dae4495bd233c35`

The conversion inputs live under the same include path as upstream:
`oneflow/core/cuda/`.

| Fixture | SHA256 |
| --- | --- |
| `oneflow/core/cuda/softmax.cuh` | `95ad8fed7bdcdf9d8a755c30116c8909e2c655b3f08f19b5965f7289c1beb8cf` |
| `oneflow/core/cuda/layer_norm.cuh` | `41f7d02e188dcc3908b142e33fc3120532f06dabc49923a4f9e2fce92990cd34` |
| `oneflow/core/cuda/rms_norm.cuh` | `9895efb57f9195cfafab8e812f54c2f651ccd704a199d5e1e107b0465d0b629a` |

The reduced affine caller adapter is versioned separately at
`tests/softmax_rmsnorm_950/inputs/rmsnorm_affine_store.cuh` with SHA256
`463127ade4e8d6a4f956c856a62515ccc028be6bdd62e4021c143ba2f26dee3c`.

`softmax.cuh` and `layer_norm.cuh` are byte-identical to the files at the
pinned commit. The upstream `rms_norm.cuh` blob has SHA256
`22aded7bc82d7894a0f58f5213fa40077066a42c0ce9ae80d9e1483ddf7323c5`;
the fixture includes this project's `rows_per_access` tail fix.

The local RMSNorm patch is limited to the forward warp kernel and its launch
geometry (7 insertions, 2 deletions):

- derive `num_row_groups` with ceiling division by `rows_per_access`;
- iterate the kernel over row groups instead of comparing a row-group index
  directly with `nrow`;
- skip `row >= nrow` in both per-row processing phases; and
- calculate the launch block count from the ceiling-divided row-group count.

These changes preserve complete row coverage when `rows_per_access > 1` and
prevent out-of-bounds accesses for a non-divisible tail. There are no other
differences from upstream `oneflow/core/cuda/rms_norm.cuh` at the pinned
commit.

Verify the vendored files from the repository root with:

```bash
sha256sum tests/fixtures/oneflow/oneflow/core/cuda/{softmax.cuh,layer_norm.cuh,rms_norm.cuh}
```
