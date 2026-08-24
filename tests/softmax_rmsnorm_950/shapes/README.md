# Shape manifests

Formal release gates consume only:

- `correctness.csv`: route boundaries and numerical coverage;
- `unified_tune.csv`: the frozen Softmax, RMSNorm plain, and RMSNorm affine
  performance set.

The other CSV files are versioned tuning sweeps used to establish route
thresholds, block sizes, packed-access behavior, and boundary coverage. They
are not added to the formal acceptance set implicitly; changing a formal
manifest requires a new same-commit 910C/950PR replay.

`rowwise_simd_v1_correctness.csv` is a focused non-performance gate for the
explicit SIMD recipe. It covers Softmax and RMSNorm selector hits and misses,
both RMSNorm mixed routes, affine RMSNorm, and the version 1 column boundaries.
It does not replace `correctness.csv` or promote a performance result.

LayerNorm uses fixed cases in `layernorm_hybrid_check.cce` rather than this
CSV because it is an independent registry-extension gate, not part of the
frozen Softmax/RMSNorm acceptance corpus. The check covers columns 8 through
8192, a non-vector tail, cross-AIV row scheduling, exact in-place output,
invalid-domain rejection, and one ACL-event profile shape.
