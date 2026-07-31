# Shape manifests

Formal release gates consume only:

- `correctness.csv`: route boundaries and numerical coverage;
- `unified_tune.csv`: the frozen Softmax, RMSNorm plain, and RMSNorm affine
  performance set.

The other CSV files are versioned tuning sweeps used to establish route
thresholds, block sizes, packed-access behavior, and boundary coverage. They
are not added to the formal acceptance set implicitly; changing a formal
manifest requires a new same-commit 910C/950PR replay.
