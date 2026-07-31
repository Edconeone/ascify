# Changelog

## Unreleased

### Added

- Opt-in `dav-c310-vec + fast` AST-gated row-wise recipes for FP16 Softmax and
  RMSNorm, including affine RMSNorm.
- Versioned dav-c310 target headers, thin CUDA-to-ACL compatibility helpers,
  fail-close mutation coverage, and 950PR correctness/performance gates.
- Versioned OneFlow conversion fixtures with source hashes and third-party
  licensing metadata.
- One host-only release-check entry point and CTest integration.

### Changed

- Device/SIMT mappings now preserve unsupported semantics conservatively and
  use AST context for device scalar-double lowering.
- All conversion, build, benchmark, and evidence outputs are confined to one
  ignored `.work/softmax_rmsnorm_950` tree.
