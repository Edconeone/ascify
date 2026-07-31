# ADR-0006: Version conversion inputs and keep run evidence out of Git

- Status: Accepted
- Date: 2026-07-31

## Context

The generated Softmax and RMSNorm headers depend on exact CUDA inputs, including
a local RMSNorm tail-row fix that is not present in the recorded upstream
OneFlow commit. Keeping those files only below ignored `.work` makes a clean
checkout unable to reproduce the conversion. Conversely, converter binaries,
generated headers, CCEC binaries, device snapshots, raw CSV, and manifests are
machine-specific and too large or volatile for source control.

The 910C CUDA parsing layout and the 950PR CANN package are user-owned external
toolchains. Vendoring either would duplicate platform installations and blur
the boundary between source and environment.

## Decision

- Version the exact OneFlow `softmax.cuh`, `layer_norm.cuh`, and patched
  `rms_norm.cuh` under `tests/fixtures/oneflow/`.
- Record the upstream repository, commit, per-file SHA256, Apache-2.0 notice,
  and the local RMSNorm patch in the fixture manifest.
- Keep the CUDA parsing root, Clang resource directory, LLVM build, and CANN
  package external and pass their roots explicitly.
- Keep all generated and measured artifacts below the single ignored
  `.work/softmax_rmsnorm_950` root.
- Bind a formal run to the Git commit plus hashes of the converter, recipe,
  fixtures, generated outputs, target headers, harness, and binaries.
- Use three release gates:
  1. host-only rewrite contracts and Python unit tests;
  2. clean 910C build, conversion, and 38-case mutation matrix;
  3. same-commit 950PR correctness and direct-A/native/direct-B performance.
- Store only small technical summaries in Git. Publish a complete evidence
  bundle separately when long-term archival is required.

## Alternatives considered

### Fetch OneFlow inputs at test time

Rejected because the tested RMSNorm input includes a local patch and a network
fetch can change availability without changing this repository.

### Commit the complete `.work` tree

Rejected because it includes platform binaries, generated output, raw evidence,
locks, and transient paths rather than reviewable source.

### Vendor the CUDA and CANN layouts

Rejected because they are external toolchains already managed in user-owned
directories and are not source fixtures.

## Consequences

- A clean checkout contains every source input needed for the conversion.
- Reproduction still requires compatible external LLVM/CUDA/CANN toolchains,
  and their hashes belong in the run manifest.
- Generated headers remain demonstrably unedited because they are recreated
  from committed fixtures and validated before being transferred.
- Repository history stays reviewable while formal evidence remains
  cryptographically bound to the tested commit.
