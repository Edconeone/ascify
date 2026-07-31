# ADR-0001: Use user-owned toolchains and one project root per remote

- Status: Accepted
- Date: 2026-07-30

## Context

Ascify is built on `910C`, while translated operators are compiled and
benchmarked on `950PR`. Both hosts are shared environments. Creating a new
top-level directory for every experiment makes results hard to reproduce and
risks mixing toolchains. Updating the system CANN installation or driver is
outside this project's scope.

## Decision

Use exactly one checkout on each host. Resolve it at runtime as `REPO_ROOT`
rather than embedding a user or host path in source or scripts.

Committed sources and harnesses live in the repository. All generated headers,
binaries, locks, manifests, profiles, and results live below the ignored
`.work/softmax_rmsnorm_950/` tree.

The `950PR` build and run path is supplied explicitly through `CANN_ROOT`.
Scripts may source `${CANN_ROOT}/set_env.sh` in their own process, but must not
modify that package, the system driver, or global environment configuration.
The 910C LLVM and CUDA parsing roots follow the same rule.

## Consequences

- A run is identified by source hashes and a manifest, not by parallel
  `v1`, `v2`, or `final2` directories.
- Device locks and health checks are project-local and reproducible.
- Switching CANN versions is an explicit future decision and requires a fresh
  compile/correctness/performance baseline.
- A clean checkout can move between user directories without patching scripts.
