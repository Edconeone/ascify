# Contributing

Run the host-only release gate before committing:

```bash
sh tests/run_release_checks.sh
```

For translator golden checks, build `ascify-clang` and provide its CUDA parsing
dependencies:

```bash
ASCIFY_BINARY=build/ascify-clang \
ASCIFY_CUDA_PATH=/path/to/cuda \
ASCIFY_CLANG_RESOURCE_DIRECTORY=ascify_install/include/ascify \
sh tests/run_release_checks.sh
```

Changes to a target recipe require:

1. positive and adversarial negative rewrite coverage;
2. a clean 910C conversion and mutation replay;
3. 950PR direct/native correctness and performance gates;
4. an ADR update when the semantic proof, runtime domain, or acceptance
   threshold changes.

Do not commit `.work`, generated headers, binaries, raw benchmark CSV, device
locks, or evidence bundles. Commit only source, fixed fixtures, expected
outputs, and small evidence summaries bound to immutable hashes.
