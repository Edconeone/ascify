# ADR-0003: Gate SIMT performance rewrites by target policy and semantic proof

- Status: Accepted
- Date: 2026-07-30

## Context

The 950PR probes show that native `asc_reduce_*` is materially faster than a
shuffle butterfly, and that shape-aware 16-byte access and block-per-row
tiling can produce large Softmax/RMSNorm gains. Applying those transformations
as unconditional token substitutions would be unsafe:

- CUDA shuffle masks and sub-warp widths have semantics not represented by the
  available native helper;
- an arbitrary reduction functor may have side effects or overload-dependent
  behavior;
- fast half math changes rounding and must pass an operator-level accuracy
  budget;
- row/block thresholds depend on the selected device rather than an operator
  name.

The traditional Ascify framework should gain these capabilities without
becoming a OneFlow-specific optimizer or being replaced by a new compiler
pipeline.

## Decision

Keep `--target-policy=portable` and `--simt-math=precise` as defaults. Enable
950PR SIMT semantic rewrites only when the caller explicitly selects:

```text
--target-policy=dav-c310-vec --simt-math=fast
```

Within that policy:

- replace a shuffle add loop only after proving a full 32-bit mask, width 32,
  offsets 16/8/4/2/1, one pure accumulator update, a supported 32-bit type,
  and a system CUDA shuffle intrinsic;
- annotate a binary reduction functor only after proving a const, two-argument,
  single-return, side-effect-free `+`, conditional max, or conditional min
  body with identical argument/result types;
- expose the proof through the generic nested aliases
  `ascify_reduction_tag`, `ascify_reduction_value_type`, and
  `ascify_reduction_owner_type`;
- inject those aliases only when the CUB compatibility include appears earlier
  in the same main file, so the public `aclcub` tag is already declared;
- let `aclcub` use the native path only when the marker value type exactly
  matches `BlockReduce<T>` after cv removal, the owner is the actual functor
  rather than an inherited marker, and `T` is `float`, `int32_t`, or
  `uint32_t`;
- keep all unproven functors and unsupported types on the existing generic
  callable fallback;
- do not put OneFlow namespaces, functor names, or business shape constants in
  `aclcub` or the matcher;
- make AST rewrite spans own their source ranges so the later raw-token pass
  cannot overlap them; report replacement conflicts instead of silently
  discarding one rewrite.

The earlier plan to leave 16-byte access, block-per-row routing, and refined
half2 exponentials only as external native candidates is superseded by
[ADR-0005](0005-add-versioned-dav-c310-rowwise-recipes.md): they are now
available through a versioned target recipe after whole-operator proof. They
are still not enabled as isolated token rewrites.

## Consequences

- Default Ascify output does not acquire target-specific numerical changes.
- The fast policy can reuse 950PR hardware reduction capability across
  unrelated source projects without operator-name coupling.
- Dependent calls such as an unproven ADL `max(a, b)`, partial masks, sub-warps,
  overloaded functors, extra statements, and device `double` remain
  conservative.
- A canonical functor declared before the CUB include, or instantiated with a
  different `BlockReduce<T>` value type, remains on the generic path.
- A derived functor that inherits a marker but overrides `operator()` also
  remains on the generic path because its owner marker does not match.
- New performance rewrites require both positive and negative golden tests,
  a real LLVM/CUDA translation run, and a CCEC compile/correctness gate.
- Shape-aware tiling evidence can be added incrementally without changing the
  overall Ascify architecture.
