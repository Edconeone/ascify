# ADR 0013: Fail closed on local-header artifact aliases and restore errors

- Status: Accepted; real-binary rollback and remote build regression are release gates
- Date: 2026-08-24
- Scope: quoted local-header closure planning and two-artifact publication

## Context

Local-header closure canonicalized source inputs, but its root output and
bundle plan used only an absolute lexical path.  An existing symlink in an
output parent could therefore make a lexical output name refer to a selected
or excluded input in a different coordinate system.  The publication helper
also discarded several remove/rename errors while restoring or deleting
transaction backups.  A failed restore could consequently be described as a
rollback even though the old pair had not been recovered.

## Decision

1. Preserve lexical artifact names for the filesystem operations, but derive
   artifact identities with `weakly_canonical`.  This resolves every existing
   parent component while still supporting a not-yet-created leaf.
2. Record the canonical root input and every resolved include dependency,
   including selected, excluded and resolved angle edges.  Before publication,
   compare the weak-canonical root, bundle and two backup identities with that
   one dependency set.  Reject exact root/backup aliases and bundle/backup
   aliases or containment.  Explicit in-place replacement of the root input is
   the sole exception.
3. Inspect paths without following the final symlink, reject stale backups,
   and check every transaction rename/remove.  Rollback removes a partial
   artifact and restores its backup with checked operations.  A first rollback
   failure is reported and retried once; a persistent failure remains an
   explicit command failure.  Backup cleanup errors are also command failures.
4. Keep fault injection outside product code.  The real-binary Linux gate uses
   a one-shot `rename` interposer to fail root publication and the first bundle
   restore.  It requires nonzero conversion status, byte-identical inputs and
   old outputs, the checked-retry diagnostics, and no leaked backup.

## Alternatives rejected

- Lexical prefix checks: they do not resolve a symlinked parent and therefore
  compare output and input in different coordinate systems.
- `canonical` for outputs: the final output leaf often does not exist before
  publication; requiring it would reject valid destinations.
- Treating excluded headers as irrelevant: publication can still overwrite an
  excluded dependency even though Ascify deliberately leaves its include
  unchanged.
- Ignoring rollback errors: this can claim restoration without restored bytes.
- A product environment-variable failpoint: it expands the shipped interface
  and can be triggered outside the test harness.

## Consequences

Output destinations through symlinked parents are supported only when their
resolved identity is isolated from all observed dependencies.  Explicit
`--inplace` continues to work for the root input, while selected/excluded
header aliases and backup aliases fail before move-aside.  Publication remains
a process-local two-artifact transaction, not a durable crash-recovery journal.
The host model and real-binary negatives establish filesystem behavior only;
every release source still needs exact-manifest LLVM, translated-fixture and
target gates before promotion.
