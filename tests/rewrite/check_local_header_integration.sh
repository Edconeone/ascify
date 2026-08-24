#!/bin/sh
set -eu

: "${ASCIFY_BINARY:?ASCIFY_BINARY is required}"
: "${ASCIFY_CUDA_PATH:?ASCIFY_CUDA_PATH is required}"
: "${ASCIFY_CLANG_RESOURCE_DIRECTORY:?ASCIFY_CLANG_RESOURCE_DIRECTORY is required}"

test_root=$(mktemp -d "${TMPDIR:-/tmp}/ascify-local-header-integration.XXXXXX")
cleanup() {
  case "$test_root" in
    "${TMPDIR:-/tmp}"/ascify-local-header-integration.*)
      rm -rf -- "$test_root"
      ;;
    *)
      echo "refusing to remove unexpected integration directory: $test_root" >&2
      ;;
  esac
}
trap cleanup 0 1 2 15

run_ascify() {
  input=$1
  output=$2
  include_dir=$3
  "$ASCIFY_BINARY" "$input" \
    --local-headers-recursive \
    "--cuda-path=$ASCIFY_CUDA_PATH" \
    "--clang-resource-directory=$ASCIFY_CLANG_RESOURCE_DIRECTORY" \
    -o "$output" \
    -- "-I$include_dir" -std=c++17
}

assert_no_backups() {
  search_root=$1
  leaked=$(find "$search_root" -name '*.ascify-backup' -print -quit)
  if [ -n "$leaked" ]; then
    echo "local-header transaction leaked backup: $leaked" >&2
    exit 1
  fi
}

selected_case="$test_root/selected-alias"
mkdir -p "$selected_case/src"
printf '%s\n' '#include "selected.h"' 'int selected_root;' \
  >"$selected_case/src/main.cu"
printf '%s\n' 'int selected_dependency;' >"$selected_case/src/selected.h"
cp "$selected_case/src/main.cu" "$selected_case/main.before"
cp "$selected_case/src/selected.h" "$selected_case/header.before"
ln -s "$selected_case/src" "$selected_case/output-parent"
if run_ascify "$selected_case/src/main.cu" \
    "$selected_case/output-parent/selected.h" "$selected_case/src" \
    >"$selected_case/stdout" 2>"$selected_case/stderr"; then
  echo "parent-symlink selected-header alias unexpectedly succeeded" >&2
  exit 1
fi
grep -F 'root output aliases or contains an input dependency' \
  "$selected_case/stderr" >/dev/null
cmp "$selected_case/main.before" "$selected_case/src/main.cu"
cmp "$selected_case/header.before" "$selected_case/src/selected.h"
assert_no_backups "$selected_case"

excluded_header=
for candidate in \
  "$ASCIFY_CLANG_RESOURCE_DIRECTORY/include/stddef.h" \
  "$ASCIFY_CLANG_RESOURCE_DIRECTORY/include/__stddef_size_t.h" \
  "$ASCIFY_CLANG_RESOURCE_DIRECTORY/include/iso646.h"; do
  if [ -f "$candidate" ] && [ ! -L "$candidate" ]; then
    excluded_header=$candidate
    break
  fi
done
if [ -z "$excluded_header" ]; then
  echo "no regular Clang resource header available for excluded-alias integration test" >&2
  exit 1
fi
excluded_case="$test_root/excluded-alias"
mkdir -p "$excluded_case/src"
excluded_header_name=$(basename "$excluded_header")
printf '#include "%s"\nint excluded_root;\n' "$excluded_header_name" \
  >"$excluded_case/src/main.cu"
cp "$excluded_case/src/main.cu" "$excluded_case/main.before"
cp "$excluded_header" "$excluded_case/header.before"
ln -s "$(dirname "$excluded_header")" "$excluded_case/output-parent"
if run_ascify "$excluded_case/src/main.cu" \
    "$excluded_case/output-parent/$excluded_header_name" \
    "$(dirname "$excluded_header")" \
    >"$excluded_case/stdout" 2>"$excluded_case/stderr"; then
  echo "parent-symlink excluded-header alias unexpectedly succeeded" >&2
  exit 1
fi
grep -F 'root output aliases or contains an input dependency' \
  "$excluded_case/stderr" >/dev/null
cmp "$excluded_case/main.before" "$excluded_case/src/main.cu"
cmp "$excluded_case/header.before" "$excluded_header"
assert_no_backups "$excluded_case"

unmanaged_case="$test_root/unmanaged-bundle"
mkdir -p "$unmanaged_case/src" "$unmanaged_case/out/root.dpp.headers"
printf '%s\n' '#include "selected.h"' 'int unmanaged_root;' \
  >"$unmanaged_case/src/main.cu"
printf '%s\n' 'int unmanaged_dependency;' >"$unmanaged_case/src/selected.h"
printf '%s\n' 'unmanaged-content-must-survive' \
  >"$unmanaged_case/out/root.dpp.headers/unmanaged"
cp "$unmanaged_case/src/main.cu" "$unmanaged_case/main.before"
cp "$unmanaged_case/src/selected.h" "$unmanaged_case/header.before"
cp "$unmanaged_case/out/root.dpp.headers/unmanaged" \
  "$unmanaged_case/unmanaged.before"
if run_ascify "$unmanaged_case/src/main.cu" \
    "$unmanaged_case/out/root.dpp" "$unmanaged_case/src" \
    >"$unmanaged_case/stdout" 2>"$unmanaged_case/stderr"; then
  echo "unmanaged local-header bundle unexpectedly replaced" >&2
  exit 1
fi
grep -F 'Refusing to replace unowned local-header bundle' \
  "$unmanaged_case/stderr" >/dev/null
cmp "$unmanaged_case/main.before" "$unmanaged_case/src/main.cu"
cmp "$unmanaged_case/header.before" "$unmanaged_case/src/selected.h"
cmp "$unmanaged_case/unmanaged.before" \
  "$unmanaged_case/out/root.dpp.headers/unmanaged"
assert_no_backups "$unmanaged_case"

case "$(uname -s)" in
  Linux)
    if ! command -v cc >/dev/null 2>&1; then
      echo "cc is required for deterministic local-header publish failpoints" >&2
      exit 1
    fi
    ;;
  *)
    echo "[local-header-integration] publish/rollback failpoints require Linux; alias and ownership negatives passed"
    exit 0
    ;;
esac

failpoint_source="$test_root/rename_failpoint.c"
failpoint_library="$test_root/rename_failpoint.so"
printf '%s\n' \
  '#define _GNU_SOURCE' \
  '#include <dlfcn.h>' \
  '#include <errno.h>' \
  '#include <stdlib.h>' \
  '#include <string.h>' \
  'typedef int (*rename_fn)(const char *, const char *);' \
  'static int publish_failed;' \
  'static int rollback_failed;' \
  'static int ends_with(const char *text, const char *suffix) {' \
  '  size_t n = strlen(text), m = strlen(suffix);' \
  '  return n >= m && strcmp(text + n - m, suffix) == 0;' \
  '}' \
  'int rename(const char *from, const char *to) {' \
  '  static rename_fn real_rename;' \
  '  if (!real_rename) real_rename = (rename_fn)dlsym(RTLD_NEXT, "rename");' \
  '  if (!publish_failed && getenv("ASCIFY_TEST_FAIL_PUBLISH_ROOT") &&' \
  '      strstr(from, "/.ascify-local-closure") && ends_with(from, "/root.dpp")) {' \
  '    publish_failed = 1; errno = EIO; return -1;' \
  '  }' \
  '  if (!rollback_failed && getenv("ASCIFY_TEST_FAIL_ROLLBACK_BUNDLE_RENAME") &&' \
  '      ends_with(from, ".headers.ascify-backup") && ends_with(to, ".headers")) {' \
  '    rollback_failed = 1; errno = EIO; return -1;' \
  '  }' \
  '  return real_rename(from, to);' \
  '}' >"$failpoint_source"
cc -shared -fPIC -O2 "$failpoint_source" -ldl -o "$failpoint_library"

transaction_case="$test_root/transaction"
mkdir -p "$transaction_case/src" "$transaction_case/out"
printf '%s\n' '#include "selected.h"' 'int transaction_root;' \
  >"$transaction_case/src/main.cu"
printf '%s\n' 'int transaction_dependency_v1;' \
  >"$transaction_case/src/selected.h"
transaction_output="$transaction_case/out/root.dpp"
transaction_bundle="$transaction_output.headers"
run_ascify "$transaction_case/src/main.cu" "$transaction_output" \
  "$transaction_case/src" >"$transaction_case/initial.stdout" \
  2>"$transaction_case/initial.stderr"
cp "$transaction_output" "$transaction_case/root.before"
cp -Rp "$transaction_bundle" "$transaction_case/bundle.before"
printf '%s\n' 'int transaction_dependency_v2;' \
  >"$transaction_case/src/selected.h"
cp "$transaction_case/src/main.cu" "$transaction_case/input-main.before"
cp "$transaction_case/src/selected.h" "$transaction_case/input-header.before"

if ASCIFY_TEST_FAIL_PUBLISH_ROOT=1 \
    LD_PRELOAD="$failpoint_library${LD_PRELOAD:+:$LD_PRELOAD}" \
    run_ascify "$transaction_case/src/main.cu" "$transaction_output" \
      "$transaction_case/src" >"$transaction_case/publish.stdout" \
      2>"$transaction_case/publish.stderr"; then
  echo "injected local-header root publish failure unexpectedly succeeded" >&2
  exit 1
fi
grep -F 'publishing root output' "$transaction_case/publish.stderr" >/dev/null
cmp "$transaction_case/root.before" "$transaction_output"
diff -r "$transaction_case/bundle.before" "$transaction_bundle"
cmp "$transaction_case/input-main.before" "$transaction_case/src/main.cu"
cmp "$transaction_case/input-header.before" "$transaction_case/src/selected.h"
assert_no_backups "$transaction_case"

if ASCIFY_TEST_FAIL_PUBLISH_ROOT=1 \
    ASCIFY_TEST_FAIL_ROLLBACK_BUNDLE_RENAME=1 \
    LD_PRELOAD="$failpoint_library${LD_PRELOAD:+:$LD_PRELOAD}" \
    run_ascify "$transaction_case/src/main.cu" "$transaction_output" \
      "$transaction_case/src" >"$transaction_case/rollback.stdout" \
      2>"$transaction_case/rollback.stderr"; then
  echo "injected local-header rollback retry case unexpectedly succeeded" >&2
  exit 1
fi
grep -F 'Rollback rename failed once; retrying' \
  "$transaction_case/rollback.stderr" >/dev/null
grep -F 'Rollback rename recovered after a checked retry' \
  "$transaction_case/rollback.stderr" >/dev/null
cmp "$transaction_case/root.before" "$transaction_output"
diff -r "$transaction_case/bundle.before" "$transaction_bundle"
cmp "$transaction_case/input-main.before" "$transaction_case/src/main.cu"
cmp "$transaction_case/input-header.before" "$transaction_case/src/selected.h"
assert_no_backups "$transaction_case"

printf '%s\n' \
  'real ascify local-header alias, ownership, publish, and rollback negatives passed'
