#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cxx=${CXX:-c++}
python=${PYTHON:-python3}
test_tmp=$(mktemp -d "${TMPDIR:-/tmp}/ascify-migration-receipt.XXXXXX")

cleanup() {
  case "$test_tmp" in
    "${TMPDIR:-/tmp}"/ascify-migration-receipt.*)
      rm -rf -- "$test_tmp"
      ;;
    *)
      echo "refusing to remove unexpected migration receipt test directory" >&2
      ;;
  esac
}
trap cleanup 0 1 2 15

"$cxx" -std=c++17 -Wall -Wextra -Werror \
  -I"$repo_root/src" \
  "$repo_root/src/MigrationReceipt.cpp" \
  "$repo_root/tests/rewrite/migration_receipt_test.cpp" \
  -o "$test_tmp/migration-receipt-test"

"$test_tmp/migration-receipt-test" > "$test_tmp/receipt.json"
"$python" -B -c \
  'import json, pathlib, sys; data=json.loads(pathlib.Path(sys.argv[1]).read_text()); assert data["schema"] == "ascify.migration-receipt"; assert data["schema_version"] == 1; assert data["status"] == "succeeded"; assert len(data["inputs"]) == 2' \
  "$test_tmp/receipt.json"

grep -F 'cl::opt<std::string> MigrationReceiptPath("migration-receipt"' \
  "$repo_root/src/ArgParse.cpp" >/dev/null
grep -F 'std::string(MigrationReceiptPath.ArgStr)' \
  "$repo_root/src/ArgParse.cpp" >/dev/null
grep -F 'createUniqueFile' "$repo_root/src/MigrationReceiptIO.cpp" >/dev/null
grep -F 'sys::fs::rename' "$repo_root/src/MigrationReceiptIO.cpp" >/dev/null
grep -F 'MigrationReceiptStatus::InProgress' \
  "$repo_root/src/MigrationReceipt.h" >/dev/null
grep -F 'publishMigrationReceiptAtomically' "$repo_root/src/main.cpp" >/dev/null
