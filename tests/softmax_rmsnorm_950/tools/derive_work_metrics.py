#!/usr/bin/env python3
"""Derive reproducible Softmax/RMSNorm work metrics from benchmark CSV rows."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import re
import sys
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, TextIO, Tuple


REQUIRED_PERF_COLUMNS = {
    "run_id",
    "op",
    "variant",
    "case_id",
    "rows",
    "cols",
    "lat_ms_median",
    "status",
}

METRIC_COLUMNS = [
    "metric_schema_version",
    "metric_model",
    "fp32_ops",
    "fp16_ops",
    "arith_ops",
    "arith_tflops",
    "special_op_kind",
    "special_ops",
    "special_gops",
    "compare_ops",
    "compare_gops",
]

TRUE_VALUES = {"1", "true", "yes", "y"}
FALSE_VALUES = {"0", "false", "no", "n"}


@dataclass(frozen=True)
class ManifestEntry:
    affine: bool
    rows: Optional[int]
    cols: Optional[int]
    line_number: int


@dataclass(frozen=True)
class WorkMetrics:
    model: str
    fp32_ops: int
    fp16_ops: int
    special_op_kind: str
    special_ops: int
    compare_ops: int

    @property
    def arith_ops(self) -> int:
        return self.fp32_ops + self.fp16_ops


def normalized_op(value: str, context: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "", value.strip().lower())
    if key == "softmax":
        return "softmax"
    if key == "rmsnorm":
        return "rms_norm"
    raise ValueError(f"{context}: unsupported op {value!r}; expected softmax or rms_norm")


def parse_positive_integer(value: str, field: str, context: str) -> int:
    text = value.strip()
    try:
        parsed = int(text, 10)
    except ValueError as error:
        raise ValueError(f"{context}: {field} is not an integer: {value!r}") from error
    if parsed <= 0:
        raise ValueError(f"{context}: {field} must be positive, got {parsed}")
    return parsed


def parse_boolean(value: str, field: str, context: str) -> bool:
    text = value.strip().lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    raise ValueError(
        f"{context}: {field} must be one of "
        f"{sorted(TRUE_VALUES | FALSE_VALUES)}, got {value!r}"
    )


def parse_positive_finite_float(value: str, field: str, context: str) -> float:
    text = value.strip()
    try:
        parsed = float(text)
    except ValueError as error:
        raise ValueError(f"{context}: {field} is not numeric: {value!r}") from error
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{context}: {field} must be positive and finite, got {value!r}")
    return parsed


def validate_header(
    fieldnames: Optional[Sequence[str]], required: Iterable[str], context: str
) -> List[str]:
    if fieldnames is None:
        raise ValueError(f"{context}: CSV has no header")
    names = list(fieldnames)
    if any(not name for name in names):
        raise ValueError(f"{context}: CSV header contains an empty column name")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"{context}: duplicate columns: {', '.join(duplicates)}")
    missing = sorted(set(required).difference(names))
    if missing:
        raise ValueError(f"{context}: missing required columns: {', '.join(missing)}")
    return names


def clean_csv_row(
    raw_row: Mapping[Optional[str], Optional[str]], line_number: int, context: str
) -> Dict[str, str]:
    if None in raw_row:
        raise ValueError(f"{context}: row {line_number} has more fields than the header")
    row = {str(key): (value or "") for key, value in raw_row.items()}
    if not any(value.strip() for value in row.values()):
        return {}
    return row


def read_shape_manifest(path: Path) -> Dict[Tuple[str, str], ManifestEntry]:
    context = f"shape manifest {path}"
    try:
        stream = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as error:
        raise ValueError(f"{context}: cannot open: {error}") from error

    entries: Dict[Tuple[str, str], ManifestEntry] = {}
    with stream:
        reader = csv.DictReader(stream)
        fieldnames = validate_header(reader.fieldnames, {"case_id", "affine"}, context)
        has_op = "op" in fieldnames
        has_rows = "rows" in fieldnames
        has_cols = "cols" in fieldnames
        for line_number, raw_row in enumerate(reader, start=2):
            row = clean_csv_row(raw_row, line_number, context)
            if not row:
                continue
            case_id = row["case_id"].strip()
            if not case_id:
                raise ValueError(f"{context}: row {line_number}: case_id is empty")
            op = ""
            if has_op and row["op"].strip():
                op = normalized_op(row["op"], f"{context}: row {line_number}")
            affine = parse_boolean(
                row["affine"], "affine", f"{context}: row {line_number}"
            )
            rows = (
                parse_positive_integer(
                    row["rows"], "rows", f"{context}: row {line_number}"
                )
                if has_rows and row["rows"].strip()
                else None
            )
            cols = (
                parse_positive_integer(
                    row["cols"], "cols", f"{context}: row {line_number}"
                )
                if has_cols and row["cols"].strip()
                else None
            )
            key = (op, case_id)
            if key in entries:
                previous = entries[key]
                raise ValueError(
                    f"{context}: duplicate case {case_id!r} at rows "
                    f"{previous.line_number} and {line_number}"
                )
            entries[key] = ManifestEntry(affine, rows, cols, line_number)
    if not entries:
        raise ValueError(f"{context}: contains no shape rows")
    return entries


def affine_from_variant(variant: str, context: str) -> Optional[bool]:
    tokens = set(re.findall(r"[a-z0-9]+", variant.lower()))
    has_affine = "affine" in tokens
    has_plain = "plain" in tokens
    if has_affine and has_plain:
        raise ValueError(
            f"{context}: variant contains conflicting affine/plain tokens: {variant!r}"
        )
    if has_affine:
        return True
    if has_plain:
        return False
    return None


def find_manifest_entry(
    manifest: Mapping[Tuple[str, str], ManifestEntry], op: str, case_id: str
) -> Optional[ManifestEntry]:
    exact = manifest.get((op, case_id))
    generic = manifest.get(("", case_id))
    if (
        exact is not None
        and generic is not None
        and (exact.affine, exact.rows, exact.cols)
        != (generic.affine, generic.rows, generic.cols)
    ):
        raise ValueError(
            f"case {case_id!r}: shape manifest has conflicting op-specific and generic rows"
        )
    return exact if exact is not None else generic


def resolve_affine(
    row: Mapping[str, str],
    op: str,
    rows: int,
    cols: int,
    manifest: Mapping[Tuple[str, str], ManifestEntry],
    context: str,
) -> bool:
    if op != "rms_norm":
        return False

    candidates: List[Tuple[str, bool]] = []
    if "affine" in row and row["affine"].strip():
        candidates.append(
            (
                "performance CSV affine",
                parse_boolean(row["affine"], "affine", context),
            )
        )

    case_id = row["case_id"].strip()
    entry = find_manifest_entry(manifest, op, case_id) if manifest else None
    if entry is not None:
        if entry.rows is not None and entry.rows != rows:
            raise ValueError(
                f"{context}: shape manifest rows={entry.rows} disagrees with performance "
                f"rows={rows} for case {case_id!r}"
            )
        if entry.cols is not None and entry.cols != cols:
            raise ValueError(
                f"{context}: shape manifest cols={entry.cols} disagrees with performance "
                f"cols={cols} for case {case_id!r}"
            )
        candidates.append(("shape manifest affine", entry.affine))

    variant_value = affine_from_variant(row["variant"], context)
    if variant_value is not None:
        candidates.append(("variant", variant_value))

    if not candidates:
        raise ValueError(
            f"{context}: cannot determine RMSNorm affine/plain mode; provide an affine "
            "column, an unambiguous variant token, or --shape-manifest"
        )
    values = {value for _, value in candidates}
    if len(values) != 1:
        details = ", ".join(f"{source}={int(value)}" for source, value in candidates)
        raise ValueError(f"{context}: conflicting RMSNorm affine classification: {details}")
    return candidates[0][1]


def compute_work_metrics(op: str, rows: int, cols: int, affine: bool) -> WorkMetrics:
    elements = rows * cols
    if op == "softmax":
        return WorkMetrics(
            model="softmax_fwd_v1",
            fp32_ops=3 * elements - rows,
            fp16_ops=0,
            special_op_kind="exp",
            special_ops=elements,
            compare_ops=elements - rows,
        )
    if op == "rms_norm":
        return WorkMetrics(
            model="rmsnorm_affine_fwd_v1" if affine else "rmsnorm_plain_fwd_v1",
            fp32_ops=3 * elements + rows,
            fp16_ops=elements if affine else 0,
            special_op_kind="rsqrt",
            special_ops=rows,
            compare_ops=0,
        )
    raise ValueError(f"unsupported normalized op {op!r}")


def format_rate(value: float) -> str:
    return format(value, ".17g")


def derive_rows(
    stream: TextIO,
    input_path: Path,
    manifest: Mapping[Tuple[str, str], ManifestEntry],
) -> Tuple[List[str], List[Dict[str, str]], int]:
    context = f"performance CSV {input_path}"
    reader = csv.DictReader(stream)
    fieldnames = validate_header(reader.fieldnames, REQUIRED_PERF_COLUMNS, context)
    overlap = sorted(set(fieldnames).intersection(METRIC_COLUMNS))
    if overlap:
        raise ValueError(
            f"{context}: input already contains derived metric columns: {', '.join(overlap)}"
        )

    output_rows: List[Dict[str, str]] = []
    measured_rows = 0
    for line_number, raw_row in enumerate(reader, start=2):
        row = clean_csv_row(raw_row, line_number, context)
        if not row:
            continue
        row_context = f"{context}: row {line_number}"
        op = normalized_op(row["op"], row_context)
        rows = parse_positive_integer(row["rows"], "rows", row_context)
        cols = parse_positive_integer(row["cols"], "cols", row_context)
        affine = resolve_affine(row, op, rows, cols, manifest, row_context)
        work = compute_work_metrics(op, rows, cols, affine)

        derived = {
            "metric_schema_version": "1",
            "metric_model": work.model,
            "fp32_ops": str(work.fp32_ops),
            "fp16_ops": str(work.fp16_ops),
            "arith_ops": str(work.arith_ops),
            "arith_tflops": "",
            "special_op_kind": work.special_op_kind,
            "special_ops": str(work.special_ops),
            "special_gops": "",
            "compare_ops": str(work.compare_ops),
            "compare_gops": "",
        }
        if row["status"].strip().lower() == "pass":
            median_ms = parse_positive_finite_float(
                row["lat_ms_median"], "lat_ms_median", row_context
            )
            derived["arith_tflops"] = format_rate(work.arith_ops / (median_ms * 1.0e9))
            derived["special_gops"] = format_rate(
                work.special_ops / (median_ms * 1.0e6)
            )
            derived["compare_gops"] = format_rate(
                work.compare_ops / (median_ms * 1.0e6)
            )
            measured_rows += 1
        output_rows.append({**row, **derived})

    if not output_rows:
        raise ValueError(f"{context}: contains no result rows")
    return fieldnames + METRIC_COLUMNS, output_rows, measured_rows


def write_rows(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, str]], force: bool
) -> None:
    if path.exists() and not force:
        raise ValueError(f"output {path} already exists; pass --force to replace it")
    mode = "w" if force else "x"
    try:
        with path.open(mode, encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    except OSError as error:
        raise ValueError(f"cannot write output {path}: {error}") from error


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append versioned arithmetic, exp/rsqrt, and comparison throughput "
            "metrics to a Softmax/RMSNorm performance CSV."
        )
    )
    parser.add_argument("input", type=Path, help="input perf_history.csv")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("perf_metrics_v1.csv"),
        help="derived CSV path (default: ./perf_metrics_v1.csv)",
    )
    parser.add_argument(
        "--shape-manifest",
        "--shapes",
        dest="shape_manifest",
        type=Path,
        help="optional shape CSV containing case_id, affine, and optionally op/rows/cols",
    )
    parser.add_argument(
        "--force", action="store_true", help="replace an existing output file"
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.input.resolve() == args.output.resolve():
            raise ValueError("input and output paths must be different")
        manifest = (
            read_shape_manifest(args.shape_manifest) if args.shape_manifest else {}
        )
        try:
            input_stream = args.input.open("r", encoding="utf-8-sig", newline="")
        except OSError as error:
            raise ValueError(f"performance CSV {args.input}: cannot open: {error}") from error
        with input_stream:
            fieldnames, rows, measured_rows = derive_rows(
                input_stream, args.input, manifest
            )
        write_rows(args.output, fieldnames, rows, args.force)
    except (csv.Error, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(
        f"wrote {len(rows)} rows ({measured_rows} pass rows with rates) to {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
