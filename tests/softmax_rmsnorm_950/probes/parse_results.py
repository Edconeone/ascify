#!/usr/bin/env python3
"""Validate and summarize simt_hw_probes CSV output using only the stdlib."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TextIO


REQUIRED_COLUMNS = {
    "schema_version",
    "probe",
    "variant",
    "access_bytes",
    "threads",
    "blocks",
    "elements",
    "inner_iterations",
    "timed_launches",
    "elapsed_ms_median",
    "elapsed_ms_min",
    "ns_per_launch",
    "traffic_bytes",
    "gbps",
    "operations",
    "gops",
    "checksum",
    "status",
}


@dataclass(frozen=True)
class Summary:
    section: str
    item: str
    value: float
    unit: str
    details: str


def result_key(row: dict[str, str]) -> tuple[str, str, int]:
    discriminator = 0
    if row["probe"] == "copy_bandwidth":
        discriminator = int(parse_number(row, "access_bytes"))
    elif row["probe"] == "thread_scaling":
        discriminator = int(parse_number(row, "threads"))
    return row["probe"], row["variant"], discriminator


def expected_result_keys() -> set[tuple[str, str, int]]:
    keys = {
        ("launch_floor", "one_block", 0),
        ("launch_floor", "all_aiv", 0),
        ("warp_reduce", "asc_reduce_add", 0),
        ("warp_reduce", "shuffle_xor_5stage", 0),
        ("math", "expf", 0),
        ("math", "rsqrtf", 0),
    }
    keys.update(("copy_bandwidth", "width_sweep", width) for width in (2, 4, 8, 16))
    keys.update(
        ("thread_scaling", "copy_16b", threads)
        for threads in (32, 64, 128, 256, 512, 1024)
    )
    keys.add(("thread_scaling", "copy_16b_2x1024", 2048))
    return keys


def coverage_issues(rows: Iterable[dict[str, str]]) -> list[str]:
    counts = Counter(result_key(row) for row in rows)
    expected = expected_result_keys()
    missing = sorted(expected.difference(counts))
    unexpected = sorted(set(counts).difference(expected))
    duplicates = sorted(key for key, count in counts.items() if count != 1)
    issues: list[str] = []
    if missing:
        issues.append(
            "missing=" + ";".join(f"{probe}/{variant}/{value}" for probe, variant, value in missing)
        )
    if unexpected:
        issues.append(
            "unexpected="
            + ";".join(f"{probe}/{variant}/{value}" for probe, variant, value in unexpected)
        )
    if duplicates:
        issues.append(
            "duplicates="
            + ";".join(
                f"{probe}/{variant}/{value}x{counts[(probe, variant, value)]}"
                for probe, variant, value in duplicates
            )
        )
    return issues


def parse_number(row: dict[str, str], column: str) -> float:
    text = row.get(column, "")
    try:
        value = float(text)
    except ValueError as error:
        raise ValueError(
            f"row {row.get('_line', '?')}: {column} is not numeric: {text!r}"
        ) from error
    if not math.isfinite(value):
        raise ValueError(
            f"row {row.get('_line', '?')}: {column} is not finite: {text!r}"
        )
    return value


def read_rows(stream: TextIO) -> list[dict[str, str]]:
    reader = csv.DictReader(stream)
    if reader.fieldnames is None:
        raise ValueError("input has no CSV header")
    missing = sorted(REQUIRED_COLUMNS.difference(reader.fieldnames))
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")

    rows: list[dict[str, str]] = []
    numeric_columns = REQUIRED_COLUMNS.difference({"probe", "variant", "status"})
    for line_number, raw_row in enumerate(reader, start=2):
        if None in raw_row:
            raise ValueError(f"row {line_number}: more fields than the CSV header")
        row = {key: (value or "").strip() for key, value in raw_row.items()}
        row["_line"] = str(line_number)
        if row["schema_version"] != "1":
            raise ValueError(
                f"row {line_number}: unsupported schema_version "
                f"{row['schema_version']!r}"
            )
        if not row["probe"] or not row["variant"] or not row["status"]:
            raise ValueError(f"row {line_number}: probe, variant and status are required")
        for column in numeric_columns:
            parse_number(row, column)
        rows.append(row)
    if not rows:
        raise ValueError("input contains no result rows")
    return rows


def detail(row: dict[str, str], *, include_access: bool = False) -> str:
    parts = [
        f"threads={int(parse_number(row, 'threads'))}",
        f"blocks={int(parse_number(row, 'blocks'))}",
        f"median={parse_number(row, 'elapsed_ms_median'):.6g} ms",
    ]
    if include_access:
        parts.insert(0, f"access={int(parse_number(row, 'access_bytes'))} B")
    return ", ".join(parts)


def summarize(rows: Iterable[dict[str, str]]) -> list[Summary]:
    all_rows = list(rows)
    ok_rows = [row for row in all_rows if row["status"] == "ok"]
    summaries: list[Summary] = []

    launch_rows = sorted(
        (row for row in ok_rows if row["probe"] == "launch_floor"),
        key=lambda row: row["variant"],
    )
    for row in launch_rows:
        summaries.append(
            Summary(
                "launch_floor",
                row["variant"],
                parse_number(row, "ns_per_launch"),
                "ns/launch",
                detail(row),
            )
        )

    copy_rows = [row for row in ok_rows if row["probe"] == "copy_bandwidth"]
    widths = sorted({int(parse_number(row, "access_bytes")) for row in copy_rows})
    for width in widths:
        candidates = [
            row
            for row in copy_rows
            if int(parse_number(row, "access_bytes")) == width
        ]
        best = max(candidates, key=lambda row: parse_number(row, "gbps"))
        summaries.append(
            Summary(
                "copy_bandwidth",
                f"{width}B",
                parse_number(best, "gbps"),
                "GB/s",
                detail(best),
            )
        )

    scaling_rows = sorted(
        (row for row in ok_rows if row["probe"] == "thread_scaling"),
        key=lambda row: parse_number(row, "threads"),
    )
    best_scaling = (
        max(scaling_rows, key=lambda row: parse_number(row, "gbps"))
        if scaling_rows
        else None
    )
    for row in scaling_rows:
        threads = int(parse_number(row, "threads"))
        best_suffix = " [best]" if row is best_scaling else ""
        summaries.append(
            Summary(
                "thread_scaling",
                str(threads),
                parse_number(row, "gbps"),
                "GB/s",
                detail(row, include_access=True)
                + f", variant={row['variant']}"
                + best_suffix,
            )
        )

    reduce_rows = {
        row["variant"]: row for row in ok_rows if row["probe"] == "warp_reduce"
    }
    for variant in ("asc_reduce_add", "shuffle_xor_5stage"):
        row = reduce_rows.get(variant)
        if row is not None:
            summaries.append(
                Summary(
                    "warp_reduce",
                    variant,
                    parse_number(row, "gops"),
                    "Gwarp-reductions/s",
                    detail(row),
                )
            )
    hardware = reduce_rows.get("asc_reduce_add")
    shuffle = reduce_rows.get("shuffle_xor_5stage")
    if hardware is not None and shuffle is not None:
        shuffle_rate = parse_number(shuffle, "gops")
        if shuffle_rate > 0.0:
            summaries.append(
                Summary(
                    "warp_reduce",
                    "hardware_vs_shuffle",
                    parse_number(hardware, "gops") / shuffle_rate,
                    "x",
                    ">1 means asc_reduce_add is faster",
                )
            )

    math_rows = sorted(
        (row for row in ok_rows if row["probe"] == "math"),
        key=lambda row: row["variant"],
    )
    for row in math_rows:
        summaries.append(
            Summary(
                "math",
                row["variant"],
                parse_number(row, "gops"),
                "Gcalls/s",
                detail(row),
            )
        )

    rejected = [row for row in all_rows if row["status"] != "ok"]
    if rejected:
        statuses: dict[str, int] = {}
        for row in rejected:
            statuses[row["status"]] = statuses.get(row["status"], 0) + 1
        summaries.append(
            Summary(
                "validation",
                "non_ok_rows",
                float(len(rejected)),
                "rows",
                ", ".join(f"{key}={statuses[key]}" for key in sorted(statuses)),
            )
        )
    coverage = coverage_issues(all_rows)
    if coverage:
        summaries.append(
            Summary(
                "validation",
                "coverage_issues",
                float(len(coverage)),
                "issues",
                " | ".join(coverage),
            )
        )
    return summaries


def format_value(value: float) -> str:
    if value == 0.0:
        return "0"
    if abs(value) >= 1000.0:
        return f"{value:.3f}"
    return f"{value:.6g}"


def markdown_escape(text: str) -> str:
    return text.replace("|", r"\|").replace("\n", " ")


def write_markdown(summaries: Iterable[Summary], stream: TextIO) -> None:
    stream.write("| Section | Item | Value | Unit | Details |\n")
    stream.write("|---|---|---:|---|---|\n")
    for item in summaries:
        stream.write(
            f"| {markdown_escape(item.section)} | {markdown_escape(item.item)} | "
            f"{format_value(item.value)} | {markdown_escape(item.unit)} | "
            f"{markdown_escape(item.details)} |\n"
        )


def write_csv(summaries: Iterable[Summary], stream: TextIO) -> None:
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["section", "item", "value", "unit", "details"])
    for item in summaries:
        writer.writerow(
            [item.section, item.item, format_value(item.value), item.unit, item.details]
        )


def open_input(path: str) -> tuple[TextIO, bool]:
    if path == "-":
        return sys.stdin, False
    return Path(path).open("r", encoding="utf-8", newline=""), True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and summarize simt_hw_probes structured CSV."
    )
    parser.add_argument("input", nargs="?", default="-", help="CSV path, or - for stdin")
    parser.add_argument(
        "--format", choices=("markdown", "csv"), default="markdown", help="summary format"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return non-zero if any probe row has status other than ok",
    )
    arguments = parser.parse_args()

    input_stream, should_close = open_input(arguments.input)
    try:
        rows = read_rows(input_stream)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    finally:
        if should_close:
            input_stream.close()

    summaries = summarize(rows)
    if arguments.format == "csv":
        write_csv(summaries, sys.stdout)
    else:
        write_markdown(summaries, sys.stdout)
    if arguments.strict and (
        any(row["status"] != "ok" for row in rows) or coverage_issues(rows)
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
