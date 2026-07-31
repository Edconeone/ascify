#!/usr/bin/env python3
"""Strict validation and evidence packaging for formal 950PR runs."""

from __future__ import annotations

import argparse
import ast
import csv
import ctypes
import datetime as dt
import errno
import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
import types
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Optional, Sequence


RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TIER_RANK = {"smoke": 0, "tune": 1, "full": 2}
CONVERSION_SCHEMA = "ascify-conversion-evidence-v3"
TEST_ROOT = Path(__file__).resolve().parent.parent
CANONICAL_FORMAL_SHAPES = {
    "boundary_shapes": (
        TEST_ROOT / "shapes" / "correctness.csv",
        "162debfbe0d6ec567a04018024673c17bdb72784c0956478cba7ff4d52cd5bee",
    ),
    "tune_shapes": (
        TEST_ROOT / "shapes" / "unified_tune.csv",
        "306e6cf1cd9d6af2bf391d5f730258d58932b55e2f19b9576dfa8911d4fe42f9",
    ),
}
FORMAL_PHASE_TIMING = {
    "preheat": (5, 5, 5),
    "direct_perf_a": (20, 50, 20),
    "native_perf": (20, 50, 20),
    "direct_perf_b": (20, 50, 20),
}
FORMAL_PHASE_CHECK_MAX_ELEMENTS = {
    "direct_boundary": 5_000_000,
    "native_boundary": 5_000_000,
    "direct_tune": 10_000_000,
    "native_tune": 10_000_000,
    "preheat": 10_000_000,
    "direct_perf_a": 10_000_000,
    "native_perf": 10_000_000,
    "direct_perf_b": 10_000_000,
}
FORMAL_IDLE_METRICS = (
    "Aicore Usage Rate",
    "Aivector Usage Rate",
    "HBM Bandwidth Usage Rate",
    "NPU Utilization",
)
CONVERSION_POLICY = {
    "target_policy": "dav-c310-vec",
    "simt_math": "fast",
}
VARIANTS = {
    "direct": {
        "softmax": {"converted_dispatch"},
        "rms_norm": {"converted_plain", "converted_affine"},
    },
    "native": {
        "softmax": {"native_half2_hybrid"},
        "rms_norm": {"native_plain", "native_affine"},
    },
}

COMMON_MANIFEST_FILES = {
    "shapes",
    "evidence_csv",
    "ccec_binary",
    "cann_set_env",
    "softmax_native_source",
    "rmsnorm_native_source",
    "generated_softmax_header",
    "generated_rmsnorm_header",
    "generated_layer_norm_header",
    "generated_rmsnorm_adapter",
    "rmsnorm_adapter_input",
    "dav_c310_recipe_header",
    "dav_c310_softmax_impl",
    "dav_c310_rmsnorm_impl",
    "ascify_recipe_source",
    "ascify_recipe_header",
    "runner_common_header",
    "build_script",
    "run_smoke_script",
    "select_device_script",
    "formal_runner_script",
    "validator_script",
    "validator_test",
    "safe_lock_helper",
    "derive_work_metrics_tool",
    "derive_work_metrics_test",
    "ascify_conversion_manifest",
    "formal_build_input_snapshot",
    "formal_binary_bundle_manifest",
    "aclcub_header",
    "boundary_shapes",
    "tune_shapes",
}

BRACKET_STABLE_FILES = COMMON_MANIFEST_FILES - {"evidence_csv", "shapes"}
BRACKET_STABLE_FILES |= {
    "softmax_bench_source",
    "rmsnorm_bench_source",
}

FORMAL_BUILD_INPUT_FILES = {
    "conversion_manifest",
    "generated_softmax",
    "generated_layer_norm",
    "generated_rmsnorm",
    "generated_rmsnorm_adapter",
    "target_recipe_header",
    "target_softmax_impl",
    "target_rmsnorm_impl",
    "ascify_recipe_source",
    "ascify_recipe_header",
    "softmax_native_source",
    "rmsnorm_native_source",
    "rmsnorm_adapter_input",
    "runner_common_header",
    "softmax_check_source",
    "rmsnorm_check_source",
    "softmax_bench_source",
    "rmsnorm_bench_source",
    "build_script",
    "run_smoke_script",
    "formal_runner_script",
    "validator_script",
    "validator_test",
    "safe_lock_helper",
    "derive_work_metrics_tool",
    "derive_work_metrics_test",
    "select_device_script",
    "ccec_binary",
    "cann_set_env",
    "aclcub_header",
    "boundary_shapes",
    "tune_shapes",
}

FORMAL_BINARY_NAMES = {
    "softmax_check_production_fast",
    "rmsnorm_check_production_fast",
    "softmax_bench_production_fast",
    "rmsnorm_bench_production_fast",
    "softmax_check_production_fast_native_half2_hybrid",
    "rmsnorm_check_production_fast_native",
    "softmax_bench_production_fast_native_half2_hybrid",
    "rmsnorm_bench_production_fast_native",
}

FORMAL_PHASE_SHAPE_FILE = {
    "direct_boundary": "boundary_shapes",
    "native_boundary": "boundary_shapes",
    "direct_tune": "tune_shapes",
    "native_tune": "tune_shapes",
    "preheat": "tune_shapes",
    "direct_perf_a": "tune_shapes",
    "native_perf": "tune_shapes",
    "direct_perf_b": "tune_shapes",
}
FORMAL_PHASE_PROTOCOL = {
    "direct_boundary": {
        "execution_kind": "accuracy",
        "shape_role": "boundary_shapes",
        "tier": "smoke",
        "variant_mode": "direct",
        "expected_softmax": 42,
        "expected_rmsnorm": 18,
    },
    "native_boundary": {
        "execution_kind": "accuracy",
        "shape_role": "boundary_shapes",
        "tier": "smoke",
        "variant_mode": "native",
        "expected_softmax": 42,
        "expected_rmsnorm": 18,
    },
    "direct_tune": {
        "execution_kind": "accuracy",
        "shape_role": "tune_shapes",
        "tier": "tune",
        "variant_mode": "direct",
        "expected_softmax": 5,
        "expected_rmsnorm": 10,
    },
    "native_tune": {
        "execution_kind": "accuracy",
        "shape_role": "tune_shapes",
        "tier": "tune",
        "variant_mode": "native",
        "expected_softmax": 5,
        "expected_rmsnorm": 10,
    },
    "preheat": {
        "execution_kind": "perf",
        "shape_role": "tune_shapes",
        "tier": "tune",
        "variant_mode": "direct",
        "expected_softmax": 5,
        "expected_rmsnorm": 10,
    },
    "direct_perf_a": {
        "execution_kind": "perf",
        "shape_role": "tune_shapes",
        "tier": "tune",
        "variant_mode": "direct",
        "expected_softmax": 5,
        "expected_rmsnorm": 10,
    },
    "native_perf": {
        "execution_kind": "perf",
        "shape_role": "tune_shapes",
        "tier": "tune",
        "variant_mode": "native",
        "expected_softmax": 5,
        "expected_rmsnorm": 10,
    },
    "direct_perf_b": {
        "execution_kind": "perf",
        "shape_role": "tune_shapes",
        "tier": "tune",
        "variant_mode": "direct",
        "expected_softmax": 5,
        "expected_rmsnorm": 10,
    },
}

RUN_MANIFEST_TO_FREEZE_FILE = {
    "ccec_binary": "ccec_binary",
    "cann_set_env": "cann_set_env",
    "softmax_native_source": "softmax_native_source",
    "rmsnorm_native_source": "rmsnorm_native_source",
    "generated_softmax_header": "generated_softmax",
    "generated_rmsnorm_header": "generated_rmsnorm",
    "generated_layer_norm_header": "generated_layer_norm",
    "generated_rmsnorm_adapter": "generated_rmsnorm_adapter",
    "rmsnorm_adapter_input": "rmsnorm_adapter_input",
    "dav_c310_recipe_header": "target_recipe_header",
    "dav_c310_softmax_impl": "target_softmax_impl",
    "dav_c310_rmsnorm_impl": "target_rmsnorm_impl",
    "ascify_recipe_source": "ascify_recipe_source",
    "ascify_recipe_header": "ascify_recipe_header",
    "runner_common_header": "runner_common_header",
    "softmax_check_source": "softmax_check_source",
    "rmsnorm_check_source": "rmsnorm_check_source",
    "softmax_bench_source": "softmax_bench_source",
    "rmsnorm_bench_source": "rmsnorm_bench_source",
    "build_script": "build_script",
    "run_smoke_script": "run_smoke_script",
    "formal_runner_script": "formal_runner_script",
    "validator_script": "validator_script",
    "validator_test": "validator_test",
    "safe_lock_helper": "safe_lock_helper",
    "derive_work_metrics_tool": "derive_work_metrics_tool",
    "derive_work_metrics_test": "derive_work_metrics_test",
    "select_device_script": "select_device_script",
    "ascify_conversion_manifest": "conversion_manifest",
    "aclcub_header": "aclcub_header",
    "boundary_shapes": "boundary_shapes",
    "tune_shapes": "tune_shapes",
}

CONVERSION_CORE_ARTIFACTS = {
    "ascify_binary": "converter",
    "recipe_source": "source",
    "recipe_header": "source",
    "input_softmax": "input",
    "input_layer_norm": "input",
    "input_rmsnorm": "input",
    "input_rmsnorm_adapter": "input",
    "generated_softmax_header": "output",
    "generated_layer_norm_header": "output",
    "generated_rmsnorm_header": "output",
    "generated_rmsnorm_adapter": "output",
}

CONVERSION_UNITS = {
    "softmax": ("input_softmax", "generated_softmax_header"),
    "layer_norm": ("input_layer_norm", "generated_layer_norm_header"),
    "rmsnorm": ("input_rmsnorm", "generated_rmsnorm_header"),
    "rmsnorm_adapter": (
        "input_rmsnorm_adapter",
        "generated_rmsnorm_adapter",
    ),
}

CONVERSION_FORCED_INCLUDES = {
    "softmax": [],
    "layer_norm": [
        "-include",
        "cuda_fp16.h",
        "-include",
        "cuda_bf16.h",
    ],
    "rmsnorm": [
        "-include",
        "cuda_fp16.h",
        "-include",
        "cuda_bf16.h",
    ],
    "rmsnorm_adapter": [
        "-include",
        "cuda_fp16.h",
        "-include",
        "cuda_bf16.h",
    ],
}

RECIPE_TOPOLOGY_TOKENS = {
    "target_header_includes": (
        "#include <ascify/target/dav_c310/rowwise_norm_recipes.hpp>"
    ),
    "direct_load_markers": "using ascify_target_direct_load_tag = void;",
    "direct_store_markers": "using ascify_target_direct_store_tag = void;",
    "softmax_direct_wrappers": "::ascify::target::dav_c310::TrySoftmax(",
    "rmsnorm_direct_wrappers": "::ascify::target::dav_c310::TryRmsNorm(",
}

EXPECTED_CONVERSION_TOPOLOGY = {
    "softmax": {
        "target_header_includes": 1,
        "direct_load_markers": 1,
        "direct_store_markers": 1,
        "softmax_direct_wrappers": 3,
        "rmsnorm_direct_wrappers": 0,
    },
    "layer_norm": {
        "target_header_includes": 0,
        "direct_load_markers": 1,
        "direct_store_markers": 1,
        "softmax_direct_wrappers": 0,
        "rmsnorm_direct_wrappers": 0,
    },
    "rmsnorm": {
        "target_header_includes": 1,
        "direct_load_markers": 0,
        "direct_store_markers": 0,
        "softmax_direct_wrappers": 0,
        "rmsnorm_direct_wrappers": 3,
    },
    "rmsnorm_adapter": {
        "target_header_includes": 0,
        "direct_load_markers": 0,
        "direct_store_markers": 1,
        "softmax_direct_wrappers": 0,
        "rmsnorm_direct_wrappers": 0,
    },
}

DIRECT_RECIPE_WRAPPERS = {
    "softmax": (
        "LaunchSoftmaxWarpImpl",
        "LaunchSoftmaxBlockSMemImpl",
        "LaunchSoftmaxBlockUncachedImpl",
    ),
    "layer_norm": (),
    "rmsnorm": (
        "LaunchRmsNormWarpImpl",
        "LaunchRmsNormBlockSMemImpl",
        "LaunchRmsNormBlockUncachedImpl",
    ),
    "rmsnorm_adapter": (),
}

FORBIDDEN_RECIPE_DISPATCHERS = {
    "softmax": ("DispatchSoftmax", "DispatchLogSoftmax"),
    "layer_norm": (),
    "rmsnorm": ("LaunchRmsNorm",),
    "rmsnorm_adapter": (),
}

FORBIDDEN_RECIPE_DISPATCHER_DEFINITION_COUNTS = {
    "softmax": {
        "DispatchSoftmax": 2,
        "DispatchLogSoftmax": 2,
    },
    "layer_norm": {},
    "rmsnorm": {
        "LaunchRmsNorm": 2,
    },
    "rmsnorm_adapter": {},
}


def expected_recipe_placement(unit: str) -> dict[str, object]:
    wrapper_kind = (
        "softmax" if unit == "softmax"
        else "rmsnorm" if unit == "rmsnorm"
        else ""
    )
    wrappers = {}
    for name in DIRECT_RECIPE_WRAPPERS[unit]:
        wrappers[name] = {
            "definition_count": 1,
            "cuda_launches": 1,
            "softmax_try_calls": 1 if wrapper_kind == "softmax" else 0,
            "rmsnorm_try_calls": 1 if wrapper_kind == "rmsnorm" else 0,
            "entry_recipe_prologues": (
                1 if wrapper_kind == "softmax" else 0
            ),
            "entry_recipe_argument_bindings": (
                1 if wrapper_kind == "softmax" else 0
            ),
            "post_geometry_pre_launch_recipe_prologues": (
                1 if wrapper_kind == "rmsnorm" else 0
            ),
            "post_geometry_pre_launch_recipe_argument_bindings": (
                1 if wrapper_kind == "rmsnorm" else 0
            ),
            "softmax_ksoftmax_guards": 1 if wrapper_kind == "softmax" else 0,
            "softmax_klogsoftmax_refs": 0,
        }
    return {
        "direct_wrappers": wrappers,
        "forbidden_dispatcher_definitions": (
            FORBIDDEN_RECIPE_DISPATCHER_DEFINITION_COUNTS[unit]
        ),
        "unexpected_try_calls": 0,
        "forbidden_dispatcher_try_calls": 0,
        "forbidden_logsoftmax_try_calls": 0,
    }


EXPECTED_RECIPE_PLACEMENT = {
    unit: expected_recipe_placement(unit)
    for unit in CONVERSION_UNITS
}

RECIPE_SUMMARY_PREFIX = "Ascify dav-c310 recipe:"
DIRECT_AB_SPREAD_LIMIT = 1.05
DIRECT_CASE_NATIVE_RATIO_MIN = 0.90
DIRECT_CLASS_NATIVE_GEOMEAN_MIN = 0.95
PERFORMANCE_CLASSES = ("softmax", "rms_plain", "rms_affine")
EXPECTED_FORMAL_GATES = {
    "direct_boundary": ("smoke", "42", "18", "converted", "converted"),
    "native_boundary": (
        "smoke",
        "42",
        "18",
        "native-half2-hybrid",
        "native",
    ),
    "direct_tune": ("tune", "5", "10", "converted", "converted"),
    "native_tune": (
        "tune",
        "5",
        "10",
        "native-half2-hybrid",
        "native",
    ),
}
BRACKET_SUMMARY_FIELDS = (
    "formal_set_id",
    "op",
    "case_id",
    "performance_class",
    "direct_a_run_id",
    "native_run_id",
    "direct_b_run_id",
    "direct_a_gbps",
    "direct_b_gbps",
    "direct_ab_geomean_gbps",
    "native_gbps",
    "direct_center_over_native",
    "direct_case_over_native_min_ratio",
    "direct_class_over_native_geomean",
    "direct_class_over_native_geomean_min_ratio",
    "direct_ab_spread_ratio",
    "direct_ab_relative_drift",
)
RUNTIME_CONFIG_FIELDS = (
    "runtime_config_schema_version",
    "run_id",
    "op",
    "variant",
    "device_id",
    "target_entry",
    "requested_grid_cap",
    "vector_core_count",
    "resolved_grid_cap",
    "grid_policy",
    "block_threads_policy",
)
FORMAL_METRIC_FIELDS = (
    "metric_schema_version",
    "formal_set_id",
    "run_role",
    "run_id",
    "op",
    "variant",
    "case_id",
    "rows",
    "cols",
    "affine",
    "lat_ms_median",
    "gbps",
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
)


class ValidationError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise ValidationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        fail(f"{label} must be 64 lowercase hexadecimal SHA-256 digits")
    return value


def require_nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        fail(f"{label} must be a non-empty string")
    return value


def contained_path(root: Path, relative_text: object, label: str) -> Path:
    raw = require_nonempty_string(relative_text, label)
    if "\\" in raw or "\x00" in raw:
        fail(f"{label} is not a portable POSIX relative path: {raw!r}")
    relative = PurePosixPath(raw)
    if (
        relative.is_absolute()
        or any(part in ("", ".", "..") for part in relative.parts)
        or str(relative) != raw
    ):
        fail(f"{label} is not a canonical relative path: {raw!r}")
    root_resolved = root.resolve()
    candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            fail(f"{label} traverses a symlink: {current}")
    resolved = candidate.resolve()
    try:
        common = Path(os.path.commonpath((str(root_resolved), str(resolved))))
    except ValueError:
        fail(f"{label} escapes evidence root: {raw!r}")
        raise AssertionError
    if common != root_resolved:
        fail(f"{label} escapes evidence root: {raw!r}")
    if not resolved.is_file():
        fail(f"{label} is missing or not a regular file: {resolved}")
    return resolved


def read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        fail(f"missing JSON manifest: {path}")
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        fail(f"JSON manifest root must be an object: {path}")
    return value


def _preprocessor_integer(
    name: str, macros: Mapping[str, str], resolving: Optional[set[str]] = None
) -> int:
    if name not in macros:
        return 0
    resolving = set() if resolving is None else set(resolving)
    if name in resolving:
        fail(f"recursive preprocessor value for {name}")
    resolving.add(name)
    value = macros[name].strip()
    if not value:
        return 1
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        return _preprocessor_integer(value, macros, resolving)
    integer = re.fullmatch(
        r"(?P<sign>[+-]?)(?P<body>(?:0[xX][0-9A-Fa-f]+)|(?:0[bB][01]+)|"
        r"(?:0[0-7]+)|(?:[0-9]+))[uUlL]*",
        value,
    )
    if integer:
        body = integer.group("body")
        base = (
            16
            if body.lower().startswith("0x")
            else 2
            if body.lower().startswith("0b")
            else 8
            if len(body) > 1 and body.startswith("0")
            else 10
        )
        parsed = int(body, base)
        return -parsed if integer.group("sign") == "-" else parsed
    # Feature macros often have an empty or non-numeric replacement. They are
    # still true when used as a simple preprocessor predicate.
    return 1


def _evaluate_preprocessor_expression(
    expression: str, macros: Mapping[str, str]
) -> bool:
    expression = re.sub(
        r"\bdefined\s*(?:\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)"
        r"|([A-Za-z_][A-Za-z0-9_]*))",
        lambda match: (
            "1" if (match.group(1) or match.group(2)) in macros else "0"
        ),
        expression,
    )
    expression = re.sub(
        r"\b[A-Za-z_][A-Za-z0-9_]*\b",
        lambda match: str(_preprocessor_integer(match.group(0), macros)),
        expression,
    )
    expression = re.sub(r"(?<=\d)[uUlL]+\b", "", expression)
    expression = expression.replace("&&", " and ").replace("||", " or ")
    expression = re.sub(r"!(?!=)", " not ", expression)
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as error:
        fail(f"unsupported preprocessor expression {expression!r}: {error.msg}")
        raise AssertionError

    def evaluate(node: ast.AST) -> int:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.BoolOp):
            values = [evaluate(value) != 0 for value in node.values]
            if isinstance(node.op, ast.And):
                return int(all(values))
            if isinstance(node.op, ast.Or):
                return int(any(values))
        if isinstance(node, ast.UnaryOp):
            value = evaluate(node.operand)
            if isinstance(node.op, ast.Not):
                return int(not value)
            if isinstance(node.op, ast.Invert):
                return ~value
            if isinstance(node.op, ast.UAdd):
                return value
            if isinstance(node.op, ast.USub):
                return -value
        if isinstance(node, ast.BinOp):
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, (ast.Div, ast.FloorDiv)):
                if right == 0:
                    fail("division by zero in preprocessor expression")
                return int(left / right)
            if isinstance(node.op, ast.Mod):
                if right == 0:
                    fail("modulo by zero in preprocessor expression")
                return left % right
            if isinstance(node.op, ast.LShift):
                return left << right
            if isinstance(node.op, ast.RShift):
                return left >> right
            if isinstance(node.op, ast.BitAnd):
                return left & right
            if isinstance(node.op, ast.BitOr):
                return left | right
            if isinstance(node.op, ast.BitXor):
                return left ^ right
        if isinstance(node, ast.Compare):
            left = evaluate(node.left)
            for operator, comparator in zip(node.ops, node.comparators):
                right = evaluate(comparator)
                if isinstance(operator, ast.Eq):
                    accepted = left == right
                elif isinstance(operator, ast.NotEq):
                    accepted = left != right
                elif isinstance(operator, ast.Lt):
                    accepted = left < right
                elif isinstance(operator, ast.LtE):
                    accepted = left <= right
                elif isinstance(operator, ast.Gt):
                    accepted = left > right
                elif isinstance(operator, ast.GtE):
                    accepted = left >= right
                else:
                    fail(
                        "unsupported comparison in preprocessor expression "
                        f"{expression!r}"
                    )
                    raise AssertionError
                if not accepted:
                    return 0
                left = right
            return 1
        fail(f"unsupported preprocessor expression node in {expression!r}")
        raise AssertionError

    return evaluate(tree) != 0


def active_cpp_source(
    text: str, predefined_macros: Optional[Mapping[str, str]] = None
) -> str:
    """Mask inactive preprocessor branches while preserving source offsets."""
    macros = dict(predefined_macros or {})
    original_lines = text.splitlines(keepends=True)
    masked_lines = mask_cpp_non_code(text).splitlines(keepends=True)
    if len(original_lines) != len(masked_lines):
        raise AssertionError("C++ masking changed the physical line count")
    output = list(original_lines)
    stack: list[dict[str, object]] = []
    active = True
    index = 0

    def blank_line(line: str) -> str:
        return "".join(char if char in "\r\n" else " " for char in line)

    while index < len(original_lines):
        end = index + 1
        while end < len(original_lines):
            logical_tail = masked_lines[end - 1].rstrip("\r\n")
            if not logical_tail.rstrip().endswith("\\"):
                break
            end += 1
        masked_logical = "".join(masked_lines[index:end])
        masked_logical = re.sub(r"\\\r?\n", "", masked_logical)
        directive = re.match(
            r"^[ \t]*#[ \t]*([A-Za-z_][A-Za-z0-9_]*)(.*)$",
            masked_logical,
            flags=re.DOTALL,
        )
        if directive is None:
            if not active:
                for line_index in range(index, end):
                    output[line_index] = blank_line(original_lines[line_index])
            index = end
            continue

        name = directive.group(1)
        argument = directive.group(2).strip()
        active_before = active
        if name in ("if", "ifdef", "ifndef"):
            parent_active = active
            if not parent_active:
                condition = False
            elif name == "ifdef":
                condition = argument in macros
            elif name == "ifndef":
                condition = argument not in macros
            else:
                condition = _evaluate_preprocessor_expression(argument, macros)
            current = parent_active and condition
            stack.append(
                {
                    "parent_active": parent_active,
                    "branch_taken": current,
                    "else_seen": False,
                }
            )
            active = current
        elif name == "elif":
            if not stack or bool(stack[-1]["else_seen"]):
                fail("unmatched or post-else #elif in C++ source")
            frame = stack[-1]
            eligible = bool(frame["parent_active"]) and not bool(
                frame["branch_taken"]
            )
            condition = (
                _evaluate_preprocessor_expression(argument, macros)
                if eligible
                else False
            )
            active = eligible and condition
            frame["branch_taken"] = bool(frame["branch_taken"]) or active
        elif name == "else":
            if not stack or bool(stack[-1]["else_seen"]):
                fail("unmatched or duplicate #else in C++ source")
            frame = stack[-1]
            active = bool(frame["parent_active"]) and not bool(frame["branch_taken"])
            frame["branch_taken"] = bool(frame["branch_taken"]) or active
            frame["else_seen"] = True
        elif name == "endif":
            if not stack:
                fail("unmatched #endif in C++ source")
            frame = stack.pop()
            active = bool(frame["parent_active"])
        elif active and name == "define":
            definition = re.match(r"([A-Za-z_][A-Za-z0-9_]*)(.*)", argument)
            if definition:
                macro_name = definition.group(1)
                replacement = definition.group(2)
                macros[macro_name] = (
                    "1" if replacement.startswith("(") else replacement.strip()
                )
        elif active and name == "undef":
            macro_name = argument.split(None, 1)[0] if argument else ""
            macros.pop(macro_name, None)

        keep_include = name == "include" and active_before
        if not keep_include:
            for line_index in range(index, end):
                output[line_index] = blank_line(original_lines[line_index])
        index = end

    if stack:
        fail("unterminated preprocessor conditional in C++ source")
    return "".join(output)


def recipe_topology(path: Path) -> dict[str, int]:
    text = active_cpp_source(path.read_text(encoding="utf-8", errors="strict"))
    text = mask_cpp_non_code(text)
    return {
        name: text.count(token)
        for name, token in RECIPE_TOPOLOGY_TOKENS.items()
    }


def mask_cpp_non_code(text: str) -> str:
    """Mask comments and quoted literals while preserving offsets and newlines."""
    masked = list(text)
    index = 0
    state = "code"
    quote = ""
    while index < len(text):
        current = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""
        if state == "code":
            if current == "/" and following == "/":
                masked[index] = masked[index + 1] = " "
                index += 2
                state = "line_comment"
                continue
            if current == "/" and following == "*":
                masked[index] = masked[index + 1] = " "
                index += 2
                state = "block_comment"
                continue
            if current in ("'", '"'):
                quote = current
                masked[index] = " "
                index += 1
                state = "quoted"
                continue
            index += 1
            continue
        if state == "line_comment":
            if current == "\n":
                state = "code"
            else:
                masked[index] = " "
            index += 1
            continue
        if state == "block_comment":
            if current == "*" and following == "/":
                masked[index] = masked[index + 1] = " "
                index += 2
                state = "code"
                continue
            if current != "\n":
                masked[index] = " "
            index += 1
            continue
        if current == "\\":
            masked[index] = " "
            if index + 1 < len(text):
                if text[index + 1] != "\n":
                    masked[index + 1] = " "
                index += 2
            else:
                index += 1
            continue
        if current == quote:
            masked[index] = " "
            index += 1
            state = "code"
            continue
        if current != "\n":
            masked[index] = " "
        index += 1
    return "".join(masked)


def matching_delimiter(
    text: str, start: int, opening: str, closing: str
) -> Optional[int]:
    if start >= len(text) or text[start] != opening:
        return None
    depth = 0
    for index in range(start, len(text)):
        if text[index] == opening:
            depth += 1
        elif text[index] == closing:
            depth -= 1
            if depth == 0:
                return index
    return None


def split_top_level(text: str) -> list[str]:
    pieces: list[str] = []
    start = 0
    round_depth = 0
    square_depth = 0
    brace_depth = 0
    angle_depth = 0
    for index, current in enumerate(text):
        if current == "(":
            round_depth += 1
        elif current == ")":
            round_depth -= 1
        elif current == "[":
            square_depth += 1
        elif current == "]":
            square_depth -= 1
        elif current == "{":
            brace_depth += 1
        elif current == "}":
            brace_depth -= 1
        elif current == "<":
            angle_depth += 1
        elif current == ">" and angle_depth:
            angle_depth -= 1
        elif (
            current == ","
            and round_depth == 0
            and square_depth == 0
            and brace_depth == 0
            and angle_depth == 0
        ):
            pieces.append(text[start:index].strip())
            start = index + 1
        if min(round_depth, square_depth, brace_depth, angle_depth) < 0:
            return []
    if any((round_depth, square_depth, brace_depth, angle_depth)):
        return []
    pieces.append(text[start:].strip())
    return pieces


def declaration_name(text: str) -> Optional[str]:
    without_default = text.split("=", 1)[0].strip()
    match = re.search(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^\]]*\]\s*)*$",
        without_default,
    )
    return match.group(1) if match else None


def nearest_template_parameters(masked: str, before: int) -> list[str]:
    candidates = list(re.finditer(r"\btemplate\s*<", masked[:before]))
    if not candidates:
        return []
    match = candidates[-1]
    open_angle = masked.find("<", match.start(), match.end())
    close_angle = matching_delimiter(masked, open_angle, "<", ">")
    if close_angle is None or close_angle >= before:
        return []
    between = masked[close_angle + 1:before]
    if ";" in between or "{" in between or "}" in between:
        return []
    return split_top_level(masked[open_angle + 1:close_angle])


def named_function_definitions(masked: str, name: str) -> list[dict[str, object]]:
    definitions: list[dict[str, object]] = []
    pattern = re.compile(rf"\b{re.escape(name)}\s*\(")
    for match in pattern.finditer(masked):
        open_paren = masked.find("(", match.start(), match.end())
        close_paren = matching_delimiter(masked, open_paren, "(", ")")
        if close_paren is None:
            continue
        body_start = close_paren + 1
        while body_start < len(masked) and masked[body_start].isspace():
            body_start += 1
        if body_start >= len(masked) or masked[body_start] != "{":
            continue
        body_end = matching_delimiter(masked, body_start, "{", "}")
        if body_end is None:
            continue
        parameters = split_top_level(masked[open_paren + 1:close_paren])
        definitions.append(
            {
                "body": masked[body_start + 1:body_end],
                "parameters": parameters,
                "parameter_names": [
                    declaration_name(parameter) for parameter in parameters
                ],
                "template_parameters": nearest_template_parameters(
                    masked, match.start()
                ),
            }
        )
    return definitions


def named_function_bodies(masked: str, name: str) -> list[str]:
    return [
        str(definition["body"])
        for definition in named_function_definitions(masked, name)
    ]


def skip_space(text: str, index: int = 0) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def parse_if_block(
    text: str, index: int, require_constexpr: bool
) -> Optional[tuple[str, str, int]]:
    index = skip_space(text, index)
    match = re.match(r"if\b", text[index:])
    if not match:
        return None
    index += match.end()
    index = skip_space(text, index)
    has_constexpr = text.startswith("constexpr", index)
    if has_constexpr:
        index += len("constexpr")
        index = skip_space(text, index)
    if has_constexpr != require_constexpr:
        return None
    if index >= len(text) or text[index] != "(":
        return None
    close_condition = matching_delimiter(text, index, "(", ")")
    if close_condition is None:
        return None
    condition = text[index + 1:close_condition]
    block_start = skip_space(text, close_condition + 1)
    if block_start >= len(text) or text[block_start] != "{":
        return None
    block_end = matching_delimiter(text, block_start, "{", "}")
    if block_end is None:
        return None
    return condition, text[block_start + 1:block_end], block_end + 1


def normalized_expression(text: str) -> str:
    return re.sub(r"\s+", "", text)


def compute_template_name(definition: Mapping[str, object]) -> Optional[str]:
    parameters = definition["template_parameters"]
    if not isinstance(parameters, list) or len(parameters) < 3:
        return None
    compute = parameters[2]
    if not isinstance(compute, str) or not re.match(
        r"^\s*(?:typename|class)\b", compute
    ):
        return None
    return declaration_name(compute)


def exact_try_sequence(
    text: str, try_name: str, expected_arguments: Sequence[str]
) -> bool:
    index = skip_space(text)
    declaration = re.match(
        r"const\s+auto\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
        rf"::ascify::target::dav_c310::{re.escape(try_name)}\s*\(",
        text[index:],
    )
    if not declaration:
        return False
    result_name = declaration.group(1)
    call_open = index + declaration.end() - 1
    call_close = matching_delimiter(text, call_open, "(", ")")
    if call_close is None:
        return False
    actual_arguments = split_top_level(text[call_open + 1:call_close])
    if [normalized_expression(value) for value in actual_arguments] != [
        normalized_expression(value) for value in expected_arguments
    ]:
        return False
    index = skip_space(text, call_close + 1)
    if index >= len(text) or text[index] != ";":
        return False
    remainder = text[index + 1:]
    handled = re.fullmatch(
        rf"\s*if\s*\(\s*{re.escape(result_name)}\.handled\s*\)\s*"
        rf"\{{\s*return\s+{re.escape(result_name)}\.status\s*;\s*\}}\s*",
        remainder,
        flags=re.DOTALL,
    )
    return handled is not None


WRAPPER_PARAMETER_COUNTS = {
    "softmax": {
        "LaunchSoftmaxWarpImpl": 5,
        "LaunchSoftmaxBlockSMemImpl": 6,
        "LaunchSoftmaxBlockUncachedImpl": 5,
    },
    "rmsnorm": {
        "LaunchRmsNormWarpImpl": 7,
        "LaunchRmsNormBlockSMemImpl": 8,
        "LaunchRmsNormBlockUncachedImpl": 7,
    },
}


def exact_entry_recipe_binding(
    definition: Mapping[str, object], unit: str, wrapper_name: str
) -> bool:
    parameter_names = definition["parameter_names"]
    if (
        not isinstance(parameter_names, list)
        or len(parameter_names) != WRAPPER_PARAMETER_COUNTS[unit][wrapper_name]
        or any(not isinstance(name, str) or not name for name in parameter_names)
    ):
        return False
    compute = compute_template_name(definition)
    if compute is None:
        return False
    body = str(definition["body"])
    outer = parse_if_block(body, 0, require_constexpr=False)
    if outer is None:
        return False
    # The generated recipe must be the first statement.  The original
    # converted fallback is expected after this guarded block.
    recipe_body = outer[1]
    if unit == "softmax":
        inner = parse_if_block(recipe_body, 0, require_constexpr=True)
        if inner is None or recipe_body[skip_space(recipe_body, inner[2]):].strip():
            return False
        template_parameters = definition["template_parameters"]
        if not isinstance(template_parameters, list) or not template_parameters:
            return False
        algorithm = declaration_name(str(template_parameters[-1]))
        if algorithm is None:
            return False
        guard = normalized_expression(inner[0])
        if not re.fullmatch(
            rf"{re.escape(algorithm)}==::[A-Za-z_][A-Za-z0-9_:]*::kSoftmax",
            guard,
        ):
            return False
        expected = [
            str(parameter_names[0]),
            str(parameter_names[1]),
            str(parameter_names[2]),
            str(parameter_names[-2]),
            str(parameter_names[-1]),
            f"static_cast<{compute}*>(nullptr)",
        ]
        return exact_try_sequence(inner[1], "TrySoftmax", expected)
    expected = [
        str(parameter_names[0]),
        str(parameter_names[1]),
        str(parameter_names[2]),
        str(parameter_names[-4]),
        str(parameter_names[-3]),
        str(parameter_names[-2]),
        str(parameter_names[-1]),
        f"static_cast<{compute}*>(nullptr)",
    ]
    return exact_try_sequence(recipe_body, "TryRmsNorm", expected)


RMS_GEOMETRY_CALL_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?:(?:::)?(?:[A-Za-z_][A-Za-z0-9_]*::)*)"
    r"layer_norm::GetNumBlocks\s*\("
)


def nearest_unmatched_open_brace(text: str, before: int) -> Optional[int]:
    stack: list[int] = []
    for index, current in enumerate(text[:before]):
        if current == "{":
            stack.append(index)
        elif current == "}":
            if not stack:
                return None
            stack.pop()
    return stack[-1] if stack else None


def exact_rms_post_geometry_binding(
    definition: Mapping[str, object], wrapper_name: str
) -> bool:
    parameter_names = definition["parameter_names"]
    if (
        not isinstance(parameter_names, list)
        or len(parameter_names)
        != WRAPPER_PARAMETER_COUNTS["rmsnorm"][wrapper_name]
        or any(not isinstance(name, str) or not name for name in parameter_names)
    ):
        return False
    compute = compute_template_name(definition)
    if compute is None:
        return False

    body = str(definition["body"])
    geometry_calls = list(RMS_GEOMETRY_CALL_RE.finditer(body))
    launch_offsets = [match.start() for match in re.finditer(r"<<<", body)]
    if len(geometry_calls) != 1 or len(launch_offsets) != 1:
        return False
    geometry = geometry_calls[0]
    geometry_scope_start = nearest_unmatched_open_brace(
        body, geometry.start()
    )
    if geometry_scope_start is None:
        return False
    geometry_scope_end = matching_delimiter(
        body, geometry_scope_start, "{", "}"
    )
    if geometry_scope_end is None:
        return False

    declaration_prefix = body[
        geometry_scope_start + 1:geometry.start()
    ]
    declaration = re.fullmatch(
        r"\s*(?:const\s+)?"
        r"[A-Za-z_][A-Za-z0-9_:<>]*\s+"
        r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*",
        declaration_prefix,
    )
    if declaration is None:
        return False
    error_name = declaration.group(1)

    call_open = body.find("(", geometry.start(), geometry.end())
    call_close = matching_delimiter(body, call_open, "(", ")")
    if call_close is None:
        return False
    cursor = skip_space(body, call_close + 1)
    if cursor >= len(body) or body[cursor] != ";":
        return False

    error_guard_start = skip_space(body, cursor + 1)
    error_guard = parse_if_block(
        body, error_guard_start, require_constexpr=False
    )
    if error_guard is None:
        return False
    normalized_guard = normalized_expression(error_guard[0])
    if normalized_guard not in {
        f"{error_name}!=ACL_SUCCESS",
        f"ACL_SUCCESS!={error_name}",
    }:
        return False
    if re.fullmatch(
        rf"\s*return\s+{re.escape(error_name)}\s*;\s*",
        error_guard[1],
        flags=re.DOTALL,
    ) is None:
        return False
    cursor = skip_space(body, error_guard[2])
    if re.match(r"else\b", body[cursor:]):
        return False
    if cursor != geometry_scope_end:
        return False

    cursor = skip_space(body, geometry_scope_end + 1)
    recipe = parse_if_block(body, cursor, require_constexpr=False)
    if recipe is None:
        return False
    expected_arguments = [
        str(parameter_names[0]),
        str(parameter_names[1]),
        str(parameter_names[2]),
        str(parameter_names[-4]),
        str(parameter_names[-3]),
        str(parameter_names[-2]),
        str(parameter_names[-1]),
        f"static_cast<{compute}*>(nullptr)",
    ]
    if not exact_try_sequence(
        recipe[1], "TryRmsNorm", expected_arguments
    ):
        return False

    return recipe[2] < launch_offsets[0]


def recipe_placement(path: Path, unit: str) -> dict[str, object]:
    if unit not in CONVERSION_UNITS:
        fail(f"unsupported conversion unit for placement validation: {unit!r}")
    text = active_cpp_source(path.read_text(encoding="utf-8", errors="strict"))
    masked = mask_cpp_non_code(text)
    softmax_token = RECIPE_TOPOLOGY_TOKENS["softmax_direct_wrappers"]
    rmsnorm_token = RECIPE_TOPOLOGY_TOKENS["rmsnorm_direct_wrappers"]
    wrappers = {}
    placed_try_calls = 0
    for name in DIRECT_RECIPE_WRAPPERS[unit]:
        definitions = named_function_definitions(masked, name)
        bodies = [str(definition["body"]) for definition in definitions]
        combined = "\n".join(bodies)
        softmax_calls = combined.count(softmax_token)
        rmsnorm_calls = combined.count(rmsnorm_token)
        placed_try_calls += softmax_calls + rmsnorm_calls
        exact_bindings = sum(
            1
            for definition in definitions
            if exact_entry_recipe_binding(definition, unit, name)
        )
        post_geometry_bindings = sum(
            1
            for definition in definitions
            if unit == "rmsnorm"
            and exact_rms_post_geometry_binding(definition, name)
        )
        wrappers[name] = {
            "definition_count": len(definitions),
            "cuda_launches": combined.count("<<<"),
            "softmax_try_calls": softmax_calls,
            "rmsnorm_try_calls": rmsnorm_calls,
            "entry_recipe_prologues": exact_bindings,
            "entry_recipe_argument_bindings": exact_bindings,
            "post_geometry_pre_launch_recipe_prologues": (
                post_geometry_bindings
            ),
            "post_geometry_pre_launch_recipe_argument_bindings": (
                post_geometry_bindings
            ),
            "softmax_ksoftmax_guards": len(
                re.findall(
                    r"if\s+constexpr\s*\(\s*algorithm\s*=="
                    r"\s*::[A-Za-z_][A-Za-z0-9_:]*::kSoftmax\s*\)",
                    combined,
                )
            ),
            "softmax_klogsoftmax_refs": len(
                re.findall(r"\bkLogSoftmax\b", combined)
            ),
        }
    forbidden_bodies = []
    forbidden_logsoftmax_bodies = []
    forbidden_definition_counts = {}
    for name in FORBIDDEN_RECIPE_DISPATCHERS[unit]:
        bodies = named_function_bodies(masked, name)
        forbidden_definition_counts[name] = len(bodies)
        forbidden_bodies.extend(bodies)
        if name == "DispatchLogSoftmax":
            forbidden_logsoftmax_bodies.extend(bodies)
    forbidden_text = "\n".join(forbidden_bodies)
    forbidden_logsoftmax_text = "\n".join(forbidden_logsoftmax_bodies)
    total_try_calls = masked.count(softmax_token) + masked.count(rmsnorm_token)
    return {
        "direct_wrappers": wrappers,
        "forbidden_dispatcher_definitions": forbidden_definition_counts,
        "unexpected_try_calls": total_try_calls - placed_try_calls,
        "forbidden_dispatcher_try_calls": (
            forbidden_text.count(softmax_token)
            + forbidden_text.count(rmsnorm_token)
        ),
        "forbidden_logsoftmax_try_calls": (
            forbidden_logsoftmax_text.count(softmax_token)
            + forbidden_logsoftmax_text.count(rmsnorm_token)
        ),
    }


def recipe_summary(topology: Mapping[str, int]) -> str:
    return (
        "Ascify dav-c310 recipe: adapters("
        f"load={topology['direct_load_markers']},"
        f"store={topology['direct_store_markers']}), "
        "direct_wrappers("
        f"softmax={topology['softmax_direct_wrappers']},"
        f"rmsnorm={topology['rmsnorm_direct_wrappers']})"
    )


def validate_conversion_evidence(args: argparse.Namespace) -> dict[str, str]:
    manifest = args.manifest.resolve()
    document = read_json_object(manifest)
    expected_top = {
        "schema_version",
        "conversion_set_id",
        "created_utc",
        "hostname",
        "policy",
        "converter",
        "environment",
        "artifacts",
        "conversions",
    }
    if set(document) != expected_top:
        fail(
            f"{manifest}: conversion manifest keys mismatch; "
            f"actual={sorted(document)}, expected={sorted(expected_top)}"
        )
    if document["schema_version"] != CONVERSION_SCHEMA:
        fail(
            f"{manifest}: schema_version={document['schema_version']!r}, "
            f"expected={CONVERSION_SCHEMA!r}"
        )
    conversion_set_id = require_nonempty_string(
        document["conversion_set_id"], "conversion_set_id"
    )
    validate_run_id(conversion_set_id)
    iso8601(require_nonempty_string(document["created_utc"], "created_utc"), "conversion")
    require_nonempty_string(document["hostname"], "hostname")
    if document["policy"] != CONVERSION_POLICY:
        fail(
            f"{manifest}: policy={document['policy']!r}, "
            f"expected={CONVERSION_POLICY!r}"
        )
    environment = document["environment"]
    expected_environment_keys = {
        "cuda_root_origin",
        "clang_resource_directory_origin",
        "input_include_path",
    }
    if not isinstance(environment, dict) or set(environment) != expected_environment_keys:
        fail(
            f"{manifest}: environment keys mismatch; "
            f"actual={sorted(environment) if isinstance(environment, dict) else environment!r}, "
            f"expected={sorted(expected_environment_keys)}"
        )
    cuda_root_origin = require_nonempty_string(
        environment["cuda_root_origin"], "environment.cuda_root_origin"
    )
    clang_resource_origin = require_nonempty_string(
        environment["clang_resource_directory_origin"],
        "environment.clang_resource_directory_origin",
    )
    if not Path(cuda_root_origin).is_absolute():
        fail(f"{manifest}: CUDA origin must be an absolute executed path")
    if not Path(clang_resource_origin).is_absolute():
        fail(f"{manifest}: Clang resource origin must be an absolute executed path")
    include_text = require_nonempty_string(
        environment["input_include_path"], "environment.input_include_path"
    )
    include_relative = PurePosixPath(include_text)
    if (
        include_relative.is_absolute()
        or any(part in ("", ".", "..") for part in include_relative.parts)
        or str(include_relative) != include_text
    ):
        fail(f"{manifest}: input include path is not canonical and relative")
    include_path = manifest.parent.joinpath(*include_relative.parts)
    if include_path.is_symlink() or not include_path.resolve().is_dir():
        fail(f"{manifest}: input include directory is missing or a symlink")
    ensure_contained(manifest.parent, include_path, "conversion input include")

    converter = document["converter"]
    expected_converter = {
        "binary_artifact": "ascify_binary",
        "recipe_source_artifact": "recipe_source",
        "recipe_header_artifact": "recipe_header",
    }
    if converter != expected_converter:
        fail(
            f"{manifest}: converter references={converter!r}, "
            f"expected={expected_converter!r}"
        )

    raw_artifacts = document["artifacts"]
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        fail(f"{manifest}: artifacts must be a non-empty array")
    artifacts: dict[str, dict[str, str]] = {}
    paths: dict[str, Path] = {}
    artifact_relative_paths: set[str] = set()
    for index, raw in enumerate(raw_artifacts):
        label = f"{manifest}: artifacts[{index}]"
        if not isinstance(raw, dict) or set(raw) != {
            "logical_id",
            "role",
            "path",
            "sha256",
        }:
            fail(f"{label} must contain exactly logical_id/role/path/sha256")
        logical_id = require_nonempty_string(raw["logical_id"], f"{label}.logical_id")
        if not RUN_ID_RE.fullmatch(logical_id):
            fail(f"{label}.logical_id is unsafe: {logical_id!r}")
        if logical_id in artifacts:
            fail(f"{manifest}: duplicate artifact logical_id {logical_id!r}")
        role = require_nonempty_string(raw["role"], f"{label}.role")
        digest = require_sha256(raw["sha256"], f"{label}.sha256")
        relative_text = str(raw["path"])
        if relative_text in artifact_relative_paths:
            fail(f"{manifest}: duplicate artifact path {relative_text!r}")
        artifact_path = contained_path(manifest.parent, raw["path"], f"{label}.path")
        actual = sha256_file(artifact_path)
        if actual != digest:
            fail(
                f"{manifest}: artifact hash mismatch for {logical_id}: "
                f"manifest={digest}, actual={actual}"
            )
        artifacts[logical_id] = {
            "role": role,
            "path": str(raw["path"]),
            "sha256": digest,
        }
        paths[logical_id] = artifact_path
        artifact_relative_paths.add(relative_text)

    missing_core = set(CONVERSION_CORE_ARTIFACTS) - set(artifacts)
    if missing_core:
        fail(f"{manifest}: missing core artifacts: {sorted(missing_core)}")
    for logical_id, expected_role in CONVERSION_CORE_ARTIFACTS.items():
        if artifacts[logical_id]["role"] != expected_role:
            fail(
                f"{manifest}: artifact {logical_id} role="
                f"{artifacts[logical_id]['role']!r}, expected={expected_role!r}"
            )

    expected_binary = require_sha256(
        args.ascify_binary_sha256, "--ascify-binary-sha256"
    )
    if artifacts["ascify_binary"]["sha256"] != expected_binary:
        fail(
            f"{manifest}: Ascify binary hash does not match formal expectation"
        )
    current_recipe_source = sha256_file(args.recipe_source)
    current_recipe_header = sha256_file(args.recipe_header)
    if artifacts["recipe_source"]["sha256"] != current_recipe_source:
        fail(f"{manifest}: 910C recipe source differs from current 950 source")
    if artifacts["recipe_header"]["sha256"] != current_recipe_header:
        fail(f"{manifest}: 910C recipe header differs from current 950 header")

    staged_outputs = {
        "generated_softmax_header": args.staged_softmax,
        "generated_layer_norm_header": args.staged_layer_norm,
        "generated_rmsnorm_header": args.staged_rmsnorm,
        "generated_rmsnorm_adapter": args.staged_rmsnorm_adapter,
    }
    for logical_id, staged_path in staged_outputs.items():
        if not staged_path.is_file():
            fail(f"missing staged generated artifact {logical_id}: {staged_path}")
        staged_digest = sha256_file(staged_path)
        if artifacts[logical_id]["sha256"] != staged_digest:
            fail(
                f"{manifest}: staged output differs from 910C evidence for "
                f"{logical_id}: evidence={artifacts[logical_id]['sha256']}, "
                f"staged={staged_digest}"
            )

    raw_conversions = document["conversions"]
    if not isinstance(raw_conversions, list):
        fail(f"{manifest}: conversions must be an array")
    conversions: dict[str, dict[str, object]] = {}
    used_logs: set[str] = set()
    for index, raw in enumerate(raw_conversions):
        label = f"{manifest}: conversions[{index}]"
        required = {
            "logical_id",
            "input_artifact",
            "output_artifact",
            "stdout_artifact",
            "stderr_artifact",
            "argv",
            "exit_code",
            "recipe_placement",
            "recipe_topology",
            "recipe_summary",
        }
        if not isinstance(raw, dict) or set(raw) != required:
            fail(f"{label} fields mismatch")
        logical_id = require_nonempty_string(raw["logical_id"], f"{label}.logical_id")
        if logical_id in conversions:
            fail(f"{manifest}: duplicate conversion {logical_id!r}")
        if raw["exit_code"] != 0:
            fail(f"{label}.exit_code={raw['exit_code']!r}, expected=0")
        argv = raw["argv"]
        if (
            not isinstance(argv, list)
            or not argv
            or any(not isinstance(item, str) or not item for item in argv)
        ):
            fail(f"{label}.argv must be a non-empty string array")
        if "--target-policy=dav-c310-vec" not in argv:
            fail(f"{label}.argv lacks exact target policy")
        if "--simt-math=fast" not in argv:
            fail(f"{label}.argv lacks exact SIMT math policy")
        if "-o" not in argv:
            fail(f"{label}.argv lacks an explicit output")
        for field, role in (
            ("input_artifact", "input"),
            ("output_artifact", "output"),
            ("stdout_artifact", "log"),
            ("stderr_artifact", "log"),
        ):
            reference = raw[field]
            if reference not in artifacts:
                fail(f"{label}.{field} references missing artifact {reference!r}")
            if artifacts[reference]["role"] != role:
                fail(
                    f"{label}.{field} references role "
                    f"{artifacts[reference]['role']!r}, expected={role!r}"
                )
        for field in ("stdout_artifact", "stderr_artifact"):
            reference = str(raw[field])
            if reference in used_logs:
                fail(f"{manifest}: conversion logs are reused: {reference!r}")
            used_logs.add(reference)
        conversions[logical_id] = raw

    if set(conversions) != set(CONVERSION_UNITS):
        fail(
            f"{manifest}: conversion units mismatch; "
            f"actual={sorted(conversions)}, expected={sorted(CONVERSION_UNITS)}"
        )
    for unit, (input_id, output_id) in CONVERSION_UNITS.items():
        conversion = conversions[unit]
        if (
            conversion["input_artifact"] != input_id
            or conversion["output_artifact"] != output_id
        ):
            fail(
                f"{manifest}: conversion {unit} input/output mapping mismatch"
            )
        expected_stdout = f"log_{unit}_stdout"
        expected_stderr = f"log_{unit}_stderr"
        if (
            conversion["stdout_artifact"] != expected_stdout
            or conversion["stderr_artifact"] != expected_stderr
        ):
            fail(f"{manifest}: conversion {unit} log mapping mismatch")
        expected_argv = [
            "./" + artifacts["ascify_binary"]["path"],
            artifacts[input_id]["path"],
            "--target-policy=dav-c310-vec",
            "--simt-math=fast",
            f"--cuda-path={cuda_root_origin}",
            f"--clang-resource-directory={clang_resource_origin}",
            "-o",
            artifacts[output_id]["path"],
            "--",
            f"-I{include_text}",
            "-std=c++17",
        ]
        expected_argv.extend(CONVERSION_FORCED_INCLUDES[unit])
        if conversion["argv"] != expected_argv:
            fail(
                f"{manifest}: conversion {unit} argv is not bound to its "
                f"converter/input/output/environment artifacts"
            )
        expected_topology = EXPECTED_CONVERSION_TOPOLOGY[unit]
        recorded_topology = conversion["recipe_topology"]
        if (
            not isinstance(recorded_topology, dict)
            or set(recorded_topology) != set(expected_topology)
            or any(type(value) is not int for value in recorded_topology.values())
        ):
            fail(
                f"{manifest}: conversion {unit} recipe topology is not an "
                "exact integer counter object"
            )
        if recorded_topology != expected_topology:
            fail(
                f"{manifest}: conversion {unit} recorded recipe topology="
                f"{recorded_topology!r}, expected={expected_topology!r}"
            )
        actual_topology = recipe_topology(paths[output_id])
        if actual_topology != expected_topology:
            fail(
                f"{manifest}: conversion {unit} generated recipe topology="
                f"{actual_topology!r}, expected={expected_topology!r}"
            )
        expected_placement = EXPECTED_RECIPE_PLACEMENT[unit]
        recorded_placement = conversion["recipe_placement"]
        if recorded_placement != expected_placement:
            fail(
                f"{manifest}: conversion {unit} recorded recipe placement="
                f"{recorded_placement!r}, expected={expected_placement!r}"
            )
        actual_placement = recipe_placement(paths[output_id], unit)
        if actual_placement != expected_placement:
            fail(
                f"{manifest}: conversion {unit} generated recipe placement="
                f"{actual_placement!r}, expected={expected_placement!r}"
            )
        expected_summary = recipe_summary(expected_topology)
        if conversion["recipe_summary"] != expected_summary:
            fail(
                f"{manifest}: conversion {unit} recipe summary="
                f"{conversion['recipe_summary']!r}, expected={expected_summary!r}"
            )
        stderr_lines = paths[expected_stderr].read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        summary_lines = [
            line for line in stderr_lines
            if line.startswith(RECIPE_SUMMARY_PREFIX)
        ]
        if summary_lines != [expected_summary]:
            fail(
                f"{manifest}: conversion {unit} stderr recipe summaries="
                f"{summary_lines!r}, expected={[expected_summary]!r}"
            )

    summary = {
        "ASCIFY_CONVERSION_SET_ID": conversion_set_id,
        "ASCIFY_CONVERSION_MANIFEST_SHA256": sha256_file(manifest),
        "ASCIFY_BINARY_SHA256": expected_binary,
        "ASCIFY_INPUT_SOFTMAX_SHA256": artifacts["input_softmax"]["sha256"],
        "ASCIFY_INPUT_LAYER_NORM_SHA256": artifacts["input_layer_norm"]["sha256"],
        "ASCIFY_INPUT_RMSNORM_SHA256": artifacts["input_rmsnorm"]["sha256"],
        "ASCIFY_INPUT_RMSNORM_ADAPTER_SHA256": artifacts[
            "input_rmsnorm_adapter"
        ]["sha256"],
    }
    return summary


def validate_run_id(run_id: str) -> None:
    if not RUN_ID_RE.fullmatch(run_id):
        fail(f"unsafe run id: {run_id!r}")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        fail(f"missing CSV: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            fail(f"CSV has no header: {path}")
        rows = list(reader)
    return list(reader.fieldnames), rows


def selected_shape_rows(
    shapes: Path, execution_kind: str, requested_tier: str
) -> dict[tuple[str, str], dict[str, str]]:
    fieldnames, rows = read_csv(shapes)
    required = {
        "case_id",
        "op",
        "dtype",
        "tier",
        "run_check",
        "run_bench",
        "idx",
        "rows",
        "cols",
        "scenario",
        "input_pattern",
        "cuda_pred_path",
        "affine",
        "eps",
    }
    missing = required - set(fieldnames)
    if missing:
        fail(f"shape CSV is missing columns: {sorted(missing)}")
    if execution_kind not in ("accuracy", "perf"):
        fail(f"unsupported shape execution kind: {execution_kind!r}")
    flag = "run_check" if execution_kind == "accuracy" else "run_bench"
    requested_rank = TIER_RANK.get(requested_tier)
    if requested_rank is None:
        fail(f"unsupported requested tier: {requested_tier}")
    selected: dict[tuple[str, str], dict[str, str]] = {}
    all_keys: set[tuple[str, str]] = set()
    all_indices: set[int] = set()
    for line_number, row in enumerate(rows, start=2):
        label = f"{shapes}:{line_number}"
        if not RUN_ID_RE.fullmatch(row["case_id"]):
            fail(f"{label}: unsafe or empty case_id={row['case_id']!r}")
        if row["op"] not in ("softmax", "rms_norm"):
            fail(f"{label}: unsupported op={row['op']!r}")
        if row["dtype"] != "fp16":
            fail(f"{label}: formal shape dtype must be fp16")
        if row["tier"] not in TIER_RANK:
            fail(f"{label}: unsupported tier={row['tier']!r}")
        for run_flag in ("run_check", "run_bench"):
            if row[run_flag] not in ("0", "1"):
                fail(f"{label}: {run_flag} must be 0 or 1")
        index = nonnegative_integer(row["idx"], f"{label} idx")
        if index in all_indices:
            fail(f"{label}: duplicate shape idx={index}")
        all_indices.add(index)
        for dimension in ("rows", "cols"):
            value = nonnegative_integer(
                row[dimension], f"{label} {dimension}"
            )
            if value == 0:
                fail(f"{label}: {dimension} must be positive")
        if row["affine"] not in ("0", "1"):
            fail(f"{label}: affine must be 0 or 1")
        if row["op"] == "softmax" and row["affine"] != "0":
            fail(f"{label}: Softmax affine must be 0")
        finite_number(row["eps"], f"{label} eps", positive=True)
        for name in ("scenario", "input_pattern", "cuda_pred_path"):
            if not row[name]:
                fail(f"{label}: {name} must be non-empty")
        key = (row["op"], row["case_id"])
        if key in all_keys:
            fail(f"duplicate shape key in {shapes}: {key}")
        all_keys.add(key)
        case_rank = TIER_RANK[row["tier"]]
        if row[flag] != "1" or case_rank > requested_rank:
            continue
        selected[key] = row
    return selected


def finite_number(value: str, label: str, positive: bool = False) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        fail(f"{label} is not numeric: {value!r}")
        raise AssertionError
    if not math.isfinite(parsed):
        fail(f"{label} is not finite: {value!r}")
    if positive and parsed <= 0.0:
        fail(f"{label} must be positive: {value!r}")
    return parsed


def nonnegative_integer(value: str, label: str) -> int:
    if not re.fullmatch(r"0|[1-9][0-9]*", value):
        fail(f"{label} must be a canonical non-negative integer: {value!r}")
    return int(value)


def validate_runtime_grid_config(
    path: Path, run_id: str, variant_mode: str, device: int
) -> dict[str, int]:
    if variant_mode not in VARIANTS:
        fail(f"unsupported runtime-config variant mode: {variant_mode!r}")
    if not path.is_file():
        fail(f"missing runtime grid config: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if tuple(reader.fieldnames or ()) != RUNTIME_CONFIG_FIELDS:
            fail(f"{path}: runtime grid config schema mismatch")
        rows = list(reader)
    if len(rows) != 2 or {row["op"] for row in rows} != {
        "softmax",
        "rms_norm",
    }:
        fail(f"{path}: runtime grid config must contain one Softmax and one RMSNorm row")
    expected = {
        "direct": {
            "softmax": (
                "converted_dispatch",
                "try_softmax_default_grid",
                "adaptive",
            ),
            "rms_norm": (
                "converted_plain+converted_affine",
                "try_rmsnorm_default_grid",
                "plain512-affine256-auto",
            ),
        },
        "native": {
            "softmax": (
                "native_half2_hybrid",
                "softmax_fp16_explicit_grid0",
                "adaptive",
            ),
            "rms_norm": (
                "native_plain+native_affine",
                "rmsnorm_fp16_explicit_grid0",
                "plain512-affine256-auto",
            ),
        },
    }[variant_mode]
    vector_counts: set[int] = set()
    resolved_caps: set[int] = set()
    for row in rows:
        op = row["op"]
        expected_variant, expected_entry, expected_blocks = expected[op]
        expected_values = {
            "runtime_config_schema_version": "1",
            "run_id": run_id,
            "variant": expected_variant,
            "device_id": str(device),
            "target_entry": expected_entry,
            "requested_grid_cap": "0",
            "grid_policy": "auto-aiv-x32",
            "block_threads_policy": expected_blocks,
        }
        for name, value in expected_values.items():
            if row[name] != value:
                fail(
                    f"{path}: {op} {name}={row[name]!r}, expected={value!r}"
                )
        vector_count = nonnegative_integer(
            row["vector_core_count"], f"{path}: {op} vector_core_count"
        )
        resolved_cap = nonnegative_integer(
            row["resolved_grid_cap"], f"{path}: {op} resolved_grid_cap"
        )
        if vector_count <= 0 or vector_count > 1024 or resolved_cap != vector_count * 32:
            fail(
                f"{path}: {op} grid resolution mismatch: "
                f"AIV={vector_count}, resolved={resolved_cap}"
            )
        vector_counts.add(vector_count)
        resolved_caps.add(resolved_cap)
    if len(vector_counts) != 1 or len(resolved_caps) != 1:
        fail(f"{path}: Softmax/RMSNorm resolved different runtime grid caps")
    return {
        "vector_core_count": next(iter(vector_counts)),
        "resolved_grid_cap": next(iter(resolved_caps)),
    }


def require_normalized_fragment(
    path: Path,
    fragment: str,
    label: str,
    predefined_macros: Optional[Mapping[str, str]] = None,
) -> None:
    source = normalized_expression(
        mask_cpp_non_code(
            active_cpp_source(
                path.read_text(encoding="utf-8", errors="strict"),
                predefined_macros,
            )
        )
    )
    expected = normalized_expression(fragment)
    if source.count(expected) != 1:
        fail(
            f"{path}: expected exactly one normalized {label} contract, "
            f"found {source.count(expected)}"
        )


def validate_runtime_source_contracts(args: argparse.Namespace) -> None:
    if (
        recipe_placement(args.generated_softmax, "softmax")
        != EXPECTED_RECIPE_PLACEMENT["softmax"]
    ):
        fail("generated Softmax does not prove the default TrySoftmax entry path")
    if (
        recipe_placement(args.generated_rmsnorm, "rmsnorm")
        != EXPECTED_RECIPE_PLACEMENT["rmsnorm"]
    ):
        fail("generated RMSNorm does not prove the default TryRmsNorm entry path")
    require_normalized_fragment(
        args.softmax_bench,
        """
        return softmax_native_dispatch(
            stream, input, output, rows, static_cast<int>(cols), 0, 0);
        """,
        "native Softmax explicit grid_cap=0",
        {
            "ASCIFY950_USE_TUNED_SOFTMAX": "1",
            "ASCIFY950_USE_BLOCK_SOFTMAX": "1",
        },
    )
    require_normalized_fragment(
        args.softmax_native,
        """
        return ::ascify::target::dav_c310::SoftmaxFp16(
            stream, input, output, rows, columns, block_threads, grid_cap);
        """,
        "native Softmax wrapper forwarding",
    )
    require_normalized_fragment(
        args.rmsnorm_native,
        """
        return ::ascify::target::dav_c310::v1::detail::rmsnorm_fp16::Launch(
            stream, input, output, weight, inverse_rms, rows, columns, epsilon,
            0, 0);
        """,
        "native RMSNorm explicit grid_cap=0",
    )
    require_normalized_fragment(
        args.target_header,
        """
        return TrySoftmax(stream, LoadAccess::Data(load),
                          StoreAccess::Data(store), rows, columns);
        """,
        "direct Softmax adapter default dispatch",
    )
    require_normalized_fragment(
        args.target_header,
        """
        return TryRmsNorm(stream, LoadAccess::Data(load),
                          StoreAccess::Data(store), weight, rows, columns,
                          epsilon, inverse_rms);
        """,
        "direct RMSNorm adapter default dispatch",
    )
    require_normalized_fragment(
        args.target_softmax_impl,
        "if (grid_cap == 0) grid_cap = smn_vector_core_count() * 32;",
        "Softmax auto AIV grid resolution",
    )
    require_normalized_fragment(
        args.target_rmsnorm_impl,
        "if (grid_cap == 0) { grid_cap = VectorCoreCount() * 32; }",
        "RMSNorm auto AIV grid resolution",
    )


def expected_variant_for_shape(
    variant_mode: str, op: str, shape: Mapping[str, str]
) -> str:
    if op == "softmax":
        if shape["affine"] != "0":
            fail(f"Softmax shape {shape['case_id']} must have affine=0")
        return (
            "converted_dispatch"
            if variant_mode == "direct"
            else "native_half2_hybrid"
        )
    if shape["affine"] not in ("0", "1"):
        fail(
            f"RMSNorm shape {shape['case_id']} has invalid affine="
            f"{shape['affine']!r}"
        )
    affine = shape["affine"] == "1"
    if variant_mode == "direct":
        return "converted_affine" if affine else "converted_plain"
    return "native_affine" if affine else "native_plain"


def validate_accuracy_numbers(row: Mapping[str, str], key: tuple[str, str]) -> None:
    metrics = {
        name: finite_number(row[name], f"{key} {name}")
        for name in (
            "max_abs_error",
            "max_scaled_rel_error",
            "max_row_sum_error",
            "max_aux_abs_error",
            "max_aux_scaled_rel_error",
        )
    }
    if any(value < 0.0 for value in metrics.values()):
        fail(f"{key} accuracy metrics must be non-negative")
    counts = {
        name: nonnegative_integer(row[name], f"{key} {name}")
        for name in (
            "nonfinite_count",
            "guard_mismatch_count",
            "canary_mismatch_count",
        )
    }
    if any(counts.values()):
        fail(f"{key} accuracy safety counters are non-zero: {counts}")
    if row["op"] == "softmax":
        limits = {
            "max_abs_error": 5.0e-4,
            "max_scaled_rel_error": 2.0e-3,
            "max_row_sum_error": 2.0e-3,
            "max_aux_abs_error": 0.0,
            "max_aux_scaled_rel_error": 0.0,
        }
    else:
        limits = {
            "max_abs_error": 5.0e-3,
            "max_scaled_rel_error": 5.0e-3,
            "max_aux_abs_error": 2.0e-3,
            "max_aux_scaled_rel_error": 5.0e-3,
        }
    for name, limit in limits.items():
        if metrics[name] > limit:
            fail(f"{key} {name}={metrics[name]} exceeds formal limit {limit}")


def validate_perf_numbers(
    row: Mapping[str, str], shape: Mapping[str, str], key: tuple[str, str]
) -> None:
    mean_ms = finite_number(row["lat_ms_mean"], f"{key} lat_ms_mean", positive=True)
    median_ms = finite_number(
        row["lat_ms_median"], f"{key} lat_ms_median", positive=True
    )
    minimum_ms = finite_number(
        row["lat_ms_min"], f"{key} lat_ms_min", positive=True
    )
    p90_ms = finite_number(row["lat_ms_p90"], f"{key} lat_ms_p90", positive=True)
    if not (minimum_ms <= median_ms <= p90_ms):
        fail(
            f"{key} latency order is invalid: "
            f"min={minimum_ms}, median={median_ms}, p90={p90_ms}"
        )
    if mean_ms < minimum_ms:
        fail(f"{key} mean latency is below minimum latency")

    logical_bytes = nonnegative_integer(row["logical_bytes"], f"{key} logical_bytes")
    rows = int(shape["rows"])
    cols = int(shape["cols"])
    if row["op"] == "softmax":
        expected_bytes = 4 * rows * cols
    else:
        expected_bytes = 4 * rows * cols + 4 * rows
        if shape["affine"] == "1":
            expected_bytes += 2 * cols
    if logical_bytes != expected_bytes:
        fail(
            f"{key} logical_bytes={logical_bytes}, expected={expected_bytes}"
        )
    gbps = finite_number(row["gbps"], f"{key} gbps", positive=True)
    expected_gbps = logical_bytes / (median_ms * 1.0e-3) / 1.0e9
    if not math.isclose(gbps, expected_gbps, rel_tol=1.0e-12, abs_tol=1.0e-12):
        fail(f"{key} gbps={gbps}, recomputed={expected_gbps}")


def validate_csv_rows(args: argparse.Namespace) -> tuple[list[str], list[dict[str, str]]]:
    validate_run_id(args.run_id)
    fieldnames, all_rows = read_csv(args.csv)
    required = {
        "run_id",
        "op",
        "dtype",
        "math_mode",
        "variant",
        "case_id",
        "idx",
        "tier",
        "rows",
        "cols",
        "device_id",
        "status",
        "reason",
        "scenario",
        "input_pattern",
        "cuda_pred_path",
    }
    if args.execution_kind == "perf":
        required |= {
            "warmup",
            "samples",
            "inner_repeats",
            "lat_ms_mean",
            "lat_ms_median",
            "lat_ms_min",
            "lat_ms_p90",
            "logical_bytes",
            "gbps",
        }
    else:
        required |= {
            "max_abs_error",
            "max_scaled_rel_error",
            "max_row_sum_error",
            "max_aux_abs_error",
            "max_aux_scaled_rel_error",
            "nonfinite_count",
            "guard_mismatch_count",
            "canary_mismatch_count",
        }
    missing = required - set(fieldnames)
    if missing:
        fail(f"{args.csv} is missing columns: {sorted(missing)}")

    rows = [row for row in all_rows if row["run_id"] == args.run_id]
    if not rows:
        fail(f"run_id={args.run_id} has no rows in {args.csv}")

    keys = [(row["op"], row["case_id"]) for row in rows]
    duplicates = [key for key, count in Counter(keys).items() if count != 1]
    if duplicates:
        fail(f"run_id={args.run_id} has duplicate case rows: {sorted(duplicates)}")

    expected_shapes = selected_shape_rows(args.shapes, args.execution_kind, args.tier)
    if set(keys) != set(expected_shapes):
        missing_keys = sorted(set(expected_shapes) - set(keys))
        extra_keys = sorted(set(keys) - set(expected_shapes))
        fail(
            f"run_id={args.run_id} shape coverage mismatch; "
            f"missing={missing_keys}, extra={extra_keys}"
        )

    counts = Counter(row["op"] for row in rows)
    expected_counts = {
        "softmax": args.expected_softmax,
        "rms_norm": args.expected_rmsnorm,
    }
    if counts != Counter(expected_counts):
        fail(
            f"run_id={args.run_id} count mismatch: "
            f"actual={dict(counts)}, expected={expected_counts}"
        )

    expected_variants = VARIANTS[args.variant_mode]
    actual_variants: dict[str, set[str]] = {
        op: {row["variant"] for row in rows if row["op"] == op}
        for op in expected_variants
    }
    for op, actual in actual_variants.items():
        allowed = expected_variants[op]
        if not actual or not actual <= allowed:
            fail(
                f"run_id={args.run_id} {op} variants mismatch: "
                f"actual={sorted(actual)}, allowed={sorted(allowed)}"
            )
    # Both affine and plain RMSNorm routes are deliberately represented by the
    # formal shape sets. Requiring both catches accidental route collapse.
    if actual_variants["rms_norm"] != expected_variants["rms_norm"]:
        fail(
            f"run_id={args.run_id} RMSNorm route coverage mismatch: "
            f"actual={sorted(actual_variants['rms_norm'])}, "
            f"expected={sorted(expected_variants['rms_norm'])}"
        )

    for row in rows:
        key = (row["op"], row["case_id"])
        shape = expected_shapes[key]
        for column in (
            "dtype",
            "idx",
            "tier",
            "rows",
            "cols",
            "scenario",
            "input_pattern",
            "cuda_pred_path",
        ):
            if row[column] != shape[column]:
                fail(
                    f"run_id={args.run_id} {key} {column} mismatch: "
                    f"result={row[column]!r}, shape={shape[column]!r}"
                )
        if row["status"] != "pass":
            fail(
                f"run_id={args.run_id} {key} status={row['status']!r}, "
                f"reason={row.get('reason', '')!r}"
            )
        if row["reason"] != "":
            fail(f"run_id={args.run_id} {key} pass row has a reason")
        if row["dtype"] != "fp16":
            fail(f"run_id={args.run_id} {key} dtype={row['dtype']!r}")
        if row["math_mode"] != args.math_mode:
            fail(
                f"run_id={args.run_id} {key} math_mode={row['math_mode']!r}, "
                f"expected={args.math_mode!r}"
            )
        if row["device_id"] != str(args.device):
            fail(
                f"run_id={args.run_id} {key} device_id={row['device_id']!r}, "
                f"expected={args.device}"
            )
        expected_variant = expected_variant_for_shape(
            args.variant_mode, row["op"], shape
        )
        if row["variant"] != expected_variant:
            fail(
                f"run_id={args.run_id} {key} variant={row['variant']!r}, "
                f"expected={expected_variant!r}"
            )
        if args.execution_kind == "perf":
            expected_timing = {
                "warmup": str(args.warmup),
                "samples": str(args.samples),
                "inner_repeats": str(args.inner_repeats),
            }
            for column, expected in expected_timing.items():
                if row[column] != expected:
                    fail(
                        f"run_id={args.run_id} {key} {column}={row[column]!r}, "
                        f"expected={expected}"
                    )
            validate_perf_numbers(row, shape, key)
        else:
            validate_accuracy_numbers(row, key)

    return fieldnames, rows


def write_snapshot(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        stream = path.open("x", newline="", encoding="utf-8")
    except FileExistsError:
        fail(f"refusing to overwrite evidence snapshot: {path}")
    with stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_manifest(path: Path, verify_files: bool = True) -> dict[str, dict[str, str]]:
    if not path.is_file():
        fail(f"missing manifest: {path}")
    values: dict[str, str] = {}
    files: dict[str, str] = {}
    file_paths: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration:
            fail(f"empty manifest: {path}")
        if header != ["kind", "name", "value_or_path", "sha256"]:
            fail(f"manifest schema mismatch: {path}")
        for line_number, fields in enumerate(reader, start=2):
            if len(fields) != 4:
                fail(f"{path}:{line_number}: expected four TSV fields")
            kind, name, value_or_path, digest = fields
            if kind == "value":
                if name in values:
                    fail(f"{path}:{line_number}: duplicate value {name!r}")
                if digest:
                    fail(f"{path}:{line_number}: value row has a digest")
                values[name] = value_or_path
            elif kind == "file":
                if name in files:
                    fail(f"{path}:{line_number}: duplicate file {name!r}")
                if not SHA256_RE.fullmatch(digest):
                    fail(f"{path}:{line_number}: invalid SHA-256 for {name!r}")
                files[name] = digest
                file_paths[name] = value_or_path
            else:
                fail(f"{path}:{line_number}: unsupported kind {kind!r}")
    if verify_files:
        for name, digest in files.items():
            artifact = Path(file_paths[name])
            if not artifact.is_absolute():
                artifact = path.parent / artifact
            if not artifact.is_file():
                fail(f"{path}: recorded artifact is missing: {name}={artifact}")
            actual = sha256_file(artifact)
            if actual != digest:
                fail(
                    f"{path}: artifact hash drift for {name}: "
                    f"manifest={digest}, actual={actual}"
                )
            file_paths[name] = str(artifact.resolve())
    return {"values": values, "files": files, "paths": file_paths}


def require_values(
    manifest: Path, values: Mapping[str, str], expected: Mapping[str, str]
) -> None:
    for name, expected_value in expected.items():
        actual = values.get(name)
        if actual != expected_value:
            fail(
                f"{manifest}: value {name!r}={actual!r}, expected={expected_value!r}"
            )


def require_phase_shape_closure(
    manifest: Path,
    phase: str,
    files: Mapping[str, str],
) -> str:
    expected_role = FORMAL_PHASE_SHAPE_FILE.get(phase)
    if expected_role is None:
        fail(f"{manifest}: unsupported formal phase for shape closure: {phase!r}")
    shape_digest = files.get("shapes")
    expected_digest = files.get(expected_role)
    if shape_digest is None or expected_digest is None:
        fail(
            f"{manifest}: shape closure lacks shapes/{expected_role} hashes"
        )
    if shape_digest != expected_digest:
        fail(
            f"{manifest}: phase {phase!r} must use {expected_role}; "
            f"shapes={shape_digest}, {expected_role}={expected_digest}"
        )
    return expected_role


def require_canonical_formal_shapes(
    manifest: Path,
    files: Mapping[str, str],
    paths: Optional[Mapping[str, str]] = None,
) -> None:
    for role, (canonical_path, expected_digest) in (
        CANONICAL_FORMAL_SHAPES.items()
    ):
        actual_digest = files.get(role)
        if actual_digest != expected_digest:
            fail(
                f"{manifest}: non-canonical {role} hash: "
                f"actual={actual_digest!r}, expected={expected_digest}"
            )
        if paths is not None:
            actual_path = paths.get(role)
            if actual_path is None or Path(actual_path).resolve() != (
                canonical_path.resolve()
            ):
                fail(
                    f"{manifest}: non-canonical {role} path: "
                    f"actual={actual_path!r}, "
                    f"expected={str(canonical_path.resolve())!r}"
                )


def require_formal_timing(
    manifest: Path,
    phase: str,
    execution_kind: str,
    warmup: int,
    samples: int,
    inner_repeats: int,
) -> None:
    if execution_kind != "perf":
        return
    expected = FORMAL_PHASE_TIMING.get(phase)
    if expected is None:
        fail(f"{manifest}: unsupported formal performance phase {phase!r}")
    actual = (warmup, samples, inner_repeats)
    if actual != expected:
        fail(
            f"{manifest}: non-canonical timing for {phase}: "
            f"actual={actual}, expected={expected}"
        )


def require_canonical_phase_shape(
    label: Path,
    phase: str,
    shapes: Path,
) -> str:
    protocol = FORMAL_PHASE_PROTOCOL.get(phase)
    if protocol is None:
        fail(f"{label}: unsupported formal phase {phase!r}")
    role = str(protocol["shape_role"])
    canonical_path, expected_digest = CANONICAL_FORMAL_SHAPES[role]
    try:
        actual_path = shapes.resolve(strict=True)
        expected_path = canonical_path.resolve(strict=True)
    except FileNotFoundError as error:
        fail(f"{label}: formal shape path is missing: {error}")
        raise AssertionError
    if actual_path != expected_path:
        fail(
            f"{label}: phase {phase!r} active shapes path is non-canonical: "
            f"actual={str(actual_path)!r}, expected={str(expected_path)!r}"
        )
    actual_digest = sha256_file(actual_path)
    if actual_digest != expected_digest:
        fail(
            f"{label}: phase {phase!r} active shapes hash is non-canonical: "
            f"actual={actual_digest}, expected={expected_digest}"
        )
    return role


def validate_formal_run_contract(
    label: Path,
    *,
    phase: str,
    execution_kind: str,
    shapes: Path,
    tier: str,
    variant_mode: str,
    math_mode: str,
    expected_softmax: int,
    expected_rmsnorm: int,
    warmup: int,
    samples: int,
    inner_repeats: int,
    check_max_elements: int,
    device_util_max: str,
    device_hbm_bw_max: str,
) -> None:
    protocol = FORMAL_PHASE_PROTOCOL.get(phase)
    if protocol is None:
        fail(f"{label}: unsupported formal phase {phase!r}")
    if math_mode != "production-fast":
        fail(
            f"{label}: formal math_mode={math_mode!r}, "
            "expected='production-fast'"
        )
    actual_protocol = {
        "execution_kind": execution_kind,
        "tier": tier,
        "variant_mode": variant_mode,
        "expected_softmax": expected_softmax,
        "expected_rmsnorm": expected_rmsnorm,
    }
    for name, actual in actual_protocol.items():
        expected = protocol[name]
        if actual != expected:
            fail(
                f"{label}: phase {phase!r} {name}={actual!r}, "
                f"expected={expected!r}"
            )
    require_canonical_phase_shape(label, phase, shapes)
    require_formal_timing(
        label,
        phase,
        execution_kind,
        warmup,
        samples,
        inner_repeats,
    )
    expected_check_max = FORMAL_PHASE_CHECK_MAX_ELEMENTS.get(phase)
    if check_max_elements != expected_check_max:
        fail(
            f"{label}: phase {phase!r} check_max_elements="
            f"{check_max_elements!r}, expected={expected_check_max!r}"
        )
    if device_util_max != "0" or device_hbm_bw_max != "0":
        fail(
            f"{label}: formal device thresholds must be exact 0/0, "
            f"actual={device_util_max!r}/{device_hbm_bw_max!r}"
        )


def validate_idle_device_snapshot(
    path: Path,
    device: int,
    phases: Sequence[str] = ("pre", "post"),
    expected_run_id: Optional[str] = None,
    expected_hostname: Optional[str] = None,
) -> dict[str, dt.datetime]:
    if not path.is_file():
        fail(f"missing device snapshot: {path}")
    if not phases or len(phases) != len(set(phases)):
        fail(f"{path}: device snapshot phases must be unique and non-empty")
    if any(phase not in ("pre", "post") for phase in phases):
        fail(f"{path}: unsupported device snapshot phases: {phases}")
    snapshot_lines = path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()
    phase_indices = [
        index
        for index, line in enumerate(snapshot_lines)
        if line.startswith("phase=")
    ]
    actual_phase_lines = [snapshot_lines[index] for index in phase_indices]
    expected_phase_lines = [f"phase={phase}" for phase in phases]
    if (
        actual_phase_lines != expected_phase_lines
        or not phase_indices
        or phase_indices[0] != 0
    ):
        fail(
            f"{path}: device snapshot phase sequence mismatch: "
            f"actual={actual_phase_lines}, expected={expected_phase_lines}"
        )
    timestamps: dict[str, dt.datetime] = {}
    section_headers = ("[mapping]", "[health]", "[usages]", "[processes]")
    for phase_index, phase in enumerate(phases):
        start = phase_indices[phase_index]
        end = (
            phase_indices[phase_index + 1]
            if phase_index + 1 < len(phase_indices)
            else len(snapshot_lines)
        )
        block_lines = snapshot_lines[start:end]
        block_label = f"{path}: phase={phase}"
        if len(block_lines) < 9 or not block_lines[1].startswith("utc_time="):
            fail(f"{block_label}: malformed metadata prefix")
        actual_hostname = block_lines[2].removeprefix("hostname=")
        actual_run_id = block_lines[3].removeprefix("run_id=")
        if (
            not block_lines[2].startswith("hostname=")
            or not actual_hostname
            or not block_lines[3].startswith("run_id=")
            or not RUN_ID_RE.fullmatch(actual_run_id)
            or block_lines[4] != f"device={device}"
        ):
            fail(f"{block_label}: malformed or mismatched identity prefix")
        if block_lines.count(f"device={device}") != 1:
            fail(f"{block_label}: device identity mismatch")
        if expected_run_id is not None and actual_run_id != expected_run_id:
            fail(f"{block_label}: run_id identity mismatch")
        if (
            expected_hostname is not None
            and actual_hostname != expected_hostname
        ):
            fail(f"{block_label}: hostname identity mismatch")
        timestamp_lines = [
            line for line in block_lines if line.startswith("utc_time=")
        ]
        if len(timestamp_lines) != 1:
            fail(f"{block_label}: expected exactly one utc_time")
        timestamps[phase] = iso8601(
            timestamp_lines[0].removeprefix("utc_time="),
            f"{block_label} utc_time",
        )
        section_positions = []
        for section in section_headers:
            if block_lines.count(section) != 1:
                fail(f"{block_label}: expected exactly one {section}")
            section_positions.append(block_lines.index(section))
        if (
            section_positions != sorted(section_positions)
            or section_positions[0] != 5
        ):
            fail(f"{block_label}: device sections are out of order")
        mapping_text = "\n".join(
            block_lines[
                section_positions[0] + 1 : section_positions[1]
            ]
        )
        health_text = "\n".join(
            block_lines[
                section_positions[1] + 1 : section_positions[2]
            ]
        )
        usages_text = "\n".join(
            block_lines[
                section_positions[2] + 1 : section_positions[3]
            ]
        )
        processes_text = "\n".join(
            block_lines[section_positions[3] + 1 :]
        )
        mapping_rows = re.findall(
            r"^\s*([0-9]+)\s+[0-9]+\s+[0-9]+\s+[0-9]+\s+(\S+)\s*$",
            mapping_text,
            flags=re.MULTILINE,
        )
        selected_mapping = [
            chip_name
            for npu_id, chip_name in mapping_rows
            if npu_id == str(device)
        ]
        if selected_mapping != ["Ascend950PR"]:
            fail(
                f"{block_label}: selected mapping is not exactly one "
                f"Ascend950PR row: {selected_mapping}"
            )
        for section_name, section_text in (
            ("health", health_text),
            ("usages", usages_text),
            ("processes", processes_text),
        ):
            npu_ids = re.findall(
                r"^\s*NPU ID\s*:\s*([0-9]+)\s*$",
                section_text,
                flags=re.IGNORECASE | re.MULTILINE,
            )
            if npu_ids != [str(device)]:
                fail(
                    f"{block_label}: {section_name} NPU identity "
                    f"mismatch: {npu_ids}"
                )
        health_status = re.findall(
            r"^\s*Health Status\s*:\s*(.*?)\s*$",
            health_text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if health_status != ["OK"]:
            fail(f"{block_label}: device health is not exactly one OK")
        no_process = re.findall(
            r"^\s*No process in device\.\s*$",
            processes_text,
            flags=re.MULTILINE,
        )
        process_ids = re.findall(
            r"^\s*Process\s+id\s*:",
            processes_text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if len(no_process) != 1 or process_ids:
            fail(f"{block_label}: device process state is not idle")
        for metric in FORMAL_IDLE_METRICS:
            metric_values = re.findall(
                rf"^\s*{re.escape(metric)}(?:\(%\))?\s*:\s*"
                r"([0-9]+(?:\.[0-9]+)?)\s*$",
                usages_text,
                flags=re.IGNORECASE | re.MULTILINE,
            )
            if len(metric_values) != 1 or float(metric_values[0]) != 0.0:
                fail(
                    f"{block_label}: device metric {metric!r} was not "
                    f"exactly one parseable zero: {metric_values}"
                )
    if list(timestamps.values()) != sorted(timestamps.values()):
        fail(f"{path}: device snapshot timestamps are out of order")
    return timestamps


def require_device_snapshot_interval(
    label: Path,
    timestamps: Mapping[str, dt.datetime],
    run_start: dt.datetime,
    run_end: dt.datetime,
) -> None:
    if set(timestamps) != {"pre", "post"}:
        fail(f"{label}: complete snapshot must contain pre and post timestamps")
    if not (
        run_start
        <= timestamps["pre"]
        <= timestamps["post"]
        <= run_end
    ):
        fail(
            f"{label}: snapshot timestamps fall outside the run interval: "
            f"run={run_start.isoformat()}..{run_end.isoformat()}, "
            f"pre={timestamps['pre'].isoformat()}, "
            f"post={timestamps['post'].isoformat()}"
        )


def bundled_binary_names(
    execution_kind: str, variant_mode: str
) -> dict[str, str]:
    binary_kind = "check" if execution_kind == "accuracy" else "bench"
    if variant_mode == "direct":
        return {
            f"softmax_{binary_kind}_binary": (
                f"softmax_{binary_kind}_production_fast"
            ),
            f"rmsnorm_{binary_kind}_binary": (
                f"rmsnorm_{binary_kind}_production_fast"
            ),
        }
    if variant_mode == "native":
        return {
            f"softmax_{binary_kind}_binary": (
                f"softmax_{binary_kind}_production_fast_native_half2_hybrid"
            ),
            f"rmsnorm_{binary_kind}_binary": (
                f"rmsnorm_{binary_kind}_production_fast_native"
            ),
        }
    fail(f"unsupported formal binary variant mode: {variant_mode!r}")
    raise AssertionError


def validate_manifest(args: argparse.Namespace) -> dict[str, dict[str, str]]:
    parsed = parse_manifest(args.manifest)
    values = parsed["values"]
    files = parsed["files"]
    published_manifest = getattr(args, "final_manifest_path", None)
    if published_manifest is None:
        published_manifest = args.manifest
    else:
        expected_staged_name = re.fullmatch(
            re.escape(published_manifest.name) + r"\.partial\.[1-9][0-9]*",
            args.manifest.name,
        )
        if (
            args.manifest.parent.resolve()
            != published_manifest.parent.resolve()
            or expected_staged_name is None
        ):
            fail(
                f"{args.manifest}: invalid staged manifest path for "
                f"{published_manifest}"
            )
    expected_manifest = (
        published_manifest.parent / f"{args.run_id}.tsv"
    )
    if published_manifest.resolve() != expected_manifest.resolve():
        fail(
            f"{args.manifest}: published manifest path is not bound to "
            f"run_id {args.run_id!r}: {published_manifest}"
        )
    check_max_elements = nonnegative_integer(
        values.get("check_max_elements", ""),
        f"{args.manifest}: check_max_elements",
    )
    if check_max_elements == 0:
        fail(f"{args.manifest}: check_max_elements must be positive")
    validate_formal_run_contract(
        args.manifest,
        phase=args.formal_phase,
        execution_kind=args.execution_kind,
        shapes=args.shapes,
        tier=args.tier,
        variant_mode=args.variant_mode,
        math_mode=args.math_mode,
        expected_softmax=args.expected_softmax,
        expected_rmsnorm=args.expected_rmsnorm,
        warmup=args.warmup,
        samples=args.samples,
        inner_repeats=args.inner_repeats,
        check_max_elements=check_max_elements,
        device_util_max=values.get("device_util_max", ""),
        device_hbm_bw_max=values.get("device_hbm_bw_max", ""),
    )
    expected_check_max = FORMAL_PHASE_CHECK_MAX_ELEMENTS.get(
        args.formal_phase
    )
    if expected_check_max is None:
        fail(
            f"{args.manifest}: unsupported formal phase "
            f"{args.formal_phase!r}"
        )
    expected = {
        "manifest_schema_version": "2",
        "formal_manifest": "1",
        "run_status": "complete",
        "run_id": args.run_id,
        "formal_set_id": args.formal_set_id,
        "formal_phase": args.formal_phase,
        "execution_kind": args.execution_kind,
        "device": str(args.device),
        "math_mode": args.math_mode,
        "softmax_variant": (
            "converted" if args.variant_mode == "direct" else "native-half2-hybrid"
        ),
        "rmsnorm_variant": "converted" if args.variant_mode == "direct" else "native",
        "tier": args.tier,
        "op_filter": "both",
        "skip_build": "1",
        "device_util_max": "0",
        "device_hbm_bw_max": "0",
        "check_max_elements": str(expected_check_max),
        "expected_softmax_rows": str(args.expected_softmax),
        "expected_rmsnorm_rows": str(args.expected_rmsnorm),
        "ascify_binary_sha256": args.ascify_binary_sha256,
        "ascify_target_policy": "dav-c310-vec",
        "ascify_simt_math": "fast",
        "softmax_block_threads": "adaptive",
        "softmax_block_row_max": "2048",
        "softmax_grid_cap": "auto-aiv-x32",
        "rmsnorm_block_row_threads": "512",
        "rmsnorm_block_row_affine_threads": "256",
        "rmsnorm_grid_cap": "auto-aiv-x32",
    }
    if args.execution_kind == "accuracy":
        expected.update({"skip_check": "0", "check_only": "1"})
    else:
        expected.update(
            {
                "skip_check": "1",
                "check_only": "0",
                "warmup": str(args.warmup),
                "samples": str(args.samples),
                "inner_repeats": str(args.inner_repeats),
            }
        )
    require_values(args.manifest, values, expected)
    if not values.get("hostname"):
        fail(f"{args.manifest}: hostname is empty")
    if not values.get("device_lock"):
        fail(f"{args.manifest}: device_lock is empty")
    if not values.get("formal_result_lock"):
        fail(f"{args.manifest}: formal_result_lock is empty")
    if not values.get("utc_time") or not values.get("run_end_utc_time"):
        fail(f"{args.manifest}: start/end UTC timestamps are incomplete")
    run_start = iso8601(values["utc_time"], f"{args.manifest} start")
    run_end = iso8601(values["run_end_utc_time"], f"{args.manifest} end")
    if run_start > run_end:
        fail(f"{args.manifest}: run end precedes run start")
    for name in (
        "cann_root",
        "ccec_version",
        "work_root",
        "generated_root",
        "bin_dir",
        "result_dir",
        "accuracy_csv",
        "perf_csv",
        "check_max_elements",
        "ascify_conversion_set_id",
    ):
        if values.get(name) in (None, "", "unknown", "unavailable"):
            fail(f"{args.manifest}: value {name!r} is incomplete")
    if not SHA256_RE.fullmatch(values["ascify_binary_sha256"]):
        fail(f"{args.manifest}: invalid Ascify binary SHA-256")
    for name in (
        "ascify_conversion_manifest_sha256",
        "ascify_input_softmax_sha256",
        "ascify_input_layer_norm_sha256",
        "ascify_input_rmsnorm_sha256",
        "ascify_input_rmsnorm_adapter_sha256",
    ):
        require_sha256(values.get(name), f"{args.manifest}: value {name!r}")

    required_files = set(COMMON_MANIFEST_FILES)
    required_files.add("device_snapshot")
    if args.execution_kind == "accuracy":
        required_files.update(
            {
                "softmax_check_binary",
                "rmsnorm_check_binary",
                "softmax_check_source",
                "rmsnorm_check_source",
            }
        )
    else:
        required_files.update(
            {
                "softmax_bench_binary",
                "rmsnorm_bench_binary",
                "softmax_bench_source",
                "rmsnorm_bench_source",
                "runtime_grid_config",
            }
        )
    missing = required_files - files.keys()
    if missing:
        fail(f"{args.manifest}: missing required file hashes: {sorted(missing)}")
    if files["shapes"] != sha256_file(args.shapes):
        fail(f"{args.manifest}: shape hash does not match {args.shapes}")
    expected_shape_role = require_phase_shape_closure(
        args.manifest,
        args.formal_phase,
        files,
    )
    require_canonical_formal_shapes(
        args.manifest,
        files,
        parsed["paths"],
    )
    active_shape_role = require_canonical_phase_shape(
        args.manifest,
        args.formal_phase,
        Path(parsed["paths"].get("shapes", "")),
    )
    if active_shape_role != expected_shape_role:
        fail(
            f"{args.manifest}: active shape role mismatch: "
            f"actual={active_shape_role}, expected={expected_shape_role}"
        )
    conversion_manifest = Path(parsed["paths"]["ascify_conversion_manifest"])
    build_input_snapshot = Path(
        parsed["paths"]["formal_build_input_snapshot"]
    )
    freeze = check_freeze_snapshot(
        build_input_snapshot,
        formal_contract=True,
    )
    for run_name, freeze_name in RUN_MANIFEST_TO_FREEZE_FILE.items():
        if run_name not in files:
            continue
        if files[run_name] != freeze["files"].get(freeze_name):
            fail(
                f"{args.manifest}: run/freeze artifact mismatch: "
                f"{run_name} != {freeze_name}"
            )
    bundle_manifest = Path(parsed["paths"]["formal_binary_bundle_manifest"])
    bundle = check_binary_bundle(
        bundle_manifest,
        args.formal_set_id,
        build_input_snapshot,
    )
    bundle_dir = bundle_manifest.parent.resolve()
    if Path(values["bin_dir"]).resolve() != bundle_dir:
        fail(
            f"{args.manifest}: bin_dir does not name the immutable bundle: "
            f"{values['bin_dir']!r} != {str(bundle_dir)!r}"
        )
    for run_name, bundle_name in bundled_binary_names(
        args.execution_kind, args.variant_mode
    ).items():
        if files.get(run_name) != bundle["files"].get(bundle_name):
            fail(
                f"{args.manifest}: selected binary hash is outside bundle "
                f"contract: {run_name} != {bundle_name}"
            )
        if Path(parsed["paths"][run_name]) != Path(bundle["paths"][bundle_name]):
            fail(
                f"{args.manifest}: selected binary path is not bundled: "
                f"{parsed['paths'][run_name]} != {bundle['paths'][bundle_name]}"
            )
    conversion_summary = validate_conversion_evidence(
        argparse.Namespace(
            manifest=conversion_manifest,
            ascify_binary_sha256=args.ascify_binary_sha256,
            recipe_source=Path(parsed["paths"]["ascify_recipe_source"]),
            recipe_header=Path(parsed["paths"]["ascify_recipe_header"]),
            staged_softmax=Path(parsed["paths"]["generated_softmax_header"]),
            staged_layer_norm=Path(parsed["paths"]["generated_layer_norm_header"]),
            staged_rmsnorm=Path(parsed["paths"]["generated_rmsnorm_header"]),
            staged_rmsnorm_adapter=Path(
                parsed["paths"]["generated_rmsnorm_adapter"]
            ),
        )
    )
    expected_conversion_values = {
        "ascify_conversion_set_id": conversion_summary[
            "ASCIFY_CONVERSION_SET_ID"
        ],
        "ascify_conversion_manifest_sha256": conversion_summary[
            "ASCIFY_CONVERSION_MANIFEST_SHA256"
        ],
        "ascify_input_softmax_sha256": conversion_summary[
            "ASCIFY_INPUT_SOFTMAX_SHA256"
        ],
        "ascify_input_layer_norm_sha256": conversion_summary[
            "ASCIFY_INPUT_LAYER_NORM_SHA256"
        ],
        "ascify_input_rmsnorm_sha256": conversion_summary[
            "ASCIFY_INPUT_RMSNORM_SHA256"
        ],
        "ascify_input_rmsnorm_adapter_sha256": conversion_summary[
            "ASCIFY_INPUT_RMSNORM_ADAPTER_SHA256"
        ],
    }
    require_values(args.manifest, values, expected_conversion_values)
    artifact_paths = {
        "evidence_csv": (
            Path(parsed["paths"]["evidence_csv"]),
            published_manifest.parent / f"{args.run_id}.csv",
        ),
        "device_snapshot": (
            Path(parsed["paths"]["device_snapshot"]),
            published_manifest.parent / f"{args.run_id}.device.txt",
        ),
    }
    if args.execution_kind == "perf":
        artifact_paths["runtime_grid_config"] = (
            Path(parsed["paths"]["runtime_grid_config"]),
            published_manifest.parent / f"{args.run_id}.runtime_config.tsv",
        )
    for role, (actual_path, expected_path) in artifact_paths.items():
        try:
            actual_resolved = actual_path.resolve(strict=True)
            expected_resolved = expected_path.resolve(strict=True)
        except FileNotFoundError as error:
            fail(f"{args.manifest}: missing run artifact for {role}: {error}")
            raise AssertionError
        if actual_resolved != expected_resolved:
            fail(
                f"{args.manifest}: {role} path is not bound to run_id "
                f"{args.run_id!r}: actual={actual_resolved}, "
                f"expected={expected_resolved}"
            )
    if args.evidence.resolve(strict=True) != artifact_paths[
        "evidence_csv"
    ][1].resolve(strict=True):
        fail(
            f"{args.manifest}: supplied evidence path is not canonical: "
            f"{args.evidence}"
        )
    evidence_args = argparse.Namespace(**vars(args))
    evidence_args.csv = artifact_paths["evidence_csv"][0]
    _, evidence_rows = validate_csv_rows(evidence_args)
    _, all_evidence_rows = read_csv(evidence_args.csv)
    if (
        len(all_evidence_rows) != len(evidence_rows)
        or any(
            row.get("run_id") != args.run_id
            for row in all_evidence_rows
        )
    ):
        fail(
            f"{args.manifest}: immutable evidence snapshot contains rows "
            f"outside run_id {args.run_id!r}"
        )
    snapshot_times = validate_idle_device_snapshot(
        artifact_paths["device_snapshot"][0],
        args.device,
        expected_run_id=args.run_id,
        expected_hostname=values["hostname"],
    )
    require_device_snapshot_interval(
        args.manifest,
        snapshot_times,
        run_start,
        run_end,
    )
    if args.execution_kind == "perf":
        validate_runtime_grid_config(
            artifact_paths["runtime_grid_config"][0],
            args.run_id,
            args.variant_mode,
            args.device,
        )
    return parsed


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not RUN_ID_RE.fullmatch(name):
        raise argparse.ArgumentTypeError(f"unsafe artifact name: {name!r}")
    if not raw_path:
        raise argparse.ArgumentTypeError("artifact path is empty")
    return name, Path(raw_path)


def write_freeze_snapshot(args: argparse.Namespace) -> None:
    artifacts = list(args.artifact)
    names = [name for name, _ in artifacts]
    if len(names) != len(set(names)):
        fail("freeze artifact names must be unique")
    if getattr(args, "formal_contract", False) and set(names) != FORMAL_BUILD_INPUT_FILES:
        fail(
            "formal freeze artifact set mismatch: "
            f"missing={sorted(FORMAL_BUILD_INPUT_FILES - set(names))}, "
            f"extra={sorted(set(names) - FORMAL_BUILD_INPUT_FILES)}"
        )
    if getattr(args, "formal_contract", False):
        artifact_paths = dict(artifacts)
        shape_files = {
            role: sha256_file(artifact_paths[role])
            for role in CANONICAL_FORMAL_SHAPES
        }
        shape_paths = {
            role: str(artifact_paths[role].resolve())
            for role in CANONICAL_FORMAL_SHAPES
        }
        require_canonical_formal_shapes(
            args.output,
            shape_files,
            shape_paths,
        )
    rows: list[tuple[str, str, str, str]] = [
        ("value", "snapshot_schema_version", "1", ""),
    ]
    for name, path in sorted(artifacts):
        if not path.is_file():
            fail(f"freeze artifact is missing: {name}={path}")
        rows.append(("file", name, str(path.resolve()), sha256_file(path)))
    write_tsv_manifest(args.output, rows)


def check_freeze_snapshot(
    path: Path, formal_contract: bool = False
) -> dict[str, dict[str, str]]:
    parsed = parse_manifest(path)
    require_values(
        path,
        parsed["values"],
        {"snapshot_schema_version": "1"},
    )
    if not parsed["files"]:
        fail(f"{path}: freeze snapshot has no artifacts")
    if formal_contract and set(parsed["files"]) != FORMAL_BUILD_INPUT_FILES:
        fail(
            f"{path}: formal freeze artifact set mismatch: "
            f"missing={sorted(FORMAL_BUILD_INPUT_FILES - set(parsed['files']))}, "
            f"extra={sorted(set(parsed['files']) - FORMAL_BUILD_INPUT_FILES)}"
        )
    if formal_contract:
        require_canonical_formal_shapes(
            path,
            parsed["files"],
            parsed["paths"],
        )
    return parsed


def check_binary_bundle(
    manifest: Path,
    formal_set_id: str,
    build_input_snapshot: Path,
) -> dict[str, dict[str, str]]:
    if manifest.is_symlink():
        fail(f"binary bundle manifest must not be a symlink: {manifest}")
    try:
        manifest_stat = manifest.stat()
        bundle_stat = manifest.parent.stat()
    except FileNotFoundError as error:
        fail(f"binary bundle is missing: {error}")
        raise AssertionError
    if (
        not stat.S_ISREG(manifest_stat.st_mode)
        or manifest_stat.st_mode & 0o222
        or manifest.parent.is_symlink()
        or not stat.S_ISDIR(bundle_stat.st_mode)
        or bundle_stat.st_mode & 0o222
    ):
        fail(f"binary bundle and manifest must be read-only: {manifest}")
    if build_input_snapshot.is_symlink():
        fail(
            "binary bundle build-input snapshot must not be a symlink: "
            f"{build_input_snapshot}"
        )
    check_freeze_snapshot(build_input_snapshot, formal_contract=True)
    try:
        snapshot_path = build_input_snapshot.resolve(strict=True)
    except FileNotFoundError as error:
        fail(f"binary bundle build-input snapshot is missing: {error}")
        raise AssertionError
    parsed = parse_manifest(manifest)
    expected_values = {
        "binary_bundle_schema_version": "2",
        "formal_set_id": formal_set_id,
        "build_input_snapshot_path": str(snapshot_path),
        "build_input_snapshot_sha256": sha256_file(snapshot_path),
    }
    require_values(manifest, parsed["values"], expected_values)
    if set(parsed["files"]) != FORMAL_BINARY_NAMES:
        fail(
            f"{manifest}: binary bundle set mismatch: "
            f"missing={sorted(FORMAL_BINARY_NAMES - set(parsed['files']))}, "
            f"extra={sorted(set(parsed['files']) - FORMAL_BINARY_NAMES)}"
        )
    bundle_dir = manifest.parent.resolve()
    for name, resolved in parsed["paths"].items():
        candidate = manifest.parent / name
        try:
            candidate_stat = candidate.lstat()
        except FileNotFoundError:
            fail(f"{manifest}: bundled binary is missing: {name}")
            raise AssertionError
        if (
            stat.S_ISLNK(candidate_stat.st_mode)
            or not stat.S_ISREG(candidate_stat.st_mode)
            or candidate_stat.st_mode & 0o222
            or candidate.resolve() != bundle_dir / name
            or Path(resolved) != bundle_dir / name
            or not os.access(candidate, os.X_OK)
        ):
            fail(f"{manifest}: bundled binary is misnamed or not executable: {name}")
    return parsed


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is not None:
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,
        )
        if result == 0:
            return
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            fail(f"refusing to overwrite binary bundle: {destination}")
        fail(
            f"atomic binary bundle publish failed: "
            f"{os.strerror(error)} ({source} -> {destination})"
        )
    if os.path.lexists(destination):
        fail(f"refusing to overwrite binary bundle: {destination}")
    # renameat2 is available on the Linux evidence hosts. This fallback keeps
    # unit fixtures portable on platforms that lack the Linux syscall.
    os.rename(source, destination)


def write_binary_bundle(args: argparse.Namespace) -> None:
    validate_run_id(args.formal_set_id)
    if args.build_input_snapshot.is_symlink():
        fail(
            "binary bundle build-input snapshot must not be a symlink: "
            f"{args.build_input_snapshot}"
        )
    check_freeze_snapshot(
        args.build_input_snapshot,
        formal_contract=True,
    )
    try:
        snapshot_path = args.build_input_snapshot.resolve(strict=True)
    except FileNotFoundError as error:
        fail(f"binary bundle build-input snapshot is missing: {error}")
        raise AssertionError
    artifacts = list(args.artifact)
    names = [name for name, _ in artifacts]
    if set(names) != FORMAL_BINARY_NAMES or len(names) != len(set(names)):
        fail(
            "binary bundle input set mismatch: "
            f"missing={sorted(FORMAL_BINARY_NAMES - set(names))}, "
            f"extra={sorted(set(names) - FORMAL_BINARY_NAMES)}"
        )
    output_dir = args.output_dir.absolute()
    if os.path.lexists(output_dir):
        fail(f"refusing to overwrite binary bundle: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    partial = output_dir.with_name(f"{output_dir.name}.partial.{os.getpid()}")
    if os.path.lexists(partial):
        fail(f"binary bundle partial already exists: {partial}")
    partial.mkdir(mode=0o700)
    rows: list[tuple[str, str, str, str]] = [
        ("value", "binary_bundle_schema_version", "2", ""),
        ("value", "formal_set_id", args.formal_set_id, ""),
        ("value", "build_input_snapshot_path", str(snapshot_path), ""),
        (
            "value",
            "build_input_snapshot_sha256",
            sha256_file(snapshot_path),
            "",
        ),
    ]
    for name, source in sorted(artifacts):
        if source.name != name:
            fail(f"binary bundle logical name must equal basename: {name}={source}")
        if source.is_symlink() or not source.is_file() or not os.access(source, os.X_OK):
            fail(f"binary bundle source is not a regular executable: {source}")
        before = source.stat()
        before_digest = sha256_file(source)
        destination = partial / name
        shutil.copy2(source, destination, follow_symlinks=False)
        after = source.stat()
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before_digest,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            sha256_file(source),
        ):
            fail(f"binary changed while bundling: {source}")
        if sha256_file(destination) != before_digest or not os.access(
            destination, os.X_OK
        ):
            fail(f"bundled binary copy differs from source: {source}")
        destination.chmod(0o555)
        rows.append(("file", name, name, before_digest))
    write_tsv_manifest(partial / "manifest.tsv", rows)
    (partial / "manifest.tsv").chmod(0o444)
    _rename_directory_noreplace(partial, output_dir)
    output_dir.chmod(0o555)
    check_binary_bundle(
        output_dir / "manifest.tsv",
        args.formal_set_id,
        snapshot_path,
    )


def canonical_path(path: Path) -> Path:
    return path.resolve()


def ensure_contained(parent: Path, child: Path, label: str) -> None:
    parent_resolved = parent.resolve()
    child_resolved = child.resolve()
    try:
        common = Path(
            os.path.commonpath((str(parent_resolved), str(child_resolved)))
        )
    except ValueError:
        fail(f"{label} is not under {parent_resolved}: {child_resolved}")
        raise AssertionError
    if common != parent_resolved:
        fail(f"{label} is not under {parent_resolved}: {child_resolved}")


def validate_formal_paths(args: argparse.Namespace) -> None:
    result_dir = args.result_dir.resolve()
    work_root = args.work_root.resolve()
    ensure_contained(work_root, result_dir, "result directory")
    paths = {
        "accuracy CSV": args.accuracy_csv,
        "performance CSV": args.perf_csv,
        "preheat CSV": args.preheat_csv,
        "manifest directory": args.manifest_dir,
        "set manifest": args.set_manifest,
        "bracket summary": args.bracket_summary,
        "formal metrics": args.formal_metrics,
        "result lock": args.result_lock,
        "build input snapshot": args.build_input_snapshot,
        "formal binary bundle": args.binary_bundle_dir,
    }
    for label, path in paths.items():
        ensure_contained(result_dir, path, label)
        if path.is_symlink():
            fail(f"{label} must not be a symlink: {path}")
    csv_paths = [
        args.accuracy_csv.resolve(),
        args.perf_csv.resolve(),
        args.preheat_csv.resolve(),
    ]
    if len(set(str(path) for path in csv_paths)) != len(csv_paths):
        fail(f"accuracy/performance/preheat CSV paths alias: {csv_paths}")
    existing = [path for path in csv_paths if path.exists()]
    for index, left in enumerate(existing):
        for right in existing[index + 1 :]:
            if os.path.samefile(str(left), str(right)):
                fail(f"formal CSV paths share one inode: {left}, {right}")


def validate_lock_fd(args: argparse.Namespace) -> None:
    fd_path = Path(f"/proc/self/fd/{args.fd}")
    try:
        actual = fd_path.resolve(strict=True)
        declared = args.declared.resolve(strict=True)
        expected = args.expected.resolve(strict=True)
    except FileNotFoundError as error:
        fail(f"formal lock path is missing: {error}")
        raise AssertionError
    if actual != declared or actual != expected:
        fail(
            f"fd {args.fd} lock identity mismatch: "
            f"actual={actual}, declared={declared}, expected={expected}"
        )


def manifest_validation_args(
    manifest_path: Path, parsed: Mapping[str, Mapping[str, str]]
) -> argparse.Namespace:
    values = parsed["values"]
    paths = parsed["paths"]
    execution_kind = values.get("execution_kind", "")
    variant_pair = (
        values.get("softmax_variant", ""),
        values.get("rmsnorm_variant", ""),
    )
    if variant_pair == ("converted", "converted"):
        variant_mode = "direct"
    elif variant_pair == ("native-half2-hybrid", "native"):
        variant_mode = "native"
    else:
        fail(f"{manifest_path}: cannot infer formal variant mode: {variant_pair}")
        raise AssertionError
    try:
        return argparse.Namespace(
            execution_kind=execution_kind,
            run_id=values["run_id"],
            shapes=Path(paths["shapes"]),
            tier=values["tier"],
            variant_mode=variant_mode,
            expected_softmax=positive_int(values["expected_softmax_rows"]),
            expected_rmsnorm=positive_int(values["expected_rmsnorm_rows"]),
            device=int(values["device"]),
            math_mode=values["math_mode"],
            warmup=positive_int(values["warmup"]),
            samples=positive_int(values["samples"]),
            inner_repeats=positive_int(values["inner_repeats"]),
            csv=Path(paths["evidence_csv"]),
            manifest=manifest_path,
            evidence=Path(paths["evidence_csv"]),
            formal_set_id=values["formal_set_id"],
            formal_phase=values["formal_phase"],
            ascify_binary_sha256=values["ascify_binary_sha256"],
        )
    except (KeyError, ValueError, argparse.ArgumentTypeError) as error:
        fail(f"{manifest_path}: cannot infer complete validation contract: {error}")
        raise AssertionError


def validate_complete_manifest(
    path: Path,
) -> tuple[dict[str, dict[str, str]], argparse.Namespace]:
    parsed = parse_manifest(path)
    args = manifest_validation_args(path, parsed)
    validate_csv_rows(args)
    validated = validate_manifest(args)
    return validated, args


def preflight(args: argparse.Namespace) -> None:
    run_ids = list(args.run_id)
    prepared = bool(getattr(args, "prepared", False))
    if prepared and not args.set_id:
        fail("prepared preflight requires --set-id")
    if prepared and (
        args.manifest_dir.is_symlink() or not args.manifest_dir.is_dir()
    ):
        fail(
            "prepared preflight requires a real manifest directory: "
            f"{args.manifest_dir}"
        )
    if len(run_ids) != len(set(run_ids)):
        fail("formal run IDs must be unique")
    for run_id in run_ids:
        validate_run_id(run_id)
    for history in (args.accuracy_csv, args.perf_csv):
        if not history.exists():
            continue
        _, rows = read_csv(history)
        used = sorted({row.get("run_id", "") for row in rows} & set(run_ids))
        if used:
            fail(f"{history}: formal run IDs already have rows: {used}")
    if args.manifest_dir.exists():
        conflicts: list[Path] = []
        for run_id in run_ids:
            for suffix in (
                ".tsv",
                ".csv",
                ".device.txt",
                ".runtime_config.tsv",
            ):
                candidate = args.manifest_dir / f"{run_id}{suffix}"
                if os.path.lexists(candidate):
                    conflicts.append(candidate)
            conflicts.extend(args.manifest_dir.glob(f"{run_id}.partial*"))
            conflicts.extend(args.manifest_dir.glob(f"{run_id}.*.partial*"))
        if args.set_id:
            candidate = args.manifest_dir / f"{args.set_id}.tsv"
            if os.path.lexists(candidate):
                conflicts.append(candidate)
            conflicts.extend(args.manifest_dir.glob(f"{args.set_id}.partial*"))
            conflicts.extend(args.manifest_dir.glob(f"{args.set_id}.*.partial*"))
            conflicts.extend(
                args.manifest_dir.glob(f"{args.set_id}.bracket_summary.csv")
            )
            conflicts.extend(
                args.manifest_dir.glob(f"{args.set_id}.formal_metrics.csv")
            )
            snapshot = (
                args.manifest_dir / f"{args.set_id}.build_inputs.tsv"
            )
            bundle = args.manifest_dir / f"{args.set_id}.binaries"
            conflicts.extend(
                args.manifest_dir.glob(f"{args.set_id}.binaries.partial.*")
            )
            if prepared:
                if (
                    snapshot.is_symlink()
                    or not snapshot.is_file()
                    or bundle.is_symlink()
                    or not bundle.is_dir()
                ):
                    fail(
                        "prepared formal inputs are missing or unsafe: "
                        f"snapshot={snapshot}, bundle={bundle}"
                    )
            else:
                if os.path.lexists(snapshot):
                    conflicts.append(snapshot)
                if os.path.lexists(bundle):
                    conflicts.append(bundle)
        if conflicts:
            fail(
                "formal evidence paths already exist: "
                + ", ".join(str(path) for path in sorted(set(conflicts)))
            )
    if args.set_id and args.set_id in set(run_ids):
        fail("formal set ID must be distinct from every run ID")


def iso8601(value: str, label: str) -> dt.datetime:
    try:
        return dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=dt.timezone.utc
        )
    except ValueError as error:
        fail(f"invalid {label} timestamp: {value!r}")
        raise AssertionError from error


def manifest_rows(
    values: Sequence[tuple[str, str]],
    files: Sequence[tuple[str, Path]],
) -> list[tuple[str, str, str, str]]:
    rows = [("value", name, value, "") for name, value in values]
    rows.extend(
        ("file", name, str(path), sha256_file(path)) for name, path in files
    )
    return rows


def write_tsv_manifest(
    path: Path, rows: Sequence[tuple[str, str, str, str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        stream = path.open("x", newline="", encoding="utf-8")
    except FileExistsError:
        fail(f"refusing to overwrite formal set manifest: {path}")
    with stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(("kind", "name", "value_or_path", "sha256"))
        writer.writerows(rows)


def keyed_perf_rows(
    rows: Sequence[Mapping[str, str]], label: str
) -> dict[tuple[str, str], Mapping[str, str]]:
    keyed: dict[tuple[str, str], Mapping[str, str]] = {}
    for row in rows:
        key = (row.get("op", ""), row.get("case_id", ""))
        if not all(key) or key in keyed:
            fail(f"{label} has an empty or duplicate performance key: {key}")
        keyed[key] = row
    return keyed


def performance_class(
    key: tuple[str, str],
    shapes: Mapping[tuple[str, str], Mapping[str, str]],
) -> str:
    shape = shapes.get(key)
    if shape is None:
        fail(f"A/native/B performance key has no frozen shape: {key}")
    if key[0] == "softmax":
        return "softmax"
    if key[0] == "rms_norm":
        affine = shape.get("affine")
        if affine == "0":
            return "rms_plain"
        if affine == "1":
            return "rms_affine"
        fail(f"frozen RMSNorm shape has invalid affine field for {key}: {affine!r}")
    fail(f"unsupported performance op for class gate: {key[0]!r}")
    raise AssertionError


def build_bracket_summary(
    formal_set_id: str,
    run_ids: tuple[str, str, str],
    rows_a: Sequence[Mapping[str, str]],
    rows_native: Sequence[Mapping[str, str]],
    rows_b: Sequence[Mapping[str, str]],
    shapes: Mapping[tuple[str, str], Mapping[str, str]],
) -> tuple[list[dict[str, str]], float, dict[str, float]]:
    a = keyed_perf_rows(rows_a, "direct A")
    native = keyed_perf_rows(rows_native, "native")
    b = keyed_perf_rows(rows_b, "direct B")
    if set(a) != set(native) or set(a) != set(b):
        fail(
            "A/native/B performance keys differ: "
            f"A={sorted(a)}, native={sorted(native)}, B={sorted(b)}"
        )
    measurements: list[dict[str, object]] = []
    ratios_by_class: dict[str, list[float]] = {
        name: [] for name in PERFORMANCE_CLASSES
    }
    max_spread = 1.0
    for key in sorted(a):
        a_gbps = finite_number(a[key]["gbps"], f"direct A {key} gbps", positive=True)
        native_gbps = finite_number(
            native[key]["gbps"], f"native {key} gbps", positive=True
        )
        b_gbps = finite_number(b[key]["gbps"], f"direct B {key} gbps", positive=True)
        spread = max(a_gbps, b_gbps) / min(a_gbps, b_gbps)
        max_spread = max(max_spread, spread)
        if spread > DIRECT_AB_SPREAD_LIMIT:
            fail(
                f"direct A/B performance drift exceeds "
                f"{DIRECT_AB_SPREAD_LIMIT:.2f} for {key}: "
                f"A={a_gbps:.17g} GB/s, B={b_gbps:.17g} GB/s, "
                f"spread={spread:.17g}"
            )
        center = math.sqrt(a_gbps) * math.sqrt(b_gbps)
        direct_native_ratio = center / native_gbps
        if direct_native_ratio < DIRECT_CASE_NATIVE_RATIO_MIN:
            fail(
                f"direct/native case performance is below "
                f"{DIRECT_CASE_NATIVE_RATIO_MIN:.2f} for {key}: "
                f"direct_geomean={center:.17g} GB/s, "
                f"native={native_gbps:.17g} GB/s, "
                f"ratio={direct_native_ratio:.17g}"
            )
        class_name = performance_class(key, shapes)
        ratios_by_class[class_name].append(direct_native_ratio)
        measurements.append(
            {
                "key": key,
                "class_name": class_name,
                "a_gbps": a_gbps,
                "native_gbps": native_gbps,
                "b_gbps": b_gbps,
                "spread": spread,
                "center": center,
                "direct_native_ratio": direct_native_ratio,
            }
        )
    if not measurements:
        fail("A/native/B bracket contains no performance rows")
    class_geomeans: dict[str, float] = {}
    for class_name in PERFORMANCE_CLASSES:
        ratios = ratios_by_class[class_name]
        if not ratios:
            fail(f"A/native/B bracket has no rows for class {class_name}")
        class_geomean = math.exp(
            math.fsum(math.log(ratio) for ratio in ratios) / len(ratios)
        )
        class_geomeans[class_name] = class_geomean
        if class_geomean < DIRECT_CLASS_NATIVE_GEOMEAN_MIN:
            fail(
                f"direct/native class performance is below "
                f"{DIRECT_CLASS_NATIVE_GEOMEAN_MIN:.2f} for {class_name}: "
                f"geomean={class_geomean:.17g}"
            )
    summary: list[dict[str, str]] = []
    for measurement in measurements:
        key = measurement["key"]
        if not isinstance(key, tuple):
            raise AssertionError("internal bracket key type drift")
        class_name = str(measurement["class_name"])
        summary.append(
            {
                "formal_set_id": formal_set_id,
                "op": key[0],
                "case_id": key[1],
                "performance_class": class_name,
                "direct_a_run_id": run_ids[0],
                "native_run_id": run_ids[1],
                "direct_b_run_id": run_ids[2],
                "direct_a_gbps": f"{float(measurement['a_gbps']):.17g}",
                "direct_b_gbps": f"{float(measurement['b_gbps']):.17g}",
                "direct_ab_geomean_gbps": (
                    f"{float(measurement['center']):.17g}"
                ),
                "native_gbps": f"{float(measurement['native_gbps']):.17g}",
                "direct_center_over_native": (
                    f"{float(measurement['direct_native_ratio']):.17g}"
                ),
                "direct_case_over_native_min_ratio": (
                    f"{DIRECT_CASE_NATIVE_RATIO_MIN:.17g}"
                ),
                "direct_class_over_native_geomean": (
                    f"{class_geomeans[class_name]:.17g}"
                ),
                "direct_class_over_native_geomean_min_ratio": (
                    f"{DIRECT_CLASS_NATIVE_GEOMEAN_MIN:.17g}"
                ),
                "direct_ab_spread_ratio": (
                    f"{float(measurement['spread']):.17g}"
                ),
                "direct_ab_relative_drift": (
                    f"{float(measurement['spread']) - 1.0:.17g}"
                ),
            }
        )
    return summary, max_spread, class_geomeans


def write_bracket_summary(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        stream = path.open("x", newline="", encoding="utf-8")
    except FileExistsError:
        fail(f"refusing to overwrite bracket summary: {path}")
    with stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(BRACKET_SUMMARY_FIELDS),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def load_metric_model(path: Path) -> Any:
    if not path.is_file():
        fail(f"missing frozen metric model: {path}")
    module_name = "_ascify_formal_metric_model"
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    sys.modules[module_name] = module
    try:
        source = path.read_bytes()
        code = compile(source, str(path), "exec")
        exec(code, module.__dict__)
    except Exception as error:
        fail(f"cannot load frozen metric model {path}: {error}")
    for name in ("compute_work_metrics", "format_rate"):
        if not callable(getattr(module, name, None)):
            fail(f"{path}: frozen metric model lacks callable {name}")
    return module


def build_formal_metric_rows(
    formal_set_id: str,
    roles: Sequence[str],
    evidence_rows: Sequence[Sequence[Mapping[str, str]]],
    shapes: Mapping[tuple[str, str], Mapping[str, str]],
    metric_model: Any,
) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for role, run_rows in zip(roles, evidence_rows):
        for row in sorted(run_rows, key=lambda value: (value["op"], value["case_id"])):
            key = (row["op"], row["case_id"])
            if key not in shapes:
                fail(f"formal metric row has no frozen shape: {key}")
            shape = shapes[key]
            rows = int(shape["rows"])
            cols = int(shape["cols"])
            affine = shape["affine"] == "1"
            work = metric_model.compute_work_metrics(row["op"], rows, cols, affine)
            median_ms = finite_number(
                row["lat_ms_median"], f"formal metric {role} {key} latency",
                positive=True,
            )
            gbps = finite_number(
                row["gbps"], f"formal metric {role} {key} gbps", positive=True
            )
            output.append(
                {
                    "metric_schema_version": "1",
                    "formal_set_id": formal_set_id,
                    "run_role": role,
                    "run_id": row["run_id"],
                    "op": row["op"],
                    "variant": row["variant"],
                    "case_id": row["case_id"],
                    "rows": str(rows),
                    "cols": str(cols),
                    "affine": "1" if affine else "0",
                    "lat_ms_median": row["lat_ms_median"],
                    "gbps": row["gbps"],
                    "metric_model": work.model,
                    "fp32_ops": str(work.fp32_ops),
                    "fp16_ops": str(work.fp16_ops),
                    "arith_ops": str(work.arith_ops),
                    "arith_tflops": metric_model.format_rate(
                        work.arith_ops / (median_ms * 1.0e9)
                    ),
                    "special_op_kind": work.special_op_kind,
                    "special_ops": str(work.special_ops),
                    "special_gops": metric_model.format_rate(
                        work.special_ops / (median_ms * 1.0e6)
                    ),
                    "compare_ops": str(work.compare_ops),
                    "compare_gops": metric_model.format_rate(
                        work.compare_ops / (median_ms * 1.0e6)
                    ),
                }
            )
    expected_rows = len(roles) * len(shapes)
    if len(output) != expected_rows:
        fail(
            f"formal metric coverage mismatch: actual={len(output)}, "
            f"expected={expected_rows}"
        )
    return output


def write_formal_metrics(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        stream = path.open("x", newline="", encoding="utf-8")
    except FileExistsError:
        fail(f"refusing to overwrite formal metrics: {path}")
    with stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(FORMAL_METRIC_FIELDS), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def require_distinct_files(label: str, paths: Sequence[Path]) -> None:
    resolved: list[Path] = []
    identities: list[tuple[int, int]] = []
    for path in paths:
        try:
            canonical = path.resolve(strict=True)
            file_stat = canonical.stat()
        except FileNotFoundError as error:
            fail(f"{label}: missing artifact: {error}")
            raise AssertionError
        if not canonical.is_file():
            fail(f"{label}: artifact is not a regular file: {canonical}")
        resolved.append(canonical)
        identities.append((file_stat.st_dev, file_stat.st_ino))
    if len(resolved) != len(set(resolved)):
        fail(f"{label}: artifact paths must be pairwise distinct: {resolved}")
    if len(identities) != len(set(identities)):
        fail(f"{label}: artifacts must not share an inode")


def require_distinct_formal_runs(
    formal_set_id: str,
    run_ids: Sequence[str],
    manifests: Sequence[Path],
    evidence: Sequence[Path],
    device_snapshots: Sequence[Path],
    runtime_configs: Sequence[Path],
) -> None:
    if len(run_ids) != 8:
        fail(f"formal set must contain exactly 8 runs, got {len(run_ids)}")
    if len(run_ids) != len(set(run_ids)):
        fail(f"formal run IDs must be pairwise distinct: {list(run_ids)}")
    if formal_set_id in run_ids:
        fail("formal set ID must not reuse a formal run ID")
    if (
        len(manifests) != 8
        or len(evidence) != 8
        or len(device_snapshots) != 8
    ):
        fail(
            "formal set must bind 8 distinct manifests, evidence files, "
            f"and device snapshots: manifests={len(manifests)}, "
            f"evidence={len(evidence)}, snapshots={len(device_snapshots)}"
        )
    if len(runtime_configs) != 4:
        fail(
            "formal set must bind 4 distinct performance runtime configs, "
            f"got {len(runtime_configs)}"
        )
    require_distinct_files("formal run manifests", manifests)
    require_distinct_files("formal run evidence", evidence)
    require_distinct_files("formal device snapshots", device_snapshots)
    require_distinct_files("formal runtime configs", runtime_configs)


def compare_bracket(args: argparse.Namespace) -> None:
    manifests = [args.manifest_a, args.manifest_native, args.manifest_b]
    validated = [validate_complete_manifest(path) for path in manifests]
    parsed = [item[0] for item in validated]
    a, native, b = parsed
    formal_run_ids = [item["values"]["run_id"] for item in parsed]
    formal_manifest_paths = list(manifests)
    formal_evidence_paths = [
        Path(item["paths"]["evidence_csv"]) for item in parsed
    ]
    formal_device_snapshot_paths = [
        Path(item["paths"]["device_snapshot"]) for item in parsed
    ]
    formal_runtime_paths = [
        Path(item["paths"]["runtime_grid_config"]) for item in parsed
    ]
    if len(set(formal_run_ids)) != 3 or args.formal_set_id in formal_run_ids:
        fail(
            "A/native/B run IDs must be pairwise distinct and differ "
            f"from the formal set ID: {formal_run_ids}"
        )
    require_distinct_files("A/native/B manifests", formal_manifest_paths)
    require_distinct_files("A/native/B evidence", formal_evidence_paths)
    require_distinct_files(
        "A/native/B device snapshots",
        formal_device_snapshot_paths,
    )
    require_distinct_files("A/native/B runtime configs", formal_runtime_paths)
    for path, item, expected_phase, expected_variant in zip(
        manifests,
        parsed,
        ("direct_perf_a", "native_perf", "direct_perf_b"),
        (("converted", "converted"), ("native-half2-hybrid", "native"), ("converted", "converted")),
    ):
        require_values(
            path,
            item["values"],
            {
                "manifest_schema_version": "2",
                "formal_manifest": "1",
                "run_status": "complete",
                "formal_set_id": args.formal_set_id,
                "formal_phase": expected_phase,
                "execution_kind": "perf",
                "softmax_variant": expected_variant[0],
                "rmsnorm_variant": expected_variant[1],
            },
        )

    stable_values = (
        "hostname",
        "device",
        "device_lock",
        "formal_result_lock",
        "device_util_max",
        "device_hbm_bw_max",
        "cann_root",
        "bin_dir",
        "math_mode",
        "tier",
        "warmup",
        "samples",
        "inner_repeats",
        "check_max_elements",
        "op_filter",
        "perf_csv",
        "ascify_binary_sha256",
        "softmax_block_threads",
        "softmax_block_row_max",
        "rmsnorm_block_row_threads",
        "rmsnorm_block_row_affine_threads",
        "rmsnorm_grid_cap",
        "ascify_target_policy",
        "ascify_simt_math",
        "ascify_conversion_set_id",
        "ascify_conversion_manifest_sha256",
        "ascify_input_softmax_sha256",
        "ascify_input_layer_norm_sha256",
        "ascify_input_rmsnorm_sha256",
        "ascify_input_rmsnorm_adapter_sha256",
    )
    for name in stable_values:
        actual = [item["values"].get(name) for item in parsed]
        if len(set(actual)) != 1 or actual[0] in (None, ""):
            fail(f"A/native/B manifest value mismatch for {name}: {actual}")

    for name in BRACKET_STABLE_FILES | {"shapes"}:
        actual = [item["files"].get(name) for item in parsed]
        if len(set(actual)) != 1 or actual[0] is None:
            fail(f"A/native/B artifact hash mismatch for {name}: {actual}")

    for name in ("softmax_bench_binary", "rmsnorm_bench_binary"):
        if a["files"].get(name) != b["files"].get(name):
            fail(f"direct A/B binary hash mismatch for {name}")
        if native["files"].get(name) == a["files"].get(name):
            fail(f"native control unexpectedly shares direct binary hash for {name}")
    softmax_grid_caps = [
        item["values"].get("softmax_grid_cap") for item in parsed
    ]
    if softmax_grid_caps != ["auto-aiv-x32"] * 3:
        fail(
            "A/native/B do not record the same auto Softmax grid cap: "
            f"{softmax_grid_caps}"
        )
    runtime_configs = [
        validate_runtime_grid_config(
            Path(item["paths"]["runtime_grid_config"]),
            item["values"]["run_id"],
            variant_mode,
            int(item["values"]["device"]),
        )
        for item, variant_mode in zip(parsed, ("direct", "native", "direct"))
    ]
    effective_grids = {
        (item["vector_core_count"], item["resolved_grid_cap"])
        for item in runtime_configs
    }
    if len(effective_grids) != 1:
        fail(f"A/native/B runtime grid resolution mismatch: {runtime_configs}")
    runtime_vector_core_count, runtime_resolved_grid_cap = next(
        iter(effective_grids)
    )

    a_start = iso8601(a["values"]["utc_time"], "A start")
    a_end = iso8601(a["values"]["run_end_utc_time"], "A end")
    native_start = iso8601(native["values"]["utc_time"], "native start")
    native_end = iso8601(native["values"]["run_end_utc_time"], "native end")
    b_start = iso8601(b["values"]["utc_time"], "B start")
    b_end = iso8601(b["values"]["run_end_utc_time"], "B end")
    if not (a_start <= a_end <= native_start <= native_end <= b_start <= b_end):
        fail("A/native/B UTC timestamps do not prove consecutive bracket order")

    evidence_rows = []
    for item in parsed:
        _, all_rows = read_csv(Path(item["paths"]["evidence_csv"]))
        run_id = item["values"]["run_id"]
        evidence_rows.append(
            [row for row in all_rows if row.get("run_id") == run_id]
        )
    metric_shapes = selected_shape_rows(
        Path(a["paths"]["shapes"]), "perf", a["values"]["tier"]
    )
    bracket_rows, max_direct_spread, class_geomeans = build_bracket_summary(
        args.formal_set_id,
        (
            a["values"]["run_id"],
            native["values"]["run_id"],
            b["values"]["run_id"],
        ),
        evidence_rows[0],
        evidence_rows[1],
        evidence_rows[2],
        metric_shapes,
    )
    recorded_metric_tool = Path(a["paths"]["derive_work_metrics_tool"])
    if recorded_metric_tool.resolve() != args.metrics_tool.resolve():
        fail(
            "formal metric tool path is not the frozen run artifact: "
            f"recorded={recorded_metric_tool}, supplied={args.metrics_tool}"
        )
    metric_model = load_metric_model(args.metrics_tool)
    formal_metric_rows = build_formal_metric_rows(
        args.formal_set_id,
        ("direct_a", "native", "direct_b"),
        evidence_rows,
        metric_shapes,
        metric_model,
    )

    values = a["values"]
    preheat, _ = validate_complete_manifest(args.preheat_manifest)
    formal_run_ids.append(preheat["values"]["run_id"])
    formal_manifest_paths.append(args.preheat_manifest)
    formal_evidence_paths.append(Path(preheat["paths"]["evidence_csv"]))
    formal_device_snapshot_paths.append(
        Path(preheat["paths"]["device_snapshot"])
    )
    formal_runtime_paths.append(
        Path(preheat["paths"]["runtime_grid_config"])
    )
    require_values(
        args.preheat_manifest,
        preheat["values"],
        {
            "manifest_schema_version": "2",
            "formal_manifest": "1",
            "run_status": "complete",
            "formal_set_id": args.formal_set_id,
            "formal_phase": "preheat",
            "execution_kind": "perf",
            "hostname": values["hostname"],
            "device": values["device"],
            "device_lock": values["device_lock"],
            "formal_result_lock": values["formal_result_lock"],
            "cann_root": values["cann_root"],
            "math_mode": values["math_mode"],
            "tier": "tune",
            "expected_softmax_rows": "5",
            "expected_rmsnorm_rows": "10",
            "softmax_variant": "converted",
            "rmsnorm_variant": "converted",
            "ascify_binary_sha256": values["ascify_binary_sha256"],
        },
    )
    if preheat["values"].get("perf_csv") == values.get("perf_csv"):
        fail("preheat rows must be isolated from the formal performance history")
    preheat_runtime = validate_runtime_grid_config(
        Path(preheat["paths"]["runtime_grid_config"]),
        preheat["values"]["run_id"],
        "direct",
        int(preheat["values"]["device"]),
    )
    if preheat_runtime != runtime_configs[0]:
        fail(
            "preheat/direct-A runtime grid resolution mismatch: "
            f"preheat={preheat_runtime}, direct-A={runtime_configs[0]}"
        )
    for name in BRACKET_STABLE_FILES | {"shapes"}:
        if preheat["files"].get(name) != a["files"].get(name):
            fail(f"{args.preheat_manifest}: preheat artifact hash mismatch for {name}")
    for name in ("softmax_bench_binary", "rmsnorm_bench_binary"):
        if preheat["files"].get(name) != a["files"].get(name):
            fail(f"{args.preheat_manifest}: preheat/direct-A binary hash mismatch for {name}")
    preheat_end = iso8601(
        preheat["values"]["run_end_utc_time"], "preheat end"
    )
    preheat_start = iso8601(preheat["values"]["utc_time"], "preheat start")
    if preheat_end > a_start:
        fail("labeled preheat did not complete before direct A started")

    gate_files: list[tuple[str, Path]] = []
    gate_values: list[tuple[str, str]] = []
    gate_start_values: list[str] = []
    seen_gate_phases: set[str] = set()
    for gate_path in args.gate_manifest:
        gate, _ = validate_complete_manifest(gate_path)
        formal_run_ids.append(gate["values"]["run_id"])
        formal_manifest_paths.append(gate_path)
        formal_evidence_paths.append(Path(gate["paths"]["evidence_csv"]))
        formal_device_snapshot_paths.append(
            Path(gate["paths"]["device_snapshot"])
        )
        gate_phase = gate["values"].get("formal_phase", "")
        if (
            gate_phase not in EXPECTED_FORMAL_GATES
            or gate_phase in seen_gate_phases
        ):
            fail(f"{gate_path}: unexpected or duplicate formal gate phase {gate_phase!r}")
        seen_gate_phases.add(gate_phase)
        tier, softmax_rows, rmsnorm_rows, softmax_variant, rmsnorm_variant = (
            EXPECTED_FORMAL_GATES[gate_phase]
        )
        require_values(
            gate_path,
            gate["values"],
            {
                "manifest_schema_version": "2",
                "formal_manifest": "1",
                "run_status": "complete",
                "formal_set_id": args.formal_set_id,
                "execution_kind": "accuracy",
                "hostname": values["hostname"],
                "device": values["device"],
                "device_lock": values["device_lock"],
                "formal_result_lock": values["formal_result_lock"],
                "cann_root": values["cann_root"],
                "math_mode": values["math_mode"],
                "ascify_binary_sha256": values["ascify_binary_sha256"],
                "tier": tier,
                "expected_softmax_rows": softmax_rows,
                "expected_rmsnorm_rows": rmsnorm_rows,
                "softmax_variant": softmax_variant,
                "rmsnorm_variant": rmsnorm_variant,
            },
        )
        for name in COMMON_MANIFEST_FILES - {"evidence_csv", "shapes"}:
            if gate["files"].get(name) != a["files"].get(name):
                fail(f"{gate_path}: formal-set artifact hash mismatch for {name}")
        gate_end = iso8601(gate["values"]["run_end_utc_time"], f"{gate_phase} end")
        gate_start_values.append(gate["values"]["utc_time"])
        if gate_end > preheat_start:
            fail(f"{gate_path}: gate did not complete before preheat started")
        evidence_path = Path(gate["paths"].get("evidence_csv", ""))
        if not evidence_path.is_file():
            fail(f"{gate_path}: missing gate evidence snapshot")
        gate_values.append((f"{gate_phase}_run_id", gate["values"]["run_id"]))
        gate_files.extend(
            [
                (f"{gate_phase}_manifest", gate_path),
                (f"{gate_phase}_evidence", evidence_path),
            ]
        )
    if seen_gate_phases != set(EXPECTED_FORMAL_GATES):
        fail(
            "formal set gate coverage mismatch: "
            f"actual={sorted(seen_gate_phases)}, "
            f"expected={sorted(EXPECTED_FORMAL_GATES)}"
        )
    require_distinct_formal_runs(
        args.formal_set_id,
        formal_run_ids,
        formal_manifest_paths,
        formal_evidence_paths,
        formal_device_snapshot_paths,
        formal_runtime_paths,
    )

    output = args.output_manifest
    freeze = check_freeze_snapshot(
        args.build_input_snapshot,
        formal_contract=True,
    )
    if (
        a["files"].get("formal_build_input_snapshot")
        != sha256_file(args.build_input_snapshot)
    ):
        fail("run manifests do not bind the supplied build input snapshot")
    # Keep the parsed object live so every frozen artifact was verified above.
    if not freeze["files"]:
        raise AssertionError("checked freeze snapshot unexpectedly has no files")
    write_bracket_summary(args.output_summary, bracket_rows)
    write_formal_metrics(args.output_metrics, formal_metric_rows)
    rows = manifest_rows(
        [
            ("manifest_schema_version", "1"),
            ("formal_set_id", args.formal_set_id),
            ("run_status", "complete"),
            ("utc_time", min(gate_start_values)),
            ("run_end_utc_time", b["values"]["run_end_utc_time"]),
            ("hostname", values["hostname"]),
            ("device", values["device"]),
            ("device_lock", values["device_lock"]),
            ("formal_result_lock", values["formal_result_lock"]),
            ("cann_root", values["cann_root"]),
            ("math_mode", values["math_mode"]),
            ("tier", values["tier"]),
            ("warmup", values["warmup"]),
            ("samples", values["samples"]),
            ("inner_repeats", values["inner_repeats"]),
            ("ascify_binary_sha256", values["ascify_binary_sha256"]),
            ("perf_csv", values["perf_csv"]),
            ("preheat_perf_csv", preheat["values"]["perf_csv"]),
            ("preheat_run_id", preheat["values"]["run_id"]),
            ("preheat_warmup", preheat["values"]["warmup"]),
            ("preheat_samples", preheat["values"]["samples"]),
            ("preheat_inner_repeats", preheat["values"]["inner_repeats"]),
            ("direct_a_run_id", values["run_id"]),
            ("native_run_id", native["values"]["run_id"]),
            ("direct_b_run_id", b["values"]["run_id"]),
            ("runtime_vector_core_count", str(runtime_vector_core_count)),
            ("runtime_resolved_grid_cap", str(runtime_resolved_grid_cap)),
            ("bracket_summary_rows", str(len(bracket_rows))),
            ("formal_metrics_rows", str(len(formal_metric_rows))),
            (
                "direct_ab_spread_limit_ratio",
                f"{DIRECT_AB_SPREAD_LIMIT:.17g}",
            ),
            (
                "direct_ab_max_spread_ratio",
                f"{max_direct_spread:.17g}",
            ),
            (
                "direct_ab_max_relative_drift",
                f"{max_direct_spread - 1.0:.17g}",
            ),
            (
                "direct_case_over_native_min_ratio",
                f"{DIRECT_CASE_NATIVE_RATIO_MIN:.17g}",
            ),
            (
                "direct_class_over_native_geomean_min_ratio",
                f"{DIRECT_CLASS_NATIVE_GEOMEAN_MIN:.17g}",
            ),
            *[
                (
                    f"direct_{class_name}_over_native_geomean_ratio",
                    f"{class_geomeans[class_name]:.17g}",
                )
                for class_name in PERFORMANCE_CLASSES
            ],
            *gate_values,
        ],
        [
            ("build_input_snapshot", args.build_input_snapshot),
            ("preheat_manifest", args.preheat_manifest),
            ("preheat_evidence", Path(preheat["paths"]["evidence_csv"])),
            (
                "preheat_runtime_grid_config",
                Path(preheat["paths"]["runtime_grid_config"]),
            ),
            ("direct_a_manifest", args.manifest_a),
            ("direct_a_evidence", Path(a["paths"]["evidence_csv"])),
            (
                "direct_a_runtime_grid_config",
                Path(a["paths"]["runtime_grid_config"]),
            ),
            ("native_manifest", args.manifest_native),
            ("native_evidence", Path(native["paths"]["evidence_csv"])),
            (
                "native_runtime_grid_config",
                Path(native["paths"]["runtime_grid_config"]),
            ),
            ("direct_b_manifest", args.manifest_b),
            ("direct_b_evidence", Path(b["paths"]["evidence_csv"])),
            (
                "direct_b_runtime_grid_config",
                Path(b["paths"]["runtime_grid_config"]),
            ),
            ("bracket_summary", args.output_summary),
            ("formal_metrics", args.output_metrics),
            (
                "formal_binary_bundle_manifest",
                Path(a["paths"]["formal_binary_bundle_manifest"]),
            ),
            *gate_files,
        ],
    )
    write_tsv_manifest(output, rows)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def add_csv_validation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--execution-kind", choices=("accuracy", "perf"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--shapes", type=Path, required=True)
    parser.add_argument("--tier", choices=tuple(TIER_RANK), required=True)
    parser.add_argument("--variant-mode", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--expected-softmax", type=positive_int, required=True)
    parser.add_argument("--expected-rmsnorm", type=positive_int, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--math-mode", required=True)
    parser.add_argument("--warmup", type=positive_int, default=1)
    parser.add_argument("--samples", type=positive_int, default=1)
    parser.add_argument("--inner-repeats", type=positive_int, default=1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    contract_parser = subparsers.add_parser(
        "run-contract",
        help="validate the immutable protocol for one formal phase",
    )
    contract_parser.add_argument("--formal-phase", required=True)
    contract_parser.add_argument(
        "--execution-kind",
        choices=("accuracy", "perf"),
        required=True,
    )
    contract_parser.add_argument("--shapes", type=Path, required=True)
    contract_parser.add_argument(
        "--tier",
        choices=tuple(TIER_RANK),
        required=True,
    )
    contract_parser.add_argument(
        "--variant-mode",
        choices=tuple(VARIANTS),
        required=True,
    )
    contract_parser.add_argument("--math-mode", required=True)
    contract_parser.add_argument(
        "--expected-softmax",
        type=positive_int,
        required=True,
    )
    contract_parser.add_argument(
        "--expected-rmsnorm",
        type=positive_int,
        required=True,
    )
    contract_parser.add_argument("--warmup", type=positive_int, required=True)
    contract_parser.add_argument("--samples", type=positive_int, required=True)
    contract_parser.add_argument(
        "--inner-repeats",
        type=positive_int,
        required=True,
    )
    contract_parser.add_argument(
        "--check-max-elements",
        type=positive_int,
        required=True,
    )
    contract_parser.add_argument("--device-util-max", required=True)
    contract_parser.add_argument("--device-hbm-bw-max", required=True)

    device_snapshot_parser = subparsers.add_parser(
        "device-snapshot",
        help="validate an idle formal device snapshot before publication",
    )
    device_snapshot_parser.add_argument(
        "--snapshot",
        type=Path,
        required=True,
    )
    device_snapshot_parser.add_argument("--device", type=int, required=True)
    device_snapshot_parser.add_argument("--run-id", required=True)
    device_snapshot_parser.add_argument("--hostname", required=True)
    device_snapshot_parser.add_argument(
        "--phase",
        choices=("pre", "post"),
        action="append",
    )
    device_snapshot_parser.add_argument("--run-start-utc")
    device_snapshot_parser.add_argument("--run-end-utc")

    csv_parser = subparsers.add_parser("csv", help="validate and snapshot one run")
    add_csv_validation_arguments(csv_parser)
    csv_parser.add_argument("--snapshot", type=Path)

    manifest_parser = subparsers.add_parser(
        "manifest", help="validate one completed formal manifest"
    )
    add_csv_validation_arguments(manifest_parser)
    manifest_parser.add_argument("--manifest", type=Path, required=True)
    manifest_parser.add_argument("--final-manifest-path", type=Path)
    manifest_parser.add_argument("--evidence", type=Path, required=True)
    manifest_parser.add_argument("--formal-set-id", required=True)
    manifest_parser.add_argument("--formal-phase", required=True)
    manifest_parser.add_argument("--ascify-binary-sha256", required=True)

    preflight_parser = subparsers.add_parser(
        "preflight", help="reject duplicate run IDs and evidence paths"
    )
    preflight_parser.add_argument("--accuracy-csv", type=Path, required=True)
    preflight_parser.add_argument("--perf-csv", type=Path, required=True)
    preflight_parser.add_argument("--manifest-dir", type=Path, required=True)
    preflight_parser.add_argument("--set-id")
    preflight_parser.add_argument("--prepared", action="store_true")
    preflight_parser.add_argument("--run-id", action="append", required=True)

    conversion_parser = subparsers.add_parser(
        "conversion", help="validate portable 910C conversion evidence"
    )
    conversion_parser.add_argument("--manifest", type=Path, required=True)
    conversion_parser.add_argument("--ascify-binary-sha256", required=True)
    conversion_parser.add_argument("--recipe-source", type=Path, required=True)
    conversion_parser.add_argument("--recipe-header", type=Path, required=True)
    conversion_parser.add_argument("--staged-softmax", type=Path, required=True)
    conversion_parser.add_argument("--staged-layer-norm", type=Path, required=True)
    conversion_parser.add_argument("--staged-rmsnorm", type=Path, required=True)
    conversion_parser.add_argument(
        "--staged-rmsnorm-adapter", type=Path, required=True
    )
    conversion_parser.add_argument("--emit-env", action="store_true")

    runtime_config_parser = subparsers.add_parser(
        "runtime-config", help="validate one measured runtime grid configuration"
    )
    runtime_config_parser.add_argument("--config", type=Path, required=True)
    runtime_config_parser.add_argument("--run-id", required=True)
    runtime_config_parser.add_argument(
        "--variant-mode", choices=sorted(VARIANTS), required=True
    )
    runtime_config_parser.add_argument("--device", type=int, required=True)

    source_config_parser = subparsers.add_parser(
        "source-config",
        help="prove direct/native grid-cap call contracts from frozen sources",
    )
    source_config_parser.add_argument(
        "--generated-softmax", type=Path, required=True
    )
    source_config_parser.add_argument(
        "--generated-rmsnorm", type=Path, required=True
    )
    source_config_parser.add_argument("--target-header", type=Path, required=True)
    source_config_parser.add_argument(
        "--target-softmax-impl", type=Path, required=True
    )
    source_config_parser.add_argument(
        "--target-rmsnorm-impl", type=Path, required=True
    )
    source_config_parser.add_argument("--softmax-bench", type=Path, required=True)
    source_config_parser.add_argument("--softmax-native", type=Path, required=True)
    source_config_parser.add_argument("--rmsnorm-native", type=Path, required=True)

    freeze_parser = subparsers.add_parser(
        "freeze", help="create or verify an immutable build-input hash snapshot"
    )
    freeze_parser.add_argument("--output", type=Path)
    freeze_parser.add_argument(
        "--artifact", type=parse_named_path, action="append", default=[]
    )
    freeze_parser.add_argument("--check", type=Path)
    freeze_parser.add_argument("--formal-contract", action="store_true")

    bundle_parser = subparsers.add_parser(
        "binary-bundle",
        help="atomically publish the exact side-by-side formal binaries",
    )
    bundle_parser.add_argument("--formal-set-id", required=True)
    bundle_parser.add_argument("--output-dir", type=Path, required=True)
    bundle_parser.add_argument(
        "--build-input-snapshot", type=Path, required=True
    )
    bundle_parser.add_argument(
        "--artifact", type=parse_named_path, action="append", default=[]
    )
    bundle_check_parser = subparsers.add_parser(
        "binary-bundle-check",
        help="verify an immutable side-by-side formal binary bundle",
    )
    bundle_check_parser.add_argument("--manifest", type=Path, required=True)
    bundle_check_parser.add_argument("--formal-set-id", required=True)
    bundle_check_parser.add_argument(
        "--build-input-snapshot", type=Path, required=True
    )

    paths_parser = subparsers.add_parser(
        "paths", help="validate canonical isolation of formal result paths"
    )
    paths_parser.add_argument("--work-root", type=Path, required=True)
    paths_parser.add_argument("--result-dir", type=Path, required=True)
    paths_parser.add_argument("--accuracy-csv", type=Path, required=True)
    paths_parser.add_argument("--perf-csv", type=Path, required=True)
    paths_parser.add_argument("--preheat-csv", type=Path, required=True)
    paths_parser.add_argument("--manifest-dir", type=Path, required=True)
    paths_parser.add_argument("--set-manifest", type=Path, required=True)
    paths_parser.add_argument("--bracket-summary", type=Path, required=True)
    paths_parser.add_argument("--formal-metrics", type=Path, required=True)
    paths_parser.add_argument("--result-lock", type=Path, required=True)
    paths_parser.add_argument("--build-input-snapshot", type=Path, required=True)
    paths_parser.add_argument("--binary-bundle-dir", type=Path, required=True)

    lock_parser = subparsers.add_parser(
        "lock", help="verify an inherited lock fd points at the declared file"
    )
    lock_parser.add_argument("--fd", type=positive_int, required=True)
    lock_parser.add_argument("--declared", type=Path, required=True)
    lock_parser.add_argument("--expected", type=Path, required=True)

    bracket_parser = subparsers.add_parser(
        "bracket", help="validate A/native/B comparability and write set manifest"
    )
    bracket_parser.add_argument("--formal-set-id", required=True)
    bracket_parser.add_argument("--manifest-a", type=Path, required=True)
    bracket_parser.add_argument("--manifest-native", type=Path, required=True)
    bracket_parser.add_argument("--manifest-b", type=Path, required=True)
    bracket_parser.add_argument("--preheat-manifest", type=Path, required=True)
    bracket_parser.add_argument(
        "--gate-manifest", type=Path, action="append", default=[]
    )
    bracket_parser.add_argument(
        "--build-input-snapshot", type=Path, required=True
    )
    bracket_parser.add_argument("--metrics-tool", type=Path, required=True)
    bracket_parser.add_argument("--output-summary", type=Path, required=True)
    bracket_parser.add_argument("--output-metrics", type=Path, required=True)
    bracket_parser.add_argument("--output-manifest", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "run-contract":
        validate_formal_run_contract(
            Path("run-contract"),
            phase=args.formal_phase,
            execution_kind=args.execution_kind,
            shapes=args.shapes,
            tier=args.tier,
            variant_mode=args.variant_mode,
            math_mode=args.math_mode,
            expected_softmax=args.expected_softmax,
            expected_rmsnorm=args.expected_rmsnorm,
            warmup=args.warmup,
            samples=args.samples,
            inner_repeats=args.inner_repeats,
            check_max_elements=args.check_max_elements,
            device_util_max=args.device_util_max,
            device_hbm_bw_max=args.device_hbm_bw_max,
        )
        print(f"[ok] immutable formal protocol: {args.formal_phase}")
    elif args.command == "device-snapshot":
        phases = tuple(args.phase or ("pre", "post"))
        timestamps = validate_idle_device_snapshot(
            args.snapshot,
            args.device,
            phases,
            expected_run_id=args.run_id,
            expected_hostname=args.hostname,
        )
        if (args.run_start_utc is None) != (args.run_end_utc is None):
            fail(
                "device-snapshot requires both --run-start-utc and "
                "--run-end-utc"
            )
        if args.run_start_utc is not None:
            require_device_snapshot_interval(
                args.snapshot,
                timestamps,
                iso8601(args.run_start_utc, "device snapshot run start"),
                iso8601(args.run_end_utc, "device snapshot run end"),
            )
        print(
            f"[ok] idle device snapshot: "
            f"device={args.device} phases={','.join(phases)}"
        )
    elif args.command == "csv":
        fieldnames, rows = validate_csv_rows(args)
        if args.snapshot:
            write_snapshot(args.snapshot, fieldnames, rows)
        print(
            f"[ok] {args.run_id}: exact "
            f"{args.expected_softmax}+{args.expected_rmsnorm} rows validated"
        )
    elif args.command == "manifest":
        validate_csv_rows(args)
        validate_manifest(args)
        print(f"[ok] {args.run_id}: formal manifest and artifact hashes validated")
    elif args.command == "preflight":
        preflight(args)
        print(f"[ok] {len(args.run_id)} formal run IDs are unused")
    elif args.command == "conversion":
        summary = validate_conversion_evidence(args)
        if args.emit_env:
            for name in sorted(summary):
                print(f"{name}\t{summary[name]}")
        else:
            print(
                f"[ok] portable conversion evidence validated: "
                f"{args.manifest}"
            )
    elif args.command == "runtime-config":
        values = validate_runtime_grid_config(
            args.config, args.run_id, args.variant_mode, args.device
        )
        print(
            f"[ok] runtime grid: AIV={values['vector_core_count']} "
            f"resolved={values['resolved_grid_cap']}"
        )
    elif args.command == "source-config":
        validate_runtime_source_contracts(args)
        print("[ok] direct/native grid-cap source contracts validated")
    elif args.command == "freeze":
        if (args.output is None) == (args.check is None):
            fail("freeze requires exactly one of --output or --check")
        if args.output is not None:
            if not args.artifact:
                fail("freeze --output requires at least one --artifact")
            write_freeze_snapshot(args)
            print(f"[ok] build inputs frozen: {args.output}")
        else:
            if args.artifact:
                fail("freeze --check does not accept --artifact")
            check_freeze_snapshot(args.check, args.formal_contract)
            print(f"[ok] frozen build inputs are unchanged: {args.check}")
    elif args.command == "binary-bundle":
        write_binary_bundle(args)
        print(f"[ok] immutable binary bundle published: {args.output_dir}")
    elif args.command == "binary-bundle-check":
        check_binary_bundle(
            args.manifest,
            args.formal_set_id,
            args.build_input_snapshot,
        )
        print(f"[ok] immutable binary bundle verified: {args.manifest}")
    elif args.command == "paths":
        validate_formal_paths(args)
        print("[ok] formal result paths are canonical and isolated")
    elif args.command == "lock":
        validate_lock_fd(args)
        print(f"[ok] inherited fd {args.fd} lock identity validated")
    elif args.command == "bracket":
        compare_bracket(args)
        print(f"[ok] A/native/B bracket validated: {args.output_manifest}")
    else:
        raise AssertionError(f"unhandled command: {args.command}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValidationError, OSError, csv.Error, ValueError) as error:
        print(f"[formal-validation] {error}", file=sys.stderr)
        sys.exit(2)
