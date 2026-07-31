#!/usr/bin/env python3
"""Small positive/negative fixtures for the formal-run evidence validator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("validate_formal_run.py")
SPEC = importlib.util.spec_from_file_location("formal_validator", str(MODULE_PATH))
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def topology_fixture(unit: str) -> str:
    pieces = [f"fixture generated {unit}\n"]
    for name, count in VALIDATOR.EXPECTED_CONVERSION_TOPOLOGY[unit].items():
        if name in ("softmax_direct_wrappers", "rmsnorm_direct_wrappers"):
            continue
        pieces.extend(
            f"{VALIDATOR.RECIPE_TOPOLOGY_TOKENS[name]}\n"
            for _ in range(count)
        )
    for name in VALIDATOR.DIRECT_RECIPE_WRAPPERS[unit]:
        if unit == "softmax":
            smem = "int smem, " if "BlockSMem" in name else ""
            pieces.append(
                "template<typename LOAD, typename STORE, typename ComputeType, "
                "int pack_size, Algorithm algorithm>\n"
                f"int {name}(aclrtStream stream, LOAD load, STORE store, "
                f"{smem}const int64_t rows, const int64_t cols) {{\n"
                "  if (rows > 0 && cols > 0) {\n"
                "    if constexpr (algorithm == "
                "::fixture::Algorithm::kSoftmax) {\n"
                "      const auto ascify_dav_c310_recipe_result = "
                "::ascify::target::dav_c310::TrySoftmax("
                "stream, load, store, rows, cols, "
                "static_cast<ComputeType*>(nullptr));\n"
                "      if (ascify_dav_c310_recipe_result.handled) { "
                "return ascify_dav_c310_recipe_result.status; }\n"
                "    }\n"
                "  }\n"
                "  Kernel<<<1, 1>>>(load, store);\n"
                "  return 0;\n"
                "}\n"
            )
        else:
            smem = "int smem_size, " if "BlockSMem" in name else ""
            pieces.append(
                "template<typename LOAD, typename STORE, typename ComputeType, "
                "int pack_size>\n"
                f"int {name}(aclrtStream stream, LOAD load, STORE store, "
                f"{smem}const int64_t nrow, const int64_t ncol, "
                "const double eps, ComputeType* inv_rms) {\n"
                "  int grid_dim_x = 0;\n"
                "  {\n"
                "    aclError err = layer_norm::GetNumBlocks("
                "Kernel, 1, 0, nrow, 32, &grid_dim_x);\n"
                "    if (err != ACL_SUCCESS) { return err; }\n"
                "  }\n"
                "  if (nrow > 0 && ncol > 0) {\n"
                "    const auto ascify_dav_c310_recipe_result = "
                "::ascify::target::dav_c310::TryRmsNorm("
                "stream, load, store, nrow, ncol, eps, inv_rms, "
                "static_cast<ComputeType*>(nullptr));\n"
                "    if (ascify_dav_c310_recipe_result.handled) { "
                "return ascify_dav_c310_recipe_result.status; }\n"
                "  }\n"
                "  Kernel<<<grid_dim_x, 1>>>(load, store);\n"
                "  return 0;\n"
                "}\n"
            )
    for name, count in (
        VALIDATOR.FORBIDDEN_RECIPE_DISPATCHER_DEFINITION_COUNTS[unit].items()
    ):
        pieces.extend(
            f"int {name}(int overload_{index}) {{ return overload_{index}; }}\n"
            for index in range(count)
        )
    return "".join(pieces)


def token_only_topology_fixture(unit: str) -> str:
    pieces = [f"token-only generated {unit}\n"]
    for name, count in VALIDATOR.EXPECTED_CONVERSION_TOPOLOGY[unit].items():
        pieces.extend(
            f"{VALIDATOR.RECIPE_TOPOLOGY_TOKENS[name]}\n"
            for _ in range(count)
        )
    return "".join(pieces)


def mutate_first_wrapper_body(
    text: str, wrapper_name: str, mutation
) -> str:
    masked = VALIDATOR.mask_cpp_non_code(text)
    match = re.search(rf"\b{re.escape(wrapper_name)}\s*\(", masked)
    if match is None:
        raise AssertionError(f"missing fixture wrapper {wrapper_name}")
    open_paren = masked.find("(", match.start(), match.end())
    close_paren = VALIDATOR.matching_delimiter(
        masked, open_paren, "(", ")"
    )
    if close_paren is None:
        raise AssertionError(f"unclosed fixture parameters for {wrapper_name}")
    body_start = VALIDATOR.skip_space(masked, close_paren + 1)
    body_end = VALIDATOR.matching_delimiter(
        masked, body_start, "{", "}"
    )
    if body_end is None:
        raise AssertionError(f"unclosed fixture body for {wrapper_name}")
    body = text[body_start + 1:body_end]
    return (
        text[:body_start + 1]
        + mutation(body)
        + text[body_end:]
    )


def rms_recipe_span(body: str) -> tuple[int, int]:
    try_offset = body.index(
        VALIDATOR.RECIPE_TOPOLOGY_TOKENS["rmsnorm_direct_wrappers"]
    )
    candidates = list(re.finditer(r"\bif\s*\(", body[:try_offset]))
    if not candidates:
        raise AssertionError("missing RMS fixture recipe guard")
    start = candidates[-1].start()
    parsed = VALIDATOR.parse_if_block(
        body, start, require_constexpr=False
    )
    if parsed is None or "TryRmsNorm" not in parsed[1]:
        raise AssertionError("malformed RMS fixture recipe guard")
    return start, parsed[2]


def rms_try_before_geometry(text: str) -> str:
    def mutate(body: str) -> str:
        recipe_start, recipe_end = rms_recipe_span(body)
        recipe = body[recipe_start:recipe_end]
        without_recipe = body[:recipe_start] + body[recipe_end:]
        geometry = without_recipe.index("layer_norm::GetNumBlocks(")
        geometry_scope = VALIDATOR.nearest_unmatched_open_brace(
            without_recipe, geometry
        )
        if geometry_scope is None:
            raise AssertionError("missing RMS fixture geometry scope")
        return (
            without_recipe[:geometry_scope]
            + recipe
            + "\n  "
            + without_recipe[geometry_scope:]
        )

    return mutate_first_wrapper_body(
        text, "LaunchRmsNormWarpImpl", mutate
    )


def rms_try_after_launch(text: str) -> str:
    def mutate(body: str) -> str:
        recipe_start, recipe_end = rms_recipe_span(body)
        recipe = body[recipe_start:recipe_end]
        without_recipe = body[:recipe_start] + body[recipe_end:]
        launch = without_recipe.index("<<<")
        launch_end = without_recipe.index(";", launch) + 1
        return (
            without_recipe[:launch_end]
            + "\n  "
            + recipe
            + without_recipe[launch_end:]
        )

    return mutate_first_wrapper_body(
        text, "LaunchRmsNormWarpImpl", mutate
    )


def rms_inverted_geometry_error_guard(text: str) -> str:
    return mutate_first_wrapper_body(
        text,
        "LaunchRmsNormWarpImpl",
        lambda body: body.replace(
            "err != ACL_SUCCESS", "err == ACL_SUCCESS", 1
        ),
    )


def rms_mismatched_geometry_error_return(text: str) -> str:
    return mutate_first_wrapper_body(
        text,
        "LaunchRmsNormWarpImpl",
        lambda body: body.replace(
            "return err;", "return ACL_SUCCESS;", 1
        ),
    )


def rms_duplicate_geometry_call(text: str) -> str:
    duplicate = (
        "\n    aclError duplicate_err = layer_norm::GetNumBlocks("
        "Kernel, 1, 0, nrow, 32, &grid_dim_x);"
    )

    def mutate(body: str) -> str:
        geometry = body.index("layer_norm::GetNumBlocks(")
        call_open = body.index("(", geometry)
        call_close = VALIDATOR.matching_delimiter(
            body, call_open, "(", ")"
        )
        if call_close is None:
            raise AssertionError("unclosed RMS fixture geometry call")
        statement_end = body.index(";", call_close) + 1
        return body[:statement_end] + duplicate + body[statement_end:]

    return mutate_first_wrapper_body(
        text, "LaunchRmsNormWarpImpl", mutate
    )


def rms_geometry_callee_namespace_drift(text: str) -> str:
    return mutate_first_wrapper_body(
        text,
        "LaunchRmsNormWarpImpl",
        lambda body: body.replace(
            "layer_norm::GetNumBlocks(",
            "fake_layer_norm::GetNumBlocks(",
            1,
        ),
    )


class FormalSourceContractTest(unittest.TestCase):
    def test_device_selector_handles_races_and_reserved_target_exits(
        self,
    ) -> None:
        repo_root = MODULE_PATH.parents[3]
        selector = (
            repo_root
            / "tests/softmax_rmsnorm_950/scripts/select_device.sh"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_npu_smi = fake_bin / "npu-smi"
            fake_npu_smi.write_text(
                """#!/usr/bin/env bash
set -eu
if [[ "$*" == "info -m" ]]; then
  printf 'NPU Chip Logic Name\\n'
  printf '0 0 0 Ascend950PR\\n'
  printf '1 1 0 Ascend950PR\\n'
  exit 0
fi
device_id=""
for argument in "$@"; do device_id="${argument}"; done
case "$*" in
  *"-t health"*)
    printf 'Health : OK\\n'
    ;;
  *"-t usages"*)
    printf 'NPU Utilization : 0\\n'
    printf 'Aicore Usage Rate : 0\\n'
    printf 'Aivector Usage Rate : 0\\n'
    printf 'HBM Bandwidth Usage Rate : 0\\n'
    ;;
  *"-t proc-mem"*)
    count_file="${FAKE_NPU_STATE}/proc-${device_id}.count"
    count=0
    if [[ -f "${count_file}" ]]; then read -r count < "${count_file}"; fi
    count=$((count + 1))
    printf '%s\\n' "${count}" > "${count_file}"
    if [[ "${FAKE_NPU_MODE}" == "race"
          && "${device_id}" == "0" && "${count}" -ge 2 ]]; then
      printf 'Process id : 123\\n'
    else
      printf 'No process in device.\\n'
    fi
    ;;
  *)
    exit 2
    ;;
esac
""",
                encoding="utf-8",
            )
            fake_flock = fake_bin / "flock"
            fake_flock.write_text(
                "#!/usr/bin/env bash\nexit 0\n",
                encoding="utf-8",
            )
            target = fake_bin / "capture-device"
            target.write_text(
                """#!/usr/bin/env bash
printf '%s\\n' "${ASCIFY_DEVICE:?}" >> "${CAPTURE_PATH:?}"
exit "${TARGET_EXIT:-0}"
""",
                encoding="utf-8",
            )
            for executable in (fake_npu_smi, fake_flock, target):
                executable.chmod(0o755)

            base_environment = os.environ.copy()
            base_environment["PATH"] = (
                str(fake_bin) + os.pathsep + base_environment["PATH"]
            )
            base_environment["WORK_ROOT"] = str(root / "work")

            def run_selector(
                name: str, mode: str, target_exit: int
            ) -> tuple[subprocess.CompletedProcess[str], list[str]]:
                state = root / f"state-{name}"
                state.mkdir()
                capture = root / f"capture-{name}.txt"
                environment = base_environment.copy()
                environment.update(
                    {
                        "FAKE_NPU_STATE": str(state),
                        "FAKE_NPU_MODE": mode,
                        "CAPTURE_PATH": str(capture),
                        "TARGET_EXIT": str(target_exit),
                    }
                )
                completed = subprocess.run(
                    ["bash", str(selector), str(target)],
                    cwd=str(repo_root),
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                )
                captured = (
                    capture.read_text(encoding="utf-8").splitlines()
                    if capture.exists()
                    else []
                )
                return completed, captured

            raced, raced_devices = run_selector("race", "race", 0)
            self.assertEqual(0, raced.returncode, raced.stderr)
            self.assertEqual(["1"], raced_devices)
            self.assertIn("state changed while acquiring lock", raced.stderr)

            for reserved_exit in (200, 201):
                with self.subTest(reserved_exit=reserved_exit):
                    completed, devices = run_selector(
                        f"reserved-{reserved_exit}",
                        "idle",
                        reserved_exit,
                    )
                    self.assertEqual(199, completed.returncode)
                    self.assertEqual(["0"], devices)
                    self.assertNotIn(
                        "state changed while acquiring lock",
                        completed.stderr,
                    )

    def test_device_selector_retries_lock_time_state_changes(self) -> None:
        repo_root = MODULE_PATH.parents[3]
        selector = (
            repo_root
            / "tests/softmax_rmsnorm_950/scripts/select_device.sh"
        ).read_text(encoding="utf-8")
        for contract in (
            "readonly LOCK_BUSY_EXIT=200",
            "readonly LOCKED_RECHECK_BUSY_EXIT=201",
            "readonly TARGET_RESERVED_EXIT=199",
            'exit "${LOCKED_RECHECK_BUSY_EXIT}"',
            '--busy-exit "${LOCK_BUSY_EXIT}"',
            "if (( lock_status == LOCKED_RECHECK_BUSY_EXIT )); then",
            "state changed while acquiring lock",
            "command_status == LOCK_BUSY_EXIT",
            "command_status == LOCKED_RECHECK_BUSY_EXIT",
            'command_status="${TARGET_RESERVED_EXIT}"',
        ):
            self.assertIn(contract, selector)
        self.assertNotEqual(
            re.search(
                r"^readonly LOCK_BUSY_EXIT=(\d+)$",
                selector,
                re.MULTILINE,
            ).group(1),
            re.search(
                r"^readonly LOCKED_RECHECK_BUSY_EXIT=(\d+)$",
                selector,
                re.MULTILINE,
            ).group(1),
        )

    def test_boundary_extrema_case_and_exact_counts_are_in_formal_gate(self) -> None:
        repo_root = MODULE_PATH.parents[3]
        boundary_shapes = (
            repo_root
            / "tests/softmax_rmsnorm_950/shapes/correctness.csv"
        )
        tune_shapes = (
            repo_root
            / "tests/softmax_rmsnorm_950/shapes/unified_tune.csv"
        )
        with boundary_shapes.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        selected = [
            row
            for row in rows
            if row["tier"] == "smoke" and row["run_check"] == "1"
        ]
        self.assertEqual(
            {"softmax": 42, "rms_norm": 18},
            {
                op: sum(row["op"] == op for row in selected)
                for op in ("softmax", "rms_norm")
            },
        )
        extrema = [
            row for row in selected if row["input_pattern"] == "fp16_extrema"
        ]
        self.assertEqual(
            {("1", "8"), ("1", "8192"), ("1120", "4104")},
            {(row["rows"], row["cols"]) for row in extrema},
        )
        with tune_shapes.open(newline="", encoding="utf-8") as stream:
            tune_rows = [
                row
                for row in csv.DictReader(stream)
                if row["tier"] == "tune" and row["run_bench"] == "1"
            ]
        self.assertEqual(
            {"softmax": 5, "rms_plain": 5, "rms_affine": 5},
            {
                "softmax": sum(row["op"] == "softmax" for row in tune_rows),
                "rms_plain": sum(
                    row["op"] == "rms_norm" and row["affine"] == "0"
                    for row in tune_rows
                ),
                "rms_affine": sum(
                    row["op"] == "rms_norm" and row["affine"] == "1"
                    for row in tune_rows
                ),
            },
        )
        self.assertEqual(
            VALIDATOR.CANONICAL_FORMAL_SHAPES["boundary_shapes"][1],
            digest(boundary_shapes),
        )
        self.assertEqual(
            VALIDATOR.CANONICAL_FORMAL_SHAPES["tune_shapes"][1],
            digest(tune_shapes),
        )
        formal_runner = (
            repo_root
            / "tests/softmax_rmsnorm_950/scripts/run_formal_recipe_v3.sh"
        ).read_text(encoding="utf-8")
        self.assertEqual(
            2,
            formal_runner.count(
                '"${BOUNDARY_SHAPES}" smoke 42 18 '
                '"${BOUNDARY_CHECK_MAX_ELEMENTS}"'
            ),
        )
        for fixed_contract in (
            'BOUNDARY_SHAPES="${TEST_ROOT}/shapes/correctness.csv"',
            'TUNE_SHAPES="${TEST_ROOT}/shapes/unified_tune.csv"',
            "BOUNDARY_CHECK_MAX_ELEMENTS=5000000",
            "TUNE_CHECK_MAX_ELEMENTS=10000000",
            "PREHEAT_WARMUP=5",
            "PREHEAT_SAMPLES=5",
            "PREHEAT_INNER_REPEATS=5",
            "FORMAL_WARMUP=20",
            "FORMAL_SAMPLES=50",
            "FORMAL_INNER_REPEATS=20",
            "FORMAL_PROCESS_SETTLE_ATTEMPTS=30",
            'FORMAL_STAGE="${ASCIFY_FORMAL_STAGE:-prepare}"',
            "measure stage requires the inherited prepare-stage result lock",
            "preflight_args+=(--prepared)",
            "immutable binaries verified; selecting a fresh idle device",
            "exec env -u DEVICE -u ASCIFY_DEVICE -u ASCIFY_DEVICE_LOCK",
            'wait_for_device_process_cleanup "${phase}"',
        ):
            self.assertIn(fixed_contract, formal_runner)
        for override in (
            'BOUNDARY_SHAPES="${BOUNDARY_SHAPES:-',
            'TUNE_SHAPES="${TUNE_SHAPES:-',
            'FORMAL_WARMUP="${FORMAL_WARMUP:-',
            'FORMAL_SAMPLES="${FORMAL_SAMPLES:-',
            'FORMAL_INNER_REPEATS="${FORMAL_INNER_REPEATS:-',
        ):
            self.assertNotIn(override, formal_runner)
        validator_source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn('"--phase-a"', validator_source)
        self.assertNotIn('"--phase-native"', validator_source)
        self.assertNotIn('"--phase-b"', validator_source)
        self.assertIn(
            '("direct_perf_a", "native_perf", "direct_perf_b")',
            validator_source,
        )
        smoke_runner = (
            repo_root
            / "tests/softmax_rmsnorm_950/scripts/run_smoke.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            '"${SCRIPT_DIR}/validate_formal_run.py" run-contract',
            smoke_runner,
        )
        self.assertNotIn("capture_idle_post_state", smoke_runner)
        self.assertIn(
            'capture_device_state post >> "${DEVICE_SNAPSHOT_OUTPUT}"',
            smoke_runner,
        )
        finalize_body = smoke_runner[
            smoke_runner.index("finalize_formal_manifest() {") :
            smoke_runner.index("# Runtime paths are scoped")
        ]
        self.assertLess(
            finalize_body.index(
                'capture_device_state post >> "${DEVICE_SNAPSHOT_OUTPUT}"'
            ),
            finalize_body.index('"${VALIDATOR_SCRIPT}" runtime-config'),
        )
        self.assertLess(
            finalize_body.index(
                'capture_device_state post >> "${DEVICE_SNAPSHOT_OUTPUT}"'
            ),
            finalize_body.index('"${VALIDATOR_SCRIPT}" csv'),
        )
        self.assertLess(
            smoke_runner.index(
                '"${VALIDATOR_SCRIPT}" manifest'
            ),
            smoke_runner.index('mv -- "${MANIFEST_OUTPUT}" "${MANIFEST}"'),
        )
        self.assertIn(
            '--build-input-snapshot "${FORMAL_BUILD_INPUT_SNAPSHOT}"',
            smoke_runner,
        )

    def test_formal_runner_builds_only_before_post_build_device_selection(
        self,
    ) -> None:
        repo_root = MODULE_PATH.parents[3]
        formal_runner = (
            repo_root
            / "tests/softmax_rmsnorm_950/scripts/run_formal_recipe_v3.sh"
        ).read_text(encoding="utf-8")
        prepare_block = formal_runner[
            formal_runner.index(
                'if [[ "${FORMAL_STAGE}" == "prepare" ]]; then\n'
                "  build_variant converted converted"
            ) :
            formal_runner.index(
                "\nfi\n\n"
                '"${VALIDATOR}" freeze \\\n'
                "  --formal-contract",
                formal_runner.index("  build_variant converted converted"),
            )
        ]
        self.assertEqual(1, prepare_block.count("build_variant converted converted"))
        self.assertEqual(
            1,
            prepare_block.count("build_variant native-half2-hybrid native"),
        )
        self.assertIn('"${VALIDATOR}" binary-bundle \\', prepare_block)
        self.assertIn(
            '--build-input-snapshot "${BUILD_INPUT_SNAPSHOT}"',
            prepare_block,
        )
        measure_tail = formal_runner[
            formal_runner.index(
                "\nfi\n\n"
                '"${VALIDATOR}" freeze \\\n'
                "  --formal-contract",
                formal_runner.index("  build_variant converted converted"),
            )
            + len("\nfi\n") :
        ]
        self.assertNotIn("\nbuild_variant converted converted\n", measure_tail)
        self.assertNotIn(
            "\nbuild_variant native-half2-hybrid native\n",
            measure_tail,
        )
        self.assertLess(
            measure_tail.index('"${VALIDATOR}" binary-bundle-check \\'),
            measure_tail.index("# Boundary coverage:"),
        )
        self.assertIn(
            '--build-input-snapshot "${BUILD_INPUT_SNAPSHOT}"',
            measure_tail,
        )
        self.assertLess(
            measure_tail.index(
                'ASCIFY_FORMAL_STAGE=measure UTIL_MAX=0 HBM_BW_MAX=0'
            ),
            measure_tail.index("# Boundary coverage:"),
        )
        self.assertIn(
            'if [[ "${FORMAL_STAGE}" != "measure" ]]; then',
            measure_tail,
        )

    def test_formal_runner_rejects_an_unknown_stage_before_host_checks(
        self,
    ) -> None:
        repo_root = MODULE_PATH.parents[3]
        runner = (
            repo_root
            / "tests/softmax_rmsnorm_950/scripts/run_formal_recipe_v3.sh"
        )
        environment = os.environ.copy()
        environment["ASCIFY_FORMAL_STAGE"] = "bogus"
        completed = subprocess.run(
            ["bash", str(runner)],
            cwd=str(repo_root),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        self.assertEqual(2, completed.returncode)
        self.assertIn(
            "ASCIFY_FORMAL_STAGE must be prepare, resume, or measure",
            completed.stderr,
        )

    def test_direct_and_native_softmax_use_the_same_auto_grid_cap(self) -> None:
        repo_root = MODULE_PATH.parents[3]
        for relative in (
            "tests/softmax_rmsnorm_950/softmax_check.cce",
            "tests/softmax_rmsnorm_950/softmax_bench.cce",
        ):
            source = (repo_root / relative).read_text(encoding="utf-8")
            self.assertIn("static_cast<int>(cols), 0, 0);", source)
            self.assertNotIn("static_cast<int>(cols), 0, 8192);", source)
        smoke_runner = (
            repo_root / "tests/softmax_rmsnorm_950/scripts/run_smoke.sh"
        ).read_text(encoding="utf-8")
        self.assertEqual(
            1,
            smoke_runner.count(
                'manifest_value softmax_grid_cap "auto-aiv-x32"'
            ),
        )
        self.assertNotIn('manifest_value softmax_grid_cap "8192"', smoke_runner)


class RuntimeGridAndMetricEvidenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_runtime_config(
        self, path: Path, run_id: str, variant_mode: str, vector_cores: int = 56
    ) -> None:
        expected = {
            "direct": (
                (
                    "softmax",
                    "converted_dispatch",
                    "try_softmax_default_grid",
                    "adaptive",
                ),
                (
                    "rms_norm",
                    "converted_plain+converted_affine",
                    "try_rmsnorm_default_grid",
                    "plain512-affine256-auto",
                ),
            ),
            "native": (
                (
                    "softmax",
                    "native_half2_hybrid",
                    "softmax_fp16_explicit_grid0",
                    "adaptive",
                ),
                (
                    "rms_norm",
                    "native_plain+native_affine",
                    "rmsnorm_fp16_explicit_grid0",
                    "plain512-affine256-auto",
                ),
            ),
        }[variant_mode]
        with path.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=list(VALIDATOR.RUNTIME_CONFIG_FIELDS),
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            for op, variant, entry, block_policy in expected:
                writer.writerow(
                    {
                        "runtime_config_schema_version": "1",
                        "run_id": run_id,
                        "op": op,
                        "variant": variant,
                        "device_id": "6",
                        "target_entry": entry,
                        "requested_grid_cap": "0",
                        "vector_core_count": str(vector_cores),
                        "resolved_grid_cap": str(vector_cores * 32),
                        "grid_policy": "auto-aiv-x32",
                        "block_threads_policy": block_policy,
                    }
                )

    def test_runtime_grid_config_is_measured_and_exact(self) -> None:
        path = self.root / "runtime.tsv"
        self.write_runtime_config(path, "run_a", "direct")
        values = VALIDATOR.validate_runtime_grid_config(
            path, "run_a", "direct", 6
        )
        self.assertEqual(
            {"vector_core_count": 56, "resolved_grid_cap": 1792}, values
        )
        rows = path.read_text(encoding="utf-8").replace("\t1792\t", "\t8192\t")
        path.write_text(rows, encoding="utf-8")
        with self.assertRaises(VALIDATOR.ValidationError):
            VALIDATOR.validate_runtime_grid_config(path, "run_a", "direct", 6)

    def test_runtime_grid_rejects_out_of_target_domain_and_extra_row(self) -> None:
        path = self.root / "runtime.tsv"
        self.write_runtime_config(path, "run_a", "native", vector_cores=1025)
        with self.assertRaises(VALIDATOR.ValidationError):
            VALIDATOR.validate_runtime_grid_config(path, "run_a", "native", 6)
        path.unlink()
        self.write_runtime_config(path, "run_a", "native")
        text = path.read_text(encoding="utf-8")
        path.write_text(text + text.splitlines()[-1] + "\n", encoding="utf-8")
        with self.assertRaises(VALIDATOR.ValidationError):
            VALIDATOR.validate_runtime_grid_config(path, "run_a", "native", 6)

    def source_contract_args(self) -> tuple[argparse.Namespace, Path, Path]:
        repo_root = MODULE_PATH.parents[3]
        softmax = self.root / "softmax.cuh"
        rmsnorm = self.root / "rms_norm.cuh"
        softmax.write_text(topology_fixture("softmax"), encoding="utf-8")
        rmsnorm.write_text(topology_fixture("rmsnorm"), encoding="utf-8")
        native_softmax = self.root / "softmax_native_kernel.cce"
        original_native = (
            repo_root
            / "tests/softmax_rmsnorm_950/kernels/softmax_native_kernel.cce"
        )
        native_softmax.write_text(
            original_native.read_text(encoding="utf-8"), encoding="utf-8"
        )
        native_rmsnorm = self.root / "rmsnorm_native_kernel.cce"
        native_rmsnorm.write_text(
            (
                repo_root
                / "tests/softmax_rmsnorm_950/kernels/rmsnorm_native_kernel.cce"
            ).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        args = argparse.Namespace(
            generated_softmax=softmax,
            generated_rmsnorm=rmsnorm,
            target_header=repo_root
            / "include/ascify/target/dav_c310/rowwise_norm_recipes.hpp",
            target_softmax_impl=repo_root
            / "include/ascify/target/dav_c310/detail/softmax_fp16_impl.hpp",
            target_rmsnorm_impl=repo_root
            / "include/ascify/target/dav_c310/detail/rmsnorm_fp16_impl.hpp",
            softmax_bench=repo_root
            / "tests/softmax_rmsnorm_950/softmax_bench.cce",
            softmax_native=native_softmax,
            rmsnorm_native=native_rmsnorm,
        )
        return args, native_softmax, native_rmsnorm

    def test_source_grid_contract_rejects_softmax_native_nonzero_cap(self) -> None:
        args, native_softmax, _ = self.source_contract_args()
        VALIDATOR.validate_runtime_source_contracts(args)
        native_softmax.write_text(
            native_softmax.read_text(encoding="utf-8").replace(
                "columns, block_threads, grid_cap",
                "columns, block_threads, 8192",
                1,
            ),
            encoding="utf-8",
        )
        with self.assertRaises(VALIDATOR.ValidationError):
            VALIDATOR.validate_runtime_source_contracts(args)

    def test_source_grid_contract_rejects_rmsnorm_native_nonzero_cap(self) -> None:
        args, _, native_rmsnorm = self.source_contract_args()
        VALIDATOR.validate_runtime_source_contracts(args)
        native_rmsnorm.write_text(
            native_rmsnorm.read_text(encoding="utf-8").replace(
                "epsilon, 0, 0);",
                "epsilon, 0, 8192);",
                1,
            ),
            encoding="utf-8",
        )
        with self.assertRaises(VALIDATOR.ValidationError):
            VALIDATOR.validate_runtime_source_contracts(args)

    def test_source_grid_contract_rejects_inactive_expected_fragment(self) -> None:
        expected_fragment = (
            "\nint dead_contract() {\n"
            "  return softmax_native_dispatch(\n"
            "      stream, input, output, rows, static_cast<int>(cols), 0, 0);\n"
            "}\n"
        )
        for directive in ("#if 0", "#ifdef ASCIFY_VALIDATOR_DEAD_BRANCH"):
            with self.subTest(directive=directive):
                args, _, _ = self.source_contract_args()
                source = Path(args.softmax_bench).read_text(encoding="utf-8")
                source = source.replace(
                    "static_cast<int>(cols), 0, 0);",
                    "static_cast<int>(cols), 0, 8192);",
                    1,
                )
                replacement = self.root / (
                    "dead_if0.cce" if directive == "#if 0" else "dead_ifdef.cce"
                )
                replacement.write_text(
                    source
                    + f"\n{directive}\n"
                    + expected_fragment
                    + "#endif\n",
                    encoding="utf-8",
                )
                args.softmax_bench = replacement
                with self.assertRaises(VALIDATOR.ValidationError):
                    VALIDATOR.validate_runtime_source_contracts(args)

    def test_formal_metric_artifact_uses_frozen_model_and_refuses_overwrite(
        self,
    ) -> None:
        repo_root = MODULE_PATH.parents[3]
        model_path = self.root / "derive_work_metrics.py"
        model_path.write_bytes(
            (
                repo_root
                / "tests/softmax_rmsnorm_950/tools/derive_work_metrics.py"
            ).read_bytes()
        )
        model = VALIDATOR.load_metric_model(model_path)
        self.assertFalse((self.root / "__pycache__").exists())
        shape = {
            "rows": "2",
            "cols": "3",
            "affine": "0",
        }
        evidence = {
            "run_id": "run_a",
            "op": "softmax",
            "variant": "converted_dispatch",
            "case_id": "sm",
            "lat_ms_median": "0.002",
            "gbps": "1.0",
        }
        rows = VALIDATOR.build_formal_metric_rows(
            "set",
            ("direct_a",),
            ((evidence,),),
            {("softmax", "sm"): shape},
            model,
        )
        self.assertEqual("16", rows[0]["arith_ops"])
        self.assertEqual("6", rows[0]["special_ops"])
        self.assertEqual("4", rows[0]["compare_ops"])
        self.assertEqual(0.000008, float(rows[0]["arith_tflops"]))
        output = self.root / "metrics.csv"
        VALIDATOR.write_formal_metrics(output, rows)
        with self.assertRaises(VALIDATOR.ValidationError):
            VALIDATOR.write_formal_metrics(output, rows)


class ConversionEvidenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.evidence = self.root / "evidence"
        self.evidence.mkdir()
        specs = [
            ("ascify_binary", "converter", "converter/ascify-clang"),
            ("recipe_source", "source", "recipe/source.cpp"),
            ("recipe_header", "source", "recipe/source.h"),
            ("input_softmax", "input", "inputs/softmax.cuh"),
            ("input_layer_norm", "input", "inputs/layer_norm.cuh"),
            ("input_rmsnorm", "input", "inputs/rms_norm.cuh"),
            ("input_rmsnorm_adapter", "input", "inputs/adapter.cuh"),
            ("generated_softmax_header", "output", "outputs/softmax.cuh"),
            ("generated_layer_norm_header", "output", "outputs/layer_norm.cuh"),
            ("generated_rmsnorm_header", "output", "outputs/rms_norm.cuh"),
            ("generated_rmsnorm_adapter", "output", "outputs/adapter.cuh"),
        ]
        units = [
            ("softmax", "input_softmax", "generated_softmax_header"),
            ("layer_norm", "input_layer_norm", "generated_layer_norm_header"),
            ("rmsnorm", "input_rmsnorm", "generated_rmsnorm_header"),
            (
                "rmsnorm_adapter",
                "input_rmsnorm_adapter",
                "generated_rmsnorm_adapter",
            ),
        ]
        for unit, _, _ in units:
            specs.append((f"log_{unit}_stdout", "log", f"logs/{unit}.stdout"))
            specs.append((f"log_{unit}_stderr", "log", f"logs/{unit}.stderr"))
        artifacts = []
        self.paths = {}
        self.artifacts = {}
        relative_by_id = {}
        output_units = {
            "generated_softmax_header": "softmax",
            "generated_layer_norm_header": "layer_norm",
            "generated_rmsnorm_header": "rmsnorm",
            "generated_rmsnorm_adapter": "rmsnorm_adapter",
        }
        stderr_units = {
            f"log_{unit}_stderr": unit for unit, _, _ in units
        }
        for logical_id, role, relative in specs:
            path = self.evidence / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if logical_id in output_units:
                content = topology_fixture(output_units[logical_id])
            elif logical_id in stderr_units:
                content = (
                    VALIDATOR.recipe_summary(
                        VALIDATOR.EXPECTED_CONVERSION_TOPOLOGY[
                            stderr_units[logical_id]
                        ]
                    )
                    + "\n"
                )
            else:
                content = f"{logical_id}\n"
            path.write_text(content, encoding="utf-8")
            self.paths[logical_id] = path
            relative_by_id[logical_id] = relative
            artifact = {
                "logical_id": logical_id,
                "role": role,
                "path": relative,
                "sha256": digest(path),
            }
            artifacts.append(artifact)
            self.artifacts[logical_id] = artifact
        conversions = []
        for unit, input_id, output_id in units:
            argv = [
                "./converter/ascify-clang",
                relative_by_id[input_id],
                "--target-policy=dav-c310-vec",
                "--simt-math=fast",
                "--cuda-path=/opt/cuda",
                "--clang-resource-directory=/opt/clang",
                "-o",
                relative_by_id[output_id],
                "--",
                "-Iinputs",
                "-std=c++17",
            ]
            argv.extend(VALIDATOR.CONVERSION_FORCED_INCLUDES[unit])
            conversions.append(
                {
                    "logical_id": unit,
                    "input_artifact": input_id,
                    "output_artifact": output_id,
                    "stdout_artifact": f"log_{unit}_stdout",
                    "stderr_artifact": f"log_{unit}_stderr",
                    "argv": argv,
                    "exit_code": 0,
                    "recipe_placement": (
                        VALIDATOR.EXPECTED_RECIPE_PLACEMENT[unit]
                    ),
                    "recipe_topology": (
                        VALIDATOR.EXPECTED_CONVERSION_TOPOLOGY[unit]
                    ),
                    "recipe_summary": VALIDATOR.recipe_summary(
                        VALIDATOR.EXPECTED_CONVERSION_TOPOLOGY[unit]
                    ),
                }
            )
        self.document = {
            "schema_version": "ascify-conversion-evidence-v3",
            "conversion_set_id": "fixture_v3",
            "created_utc": "2026-07-30T00:00:00Z",
            "hostname": "fixture",
            "policy": {
                "target_policy": "dav-c310-vec",
                "simt_math": "fast",
            },
            "converter": {
                "binary_artifact": "ascify_binary",
                "recipe_source_artifact": "recipe_source",
                "recipe_header_artifact": "recipe_header",
            },
            "environment": {
                "cuda_root_origin": "/opt/cuda",
                "clang_resource_directory_origin": "/opt/clang",
                "input_include_path": "inputs",
            },
            "artifacts": artifacts,
            "conversions": conversions,
        }
        self.manifest = self.evidence / "manifest.json"
        self.write_manifest()
        self.args = argparse.Namespace(
            manifest=self.manifest,
            ascify_binary_sha256=digest(self.paths["ascify_binary"]),
            recipe_source=self.paths["recipe_source"],
            recipe_header=self.paths["recipe_header"],
            staged_softmax=self.paths["generated_softmax_header"],
            staged_layer_norm=self.paths["generated_layer_norm_header"],
            staged_rmsnorm=self.paths["generated_rmsnorm_header"],
            staged_rmsnorm_adapter=self.paths["generated_rmsnorm_adapter"],
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_manifest(self) -> None:
        self.manifest.write_text(
            json.dumps(self.document, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    def test_conversion_positive_and_staged_mismatch(self) -> None:
        summary = VALIDATOR.validate_conversion_evidence(self.args)
        self.assertEqual(
            summary["ASCIFY_BINARY_SHA256"],
            digest(self.paths["ascify_binary"]),
        )
        staged = self.root / "wrong.cuh"
        staged.write_text("different\n", encoding="utf-8")
        self.args.staged_softmax = staged
        with self.assertRaises(VALIDATOR.ValidationError):
            VALIDATOR.validate_conversion_evidence(self.args)

    def test_conversion_rejects_relative_escape(self) -> None:
        self.document["artifacts"][0]["path"] = "../outside"
        self.write_manifest()
        with self.assertRaises(VALIDATOR.ValidationError):
            VALIDATOR.validate_conversion_evidence(self.args)

    def test_conversion_rejects_unbound_argv(self) -> None:
        self.document["conversions"][0]["argv"][1] = "inputs/rms_norm.cuh"
        self.write_manifest()
        with self.assertRaises(VALIDATOR.ValidationError):
            VALIDATOR.validate_conversion_evidence(self.args)

    def test_conversion_rejects_reused_logs(self) -> None:
        self.document["conversions"][1]["stdout_artifact"] = (
            self.document["conversions"][0]["stdout_artifact"]
        )
        self.write_manifest()
        with self.assertRaises(VALIDATOR.ValidationError):
            VALIDATOR.validate_conversion_evidence(self.args)

    def test_conversion_rejects_environment_drift(self) -> None:
        self.document["environment"]["input_include_path"] = "wrong"
        self.write_manifest()
        with self.assertRaises(VALIDATOR.ValidationError):
            VALIDATOR.validate_conversion_evidence(self.args)

    def test_conversion_rejects_recipe_topology_or_summary_drift(self) -> None:
        output = self.paths["generated_softmax_header"]
        original_output = output.read_text(encoding="utf-8")
        output.write_text(
            token_only_topology_fixture("softmax"),
            encoding="utf-8",
        )
        self.assertEqual(
            VALIDATOR.recipe_topology(output),
            VALIDATOR.EXPECTED_CONVERSION_TOPOLOGY["softmax"],
        )
        self.assertNotEqual(
            VALIDATOR.recipe_placement(output, "softmax"),
            VALIDATOR.EXPECTED_RECIPE_PLACEMENT["softmax"],
        )
        self.artifacts["generated_softmax_header"]["sha256"] = digest(output)
        self.write_manifest()
        with self.assertRaises(VALIDATOR.ValidationError):
            VALIDATOR.validate_conversion_evidence(self.args)

    def test_conversion_rejects_preprocessor_inactive_recipe(self) -> None:
        output = self.paths["generated_softmax_header"]
        original_output = output.read_text(encoding="utf-8")
        for directive in ("#if 0", "#ifdef ASCIFY_VALIDATOR_DEAD_BRANCH"):
            with self.subTest(directive=directive):
                output.write_text(
                    f"{directive}\n{original_output}#endif\n",
                    encoding="utf-8",
                )
                self.artifacts["generated_softmax_header"]["sha256"] = digest(
                    output
                )
                self.write_manifest()
                with self.assertRaises(VALIDATOR.ValidationError):
                    VALIDATOR.validate_conversion_evidence(self.args)

        output.write_text(
            original_output
            + VALIDATOR.RECIPE_TOPOLOGY_TOKENS["rmsnorm_direct_wrappers"]
            + "\n",
            encoding="utf-8",
        )
        self.artifacts["generated_softmax_header"]["sha256"] = digest(output)
        self.write_manifest()
        with self.assertRaises(VALIDATOR.ValidationError):
            VALIDATOR.validate_conversion_evidence(self.args)

        output.write_text(original_output, encoding="utf-8")
        self.artifacts["generated_softmax_header"]["sha256"] = digest(output)
        stderr = self.paths["log_softmax_stderr"]
        stderr.write_text(
            VALIDATOR.recipe_summary(
                VALIDATOR.EXPECTED_CONVERSION_TOPOLOGY["layer_norm"]
            )
            + "\n",
            encoding="utf-8",
        )
        self.artifacts["log_softmax_stderr"]["sha256"] = digest(stderr)
        self.write_manifest()
        with self.assertRaises(VALIDATOR.ValidationError):
            VALIDATOR.validate_conversion_evidence(self.args)

    def test_conversion_rejects_dispatcher_dead_or_log_misplacement(self) -> None:
        softmax = self.paths["generated_softmax_header"]
        original_softmax = softmax.read_text(encoding="utf-8")
        misplaced_dispatcher = original_softmax.replace(
            "::ascify::target::dav_c310::TrySoftmax(",
            "::ascify::target::dav_c310::TrySoftmaxNotPlaced(",
            1,
        ).replace(
            "int DispatchLogSoftmax(int overload_0) {",
            "int DispatchLogSoftmax(int overload_0) {\n"
            "  ::ascify::target::dav_c310::TrySoftmax("
            "stream, load, store, rows, cols, nullptr);",
            1,
        )
        self.assertEqual(
            misplaced_dispatcher.count(
                VALIDATOR.RECIPE_TOPOLOGY_TOKENS["softmax_direct_wrappers"]
            ),
            3,
        )
        softmax.write_text(misplaced_dispatcher, encoding="utf-8")
        self.artifacts["generated_softmax_header"]["sha256"] = digest(softmax)
        self.write_manifest()
        with self.assertRaises(VALIDATOR.ValidationError):
            VALIDATOR.validate_conversion_evidence(self.args)

    def test_conversion_rejects_try_argument_or_compute_binding_drift(self) -> None:
        softmax = self.paths["generated_softmax_header"]
        original_softmax = softmax.read_text(encoding="utf-8")
        softmax.write_text(
            original_softmax.replace(
                "stream, load, store, rows, cols,",
                "stream, store, load, rows, cols,",
                1,
            ),
            encoding="utf-8",
        )
        self.artifacts["generated_softmax_header"]["sha256"] = digest(softmax)
        self.write_manifest()
        with self.assertRaises(VALIDATOR.ValidationError):
            VALIDATOR.validate_conversion_evidence(self.args)

        softmax.write_text(
            original_softmax.replace(
                "static_cast<ComputeType*>(nullptr)",
                "static_cast<float*>(nullptr)",
                1,
            ),
            encoding="utf-8",
        )
        self.artifacts["generated_softmax_header"]["sha256"] = digest(softmax)
        self.write_manifest()
        with self.assertRaises(VALIDATOR.ValidationError):
            VALIDATOR.validate_conversion_evidence(self.args)

        softmax.write_text(original_softmax, encoding="utf-8")
        self.artifacts["generated_softmax_header"]["sha256"] = digest(softmax)
        rmsnorm = self.paths["generated_rmsnorm_header"]
        rmsnorm.write_text(
            rmsnorm.read_text(encoding="utf-8").replace(
                "nrow, ncol, eps, inv_rms,",
                "nrow, ncol, inv_rms, eps,",
                1,
            ),
            encoding="utf-8",
        )
        self.artifacts["generated_rmsnorm_header"]["sha256"] = digest(rmsnorm)
        self.write_manifest()
        with self.assertRaises(VALIDATOR.ValidationError):
            VALIDATOR.validate_conversion_evidence(self.args)

        wrong_guard = original_softmax.replace(
            "::fixture::Algorithm::kSoftmax",
            "::fixture::Algorithm::kLogSoftmax",
            1,
        )
        softmax.write_text(wrong_guard, encoding="utf-8")
        self.artifacts["generated_softmax_header"]["sha256"] = digest(softmax)
        self.write_manifest()
        with self.assertRaises(VALIDATOR.ValidationError):
            VALIDATOR.validate_conversion_evidence(self.args)

        softmax.write_text(original_softmax, encoding="utf-8")
        self.artifacts["generated_softmax_header"]["sha256"] = digest(softmax)
        rmsnorm = self.paths["generated_rmsnorm_header"]
        original_rmsnorm = rmsnorm.read_text(encoding="utf-8")
        dead_code = original_rmsnorm.replace(
            "::ascify::target::dav_c310::TryRmsNorm(",
            "::ascify::target::dav_c310::TryRmsNormNotPlaced(",
            1,
        ) + (
            "\nint DeadRmsRecipe() {\n"
            "  return ::ascify::target::dav_c310::TryRmsNorm("
            "stream, load, store, rows, cols, eps, inv_rms, nullptr).status;\n"
            "}\n"
        )
        self.assertEqual(
            dead_code.count(
                VALIDATOR.RECIPE_TOPOLOGY_TOKENS["rmsnorm_direct_wrappers"]
            ),
            3,
        )
        rmsnorm.write_text(dead_code, encoding="utf-8")
        self.artifacts["generated_rmsnorm_header"]["sha256"] = digest(rmsnorm)
        self.write_manifest()
        with self.assertRaises(VALIDATOR.ValidationError):
            VALIDATOR.validate_conversion_evidence(self.args)

    def test_conversion_rejects_rms_post_geometry_contract_mutations(
        self,
    ) -> None:
        rmsnorm = self.paths["generated_rmsnorm_header"]
        original = rmsnorm.read_text(encoding="utf-8")
        mutations = {
            "try_before_geometry": rms_try_before_geometry,
            "try_after_launch": rms_try_after_launch,
            "geometry_error_guard_inverted": (
                rms_inverted_geometry_error_guard
            ),
            "geometry_error_return_mismatch": (
                rms_mismatched_geometry_error_return
            ),
            "duplicate_geometry_call": rms_duplicate_geometry_call,
            "geometry_callee_namespace_drift": (
                rms_geometry_callee_namespace_drift
            ),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name):
                mutated = mutation(original)
                rmsnorm.write_text(mutated, encoding="utf-8")
                self.assertEqual(
                    VALIDATOR.recipe_topology(rmsnorm),
                    VALIDATOR.EXPECTED_CONVERSION_TOPOLOGY["rmsnorm"],
                )
                self.assertNotEqual(
                    VALIDATOR.recipe_placement(rmsnorm, "rmsnorm"),
                    VALIDATOR.EXPECTED_RECIPE_PLACEMENT["rmsnorm"],
                )
                self.artifacts["generated_rmsnorm_header"]["sha256"] = (
                    digest(rmsnorm)
                )
                self.write_manifest()
                with self.assertRaises(VALIDATOR.ValidationError):
                    VALIDATOR.validate_conversion_evidence(self.args)


class CsvValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.shapes = self.root / "shapes.csv"
        self.shape_fields = [
            "case_id",
            "idx",
            "op",
            "dtype",
            "tier",
            "run_check",
            "run_bench",
            "rows",
            "cols",
            "scenario",
            "input_pattern",
            "cuda_pred_path",
            "affine",
            "eps",
        ]
        self.shape_rows = [
            {
                "case_id": "sm",
                "idx": "1",
                "op": "softmax",
                "dtype": "fp16",
                "tier": "tune",
                "run_check": "1",
                "run_bench": "1",
                "rows": "2",
                "cols": "4",
                "scenario": "fixture",
                "input_pattern": "random",
                "cuda_pred_path": "warp",
                "affine": "0",
                "eps": "1e-5",
            },
            {
                "case_id": "rnp",
                "idx": "2",
                "op": "rms_norm",
                "dtype": "fp16",
                "tier": "tune",
                "run_check": "1",
                "run_bench": "1",
                "rows": "3",
                "cols": "8",
                "scenario": "fixture",
                "input_pattern": "random",
                "cuda_pred_path": "warp",
                "affine": "0",
                "eps": "1e-5",
            },
            {
                "case_id": "rna",
                "idx": "3",
                "op": "rms_norm",
                "dtype": "fp16",
                "tier": "tune",
                "run_check": "1",
                "run_bench": "1",
                "rows": "3",
                "cols": "8",
                "scenario": "fixture",
                "input_pattern": "random",
                "cuda_pred_path": "warp",
                "affine": "1",
                "eps": "1e-5",
            },
        ]
        write_csv(self.shapes, self.shape_fields, self.shape_rows)
        self.perf = self.root / "perf.csv"
        self.perf_fields = [
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
            "scenario",
            "input_pattern",
            "cuda_pred_path",
            "device_id",
            "warmup",
            "samples",
            "inner_repeats",
            "lat_ms_mean",
            "lat_ms_median",
            "lat_ms_min",
            "lat_ms_p90",
            "logical_bytes",
            "gbps",
            "status",
            "reason",
        ]
        variants = {
            "sm": "converted_dispatch",
            "rnp": "converted_plain",
            "rna": "converted_affine",
        }
        self.perf_rows = []
        for shape in self.shape_rows:
            rows = int(shape["rows"])
            cols = int(shape["cols"])
            logical_bytes = 4 * rows * cols
            if shape["op"] == "rms_norm":
                logical_bytes += 4 * rows
                if shape["affine"] == "1":
                    logical_bytes += 2 * cols
            median = 0.01
            self.perf_rows.append(
                {
                    "run_id": "fixture_run",
                    "op": shape["op"],
                    "dtype": "fp16",
                    "math_mode": "production-fast",
                    "variant": variants[shape["case_id"]],
                    "case_id": shape["case_id"],
                    "idx": shape["idx"],
                    "tier": shape["tier"],
                    "rows": shape["rows"],
                    "cols": shape["cols"],
                    "scenario": shape["scenario"],
                    "input_pattern": shape["input_pattern"],
                    "cuda_pred_path": shape["cuda_pred_path"],
                    "device_id": "3",
                    "warmup": "20",
                    "samples": "50",
                    "inner_repeats": "20",
                    "lat_ms_mean": "0.011",
                    "lat_ms_median": str(median),
                    "lat_ms_min": "0.009",
                    "lat_ms_p90": "0.012",
                    "logical_bytes": str(logical_bytes),
                    "gbps": str(logical_bytes / (median * 1.0e-3) / 1.0e9),
                    "status": "pass",
                    "reason": "",
                }
            )
        self.args = argparse.Namespace(
            execution_kind="perf",
            run_id="fixture_run",
            shapes=self.shapes,
            tier="tune",
            variant_mode="direct",
            expected_softmax=1,
            expected_rmsnorm=2,
            device=3,
            math_mode="production-fast",
            warmup=20,
            samples=50,
            inner_repeats=20,
            csv=self.perf,
        )
        self.metric_shapes = VALIDATOR.selected_shape_rows(
            self.shapes, "perf", "tune"
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_shape_parser_rejects_noncanonical_fields(self) -> None:
        mutations = (
            ("unknown-tier", 0, {"tier": "experimental"}),
            ("unsupported-op", 0, {"op": "layer_norm"}),
            ("unsupported-dtype", 0, {"dtype": "fp32"}),
            ("invalid-check-flag", 0, {"run_check": "2"}),
            ("invalid-bench-flag", 0, {"run_bench": "-1"}),
            ("noncanonical-index", 0, {"idx": "01"}),
            ("zero-rows", 0, {"rows": "0"}),
            ("negative-cols", 0, {"cols": "-1"}),
            ("invalid-affine", 1, {"affine": "2"}),
            ("softmax-affine", 0, {"affine": "1"}),
            ("nonfinite-eps", 0, {"eps": "nan"}),
            ("empty-scenario", 0, {"scenario": ""}),
            ("duplicate-index", 1, {"idx": "1"}),
            ("duplicate-key", 1, {"op": "softmax", "case_id": "sm"}),
        )
        for label, row_index, changes in mutations:
            with self.subTest(label=label):
                rows = [dict(row) for row in self.shape_rows]
                rows[row_index].update(changes)
                write_csv(self.shapes, self.shape_fields, rows)
                with self.assertRaises(VALIDATOR.ValidationError):
                    VALIDATOR.selected_shape_rows(
                        self.shapes,
                        "perf",
                        "tune",
                    )

    def write_perf(self) -> None:
        write_csv(self.perf, self.perf_fields, self.perf_rows)

    def test_perf_positive_and_nan_negative(self) -> None:
        self.write_perf()
        _, rows = VALIDATOR.validate_csv_rows(self.args)
        self.assertEqual(len(rows), 3)
        direct_a = [dict(row) for row in rows]
        native = [dict(row) for row in rows]
        direct_b = [dict(row) for row in rows]
        for row in direct_b:
            row["gbps"] = str(float(row["gbps"]) * 1.04)
        summary, max_spread, class_geomeans = VALIDATOR.build_bracket_summary(
            "fixture_set",
            ("direct_a", "native", "direct_b"),
            direct_a,
            native,
            direct_b,
            self.metric_shapes,
        )
        self.assertEqual(len(summary), 3)
        self.assertAlmostEqual(max_spread, 1.04)
        self.assertEqual(
            set(class_geomeans),
            set(VALIDATOR.PERFORMANCE_CLASSES),
        )
        self.assertTrue(
            all(
                ratio >= VALIDATOR.DIRECT_CLASS_NATIVE_GEOMEAN_MIN
                for ratio in class_geomeans.values()
            )
        )
        direct_b[0]["gbps"] = str(float(direct_a[0]["gbps"]) * 1.051)
        with self.assertRaises(VALIDATOR.ValidationError):
            VALIDATOR.build_bracket_summary(
                "fixture_set",
                ("direct_a", "native", "direct_b"),
                direct_a,
                native,
                direct_b,
                self.metric_shapes,
            )
        self.perf_rows[0]["lat_ms_median"] = "nan"
        self.write_perf()
        with self.assertRaises(VALIDATOR.ValidationError):
            VALIDATOR.validate_csv_rows(self.args)

    def test_perf_rejects_case_and_class_native_ratio_regressions(self) -> None:
        self.write_perf()
        _, rows = VALIDATOR.validate_csv_rows(self.args)
        direct_a = [dict(row) for row in rows]
        direct_b = [dict(row) for row in rows]

        slow_case_native = [dict(row) for row in rows]
        slow_case_native[0]["gbps"] = str(
            float(direct_a[0]["gbps"]) / 0.89
        )
        with self.assertRaisesRegex(
            VALIDATOR.ValidationError,
            "case performance",
        ):
            VALIDATOR.build_bracket_summary(
                "fixture_set",
                ("direct_a", "native", "direct_b"),
                direct_a,
                slow_case_native,
                direct_b,
                self.metric_shapes,
            )

        slow_class_native = [dict(row) for row in rows]
        for direct, native in zip(direct_a, slow_class_native):
            native["gbps"] = str(float(direct["gbps"]) / 0.94)
        with self.assertRaisesRegex(
            VALIDATOR.ValidationError,
            "class performance",
        ):
            VALIDATOR.build_bracket_summary(
                "fixture_set",
                ("direct_a", "native", "direct_b"),
                direct_a,
                slow_class_native,
                direct_b,
                self.metric_shapes,
            )

    def test_perf_allows_direct_to_exceed_native(self) -> None:
        self.write_perf()
        _, rows = VALIDATOR.validate_csv_rows(self.args)
        direct_a = [dict(row) for row in rows]
        direct_b = [dict(row) for row in rows]
        native = [dict(row) for row in rows]
        for direct, native_row in zip(direct_a, native):
            native_row["gbps"] = str(float(direct["gbps"]) / 1.5)
        summary, _, class_geomeans = VALIDATOR.build_bracket_summary(
            "fixture_set",
            ("direct_a", "native", "direct_b"),
            direct_a,
            native,
            direct_b,
            self.metric_shapes,
        )
        self.assertTrue(
            all(float(row["direct_center_over_native"]) > 1.0 for row in summary)
        )
        self.assertTrue(all(value > 1.0 for value in class_geomeans.values()))

    def test_perf_rejects_affine_route_and_shape_semantic_drift(self) -> None:
        self.perf_rows[1]["variant"] = "converted_affine"
        self.perf_rows[2]["variant"] = "converted_plain"
        self.write_perf()
        with self.assertRaises(VALIDATOR.ValidationError):
            VALIDATOR.validate_csv_rows(self.args)
        self.perf_rows[1]["variant"] = "converted_plain"
        self.perf_rows[2]["variant"] = "converted_affine"
        self.perf_rows[0]["scenario"] = "wrong"
        self.write_perf()
        with self.assertRaises(VALIDATOR.ValidationError):
            VALIDATOR.validate_csv_rows(self.args)

    def test_perf_rejects_logical_byte_drift(self) -> None:
        self.perf_rows[0]["logical_bytes"] = "1"
        self.write_perf()
        with self.assertRaises(VALIDATOR.ValidationError):
            VALIDATOR.validate_csv_rows(self.args)


class IsolationAndFreezeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "work"
        self.result = self.root / "results"
        self.manifests = self.result / "manifests"
        self.manifests.mkdir(parents=True)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def path_args(self) -> argparse.Namespace:
        return argparse.Namespace(
            work_root=self.root,
            result_dir=self.result,
            accuracy_csv=self.result / "accuracy.csv",
            perf_csv=self.result / "perf.csv",
            preheat_csv=self.result / "preheat.csv",
            manifest_dir=self.manifests,
            set_manifest=self.manifests / "set.tsv",
            bracket_summary=self.manifests / "set.bracket_summary.csv",
            formal_metrics=self.manifests / "set.formal_metrics.csv",
            result_lock=self.result / ".formal-results.lock",
            build_input_snapshot=self.manifests / "set.build_inputs.tsv",
            binary_bundle_dir=self.manifests / "set.binaries",
        )

    def test_paths_reject_dangling_preheat_alias(self) -> None:
        args = self.path_args()
        VALIDATOR.validate_formal_paths(args)
        args.preheat_csv.symlink_to(args.perf_csv.name)
        with self.assertRaises(VALIDATOR.ValidationError):
            VALIDATOR.validate_formal_paths(args)

    def test_freeze_detects_mutation(self) -> None:
        artifact = self.root / "input"
        artifact.write_text("v1\n", encoding="utf-8")
        snapshot = self.manifests / "freeze.tsv"
        VALIDATOR.write_freeze_snapshot(
            argparse.Namespace(
                output=snapshot,
                artifact=[("input", artifact)],
            )
        )
        VALIDATOR.check_freeze_snapshot(snapshot)
        artifact.write_text("v2\n", encoding="utf-8")
        with self.assertRaises(VALIDATOR.ValidationError):
            VALIDATOR.check_freeze_snapshot(snapshot)

    def test_formal_freeze_requires_the_exact_build_input_set(self) -> None:
        artifacts = []
        for name in sorted(VALIDATOR.FORMAL_BUILD_INPUT_FILES):
            if name in VALIDATOR.CANONICAL_FORMAL_SHAPES:
                artifact = VALIDATOR.CANONICAL_FORMAL_SHAPES[name][0]
            else:
                artifact = self.root / "inputs" / name
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_text(f"{name}\n", encoding="utf-8")
            artifacts.append((name, artifact))
        snapshot = self.manifests / "formal-freeze.tsv"
        VALIDATOR.write_freeze_snapshot(
            argparse.Namespace(
                output=snapshot,
                artifact=artifacts,
                formal_contract=True,
            )
        )
        VALIDATOR.check_freeze_snapshot(snapshot, formal_contract=True)

        with self.assertRaisesRegex(
            VALIDATOR.ValidationError,
            "formal freeze artifact set mismatch",
        ):
            VALIDATOR.write_freeze_snapshot(
                argparse.Namespace(
                    output=self.manifests / "missing-freeze.tsv",
                    artifact=artifacts[:-1],
                    formal_contract=True,
                )
            )

        mutated_boundary = self.root / "mutated-correctness.csv"
        text = VALIDATOR.CANONICAL_FORMAL_SHAPES[
            "boundary_shapes"
        ][0].read_text(encoding="utf-8")
        mutated_boundary.write_text(
            text.replace(",cols1\n", ",same_count_mutation\n", 1),
            encoding="utf-8",
        )
        mutated_artifacts = [
            (
                name,
                mutated_boundary if name == "boundary_shapes" else artifact,
            )
            for name, artifact in artifacts
        ]
        with self.assertRaisesRegex(
            VALIDATOR.ValidationError,
            "non-canonical boundary_shapes",
        ):
            VALIDATOR.write_freeze_snapshot(
                argparse.Namespace(
                    output=self.manifests / "mutated-shape-freeze.tsv",
                    artifact=mutated_artifacts,
                    formal_contract=True,
                )
            )

    def test_phase_shape_closure_rejects_wrong_grid(self) -> None:
        boundary_digest = "a" * 64
        tune_digest = "b" * 64
        VALIDATOR.require_phase_shape_closure(
            self.manifests / "boundary.tsv",
            "direct_boundary",
            {
                "shapes": boundary_digest,
                "boundary_shapes": boundary_digest,
                "tune_shapes": tune_digest,
            },
        )
        VALIDATOR.require_phase_shape_closure(
            self.manifests / "perf.tsv",
            "native_perf",
            {
                "shapes": tune_digest,
                "boundary_shapes": boundary_digest,
                "tune_shapes": tune_digest,
            },
        )
        with self.assertRaisesRegex(
            VALIDATOR.ValidationError,
            "must use boundary_shapes",
        ):
            VALIDATOR.require_phase_shape_closure(
                self.manifests / "wrong.tsv",
                "native_boundary",
                {
                    "shapes": tune_digest,
                    "boundary_shapes": boundary_digest,
                    "tune_shapes": tune_digest,
                },
            )

    def test_formal_timing_is_fixed_by_phase(self) -> None:
        manifest = self.manifests / "timing.tsv"
        VALIDATOR.require_formal_timing(
            manifest,
            "direct_perf_a",
            "perf",
            20,
            50,
            20,
        )
        VALIDATOR.require_formal_timing(
            manifest,
            "preheat",
            "perf",
            5,
            5,
            5,
        )
        with self.assertRaisesRegex(
            VALIDATOR.ValidationError,
            "non-canonical timing",
        ):
            VALIDATOR.require_formal_timing(
                manifest,
                "native_perf",
                "perf",
                1,
                1,
                1,
            )

    def test_formal_run_contract_binds_phase_role_and_active_shape_path(self) -> None:
        canonical = VALIDATOR.CANONICAL_FORMAL_SHAPES["tune_shapes"][0]
        common = {
            "phase": "direct_perf_a",
            "execution_kind": "perf",
            "shapes": canonical,
            "tier": "tune",
            "variant_mode": "direct",
            "math_mode": "production-fast",
            "expected_softmax": 5,
            "expected_rmsnorm": 10,
            "warmup": 20,
            "samples": 50,
            "inner_repeats": 20,
            "check_max_elements": 10_000_000,
            "device_util_max": "0",
            "device_hbm_bw_max": "0",
        }
        VALIDATOR.validate_formal_run_contract(
            self.manifests / "contract.tsv",
            **common,
        )
        for label, changes in (
            ("wrong-kind", {"execution_kind": "accuracy"}),
            ("wrong-tier", {"tier": "smoke"}),
            ("wrong-variant", {"variant_mode": "native"}),
            ("wrong-math-mode", {"math_mode": "compiler-fast"}),
            ("wrong-count", {"expected_softmax": 4}),
            ("weak-timing", {"samples": 1}),
            ("weak-max-elements", {"check_max_elements": 1}),
            ("weak-util-threshold", {"device_util_max": "1"}),
        ):
            with self.subTest(label=label):
                mutated = dict(common)
                mutated.update(changes)
                with self.assertRaises(VALIDATOR.ValidationError):
                    VALIDATOR.validate_formal_run_contract(
                        self.manifests / "contract.tsv",
                        **mutated,
                    )

        copied = self.root / "same-bytes-shapes.csv"
        shutil.copyfile(canonical, copied)
        copied_contract = dict(common)
        copied_contract["shapes"] = copied
        with self.assertRaisesRegex(
            VALIDATOR.ValidationError,
            "active shapes path is non-canonical",
        ):
            VALIDATOR.validate_formal_run_contract(
                self.manifests / "contract.tsv",
                **copied_contract,
            )

    def test_formal_device_snapshot_requires_zero_idle_metrics(self) -> None:
        snapshot = self.manifests / "device.txt"
        section = """\
phase={phase}
utc_time=2026-07-30T00:00:01Z
hostname=fixture-host
run_id=fixture-run
device=5
[mapping]
NPU ID Slot ID Chip ID Chip Phy-ID Chip Name
5 22 0 5 Ascend950PR
[health]
NPU ID : 5
Health Status : OK
[usages]
NPU ID : 5
Aicore Usage Rate(%) : 0
Aivector Usage Rate(%) : 0
HBM Bandwidth Usage Rate(%) : 0
NPU Utilization(%) : 0
[processes]
NPU ID : 5
No process in device.
"""
        idle_text = "".join(
            section.format(phase=phase) for phase in ("pre", "post")
        )
        snapshot.write_text(idle_text, encoding="utf-8")
        timestamps = VALIDATOR.validate_idle_device_snapshot(
            snapshot,
            5,
            expected_run_id="fixture-run",
            expected_hostname="fixture-host",
        )
        VALIDATOR.require_device_snapshot_interval(
            snapshot,
            timestamps,
            VALIDATOR.iso8601(
                "2026-07-30T00:00:00Z",
                "fixture start",
            ),
            VALIDATOR.iso8601(
                "2026-07-30T00:00:02Z",
                "fixture end",
            ),
        )
        post_only = self.manifests / "device-post.txt"
        post_only.write_text(section.format(phase="post"), encoding="utf-8")
        VALIDATOR.validate_idle_device_snapshot(
            post_only,
            5,
            ("post",),
            expected_run_id="fixture-run",
            expected_hostname="fixture-host",
        )
        stuffed = section.format(phase="post").replace(
            "[mapping]\n",
            "[mapping]\nAicore Usage Rate(%) : 0\n",
            1,
        ).replace(
            "[usages]\nNPU ID : 5\nAicore Usage Rate(%) : 0\n",
            "[usages]\nNPU ID : 5\n",
            1,
        )
        post_only.write_text(stuffed, encoding="utf-8")
        with self.assertRaisesRegex(
            VALIDATOR.ValidationError,
            "Aicore Usage Rate",
        ):
            VALIDATOR.validate_idle_device_snapshot(
                post_only,
                5,
                ("post",),
                expected_run_id="fixture-run",
                expected_hostname="fixture-host",
            )
        empty_mapping = section.format(phase="post").replace(
            "5 22 0 5 Ascend950PR\n",
            "",
            1,
        )
        post_only.write_text(empty_mapping, encoding="utf-8")
        with self.assertRaisesRegex(
            VALIDATOR.ValidationError,
            "selected mapping",
        ):
            VALIDATOR.validate_idle_device_snapshot(
                post_only,
                5,
                ("post",),
                expected_run_id="fixture-run",
                expected_hostname="fixture-host",
            )
        unparseable_process = section.format(phase="post").replace(
            "No process in device.",
            "No permission to inspect process table.",
            1,
        )
        post_only.write_text(unparseable_process, encoding="utf-8")
        with self.assertRaisesRegex(
            VALIDATOR.ValidationError,
            "process state is not idle",
        ):
            VALIDATOR.validate_idle_device_snapshot(
                post_only,
                5,
                ("post",),
                expected_run_id="fixture-run",
                expected_hostname="fixture-host",
            )
        wrong_npu = section.format(phase="post").replace(
            "[health]\nNPU ID : 5",
            "[health]\nNPU ID : 99",
            1,
        )
        post_only.write_text(wrong_npu, encoding="utf-8")
        with self.assertRaisesRegex(
            VALIDATOR.ValidationError,
            "health NPU identity mismatch",
        ):
            VALIDATOR.validate_idle_device_snapshot(
                post_only,
                5,
                ("post",),
                expected_run_id="fixture-run",
                expected_hostname="fixture-host",
            )
        bad_plus_ok = section.format(phase="post").replace(
            "Health Status : OK",
            "Health Status : BAD\nHealth Query Cached : OK",
            1,
        )
        post_only.write_text(bad_plus_ok, encoding="utf-8")
        with self.assertRaisesRegex(
            VALIDATOR.ValidationError,
            "health is not exactly one OK",
        ):
            VALIDATOR.validate_idle_device_snapshot(
                post_only,
                5,
                ("post",),
                expected_run_id="fixture-run",
                expected_hostname="fixture-host",
            )

        snapshot.write_text(
            idle_text.replace(
                "Aivector Usage Rate(%) : 0",
                "Aivector Usage Rate(%) : 1",
                1,
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            VALIDATOR.ValidationError,
            "Aivector Usage Rate",
        ):
            VALIDATOR.validate_idle_device_snapshot(snapshot, 5)

    def test_formal_set_requires_eight_distinct_run_artifact_sets(self) -> None:
        manifests = []
        evidence = []
        snapshots = []
        runtime_configs = []
        for index in range(8):
            for paths, suffix in (
                (manifests, "tsv"),
                (evidence, "csv"),
                (snapshots, "device.txt"),
            ):
                path = self.manifests / f"run-{index}.{suffix}"
                path.write_text(f"{index}:{suffix}\n", encoding="utf-8")
                paths.append(path)
        for index in range(4):
            path = self.manifests / f"run-{index}.runtime.tsv"
            path.write_text(f"{index}:runtime\n", encoding="utf-8")
            runtime_configs.append(path)
        run_ids = [f"run-{index}" for index in range(8)]
        VALIDATOR.require_distinct_formal_runs(
            "formal-set",
            run_ids,
            manifests,
            evidence,
            snapshots,
            runtime_configs,
        )

        duplicated_ids = list(run_ids)
        duplicated_ids[-1] = duplicated_ids[0]
        with self.assertRaisesRegex(
            VALIDATOR.ValidationError,
            "run IDs must be pairwise distinct",
        ):
            VALIDATOR.require_distinct_formal_runs(
                "formal-set",
                duplicated_ids,
                manifests,
                evidence,
                snapshots,
                runtime_configs,
            )

        duplicated_snapshots = list(snapshots)
        duplicated_snapshots[-1] = duplicated_snapshots[0]
        with self.assertRaisesRegex(
            VALIDATOR.ValidationError,
            "artifact paths must be pairwise distinct",
        ):
            VALIDATOR.require_distinct_formal_runs(
                "formal-set",
                run_ids,
                manifests,
                evidence,
                duplicated_snapshots,
                runtime_configs,
            )

    def test_formal_boundary_gate_counts_match_the_frozen_shape_table(self) -> None:
        shapes = MODULE_PATH.parent.parent / "shapes" / "correctness.csv"
        selected = VALIDATOR.selected_shape_rows(shapes, "accuracy", "smoke")
        counts = {
            op: sum(key[0] == op for key in selected)
            for op in ("softmax", "rms_norm")
        }
        for phase in ("direct_boundary", "native_boundary"):
            contract = VALIDATOR.EXPECTED_FORMAL_GATES[phase]
            self.assertEqual(int(contract[1]), counts["softmax"])
            self.assertEqual(int(contract[2]), counts["rms_norm"])

    def test_preflight_rejects_runtime_and_metric_evidence_content(self) -> None:
        args = argparse.Namespace(
            run_id=["run_a"],
            accuracy_csv=self.result / "accuracy.csv",
            perf_csv=self.result / "perf.csv",
            manifest_dir=self.manifests,
            set_id="set",
        )
        runtime = self.manifests / "run_a.runtime_config.tsv"
        runtime.write_text("preexisting\n", encoding="utf-8")
        with self.assertRaises(VALIDATOR.ValidationError):
            VALIDATOR.preflight(args)
        runtime.unlink()

        metrics = self.manifests / "set.formal_metrics.csv"
        metrics.write_text("preexisting\n", encoding="utf-8")
        with self.assertRaises(VALIDATOR.ValidationError):
            VALIDATOR.preflight(args)

        metrics.unlink()
        bundle = self.manifests / "set.binaries"
        bundle.mkdir()
        with self.assertRaises(VALIDATOR.ValidationError):
            VALIDATOR.preflight(args)

    def test_prepared_preflight_requires_only_frozen_inputs_to_exist(
        self,
    ) -> None:
        args = argparse.Namespace(
            run_id=["run_a", "run_b"],
            accuracy_csv=self.result / "accuracy.csv",
            perf_csv=self.result / "perf.csv",
            manifest_dir=self.manifests,
            set_id="set",
            prepared=True,
        )
        with self.assertRaisesRegex(
            VALIDATOR.ValidationError,
            "prepared formal inputs are missing or unsafe",
        ):
            VALIDATOR.preflight(args)

        snapshot = self.manifests / "set.build_inputs.tsv"
        bundle = self.manifests / "set.binaries"
        snapshot.write_text("frozen\n", encoding="utf-8")
        bundle.mkdir()
        VALIDATOR.preflight(args)

        completed_set = self.manifests / "set.tsv"
        completed_set.write_text("complete\n", encoding="utf-8")
        with self.assertRaisesRegex(
            VALIDATOR.ValidationError,
            "formal evidence paths already exist",
        ):
            VALIDATOR.preflight(args)
        completed_set.unlink()

        dangling_run = self.manifests / "run_a.csv"
        dangling_run.symlink_to(self.manifests / "missing.csv")
        with self.assertRaisesRegex(
            VALIDATOR.ValidationError,
            "formal evidence paths already exist",
        ):
            VALIDATOR.preflight(args)

        missing_dir_args = argparse.Namespace(
            **{
                **vars(args),
                "manifest_dir": self.root / "missing-manifests",
            }
        )
        with self.assertRaisesRegex(
            VALIDATOR.ValidationError,
            "requires a real manifest directory",
        ):
            VALIDATOR.preflight(missing_dir_args)


class BinaryBundleAndLockSafetyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        for candidate in self.root.glob("*.binaries"):
            if candidate.is_dir():
                candidate.chmod(0o755)
        self.temp.cleanup()

    def binary_artifacts(self, directory: Path) -> list[tuple[str, Path]]:
        directory.mkdir(parents=True)
        artifacts = []
        for name in sorted(VALIDATOR.FORMAL_BINARY_NAMES):
            path = directory / name
            path.write_text(f"#!/bin/sh\n# {name}\nexit 0\n", encoding="utf-8")
            path.chmod(0o755)
            artifacts.append((name, path))
        return artifacts

    def formal_snapshot(self, label: str) -> Path:
        artifacts = []
        for name in sorted(VALIDATOR.FORMAL_BUILD_INPUT_FILES):
            if name in VALIDATOR.CANONICAL_FORMAL_SHAPES:
                artifact = VALIDATOR.CANONICAL_FORMAL_SHAPES[name][0]
            else:
                artifact = self.root / label / "inputs" / name
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_text(
                    f"{label}:{name}\n",
                    encoding="utf-8",
                )
            artifacts.append((name, artifact))
        snapshot = self.root / f"{label}.build_inputs.tsv"
        VALIDATOR.write_freeze_snapshot(
            argparse.Namespace(
                output=snapshot,
                artifact=artifacts,
                formal_contract=True,
            )
        )
        return snapshot

    def test_binary_bundle_is_complete_non_overwriting_and_read_only(self) -> None:
        output = self.root / "set.binaries"
        artifacts = self.binary_artifacts(self.root / "scratch")
        snapshot = self.formal_snapshot("set")
        args = argparse.Namespace(
            formal_set_id="formal_set",
            output_dir=output,
            build_input_snapshot=snapshot,
            artifact=artifacts,
        )
        VALIDATOR.write_binary_bundle(args)
        parsed = VALIDATOR.check_binary_bundle(
            output / "manifest.tsv",
            "formal_set",
            snapshot,
        )
        self.assertEqual(set(parsed["files"]), VALIDATOR.FORMAL_BINARY_NAMES)
        self.assertEqual(
            parsed["values"]["build_input_snapshot_path"],
            str(snapshot.resolve()),
        )
        self.assertEqual(
            parsed["values"]["build_input_snapshot_sha256"],
            digest(snapshot),
        )
        self.assertEqual(
            output.stat().st_mode & 0o222,
            0,
        )
        self.assertEqual(
            (output / "manifest.tsv").stat().st_mode & 0o222,
            0,
        )
        with self.assertRaisesRegex(
            VALIDATOR.ValidationError,
            "refusing to overwrite binary bundle",
        ):
            VALIDATOR.write_binary_bundle(args)

        other_snapshot = self.formal_snapshot("other")
        with self.assertRaises(VALIDATOR.ValidationError):
            VALIDATOR.check_binary_bundle(
                output / "manifest.tsv",
                "formal_set",
                other_snapshot,
            )

        manifest = output / "manifest.tsv"
        manifest.chmod(0o644)
        with self.assertRaisesRegex(
            VALIDATOR.ValidationError,
            "must be read-only",
        ):
            VALIDATOR.check_binary_bundle(
                manifest,
                "formal_set",
                snapshot,
            )
        manifest.chmod(0o444)

        output.chmod(0o755)
        with self.assertRaisesRegex(
            VALIDATOR.ValidationError,
            "must be read-only",
        ):
            VALIDATOR.check_binary_bundle(
                manifest,
                "formal_set",
                snapshot,
            )
        output.chmod(0o555)

        binary = output / sorted(VALIDATOR.FORMAL_BINARY_NAMES)[0]
        binary.chmod(0o755)
        with self.assertRaisesRegex(
            VALIDATOR.ValidationError,
            "misnamed or not executable",
        ):
            VALIDATOR.check_binary_bundle(
                output / "manifest.tsv",
                "formal_set",
                snapshot,
            )
        binary.chmod(0o555)

        snapshot_files = VALIDATOR.parse_manifest(snapshot)["paths"]
        bound_input = Path(snapshot_files["aclcub_header"])
        bound_input.write_text("mutated after bundling\n", encoding="utf-8")
        with self.assertRaises(VALIDATOR.ValidationError):
            VALIDATOR.check_binary_bundle(
                manifest,
                "formal_set",
                snapshot,
            )

    def test_binary_bundle_rejects_symlink_source(self) -> None:
        artifacts = self.binary_artifacts(self.root / "unsafe-scratch")
        snapshot = self.formal_snapshot("unsafe")
        name, source = artifacts[0]
        target = source.with_suffix(".target")
        source.rename(target)
        source.symlink_to(target.name)
        with self.assertRaisesRegex(
            VALIDATOR.ValidationError,
            "not a regular executable",
        ):
            VALIDATOR.write_binary_bundle(
                argparse.Namespace(
                    formal_set_id="unsafe_set",
                    output_dir=self.root / "unsafe.binaries",
                    build_input_snapshot=snapshot,
                    artifact=[(name, source), *artifacts[1:]],
                )
            )

    def test_safe_lock_does_not_truncate_existing_file(self) -> None:
        lock = self.root / "regular.lock"
        lock.write_text("preserve-this-content\n", encoding="utf-8")
        helper = MODULE_PATH.with_name("safe_lock_exec.py")
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(helper),
                "--fd",
                "9",
                "--path",
                str(lock),
                "--",
                sys.executable,
                "-c",
                "import os; assert os.fstat(9).st_nlink == 1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout={completed.stdout}\nstderr={completed.stderr}",
        )
        self.assertEqual(lock.read_text(encoding="utf-8"), "preserve-this-content\n")

    def test_safe_lock_fails_closed_on_symlink_and_hardlink(self) -> None:
        helper = MODULE_PATH.with_name("safe_lock_exec.py")
        victim = self.root / "victim"
        victim.write_text("do-not-touch\n", encoding="utf-8")
        symlink_lock = self.root / "symlink.lock"
        symlink_lock.symlink_to(victim.name)
        hardlink_lock = self.root / "hardlink.lock"
        os.link(victim, hardlink_lock)
        for lock in (symlink_lock, hardlink_lock):
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(helper),
                    "--fd",
                    "9",
                    "--path",
                    str(lock),
                    "--",
                    sys.executable,
                    "-c",
                    "raise SystemExit(0)",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(victim.read_text(encoding="utf-8"), "do-not-touch\n")


class ConversionRunnerIntegrationTest(unittest.TestCase):
    def test_runner_packages_and_self_validates_portable_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work with spaces"
            inputs = work / "inputs"
            (inputs / "oneflow/core/cuda").mkdir(parents=True)
            for relative in (
                "oneflow/core/cuda/softmax.cuh",
                "oneflow/core/cuda/layer_norm.cuh",
                "oneflow/core/cuda/rms_norm.cuh",
                "rmsnorm_affine_store.cuh",
            ):
                path = inputs / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"fixture {relative}\n", encoding="utf-8")
            cuda_root = work / "cuda"
            cuda_root.mkdir()
            resource = work / "resource"
            (resource / "include").mkdir(parents=True)
            (resource / "include/__clang_cuda_runtime_wrapper.h").write_text(
                "fixture\n", encoding="utf-8"
            )
            fake = root / "fake-ascify"
            fake.write_text(
                "#!/usr/bin/env python3\n"
                "import os, pathlib, shutil, sys, time\n"
                "source = pathlib.Path(sys.argv[1])\n"
                "fail_on = os.environ.get('FAKE_ASCIFY_FAIL_ON')\n"
                "if fail_on and fail_on in str(source):\n"
                "    print('fixture requested failure', file=sys.stderr)\n"
                "    sys.exit(7)\n"
                "name = source.name\n"
                "unit = {'softmax.cuh': 'softmax', "
                "'layer_norm.cuh': 'layer_norm', "
                "'rms_norm.cuh': 'rmsnorm'}.get("
                "name, 'rmsnorm_adapter')\n"
                "forced = sys.argv[sys.argv.index('--') + 1:]\n"
                "expected = ['-Iinputs', '-std=c++17']\n"
                "if unit != 'softmax':\n"
                "    expected += ['-include', 'cuda_fp16.h', "
                "'-include', 'cuda_bf16.h']\n"
                "if forced != expected:\n"
                "    print('unexpected forced includes', file=sys.stderr)\n"
                "    sys.exit(9)\n"
                "time.sleep(float(os.environ.get('FAKE_ASCIFY_DELAY', '0')))\n"
                "output = pathlib.Path(sys.argv[sys.argv.index('-o') + 1])\n"
                "output.parent.mkdir(parents=True, exist_ok=True)\n"
                "shutil.copyfile(str(source), str(output))\n"
                f"topologies = {VALIDATOR.EXPECTED_CONVERSION_TOPOLOGY!r}\n"
                f"fixtures = {dict((unit, topology_fixture(unit)) for unit in VALIDATOR.CONVERSION_UNITS)!r}\n"
                "with output.open('a', encoding='utf-8') as stream:\n"
                "    stream.write(fixtures[unit])\n"
                "topology = topologies[unit]\n"
                "print('Ascify dav-c310 recipe: adapters('"
                "f\"load={topology['direct_load_markers']},\""
                "f\"store={topology['direct_store_markers']}), \""
                "'direct_wrappers('"
                "f\"softmax={topology['softmax_direct_wrappers']},\""
                "f\"rmsnorm={topology['rmsnorm_direct_wrappers']})\", "
                "file=sys.stderr)\n"
                "print('fixture conversion')\n",
                encoding="utf-8",
            )
            fake.chmod(0o755)
            tools = root / "portable-tools"
            tools.mkdir()
            flock = tools / "flock"
            flock.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            flock.chmod(0o755)
            move = tools / "mv"
            move.write_text(
                "#!/usr/bin/env python3\n"
                "import os, sys\n"
                "args = [arg for arg in sys.argv[1:] "
                "if arg not in ('-T', '-n', '--')]\n"
                "if len(args) != 2 or os.path.lexists(args[1]):\n"
                "    sys.exit(1)\n"
                "os.rename(args[0], args[1])\n",
                encoding="utf-8",
            )
            move.chmod(0o755)
            evidence = work / "conversion/evidence_v3"
            runner = MODULE_PATH.with_name("run_910_conversion_v3.sh")
            runner_cwd = MODULE_PATH.parents[3]
            environment = os.environ.copy()
            if os.uname().sysname == "Darwin" or shutil.which("flock") is None:
                environment["PATH"] = (
                    f"{tools}{os.pathsep}{environment['PATH']}"
                )
            environment.update(
                {
                    "WORK_ROOT": str(work),
                    "INPUT_ROOT": str(inputs),
                    "EVIDENCE_ROOT": os.path.relpath(evidence, runner_cwd),
                    "ASCIFY_BINARY": str(fake),
                    "CUDA_ROOT": str(cuda_root),
                    "CLANG_RESOURCE_DIRECTORY": str(resource),
                    "CONVERSION_SET_ID": "runner_fixture_v3",
                }
            )
            completed = subprocess.run(
                ["bash", str(runner)],
                cwd=str(runner_cwd),
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"stdout={completed.stdout}\nstderr={completed.stderr}",
            )
            document = json.loads(
                (evidence / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                document["schema_version"], "ascify-conversion-evidence-v3"
            )
            self.assertEqual(len(document["conversions"]), 4)
            second = subprocess.run(
                ["bash", str(runner)],
                cwd=str(runner_cwd),
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("refusing to overwrite", second.stderr)

            if os.uname().sysname != "Darwin" and shutil.which("flock"):
                race_evidence = work / "conversion/race evidence"
                race_environment = environment.copy()
                race_environment.update(
                    {
                        "EVIDENCE_ROOT": os.path.relpath(
                            race_evidence, runner_cwd
                        ),
                        "CONVERSION_SET_ID": "runner_race_fixture_v3",
                        "FAKE_ASCIFY_DELAY": "0.2",
                    }
                )
                first = subprocess.Popen(
                    ["bash", str(runner)],
                    cwd=str(runner_cwd),
                    env=race_environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                )
                deadline = time.monotonic() + 5.0
                while (
                    not list(
                        race_evidence.parent.glob(
                            f"{race_evidence.name}.partial.*"
                        )
                    )
                    and first.poll() is None
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.01)
                raced = subprocess.run(
                    ["bash", str(runner)],
                    cwd=str(runner_cwd),
                    env=race_environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                )
                first_stdout, first_stderr = first.communicate(timeout=10)
                self.assertEqual(
                    first.returncode,
                    0,
                    msg=f"stdout={first_stdout}\nstderr={first_stderr}",
                )
                self.assertNotEqual(raced.returncode, 0)
                self.assertIn("[safe-lock] lock is already held:", raced.stderr)
                self.assertTrue((race_evidence / "manifest.json").is_file())

            failed_evidence = work / "conversion/failed evidence"
            failed_environment = environment.copy()
            failed_environment.update(
                {
                    "EVIDENCE_ROOT": os.path.relpath(
                        failed_evidence, runner_cwd
                    ),
                    "CONVERSION_SET_ID": "runner_failure_fixture_v3",
                    "FAKE_ASCIFY_FAIL_ON": "rms_norm.cuh",
                }
            )
            failed = subprocess.run(
                ["bash", str(runner)],
                cwd=str(runner_cwd),
                env=failed_environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("evidence retained", failed.stderr)
            self.assertFalse(failed_evidence.exists())
            partials = list(
                failed_evidence.parent.glob(
                    f"{failed_evidence.name}.partial.*"
                )
            )
            self.assertEqual(len(partials), 1)
            self.assertTrue((partials[0] / "logs/rmsnorm.stderr").is_file())


if __name__ == "__main__":
    unittest.main()
