#!/usr/bin/env python3

from __future__ import annotations

import io
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parent))

from derive_work_metrics import (  # noqa: E402
    ManifestEntry,
    compute_work_metrics,
    derive_rows,
)


PERF_HEADER = (
    "run_id,op,variant,case_id,rows,cols,lat_ms_median,status\n"
)


class WorkMetricFormulaTest(unittest.TestCase):
    def test_semantic_operation_counts(self) -> None:
        softmax = compute_work_metrics("softmax", rows=2, cols=3, affine=False)
        self.assertEqual(softmax.fp32_ops, 16)
        self.assertEqual(softmax.fp16_ops, 0)
        self.assertEqual(softmax.special_ops, 6)
        self.assertEqual(softmax.compare_ops, 4)

        plain = compute_work_metrics("rms_norm", rows=2, cols=3, affine=False)
        self.assertEqual(plain.fp32_ops, 20)
        self.assertEqual(plain.fp16_ops, 0)
        self.assertEqual(plain.special_ops, 2)

        affine = compute_work_metrics("rms_norm", rows=2, cols=3, affine=True)
        self.assertEqual(affine.fp32_ops, 20)
        self.assertEqual(affine.fp16_ops, 6)
        self.assertEqual(affine.arith_ops, 26)

    def test_manifest_resolves_ambiguous_rmsnorm_variant(self) -> None:
        source = io.StringIO(
            PERF_HEADER
            + "run,rms_norm,direct_recipe,rn_case,2,3,0.002,pass\n"
        )
        manifest = {
            ("rms_norm", "rn_case"): ManifestEntry(
                affine=True, rows=2, cols=3, line_number=2
            )
        }
        _, rows, measured = derive_rows(source, Path("input.csv"), manifest)
        self.assertEqual(measured, 1)
        self.assertEqual(rows[0]["metric_model"], "rmsnorm_affine_fwd_v1")
        self.assertEqual(rows[0]["arith_ops"], "26")
        self.assertEqual(float(rows[0]["arith_tflops"]), 0.000013)
        self.assertEqual(float(rows[0]["special_gops"]), 0.001)

    def test_variant_classification_and_nonpass_rates(self) -> None:
        source = io.StringIO(
            PERF_HEADER
            + "run,rms_norm,converted_plain,rn_case,2,3,0,error\n"
        )
        _, rows, measured = derive_rows(source, Path("input.csv"), {})
        self.assertEqual(measured, 0)
        self.assertEqual(rows[0]["metric_model"], "rmsnorm_plain_fwd_v1")
        self.assertEqual(rows[0]["arith_tflops"], "")
        self.assertEqual(rows[0]["special_gops"], "")

    def test_missing_schema_column_is_reported(self) -> None:
        source = io.StringIO(
            "run_id,op,variant,case_id,rows,cols,status\n"
            "run,softmax,converted,sm_case,2,3,pass\n"
        )
        with self.assertRaisesRegex(ValueError, "missing required columns: lat_ms_median"):
            derive_rows(source, Path("input.csv"), {})

    def test_conflicting_affine_sources_are_rejected(self) -> None:
        source = io.StringIO(
            "run_id,op,variant,case_id,rows,cols,lat_ms_median,status,affine\n"
            "run,rms_norm,converted_plain,rn_case,2,3,0.002,pass,1\n"
        )
        with self.assertRaisesRegex(ValueError, "conflicting RMSNorm affine"):
            derive_rows(source, Path("input.csv"), {})

    def test_equivalent_specific_and_generic_manifest_rows_can_coexist(self) -> None:
        source = io.StringIO(
            PERF_HEADER
            + "run,rms_norm,direct_recipe,rn_case,2,3,0.002,pass\n"
        )
        manifest = {
            ("rms_norm", "rn_case"): ManifestEntry(
                affine=True, rows=2, cols=3, line_number=2
            ),
            ("", "rn_case"): ManifestEntry(
                affine=True, rows=2, cols=3, line_number=9
            ),
        }
        _, rows, measured = derive_rows(source, Path("input.csv"), manifest)
        self.assertEqual(measured, 1)
        self.assertEqual(rows[0]["metric_model"], "rmsnorm_affine_fwd_v1")

    def test_conflicting_specific_and_generic_manifest_rows_are_rejected(self) -> None:
        source = io.StringIO(
            PERF_HEADER
            + "run,rms_norm,direct_recipe,rn_case,2,3,0.002,pass\n"
        )
        manifest = {
            ("rms_norm", "rn_case"): ManifestEntry(
                affine=True, rows=2, cols=3, line_number=2
            ),
            ("", "rn_case"): ManifestEntry(
                affine=False, rows=2, cols=3, line_number=9
            ),
        }
        with self.assertRaisesRegex(
            ValueError, "conflicting op-specific and generic rows"
        ):
            derive_rows(source, Path("input.csv"), manifest)


if __name__ == "__main__":
    unittest.main()
