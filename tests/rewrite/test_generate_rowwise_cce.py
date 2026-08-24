#!/usr/bin/env python3
"""Host-only contract tests for the row-wise ascify-clang wrapper CLI."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[2]
GENERATOR = REPO / "tools/generate_rowwise_cce.py"


class GenerateRowwiseCceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="ascify-rowwise-wrapper-"
        )
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root / "rowwise.cu"
        self.source.write_bytes(b"__global__ void rowwise() {}\n")
        self.output = self.root / "rowwise.cce"
        self.report = self.root / "rowwise.report.json"
        self.argv_log = self.root / "ascify-argv.json"
        self.binary = self.root / "ascify-clang"
        self.binary.write_text(
            """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

arguments = sys.argv[1:]
Path(os.environ["FAKE_ASCIFY_ARGV"]).write_text(
    json.dumps(arguments), encoding="utf-8")
destination = None
for index, argument in enumerate(arguments):
    if argument == "-o" and index + 1 < len(arguments):
        destination = arguments[index + 1]
if destination is None:
    raise SystemExit(97)
if os.environ.get("FAKE_ASCIFY_SKIP_OUTPUT") != "1":
    if "FAKE_ASCIFY_OUTPUT_HEX" in os.environ:
        Path(destination).write_bytes(
            bytes.fromhex(os.environ["FAKE_ASCIFY_OUTPUT_HEX"]))
    else:
        Path(destination).write_text(
            os.environ.get("FAKE_ASCIFY_OUTPUT", "generated cce\\n"),
            encoding="utf-8")
raise SystemExit(int(os.environ.get("FAKE_ASCIFY_EXIT", "0")))
""",
            encoding="utf-8",
        )
        self.binary.chmod(self.binary.stat().st_mode | stat.S_IXUSR)
        self.environment = os.environ.copy()
        self.environment.update(
            {
                "ASCIFY_CLANG": str(self.binary),
                "FAKE_ASCIFY_ARGV": str(self.argv_log),
                "FAKE_ASCIFY_OUTPUT": "generated cce\n",
                "FAKE_ASCIFY_EXIT": "0",
            }
        )

    def run_generator(
        self,
        mode: str,
        *extra: str,
        provide_binary: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        command = [
            sys.executable,
            "-I",
            "-B",
            str(GENERATOR),
            str(self.source),
            str(self.output),
            "--mode",
            mode,
            "--report",
            str(self.report),
        ]
        if provide_binary:
            command.extend(["--ascify-clang", str(self.binary)])
        command.extend(extra)
        return subprocess.run(
            command,
            cwd=REPO,
            env=self.environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def read_argv(self) -> list[str]:
        return json.loads(self.argv_log.read_text(encoding="utf-8"))

    def read_report(self) -> dict[str, object]:
        return json.loads(self.report.read_text(encoding="utf-8"))

    def test_simt_invocation_and_compiler_argument_passthrough(self) -> None:
        include = self.root / "include"
        include.mkdir()
        cuda_path = self.root / "cuda toolkit"
        clang_resource_directory = self.root / "clang resource"
        result = self.run_generator(
            "simt",
            "--cuda-path",
            str(cuda_path),
            "--clang-resource-directory",
            str(clang_resource_directory),
            "--include",
            str(include),
            "--compiler-arg=-std=c++17",
            "--",
            "-DROW_COUNT=8",
            provide_binary=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = self.read_argv()
        self.assertEqual(
            arguments[:6],
            [
                str(self.source),
                "--target-policy=portable",
                "--simt-math=precise",
                f"--cuda-path={cuda_path}",
                f"--clang-resource-directory={clang_resource_directory}",
                "-o",
            ],
        )
        self.assertEqual(Path(arguments[6]).name, self.output.name)
        self.assertNotEqual(Path(arguments[6]), self.output)
        self.assertEqual(
            arguments[7:],
            ["--", f"-I{include}", "-std=c++17", "-DROW_COUNT=8"],
        )
        separator = arguments.index("--")
        self.assertLess(arguments.index(f"--cuda-path={cuda_path}"), separator)
        self.assertLess(
            arguments.index(
                f"--clang-resource-directory={clang_resource_directory}"
            ),
            separator,
        )
        self.assertNotIn("--target-recipe=dav-3510-rowwise-simd-v1", arguments)
        self.assertEqual(self.output.read_bytes(), b"generated cce\n")

        report = self.read_report()
        self.assertEqual(report["mode"], "simt")
        self.assertEqual(report["target_abi"], 1)
        self.assertEqual(report["input_kernel_launch_count"], 0)
        self.assertFalse(report["recognized"])
        self.assertEqual(
            report["input_sha256"], hashlib.sha256(self.source.read_bytes()).hexdigest()
        )
        self.assertEqual(
            report["output_sha256"], hashlib.sha256(self.output.read_bytes()).hexdigest()
        )
        self.assertEqual(report["schema"], "ascify.rowwise-cce-report.v3")
        self.assertEqual(
            report["ascify_binary_sha256"],
            hashlib.sha256(self.binary.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            report["generator_sha256"],
            hashlib.sha256(GENERATOR.read_bytes()).hexdigest(),
        )
        self.assertEqual(report["normalized_converter_argv"][0], "$ASCIFY_CLANG")
        self.assertEqual(report["normalized_converter_argv"][1], "$SOURCE")
        self.assertIn("$OUTPUT", report["normalized_converter_argv"])
        self.assertRegex(report["conversion_id"], r"^sha256:[0-9a-f]{64}$")

    def test_publishes_frontend_bytes_without_review_payload_lookup(self) -> None:
        frontend_output = "fake ascify output only: 7ad52a93\n"
        self.environment["FAKE_ASCIFY_OUTPUT"] = frontend_output
        result = self.run_generator("simt")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.argv_log.exists())
        self.assertEqual(self.output.read_text(encoding="utf-8"), frontend_output)
        generator_source = GENERATOR.read_text(encoding="utf-8")
        self.assertNotIn("generate_rowwise_review_cce", generator_source)
        self.assertNotIn("operator_recipes/payloads", generator_source)

    def test_simd_simt_invocation_counts_markers_and_requires_recipe(self) -> None:
        self.source.write_text(
            "__global__ void rowwise() {}\n"
            "void launch_one() { rowwise<<<1, 1>>>(); }\n"
            "void launch_two() { rowwise<<<1, 1>>>(); }\n"
            "void launch_three() { rowwise<<<1, 1>>>(); }\n",
            encoding="utf-8",
        )
        self.environment["FAKE_ASCIFY_OUTPUT"] = """// generated
#include <ascify/target/dav_c310/rowwise_simd_recipes.hpp>
aclError launch_one() {
  if ((rows > 0) && (cols > 0) && (cols % pack_size == 0)) {
    if constexpr (algorithm == ::oneflow::cuda::softmax::Algorithm::kSoftmax) {
      const auto softmax_one = ::ascify::target::dav_c310::rowwise_simd_v1::RowwiseHybridFacadeV1::TrySoftmaxHybrid(stream, load, store, rows, cols, static_cast<ComputeType*>(nullptr));
      if (softmax_one.handled) { return softmax_one.status; }
    }
  }
  kernel<<<1, 1>>>(ctx);
  return 0;
}
aclError launch_two() {
  if ((rows > 0) && (cols > 0) && (cols % pack_size == 0)) {
    if constexpr (algorithm == ::oneflow::cuda::softmax::Algorithm::kSoftmax) {
      const auto softmax_two = ::ascify::target::dav_c310::rowwise_simd_v1::RowwiseHybridFacadeV1::TrySoftmaxHybrid(stream, load, store, rows, cols, static_cast<ComputeType*>(nullptr));
      if (softmax_two.handled) {
        return softmax_two.status;
      }
    }
  }
  kernel<<<1, 1>>>(other);
  return 0;
}
aclError launch_three() {
  if ((nrow > 0) && (ncol > 0) && (ncol % pack_size == 0)) {
    const auto rmsnorm_one = ::ascify::target::dav_c310::rowwise_simd_v1::RowwiseHybridFacadeV1::TryRmsNormHybrid(stream, load, store, nrow, ncol, eps, inv_rms, static_cast<ComputeType*>(nullptr));
    if (rmsnorm_one.handled) { return rmsnorm_one.status; }
  }
  kernel<<<1, 1>>>(ctx);
  return 0;
}
"""
        result = self.run_generator("simd-simt", "--require-recipe")

        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = self.read_argv()
        self.assertEqual(
            arguments[:4],
            [
                str(self.source),
                "--target-policy=dav-c310-vec",
                "--simt-math=fast",
                "--target-recipe=dav-3510-rowwise-simd-v1",
            ],
        )
        self.assertEqual(arguments[4], "-o")
        self.assertEqual(arguments[6:], ["--"])
        report = self.read_report()
        self.assertTrue(report["recognized"])
        self.assertEqual(report["schema"], "ascify.rowwise-cce-report.v3")
        self.assertEqual(report["input_kernel_launch_count"], 3)
        self.assertEqual(report["simd_header_count"], 1)
        self.assertEqual(report["kernel_launch_count"], 3)
        self.assertEqual(report["recipe_call_token_count"], 3)
        self.assertEqual(report["dispatch_with_simt_fallback_count"], 3)
        self.assertEqual(report["dispatch_function_count"], 3)
        self.assertEqual(
            report["dispatch_execution_kind"], "intra-kernel-simd-simt"
        )
        self.assertEqual(report["try_softmax_hybrid_count"], 2)
        self.assertEqual(report["try_softmax_simd_count"], 2)
        self.assertEqual(report["try_rmsnorm_hybrid_count"], 1)
        self.assertEqual(report["try_rmsnorm_simd_count"], 1)

    def test_generated_guard_validation_is_name_independent(self) -> None:
        self.source.write_text(
            "__global__ void one() {}\n"
            "__global__ void two() {}\n"
            "void launch_one() { one<<<1, 1>>>(); }\n"
            "void launch_two() { two<<<1, 1>>>(); }\n",
            encoding="utf-8",
        )
        self.environment["FAKE_ASCIFY_OUTPUT"] = """// generated
#include <ascify/target/dav_c310/rowwise_simd_recipes.hpp>
aclError renamed_softmax() {
  if ((row_total > 0) && (column_total > 0) && (column_total % vector_width == 0) && (column_total <= (items_per_lane * lane_count)) && (has_padding || column_total == (items_per_lane * lane_count)) && (row_total % rows_each == 0)) {
    if constexpr (mode_choice == ::renamed::ops::Flavor::kVectorForward) {
      const auto selected = ::ascify::target::dav_c310::rowwise_simd_v1::RowwiseHybridFacadeV1::TrySoftmaxHybrid(queue, reader, writer, row_total, column_total, static_cast<Accumulator*>(nullptr));
      if (selected.handled) { return selected.status; }
    }
  }
  softmax_kernel<<<1, 1>>>(queue);
  return 0;
}
aclError renamed_rmsnorm() {
  if ((batch_rows > 0) && (feature_columns > 0) && (feature_columns % packed_elements == 0) && (scratch_bytes >= 0) && (static_cast<unsigned long long>(feature_columns) <= static_cast<unsigned long long>(scratch_bytes) / sizeof(RmsAccumulator))) {
    const auto selected = ::ascify::target::dav_c310::rowwise_simd_v1::RowwiseHybridFacadeV1::TryRmsNormHybrid(queue, reader, writer, batch_rows, feature_columns, epsilon_value, inverse_values, static_cast<RmsAccumulator*>(nullptr));
    if (selected.handled) { return selected.status; }
  }
  rms_kernel<<<1, 1>>>(queue);
  return 0;
}
"""
        result = self.run_generator("simd-simt", "--require-recipe")

        self.assertEqual(result.returncode, 0, result.stderr)
        report = self.read_report()
        self.assertTrue(report["recognized"])
        self.assertEqual(report["dispatch_function_count"], 2)
        self.assertEqual(
            report["dispatch_execution_kind"], "intra-kernel-simd-simt"
        )
        self.assertEqual(report["try_softmax_hybrid_count"], 1)
        self.assertEqual(report["try_softmax_simd_count"], 1)
        self.assertEqual(report["try_rmsnorm_hybrid_count"], 1)
        self.assertEqual(report["try_rmsnorm_simd_count"], 1)

    def test_malformed_simd_include_is_not_counted(self) -> None:
        self.source.write_text(
            "__global__ void rowwise() {}\n"
            "void launch() { rowwise<<<1, 1>>>(); }\n",
            encoding="utf-8",
        )
        complete = """aclError wrapper() {
  if ((rows > 0) && (cols > 0) && (cols % pack_size == 0)) {
    if constexpr (algorithm == ::oneflow::cuda::softmax::Algorithm::kSoftmax) {
      const auto result = ::ascify::target::dav_c310::rowwise_simd_v1::RowwiseHybridFacadeV1::TrySoftmaxHybrid(stream, load, store, rows, cols, static_cast<ComputeType*>(nullptr));
      if (result.handled) { return result.status; }
    }
  }
  kernel<<<1, 1>>>(ctx);
  return 0;
}
"""
        malformed_headers = {
            "multiline header name": "#include <\nascify/target/dav_c310/rowwise_simd_recipes.hpp\n>\n",
            "spaced header name": "#include < ascify/target/dav_c310/rowwise_simd_recipes.hpp >\n",
        }
        for label, header in malformed_headers.items():
            with self.subTest(label=label):
                self.environment["FAKE_ASCIFY_OUTPUT"] = header + complete
                result = self.run_generator("simd-simt", "--require-recipe")
                self.assertEqual(result.returncode, 2)
                self.assertIn("structurally valid SIMD dispatch", result.stderr)
                self.assertFalse(self.output.exists())
                self.assertFalse(self.report.exists())

    def test_comments_literals_and_incomplete_calls_are_not_recognized(self) -> None:
        self.source.write_text(
            "__global__ void rowwise() {}\n"
            "void launch() { rowwise<<<1, 1>>>(); }\n",
            encoding="utf-8",
        )
        self.environment["FAKE_ASCIFY_OUTPUT"] = r'''// generated
#include <ascify/target/dav_c310/rowwise_simd_recipes.hpp>
// const auto fake = ::ascify::target::dav_c310::rowwise_simd_v1::RowwiseHybridFacadeV1::TrySoftmaxHybrid(ctx);
// if (fake.handled) { return fake.status; }
const char* text = "::ascify::target::dav_c310::rowwise_simd_v1::RowwiseHybridFacadeV1::TryRmsNormHybrid(ctx)";
const char* raw = R"tag(const auto raw_fake =
  ::ascify::target::dav_c310::rowwise_simd_v1::RowwiseHybridFacadeV1::TrySoftmaxHybrid(ctx);
  if (raw_fake.handled) { return raw_fake.status; })tag";
const auto incomplete = ::ascify::target::dav_c310::rowwise_simd_v1::RowwiseHybridFacadeV1::TrySoftmaxHybrid(ctx);
kernel<<<1, 1>>>(ctx);
'''
        result = self.run_generator("simd-simt", "--require-recipe")

        self.assertEqual(result.returncode, 2)
        self.assertIn("structurally valid SIMD dispatch", result.stderr)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.report.exists())

    def test_recipe_proof_rejects_scope_and_lexical_bypasses(self) -> None:
        self.source.write_text(
            "__global__ void rowwise() {}\n"
            "void launch() { rowwise<<<1, 1>>>(); }\n",
            encoding="utf-8",
        )
        header = "#include <ascify/target/dav_c310/rowwise_simd_recipes.hpp>\n"
        complete = """aclError valid_wrapper() {
  if ((rows > 0) && (cols > 0) && (cols % pack_size == 0)) {
    if constexpr (algorithm == ::oneflow::cuda::softmax::Algorithm::kSoftmax) {
      const auto valid = ::ascify::target::dav_c310::rowwise_simd_v1::RowwiseHybridFacadeV1::TrySoftmaxHybrid(stream, load, store, rows, cols, static_cast<ComputeType*>(nullptr));
      if (valid.handled) { return valid.status; }
    }
  }
  kernel<<<1, 1>>>(ctx);
  return 0;
}
"""
        guarded_contract = """if ((rows > 0) && (cols > 0) && (cols % pack_size == 0)) {
  if constexpr (algorithm == ::oneflow::cuda::softmax::Algorithm::kSoftmax) {
    const auto result = ::ascify::target::dav_c310::rowwise_simd_v1::RowwiseHybridFacadeV1::TrySoftmaxHybrid(stream, load, store, rows, cols, static_cast<ComputeType*>(nullptr));
    if (result.handled) { return result.status; }
  }
}
"""
        cases = {
            "extra incomplete call": complete
            + "const auto extra = "
            "::ascify::target::dav_c310::rowwise_simd_v1::RowwiseHybridFacadeV1::TryRmsNormHybrid(ctx);\n",
            "fallback in unrelated function": """void unrelated() {
  kernel<<<1, 1>>>(ctx);
}
aclError no_fallback_wrapper() {
  const auto result = ::ascify::target::dav_c310::rowwise_simd_v1::RowwiseHybridFacadeV1::TrySoftmaxHybrid(ctx);
  if (result.handled) { return result.status; }
  return 0;
}
""",
            "fallback before dispatch": """aclError wrong_order_wrapper() {
  kernel<<<1, 1>>>(ctx);
  const auto result = ::ascify::target::dav_c310::rowwise_simd_v1::RowwiseHybridFacadeV1::TrySoftmaxHybrid(ctx);
  if (result.handled) { return result.status; }
  return 0;
}
""",
            "continued line comment": """// hidden recipe \\
const auto hidden = ::ascify::target::dav_c310::rowwise_simd_v1::RowwiseHybridFacadeV1::TrySoftmaxHybrid(ctx); \\
if (hidden.handled) { return hidden.status; }
void fallback() { kernel<<<1, 1>>>(ctx); }
""",
            "inactive preprocessor branch": """#if 0
aclError hidden_wrapper() {
  const auto hidden = ::ascify::target::dav_c310::rowwise_simd_v1::RowwiseHybridFacadeV1::TrySoftmaxHybrid(ctx);
  if (hidden.handled) { return hidden.status; }
  kernel<<<1, 1>>>(ctx);
  return 0;
}
#endif
void fallback() { kernel<<<1, 1>>>(ctx); }
""",
            "include guard alternative": """#ifndef ROWWISE_TEST_H_
#define ROWWISE_TEST_H_
#else
#include <ascify/target/dav_c310/rowwise_simd_recipes.hpp>
aclError hidden_alternative() {
  const auto hidden = ::ascify::target::dav_c310::rowwise_simd_v1::RowwiseHybridFacadeV1::TrySoftmaxHybrid(ctx);
  if (hidden.handled) { return hidden.status; }
  kernel<<<1, 1>>>(ctx);
  return 0;
}
#endif
""",
            "fallback in false branch": """aclError nested_fallback() {
  const auto result = ::ascify::target::dav_c310::rowwise_simd_v1::RowwiseHybridFacadeV1::TrySoftmaxHybrid(ctx);
  if (result.handled) { return result.status; }
  if (false) { kernel<<<1, 1>>>(ctx); }
  return 0;
}
""",
            "fallback in later lambda": """aclError lambda_fallback() {
  const auto result = ::ascify::target::dav_c310::rowwise_simd_v1::RowwiseHybridFacadeV1::TrySoftmaxHybrid(ctx);
  if (result.handled) { return result.status; }
  auto launch_later = []() { kernel<<<1, 1>>>(ctx); };
  return 0;
}
""",
            "dispatch in false branch": """aclError dead_dispatch() {
  if (false) {
    const auto result = ::ascify::target::dav_c310::rowwise_simd_v1::RowwiseHybridFacadeV1::TrySoftmaxHybrid(ctx);
    if (result.handled) { return result.status; }
  }
  kernel<<<1, 1>>>(ctx);
  return 0;
}
""",
            "dispatch in uncalled lambda": """aclError lambda_dispatch() {
  auto unused = []() {
    const auto result = ::ascify::target::dav_c310::rowwise_simd_v1::RowwiseHybridFacadeV1::TrySoftmaxHybrid(ctx);
    if (result.handled) { return result.status; }
    kernel<<<1, 1>>>(ctx);
    return 0;
  };
  return 0;
}
""",
            "digraph inactive branch": """%:if 0
aclError digraph_hidden() {
  const auto result = ::ascify::target::dav_c310::rowwise_simd_v1::RowwiseHybridFacadeV1::TrySoftmaxHybrid(ctx);
  if (result.handled) { return result.status; }
  kernel<<<1, 1>>>(ctx);
  return 0;
}
%:endif
""",
            "Softmax dispatch hidden by digraph braces": """aclError digraph_softmax_scope() {
  if ((rows > 0) && (cols > 0) && (cols % pack_size == 0)) {
    if constexpr (algorithm == ::oneflow::cuda::softmax::Algorithm::kSoftmax) {
      if (false) <%
        const auto result = ::ascify::target::dav_c310::rowwise_simd_v1::RowwiseHybridFacadeV1::TrySoftmaxHybrid(stream, load, store, rows, cols, static_cast<ComputeType*>(nullptr));
        if (result.handled) { return result.status; }
      %>
    }
  }
  kernel<<<1, 1>>>(ctx);
  return 0;
}
""",
            "RMSNorm dispatch hidden by digraph braces": """aclError digraph_rmsnorm_scope() {
  if ((nrow > 0) && (ncol > 0) && (ncol % pack_size == 0) && (smem >= 0) && (static_cast<unsigned long long>(ncol) <= static_cast<unsigned long long>(smem) / sizeof(ComputeType))) {
    if (false) <%
      const auto result = ::ascify::target::dav_c310::rowwise_simd_v1::RowwiseHybridFacadeV1::TryRmsNormHybrid(stream, load, store, nrow, ncol, eps, inv_rms, static_cast<ComputeType*>(nullptr));
      if (result.handled) { return result.status; }
    %>
  }
  kernel<<<1, 1>>>(ctx);
  return 0;
}
""",
            "Softmax prior return between fake raw comments": """aclError fake_raw_softmax() {
  // R"(
  return 91;
  // )"
"""
            + guarded_contract
            + """  kernel<<<1, 1>>>(ctx);
}
""",
            "RMSNorm prior return between fake raw comments": """aclError fake_raw_rmsnorm() {
  // R"tag(
  return 92;
  // )tag"
  if ((nrow > 0) && (ncol > 0) && (ncol % pack_size == 0) && (smem >= 0) && (static_cast<unsigned long long>(ncol) <= static_cast<unsigned long long>(smem) / sizeof(ComputeType))) {
    const auto result = ::ascify::target::dav_c310::rowwise_simd_v1::RowwiseHybridFacadeV1::TryRmsNormHybrid(stream, load, store, nrow, ncol, eps, inv_rms, static_cast<ComputeType*>(nullptr));
    if (result.handled) { return result.status; }
  }
  kernel<<<1, 1>>>(ctx);
}
""",
            "prior return between C++ digit-separated literals": """aclError digit_separator_return() {
  const auto before = 1'000;
  return 93;
  const auto after = 2'000;
"""
            + guarded_contract
            + """  kernel<<<1, 1>>>(ctx);
}
""",
            "prior return between u8 character literals": """aclError u8_character_return() {
  const auto before = u8'0';
  return 94;
  const auto after = u8'1';
"""
            + guarded_contract
            + """  kernel<<<1, 1>>>(ctx);
}
""",
            "vertical tab keeps recipe in macro directive": "#define HIDDEN\v"
            "aclError vt_macro(){if((rows>0)&&(cols>0)&&(cols%pack_size==0)){"
            "if constexpr(algorithm==::oneflow::cuda::softmax::Algorithm::kSoftmax){"
            "const auto result=::ascify::target::dav_c310::rowwise_simd_v1::"
            "RowwiseHybridFacadeV1::TrySoftmaxHybrid(stream,load,store,rows,cols,"
            "static_cast<ComputeType*>(nullptr));if(result.handled){return result.status;}"
            "}}kernel<<<1,1>>>(ctx);return 0;}\n",
            "form feed keeps recipe in macro directive": "#define HIDDEN\f"
            "aclError ff_macro(){if((rows>0)&&(cols>0)&&(cols%pack_size==0)){"
            "if constexpr(algorithm==::oneflow::cuda::softmax::Algorithm::kSoftmax){"
            "const auto result=::ascify::target::dav_c310::rowwise_simd_v1::"
            "RowwiseHybridFacadeV1::TrySoftmaxHybrid(stream,load,store,rows,cols,"
            "static_cast<ComputeType*>(nullptr));if(result.handled){return result.status;}"
            "}}kernel<<<1,1>>>(ctx);return 0;}\n",
            "unconditional return before dispatch": """aclError returned() {
  return 0;
"""
            + guarded_contract
            + """  kernel<<<1, 1>>>(ctx);
}
""",
            "dispatch in false for": """aclError false_for() {
  for (; false;) {
"""
            + guarded_contract
            + """  }
  kernel<<<1, 1>>>(ctx);
  return 0;
}
""",
            "dispatch in true-if else": """aclError false_else() {
  if (true) {}
  else {
"""
            + guarded_contract
            + """  }
  kernel<<<1, 1>>>(ctx);
  return 0;
}
""",
            "dispatch in impossible switch case": """aclError false_case() {
  switch (0) {
    case 1:
"""
            + guarded_contract
            + """      break;
  }
  kernel<<<1, 1>>>(ctx);
  return 0;
}
""",
            "goto skips dispatch": """aclError skipped_by_goto() {
  goto skip;
"""
            + guarded_contract
            + """  kernel<<<1, 1>>>(ctx);
skip:
  return 0;
}
""",
            "one-line macro recipe": """aclError macro_only() {
#define FAKE const auto result = ::ascify::target::dav_c310::rowwise_simd_v1::RowwiseHybridFacadeV1::TrySoftmaxHybrid(ctx); if (result.handled) { return result.status; } kernel<<<1, 1>>>(ctx);
  return 0;
}
""",
            "dispatch in hexadecimal-zero while": """aclError false_while() {
  while (0x0) {
"""
            + guarded_contract
            + """  }
  kernel<<<1, 1>>>(ctx);
  return 0;
}
""",
            "dispatch in braceless false for": """aclError braceless_for() {
  for (; false;)
"""
            + guarded_contract
            + """  kernel<<<1, 1>>>(ctx);
  return 0;
}
""",
            "dispatch in braceless hexadecimal-zero while": """aclError braceless_while() {
  while (0x0)
"""
            + guarded_contract
            + """  kernel<<<1, 1>>>(ctx);
  return 0;
}
""",
            "goto skips fallback": """aclError goto_after_dispatch() {
"""
            + guarded_contract
            + """  goto skip;
  kernel<<<1, 1>>>(ctx);
skip:
  return 0;
}
""",
            "continued macro recipe": """aclError continued_macro_only() {
#define FAKE const auto result = ::ascify::target::dav_c310::rowwise_simd_v1::RowwiseHybridFacadeV1::TrySoftmaxHybrid(ctx); \\
  if (result.handled) { return result.status; } kernel<<<1, 1>>>(ctx);
  return 0;
}
""",
            "phase-2 spliced block comment opener": "/\\\n*" + complete + "*/\n",
            "phase-2 spliced return": "aclError spliced_return() {\n  ret\\\nurn 0;\n"
            + guarded_contract
            + "  kernel<<<1, 1>>>(ctx);\n}\n",
            "phase-2 spliced goto": "aclError spliced_goto() {\n  go\\\nto skip;\n"
            + guarded_contract
            + "  kernel<<<1, 1>>>(ctx);\nskip:\n  return 0;\n}\n",
            "phase-2 spaced spliced return": "aclError spaced_spliced_return() {\n  ret\\   \nurn 0;\n"
            + guarded_contract
            + "  kernel<<<1, 1>>>(ctx);\n}\n",
            "phase-2 spaced spliced goto": "aclError spaced_spliced_goto() {\n  go\\\t \nto skip;\n"
            + guarded_contract
            + "  kernel<<<1, 1>>>(ctx);\nskip:\n  return 0;\n}\n",
            "phase-2 spaced continued line comment": "// hidden recipe \\   \n"
            + complete,
            "phase-2 splice inside raw string": (
                'const char* raw = R"tag(\n)ta\\\ng"\n'
                + complete
                + ')tag";\n'
            ),
            "phase-2 manufactured raw opener": (
                'const char* raw = R\\\n"tag(\n)ta\\\ng"\n'
                + complete
                + ')tag";\n'
            ),
            "fallback in unevaluated sizeof": "aclError sizeof_fallback() {\n"
            + guarded_contract
            + "  (void)sizeof((kernel<<<1, 1>>>(ctx), 0));\n  return 0;\n}\n",
            "elifdef inactive recipe": """#if 1
void active() { kernel<<<1, 1>>>(ctx); }
#elifdef HIDDEN_RECIPE
aclError hidden_elifdef() {
"""
            + guarded_contract
            + """  kernel<<<1, 1>>>(ctx);
  return 0;
}
#endif
""",
            "guard rows differ from dispatch rows": """aclError mismatched_rows() {
  if ((guard_rows > 0) && (cols > 0) && (cols % pack_size == 0)) {
    if constexpr (algorithm == ::renamed::Mode::kSoftmax) {
      const auto result = ::ascify::target::dav_c310::rowwise_simd_v1::RowwiseHybridFacadeV1::TrySoftmaxHybrid(stream, load, store, actual_rows, cols, static_cast<ComputeType*>(nullptr));
      if (result.handled) { return result.status; }
    }
  }
  kernel<<<1, 1>>>(ctx);
  return 0;
}
""",
            "warp suffix identifiers are inconsistent": """aclError mismatched_width() {
  if ((rows > 0) && (cols > 0) && (cols % pack_size == 0) && (cols <= (maximum * lane_width)) && (padding || cols == (maximum * other_width)) && (rows % rows_each == 0)) {
    if constexpr (algorithm == ::renamed::Mode::kSoftmax) {
      const auto result = ::ascify::target::dav_c310::rowwise_simd_v1::RowwiseHybridFacadeV1::TrySoftmaxHybrid(stream, load, store, rows, cols, static_cast<ComputeType*>(nullptr));
      if (result.handled) { return result.status; }
    }
  }
  kernel<<<1, 1>>>(ctx);
  return 0;
}
""",
            "shared guard type differs from dispatch type": """aclError mismatched_type() {
  if ((nrow > 0) && (ncol > 0) && (ncol % pack_size == 0) && (smem >= 0) && (static_cast<unsigned long long>(ncol) <= static_cast<unsigned long long>(smem) / sizeof(GuardType))) {
    const auto result = ::ascify::target::dav_c310::rowwise_simd_v1::RowwiseHybridFacadeV1::TryRmsNormHybrid(stream, load, store, nrow, ncol, eps, inv_rms, static_cast<CallType*>(nullptr));
    if (result.handled) { return result.status; }
  }
  kernel<<<1, 1>>>(ctx);
  return 0;
}
""",
            "nested unconditional return before dispatch": """aclError unreachable_dispatch() {
  if (true) { return 0; }
"""
            + guarded_contract
            + """  kernel<<<1, 1>>>(ctx);
}
""",
            "nested throw before dispatch": """aclError unreachable_throw() {
  if (true) { throw 1; }
"""
            + guarded_contract
            + """  kernel<<<1, 1>>>(ctx);
}
""",
            "nested unconditional return before fallback": """aclError unreachable_fallback() {
"""
            + guarded_contract
            + """  if (true) { return 0; }
  kernel<<<1, 1>>>(ctx);
}
""",
        }

        for label, body in cases.items():
            with self.subTest(label=label):
                self.environment["FAKE_ASCIFY_OUTPUT"] = header + body
                result = self.run_generator("simd-simt", "--require-recipe")
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("structurally valid SIMD dispatch", result.stderr)
                self.assertFalse(self.output.exists())
                self.assertFalse(self.report.exists())

    def test_preexisting_recipe_code_is_rejected_before_converter(self) -> None:
        self.source.write_text(
            "#include <ascify/target/dav_c310/rowwise_simd_recipes.hpp>\n"
            "const auto existing = "
            "::ascify::target::dav_c310::rowwise_simd_v1::RowwiseHybridFacadeV1::TrySoftmaxHybrid(ctx);\n"
            "if (existing.handled) { return existing.status; }\n"
            "kernel<<<1, 1>>>(ctx);\n",
            encoding="utf-8",
        )
        result = self.run_generator("simd-simt", "--require-recipe")

        self.assertEqual(result.returncode, 2)
        self.assertIn("source already contains row-wise SIMD recipe code", result.stderr)
        self.assertFalse(self.argv_log.exists())
        self.assertFalse(self.output.exists())
        self.assertFalse(self.report.exists())

    def test_trigraph_spellings_are_rejected_before_publication(self) -> None:
        self.source.write_text(
            "// phase-1 trigraph: ??=\n"
            "__global__ void rowwise() {}\n"
            "void launch() { rowwise<<<1, 1>>>(); }\n",
            encoding="utf-8",
        )
        source_result = self.run_generator(
            "simd-simt", "--require-recipe", "--compiler-arg=-trigraphs"
        )
        self.assertEqual(source_result.returncode, 2)
        self.assertIn("source contains a C/C++ trigraph spelling", source_result.stderr)
        self.assertFalse(self.argv_log.exists())
        self.assertFalse(self.output.exists())
        self.assertFalse(self.report.exists())

        self.source.write_text(
            "__global__ void rowwise() {}\n"
            "void launch() { rowwise<<<1, 1>>>(); }\n",
            encoding="utf-8",
        )
        header = "#include <ascify/target/dav_c310/rowwise_simd_recipes.hpp>\n"
        complete = """aclError hidden_wrapper() {
  if ((rows > 0) && (cols > 0) && (cols % pack_size == 0)) {
    if constexpr (algorithm == ::oneflow::cuda::softmax::Algorithm::kSoftmax) {
      const auto hidden = ::ascify::target::dav_c310::rowwise_simd_v1::RowwiseHybridFacadeV1::TrySoftmaxHybrid(stream, load, store, rows, cols, static_cast<ComputeType*>(nullptr));
      if (hidden.handled) { return hidden.status; }
    }
  }
  kernel<<<1, 1>>>(ctx);
  return 0;
}
"""
        outputs = {
            "inactive trigraph directive": header
            + "??=if 0\n"
            + complete
            + "??=endif\n",
            "trigraph manufactured line splice": header
            + "// hidden recipe ??/\n"
            + complete,
        }
        for label, output in outputs.items():
            with self.subTest(label=label):
                self.environment["FAKE_ASCIFY_OUTPUT"] = output
                result = self.run_generator(
                    "simd-simt",
                    "--require-recipe",
                    "--compiler-arg=-trigraphs",
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(
                    "ascify-clang output contains a C/C++ trigraph spelling",
                    result.stderr,
                )
                self.assertFalse(self.output.exists())
                self.assertFalse(self.report.exists())

    def test_utf8_bom_is_rejected_before_publication(self) -> None:
        self.source.write_bytes(
            b"\xef\xbb\xbf"
            b"__global__ void rowwise() {}\n"
            b"void launch() { rowwise<<<1, 1>>>(); }\n"
        )
        source_result = self.run_generator("simd-simt", "--require-recipe")
        self.assertEqual(source_result.returncode, 2)
        self.assertIn("source starts with a UTF-8 BOM", source_result.stderr)
        self.assertFalse(self.argv_log.exists())
        self.assertFalse(self.output.exists())
        self.assertFalse(self.report.exists())

        self.source.write_text(
            "__global__ void rowwise() {}\n"
            "void launch() { rowwise<<<1, 1>>>(); }\n",
            encoding="utf-8",
        )
        self.environment["FAKE_ASCIFY_OUTPUT"] = """\ufeff#if 0
#include <ascify/target/dav_c310/rowwise_simd_recipes.hpp>
aclError hidden_by_bom() {
  if ((rows > 0) && (cols > 0) && (cols % pack_size == 0)) {
    if constexpr (algorithm == ::oneflow::cuda::softmax::Algorithm::kSoftmax) {
      const auto hidden = ::ascify::target::dav_c310::rowwise_simd_v1::RowwiseHybridFacadeV1::TrySoftmaxHybrid(stream, load, store, rows, cols, static_cast<ComputeType*>(nullptr));
      if (hidden.handled) { return hidden.status; }
    }
  }
  kernel<<<1, 1>>>(ctx);
  return 0;
}
#endif
"""
        output_result = self.run_generator("simd-simt", "--require-recipe")
        self.assertEqual(output_result.returncode, 2)
        self.assertIn(
            "ascify-clang output starts with a UTF-8 BOM",
            output_result.stderr,
        )
        self.assertFalse(self.output.exists())
        self.assertFalse(self.report.exists())

    def test_nul_byte_is_rejected_before_publication(self) -> None:
        self.source.write_bytes(
            b"\0__global__ void rowwise() {}\n"
            b"void launch() { rowwise<<<1, 1>>>(); }\n"
        )
        source_result = self.run_generator("simd-simt", "--require-recipe")
        self.assertEqual(source_result.returncode, 2)
        self.assertIn("source contains a NUL byte", source_result.stderr)
        self.assertFalse(self.argv_log.exists())
        self.assertFalse(self.output.exists())
        self.assertFalse(self.report.exists())

        self.source.write_text(
            "__global__ void rowwise() {}\n"
            "void launch() { rowwise<<<1, 1>>>(); }\n",
            encoding="utf-8",
        )
        nul_output = """\0#if 0
#include <ascify/target/dav_c310/rowwise_simd_recipes.hpp>
aclError hidden_by_nul() {
  if ((rows > 0) && (cols > 0) && (cols % pack_size == 0)) {
    if constexpr (algorithm == ::oneflow::cuda::softmax::Algorithm::kSoftmax) {
      const auto hidden = ::ascify::target::dav_c310::rowwise_simd_v1::RowwiseHybridFacadeV1::TrySoftmaxHybrid(stream, load, store, rows, cols, static_cast<ComputeType*>(nullptr));
      if (hidden.handled) { return hidden.status; }
    }
  }
  kernel<<<1, 1>>>(ctx);
  return 0;
}
#endif
"""
        self.environment["FAKE_ASCIFY_OUTPUT_HEX"] = nul_output.encode("utf-8").hex()
        output_result = self.run_generator("simd-simt", "--require-recipe")
        self.assertEqual(output_result.returncode, 2)
        self.assertIn(
            "ascify-clang output contains a NUL byte",
            output_result.stderr,
        )
        self.assertFalse(self.output.exists())
        self.assertFalse(self.report.exists())

    def test_invalid_utf8_is_rejected_before_publication(self) -> None:
        self.source.write_bytes(
            b"\xff__global__ void rowwise() {}\n"
            b"void launch() { rowwise<<<1, 1>>>(); }\n"
        )
        source_result = self.run_generator("simd-simt", "--require-recipe")
        self.assertEqual(source_result.returncode, 2)
        self.assertIn("source is not valid UTF-8", source_result.stderr)
        self.assertFalse(self.argv_log.exists())
        self.assertFalse(self.output.exists())
        self.assertFalse(self.report.exists())

        self.source.write_text(
            "__global__ void rowwise() {}\n"
            "void launch() { rowwise<<<1, 1>>>(); }\n",
            encoding="utf-8",
        )
        invalid_output = b"\xffplain output\n"
        self.environment["FAKE_ASCIFY_OUTPUT_HEX"] = invalid_output.hex()
        output_result = self.run_generator("simd-simt", "--require-recipe")
        self.assertEqual(output_result.returncode, 2)
        self.assertIn(
            "ascify-clang output is not valid UTF-8",
            output_result.stderr,
        )
        self.assertFalse(self.output.exists())
        self.assertFalse(self.report.exists())

    def test_unrecognized_output_is_reported_without_requirement(self) -> None:
        self.environment["FAKE_ASCIFY_OUTPUT"] = "plain SIMT output\n"
        result = self.run_generator("simd-simt")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.output.read_text(encoding="utf-8"), "plain SIMT output\n")
        report = self.read_report()
        self.assertFalse(report["recognized"])
        self.assertEqual(report["dispatch_execution_kind"], "none")
        self.assertEqual(report["try_softmax_hybrid_count"], 0)
        self.assertEqual(report["try_softmax_simd_count"], 0)
        self.assertEqual(report["try_rmsnorm_hybrid_count"], 0)
        self.assertEqual(report["try_rmsnorm_simd_count"], 0)

    def test_require_recipe_rejects_unrecognized_without_publication(self) -> None:
        self.environment["FAKE_ASCIFY_OUTPUT"] = "plain SIMT output\n"
        result = self.run_generator("simd-simt", "--require-recipe")

        self.assertEqual(result.returncode, 2)
        self.assertIn("did not contain a structurally valid SIMD dispatch", result.stderr)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.report.exists())

    def test_require_recipe_is_rejected_for_simt_before_invocation(self) -> None:
        result = self.run_generator("simt", "--require-recipe")

        self.assertEqual(result.returncode, 2)
        self.assertIn("only valid with --mode=simd-simt", result.stderr)
        self.assertFalse(self.argv_log.exists())
        self.assertFalse(self.output.exists())
        self.assertFalse(self.report.exists())

    def test_existing_output_is_never_replaced_or_converted(self) -> None:
        self.output.write_bytes(b"keep output\n")
        result = self.run_generator("simt")

        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.output.read_bytes(), b"keep output\n")
        self.assertFalse(self.report.exists())
        self.assertFalse(self.argv_log.exists())

    def test_existing_report_is_never_replaced_or_converted(self) -> None:
        self.report.write_bytes(b"keep report\n")
        result = self.run_generator("simt")

        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.report.read_bytes(), b"keep report\n")
        self.assertFalse(self.output.exists())
        self.assertFalse(self.argv_log.exists())

    def test_converter_failure_does_not_publish_temp_output(self) -> None:
        self.environment["FAKE_ASCIFY_OUTPUT"] = "partial output\n"
        self.environment["FAKE_ASCIFY_EXIT"] = "9"
        result = self.run_generator("simd-simt")

        self.assertEqual(result.returncode, 2)
        self.assertIn("ascify-clang failed with exit code 9", result.stderr)
        self.assertTrue(self.argv_log.exists())
        self.assertFalse(self.output.exists())
        self.assertFalse(self.report.exists())
        self.assertEqual(
            list(self.root.glob(f".{self.output.name}.ascify-*")), []
        )

    def test_success_without_converter_output_does_not_publish(self) -> None:
        self.environment["FAKE_ASCIFY_SKIP_OUTPUT"] = "1"
        result = self.run_generator("simt")

        self.assertEqual(result.returncode, 2)
        self.assertIn("ascify-clang output does not exist", result.stderr)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.report.exists())

    def test_empty_converter_output_does_not_publish(self) -> None:
        self.environment["FAKE_ASCIFY_OUTPUT"] = ""
        result = self.run_generator("simt")

        self.assertEqual(result.returncode, 2)
        self.assertIn("ascify-clang produced an empty output file", result.stderr)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.report.exists())


if __name__ == "__main__":
    unittest.main()
