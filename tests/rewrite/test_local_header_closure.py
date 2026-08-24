from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from tests.rewrite.local_header_closure_model import (
    ClosureError,
    ClosureModel,
)


class LocalHeaderClosureHostFixtures(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.src = self.base / "src"
        self.out = self.base / "out"
        self.src.mkdir()
        self.out.mkdir()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write(self, relative: str, text: str) -> Path:
        path = self.base / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def model(self, mode: str = "recursive", **kwargs) -> ClosureModel:
        return ClosureModel(
            self.src / "main.cu",
            self.out / "main.cu.dpp",
            mode=mode,
            **kwargs,
        )

    def test_hc_u00_disabled_preserves_include(self) -> None:
        self.write("src/main.cu", '#include "a.h"\n')
        self.write("src/a.h", "int a;\n")
        result = self.model("disabled").run()
        self.assertIn('"a.h"', result.root.read_text())
        self.assertFalse(result.bundle.exists())

    def test_hc_u01_direct_relative_redirects_root(self) -> None:
        self.write("src/main.cu", '#include "detail/a.cuh"\n')
        header = self.write("src/detail/a.cuh", "int a;\n").resolve()
        result = self.model("direct").run()
        artifact = result.nodes[header]
        self.assertTrue(artifact.is_file())
        self.assertIn('.headers/r0/detail/a.cuh.dpp"', result.root.read_text())

    def test_hc_u02_direct_preserves_new_nested_edge(self) -> None:
        self.write("src/main.cu", '#include "a.h"\n')
        self.write("src/a.h", '#include "b.h"\n')
        self.write("src/b.h", "int b;\n")
        result = self.model("direct").run()
        self.assertEqual(1, len(result.nodes))
        a_artifact = next(iter(result.nodes.values()))
        self.assertIn('"b.h"', a_artifact.read_text())

    def test_hc_u03_recursive_redirects_header_edge(self) -> None:
        self.write("src/main.cu", '#include "a.h"\n')
        a = self.write("src/a.h", '#include "sub/b.h"\n').resolve()
        b = self.write("src/sub/b.h", "int b;\n").resolve()
        result = self.model().run()
        self.assertEqual({a, b}, set(result.nodes))
        self.assertIn('sub/b.h.dpp"', result.nodes[a].read_text())
        self.assertEqual(1, result.stats["redirected_header_edges"])

    def test_hc_u04_explicit_include_root(self) -> None:
        self.write("src/main.cu", '#include "pkg/a.h"\n')
        header = self.write("include/pkg/a.h", "int a;\n").resolve()
        result = self.model(include_roots=[self.base / "include"]).run()
        self.assertIn(header, result.nodes)
        self.assertIn("/r1/pkg/a.h.dpp", str(result.nodes[header]))

    def test_hc_u05_including_directory_shadows_include_root(self) -> None:
        self.write("src/main.cu", '#include "a.h"\n')
        local = self.write("src/a.h", "int local;\n").resolve()
        other = self.write("include/a.h", "int other;\n").resolve()
        result = self.model(include_roots=[self.base / "include"]).run()
        self.assertIn(local, result.nodes)
        self.assertNotIn(other, result.nodes)

    def test_hc_u06_aliases_deduplicate_node(self) -> None:
        self.write("src/main.cu", '#include "a.h"\n#include "./a.h"\n')
        self.write("src/a.h", "int a;\n")
        result = self.model().run()
        self.assertEqual(1, len(result.nodes))
        self.assertEqual(1, result.stats["duplicate_edges"])
        self.assertEqual(2, result.stats["selected_include_edges"])

    def test_hc_u07_cycle_terminates_and_redirects(self) -> None:
        self.write("src/main.cu", '#include "a.h"\n')
        self.write("src/a.h", '#include "b.h"\n')
        self.write("src/b.h", '#include "a.h"\n')
        result = self.model().run()
        self.assertEqual(2, len(result.nodes))
        self.assertEqual(1, result.stats["cycle_edges"])

    def test_hc_u08_shared_child_translated_once(self) -> None:
        self.write("src/main.cu", '#include "a.h"\n#include "b.h"\n')
        self.write("src/a.h", '#include "shared.h"\n')
        self.write("src/b.h", '#include "shared.h"\n')
        shared = self.write("src/shared.h", "int s;\n").resolve()
        result = self.model().run()
        self.assertEqual(3, len(result.nodes))
        self.assertEqual(1, list(result.nodes).count(shared))

    def test_hc_u09_missing_is_fail_closed(self) -> None:
        self.write("src/main.cu", '#include "missing.h"\n')
        output = self.out / "main.cu.dpp"
        output.write_text("old-root\n", encoding="utf-8")
        with self.assertRaises(ClosureError):
            self.model().run()
        self.assertEqual("old-root\n", output.read_text())
        self.assertFalse(Path(str(output) + ".headers").exists())

    def test_hc_u10_angle_helper_is_excluded(self) -> None:
        self.write("src/main.cu", "#include <helper_cuda.h>\n")
        result = self.model().run()
        self.assertEqual({}, result.nodes)
        self.assertIn("<helper_cuda.h>", result.root.read_text())
        self.assertEqual(1, result.stats["angle_edges_ignored"])

    def test_hc_u11_quoted_helper_under_user_root_is_selected(self) -> None:
        self.write("src/main.cu", '#include "helper_cuda.h"\n')
        helper = self.write("Common/helper_cuda.h", "int helper;\n").resolve()
        result = self.model(include_roots=[self.base / "Common"]).run()
        self.assertIn(helper, result.nodes)

    def test_hc_u13_existing_dpp_edge_is_idempotent(self) -> None:
        self.write("src/main.cu", '#include "a.h.dpp"\n')
        result = self.model().run()
        self.assertEqual({}, result.nodes)
        self.assertNotIn(".dpp.dpp", result.root.read_text())
        self.assertEqual(1, result.stats["already_translated_edges"])

    def test_hc_u14_dpp_root_is_rejected(self) -> None:
        root = self.write("src/main.cu.dpp", "int x;\n")
        with self.assertRaises(ClosureError):
            ClosureModel(root, self.out / "again.dpp")

    def test_hc_u15_parent_path_inside_explicit_root(self) -> None:
        root = self.write("project/src/main.cu", '#include "../include/a.h"\n')
        header = self.write("project/include/a.h", "int a;\n").resolve()
        model = ClosureModel(
            root,
            self.out / "safe.dpp",
            include_roots=[self.base / "project"],
        )
        result = model.run()
        self.assertIn(header, result.nodes)

    def test_hc_u16_parent_path_escape_is_rejected(self) -> None:
        root = self.write("project/src/main.cu", '#include "../../outside/a.h"\n')
        self.write("outside/a.h", "int a;\n")
        with self.assertRaises(ClosureError):
            ClosureModel(root, self.out / "escape.dpp").run()

    def test_hc_u17_symlink_alias_deduplicates(self) -> None:
        self.write("src/main.cu", '#include "a.h"\n#include "link.h"\n')
        target = self.write("src/a.h", "int a;\n")
        try:
            (self.src / "link.h").symlink_to(target.name)
        except OSError as exc:
            self.skipTest(f"symlink unavailable: {exc}")
        result = self.model().run()
        self.assertEqual(1, len(result.nodes))

    def test_hc_u18_symlink_escape_is_rejected(self) -> None:
        self.write("src/main.cu", '#include "link.h"\n')
        target = self.write("outside/a.h", "int a;\n")
        try:
            (self.src / "link.h").symlink_to(target)
        except OSError as exc:
            self.skipTest(f"symlink unavailable: {exc}")
        with self.assertRaises(ClosureError):
            self.model().run()

    def test_hc_u19_macro_include_is_rejected_recursively(self) -> None:
        self.write("src/main.cu", '#define H "a.h"\n#include H\n')
        self.write("src/a.h", "int a;\n")
        with self.assertRaises(ClosureError):
            self.model().run()

    def test_header_back_edge_to_root_fails_without_recursing(self) -> None:
        self.write("src/main.cu", '#include "a.h"\n')
        self.write("src/a.h", '#include "main.cu"\n')
        with self.assertRaises(ClosureError):
            self.model().run()

    def test_hc_u23_two_roots_have_isolated_bundles(self) -> None:
        root_a = self.write("a/main_a.cu", '#include "h.h"\n')
        root_b = self.write("b/main_b.cu", '#include "h.h"\n')
        self.write("a/h.h", "int a;\n")
        self.write("b/h.h", "int b;\n")
        a = ClosureModel(root_a, self.out / "main_a.cu.dpp").run()
        b = ClosureModel(root_b, self.out / "main_b.cu.dpp").run()
        self.assertNotEqual(a.bundle, b.bundle)
        self.assertTrue(a.bundle.is_dir() and b.bundle.is_dir())

    def test_hc_u24_output_dir_collision_preflight(self) -> None:
        with self.assertRaises(ClosureError):
            ClosureModel.preflight_output_names(
                [Path("a/main.cu"), Path("b/main.cu")]
            )

    def test_hc_u25_inplace_header_failure_preserves_input(self) -> None:
        root = self.write("src/main.cu", '#include "a.h"\n')
        header = self.write("src/a.h", "int a;\n")
        before = root.read_bytes()
        model = ClosureModel(root, root, inplace=True)
        with self.assertRaises(ClosureError):
            model.run(fail_on=[header])
        self.assertEqual(before, root.read_bytes())
        self.assertFalse(Path(str(root.resolve()) + ".ascify-headers").exists())

    def test_hc_u26_analysis_only_writes_nothing(self) -> None:
        self.write("src/main.cu", '#include "a.h"\n')
        self.write("src/a.h", "int a;\n")
        result = self.model().run(analysis_only=True)
        self.assertTrue(result.analysis_only)
        self.assertFalse(result.root.exists())
        self.assertFalse(result.bundle.exists())

    def test_hc_u28_header_uses_its_own_directory_context(self) -> None:
        self.write("src/main.cu", '#include "pkg/a.h"\n')
        a = self.write("include/pkg/a.h", '#include "detail/b.h"\n').resolve()
        b = self.write("include/pkg/detail/b.h", "int b;\n").resolve()
        result = self.model(include_roots=[self.base / "include"]).run()
        self.assertIn(a, result.nodes)
        self.assertIn(b, result.nodes)

    def test_unowned_bundle_collision_is_rejected(self) -> None:
        root = self.write("src/main.cu", '#include "a.h"\n')
        header = self.write("src/a.h", "int a;\n")
        root_before = root.read_bytes()
        header_before = header.read_bytes()
        bundle = Path(str(self.out / "main.cu.dpp") + ".headers")
        bundle.mkdir()
        (bundle / "unrelated").write_text("keep", encoding="utf-8")
        with self.assertRaises(ClosureError):
            self.model().run()
        self.assertEqual("keep", (bundle / "unrelated").read_text())
        self.assertEqual(root_before, root.read_bytes())
        self.assertEqual(header_before, header.read_bytes())
        self.assertFalse(
            Path(str(self.out / "main.cu.dpp") + ".ascify-backup").exists()
        )
        self.assertFalse(Path(str(bundle) + ".ascify-backup").exists())

    def test_parent_symlink_output_aliases_selected_header(self) -> None:
        root = self.write("src/main.cu", '#include "a.h"\n')
        header = self.write("src/a.h", "int selected;\n")
        root_before = root.read_bytes()
        header_before = header.read_bytes()
        alias = self.base / "selected-parent-alias"
        try:
            alias.symlink_to(self.src, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"symlink unavailable: {exc}")
        output = alias / "a.h"
        with self.assertRaisesRegex(ClosureError, "root output aliases"):
            ClosureModel(root, output).run()
        self.assertEqual(root_before, root.read_bytes())
        self.assertEqual(header_before, header.read_bytes())
        self.assertFalse(Path(str(output) + ".ascify-backup").exists())
        self.assertFalse(Path(str(output) + ".headers.ascify-backup").exists())

    def test_parent_symlink_output_aliases_excluded_header(self) -> None:
        root = self.write("src/main.cu", '#include "excluded.h"\n')
        excluded_dir = self.base / "excluded"
        header = self.write("excluded/excluded.h", "int excluded;\n")
        root_before = root.read_bytes()
        header_before = header.read_bytes()
        alias = self.base / "excluded-parent-alias"
        try:
            alias.symlink_to(excluded_dir, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"symlink unavailable: {exc}")
        output = alias / "excluded.h"
        with self.assertRaisesRegex(ClosureError, "root output aliases"):
            ClosureModel(
                root,
                output,
                include_roots=[excluded_dir],
                excluded_roots=[excluded_dir],
            ).run()
        self.assertEqual(root_before, root.read_bytes())
        self.assertEqual(header_before, header.read_bytes())
        self.assertFalse(Path(str(output) + ".ascify-backup").exists())

    def test_backup_aliases_selected_dependency(self) -> None:
        root = self.write("src/main.cu", '#include "a.h"\n')
        header = self.write("src/a.h", "int selected;\n")
        output = self.out / "main.cu.dpp"
        backup = Path(str(output) + ".ascify-backup")
        try:
            backup.symlink_to(header)
        except OSError as exc:
            self.skipTest(f"symlink unavailable: {exc}")
        with self.assertRaisesRegex(ClosureError, "root backup aliases"):
            ClosureModel(root, output).run()
        self.assertEqual(b"int selected;\n", header.read_bytes())
        self.assertTrue(backup.is_symlink())
        self.assertFalse(output.exists())

    def test_wrong_marker_schema_does_not_claim_bundle_ownership(self) -> None:
        self.write("src/main.cu", '#include "a.h"\n')
        self.write("src/a.h", "int a;\n")
        bundle = Path(str(self.out / "main.cu.dpp") + ".headers")
        bundle.mkdir()
        (bundle / ".ascify-local-closure").write_text(
            "schema=unrelated\n", encoding="utf-8"
        )
        with self.assertRaises(ClosureError):
            self.model().run()
        self.assertEqual(
            "schema=unrelated\n",
            (bundle / ".ascify-local-closure").read_text(),
        )

    def test_owned_bundle_is_replaced_on_rerun(self) -> None:
        self.write("src/main.cu", '#include "a.h"\n')
        self.write("src/a.h", "int a;\n")
        first = self.model().run()
        (first.bundle / "stale").write_text("old", encoding="utf-8")
        second = self.model().run()
        self.assertFalse((second.bundle / "stale").exists())
        self.assertTrue((second.bundle / ".ascify-local-closure").is_file())

    def test_owned_stale_bundle_removed_when_graph_becomes_empty(self) -> None:
        root = self.write("src/main.cu", '#include "a.h"\n')
        self.write("src/a.h", "int a;\n")
        first = self.model().run()
        root.write_text("int no_headers;\n", encoding="utf-8")
        second = self.model().run()
        self.assertTrue(second.root.is_file())
        self.assertFalse(first.bundle.exists())

    def test_publish_failure_rolls_back_root_and_bundle(self) -> None:
        root = self.write("src/main.cu", '#include "a.h"\n')
        self.write("src/a.h", "int v1;\n")
        old = self.model().run()
        old_root = old.root.read_bytes()
        old_header = next(iter(old.nodes.values())).read_bytes()
        (self.src / "a.h").write_text("int v2;\n", encoding="utf-8")
        input_root = root.read_bytes()
        input_header = (self.src / "a.h").read_bytes()
        with self.assertRaises(ClosureError):
            self.model().run(inject_publish_failure=True)
        self.assertEqual(old_root, old.root.read_bytes())
        self.assertEqual(old_header, next(iter(old.nodes.values())).read_bytes())
        self.assertEqual(input_root, root.read_bytes())
        self.assertEqual(input_header, (self.src / "a.h").read_bytes())
        self.assertFalse(Path(str(old.root) + ".ascify-backup").exists())
        self.assertFalse(Path(str(old.bundle) + ".ascify-backup").exists())
        self.assertEqual(root.resolve(), self.model().root)

    def test_rollback_rename_failure_is_checked_retried_and_clean(self) -> None:
        root = self.write("src/main.cu", '#include "a.h"\n')
        header = self.write("src/a.h", "int v1;\n")
        old = self.model().run()
        old_root = old.root.read_bytes()
        old_header = next(iter(old.nodes.values())).read_bytes()
        header.write_text("int v2;\n", encoding="utf-8")
        input_root = root.read_bytes()
        input_header = header.read_bytes()
        retrying_model = self.model()
        with self.assertRaises(ClosureError):
            retrying_model.run(
                inject_publish_failure=True,
                inject_rollback_failure=True,
            )
        self.assertEqual(1, retrying_model.rollback_retry_count)
        self.assertEqual(old_root, old.root.read_bytes())
        self.assertEqual(old_header, next(iter(old.nodes.values())).read_bytes())
        self.assertEqual(input_root, root.read_bytes())
        self.assertEqual(input_header, header.read_bytes())
        self.assertFalse(Path(str(old.root) + ".ascify-backup").exists())
        self.assertFalse(Path(str(old.bundle) + ".ascify-backup").exists())


if __name__ == "__main__":
    unittest.main()
