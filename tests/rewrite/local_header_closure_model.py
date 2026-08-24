"""Executable host specification for translated local include closure tests.

This model validates artifact and transaction semantics without pretending to
exercise Clang's FileEntry callback.  The C++ integration contract is checked
separately by check_local_header_closure.sh.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Dict, Iterable, List, Optional, Set, Tuple


INCLUDE_RE = re.compile(
    r"(?m)^([ \t]*#[ \t]*include[ \t]*)([<\"])([^\">]+)([>\"])"
)
MACRO_INCLUDE_RE = re.compile(
    r"(?m)^[ \t]*#[ \t]*include[ \t]+(?![<\"])([^\s/]+)"
)


class ClosureError(RuntimeError):
    pass


@dataclass
class ClosureResult:
    root: Path
    bundle: Path
    nodes: Dict[Path, Path]
    stats: Dict[str, int] = field(default_factory=dict)
    analysis_only: bool = False


class ClosureModel:
    """Small filesystem model used only by host fixtures."""

    def __init__(
        self,
        root: Path,
        output: Path,
        *,
        mode: str = "recursive",
        include_roots: Iterable[Path] = (),
        excluded_roots: Iterable[Path] = (),
        inplace: bool = False,
    ) -> None:
        if mode not in {"disabled", "direct", "recursive"}:
            raise ValueError(mode)
        self.root = root.resolve(strict=True)
        self.output_lexical = output.absolute()
        self.output = self.output_lexical.resolve(strict=False)
        self.mode = mode
        self.inplace = inplace
        if self.root.suffix == ".dpp" and mode != "disabled":
            raise ClosureError("a .dpp root is not retranslated")
        if not inplace and self.output.resolve(strict=False) == self.root:
            raise ClosureError("output aliases input without inplace")

        candidates = [self.root.parent, *include_roots]
        self.roots: List[Path] = []
        for candidate in candidates:
            canonical = Path(candidate).resolve(strict=True)
            if canonical not in self.roots:
                self.roots.append(canonical)
        self.excluded_roots = [
            Path(candidate).resolve(strict=True) for candidate in excluded_roots
        ]
        self.bundle_lexical = (
            Path(str(self.root) + ".ascify-headers")
            if inplace
            else Path(str(self.output_lexical) + ".headers")
        )
        self.bundle = self.bundle_lexical.resolve(strict=False)
        self.nodes: Dict[Path, Path] = {}
        self.depth: Dict[Path, int] = {}
        self.dependencies: Set[Path] = {self.root}
        self.rollback_retry_count = 0
        self.stats = {
            "angle_edges_ignored": 0,
            "quoted_literal_edges_seen": 0,
            "selected_include_edges": 0,
            "redirected_root_edges": 0,
            "redirected_header_edges": 0,
            "duplicate_edges": 0,
            "cycle_edges": 0,
            "already_translated_edges": 0,
            "missing_required_edges": 0,
            "external_quoted_edges": 0,
            "unsupported_macro_includes": 0,
        }

    @staticmethod
    def preflight_output_names(sources: Iterable[Path]) -> None:
        names: Set[str] = set()
        for source in sources:
            output_name = Path(source).name + ".dpp"
            if output_name in names:
                raise ClosureError("-o-dir basename collision")
            names.add(output_name)

    @staticmethod
    def _within(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def _owning_root(self, path: Path) -> Optional[Tuple[int, Path]]:
        matches = [
            (index, root)
            for index, root in enumerate(self.roots)
            if self._within(path, root)
        ]
        if not matches:
            return None
        return max(matches, key=lambda item: (len(str(item[1])), -item[0]))

    def _excluded(self, path: Path) -> bool:
        return any(self._within(path, root) for root in self.excluded_roots)

    def _resolve(self, parent: Path, spelling: str) -> Optional[Path]:
        candidates = [parent.parent / spelling]
        candidates.extend(root / spelling for root in self.roots[1:])
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve(strict=True)
        return None

    def _artifact_for(self, source: Path) -> Path:
        owning = self._owning_root(source)
        if owning is None:
            raise ClosureError("source has no eligible root")
        root_id, root = owning
        relative = source.relative_to(root)
        return self.bundle / f"r{root_id}" / Path(str(relative) + ".dpp")

    @staticmethod
    def _include_token(parent_artifact: Path, child_artifact: Path) -> str:
        relative = os.path.relpath(child_artifact, parent_artifact.parent)
        return '"' + Path(relative).as_posix() + '"'

    def _rewrite(
        self,
        source: Path,
        parent_artifact: Path,
        depth: int,
        fail_on: Set[Path],
    ) -> str:
        if source in fail_on:
            raise ClosureError(f"injected translation failure: {source}")
        text = source.read_text(encoding="utf-8")
        if self.mode == "recursive" and MACRO_INCLUDE_RE.search(text):
            self.stats["unsupported_macro_includes"] += 1
            raise ClosureError("macro include cannot form strict closure")

        def replace(match: re.Match[str]) -> str:
            prefix, opener, spelling, closer = match.groups()
            if opener == "<":
                self.stats["angle_edges_ignored"] += 1
                return match.group(0)
            self.stats["quoted_literal_edges_seen"] += 1
            if spelling.endswith(".dpp"):
                self.stats["already_translated_edges"] += 1
                return match.group(0)

            resolved = self._resolve(source, spelling)
            if resolved is None:
                self.stats["missing_required_edges"] += 1
                raise ClosureError(f"missing quoted include: {spelling}")
            self.dependencies.add(resolved)
            if resolved == self.root:
                raise ClosureError("local header resolves back to root input")
            if self._excluded(resolved):
                self.stats["external_quoted_edges"] += 1
                return match.group(0)
            owning = self._owning_root(resolved)
            if owning is None:
                lexical = source.parent / spelling
                if ".." in Path(spelling).parts or lexical.exists():
                    raise ClosureError("quoted include escapes eligible root")
                self.stats["external_quoted_edges"] += 1
                return match.group(0)

            may_select = depth == 0 or self.mode == "recursive"
            if resolved not in self.nodes and not may_select:
                return match.group(0)
            if resolved in self.nodes:
                self.stats["duplicate_edges"] += 1
                if self.depth[resolved] <= depth:
                    self.stats["cycle_edges"] += 1
            else:
                self.nodes[resolved] = self._artifact_for(resolved)
                self.depth[resolved] = depth + 1

            emitted = self._include_token(
                parent_artifact, self.nodes[resolved]
            )
            self.stats["selected_include_edges"] += 1
            key = (
                "redirected_root_edges"
                if depth == 0
                else "redirected_header_edges"
            )
            self.stats[key] += 1
            return prefix + emitted

        return INCLUDE_RE.sub(replace, text)

    def _validate_artifact_isolation(self) -> None:
        artifacts = [
            ("root output", self.output_lexical.resolve(strict=False), False),
            (
                "local-header bundle",
                self.bundle_lexical.resolve(strict=False),
                True,
            ),
            (
                "root backup",
                Path(str(self.output_lexical) + ".ascify-backup").resolve(
                    strict=False
                ),
                False,
            ),
            (
                "local-header bundle backup",
                Path(str(self.bundle_lexical) + ".ascify-backup").resolve(
                    strict=False
                ),
                True,
            ),
        ]
        for index, (_, left, left_directory) in enumerate(artifacts):
            for _, right, right_directory in artifacts[index + 1 :]:
                if (
                    left == right
                    or (left_directory and self._within(right, left))
                    or (right_directory and self._within(left, right))
                ):
                    raise ClosureError("local-header artifacts overlap")
        for label, artifact, directory_like in artifacts:
            for dependency in self.dependencies:
                explicit_inplace_root = (
                    self.inplace
                    and label == "root output"
                    and artifact == self.root
                    and dependency == self.root
                )
                if explicit_inplace_root:
                    continue
                if artifact == dependency or (
                    directory_like and self._within(dependency, artifact)
                ):
                    raise ClosureError(
                        f"{label} aliases or contains input dependency: "
                        f"{dependency}"
                    )

    @staticmethod
    def _remove_exact(path: Path) -> None:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)

    def run(
        self,
        *,
        analysis_only: bool = False,
        fail_on: Iterable[Path] = (),
        inject_publish_failure: bool = False,
        inject_rollback_failure: bool = False,
    ) -> ClosureResult:
        if self.mode == "disabled":
            if not analysis_only:
                self.output.write_text(
                    self.root.read_text(encoding="utf-8"), encoding="utf-8"
                )
            return ClosureResult(
                self.output, self.bundle, {}, dict(self.stats), analysis_only
            )

        canonical_fail = {Path(path).resolve(strict=True) for path in fail_on}
        rendered: Dict[Path, str] = {}
        rendered[self.output] = self._rewrite(
            self.root, self.output, 0, canonical_fail
        )
        index = 0
        while index < len(self.nodes):
            source = list(self.nodes)[index]
            artifact = self.nodes[source]
            rendered[artifact] = self._rewrite(
                source, artifact, self.depth[source], canonical_fail
            )
            index += 1

        self._validate_artifact_isolation()

        if analysis_only:
            return ClosureResult(
                self.output,
                self.bundle,
                dict(self.nodes),
                dict(self.stats),
                True,
            )

        self.output.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(
            tempfile.mkdtemp(
                prefix=".ascify-local-closure-", dir=self.output.parent
            )
        )
        stage_root = stage / "root.dpp"
        stage_bundle = stage / "headers"
        root_backup = Path(str(self.output) + ".ascify-backup")
        bundle_backup = Path(str(self.bundle) + ".ascify-backup")
        try:
            stage_root.write_text(rendered[self.output], encoding="utf-8")
            for source, final_artifact in self.nodes.items():
                relative = final_artifact.relative_to(self.bundle)
                staged = stage_bundle / relative
                staged.parent.mkdir(parents=True, exist_ok=True)
                staged.write_text(rendered[final_artifact], encoding="utf-8")
            if self.nodes:
                stage_bundle.mkdir(parents=True, exist_ok=True)
                (stage_bundle / ".ascify-local-closure").write_text(
                    "schema=host-spec-v1\n", encoding="utf-8"
                )

            if self.output.exists() and (
                self.output.is_symlink() or self.output.is_dir()
            ):
                raise ClosureError("unsafe existing root output")
            marker = self.bundle / ".ascify-local-closure"
            owned_bundle = (
                not self.bundle.is_symlink()
                and not marker.is_symlink()
                and marker.is_file()
                and marker.read_text(encoding="utf-8").startswith(
                    "schema=host-spec-v1\n"
                )
            )
            if self.nodes and self.bundle.exists() and not owned_bundle:
                raise ClosureError("unowned bundle collision")
            if root_backup.exists() or bundle_backup.exists():
                raise ClosureError("stale transaction backup")

            if self.bundle.exists() and (self.nodes or owned_bundle):
                os.replace(self.bundle, bundle_backup)
            if self.output.exists():
                os.replace(self.output, root_backup)
            try:
                if self.nodes:
                    os.replace(stage_bundle, self.bundle)
                if inject_publish_failure:
                    raise ClosureError("injected root publish failure")
                os.replace(stage_root, self.output)
            except Exception:
                if self.bundle.exists() and self.nodes:
                    self._remove_exact(self.bundle)
                if bundle_backup.exists():
                    if inject_rollback_failure:
                        try:
                            raise OSError("injected first rollback rename failure")
                        except OSError:
                            self.rollback_retry_count += 1
                    os.replace(bundle_backup, self.bundle)
                if root_backup.exists():
                    os.replace(root_backup, self.output)
                raise
            self._remove_exact(root_backup)
            self._remove_exact(bundle_backup)
        finally:
            if stage.exists():
                shutil.rmtree(stage)

        return ClosureResult(
            self.output,
            self.bundle,
            dict(self.nodes),
            dict(self.stats),
            False,
        )
