#!/usr/bin/env python3
"""Generate a row-wise CCE by invoking ascify-clang.

The converter writes to a private temporary directory first.  The requested CCE
and its JSON report are published only after conversion and validation succeed;
existing destinations are never replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Sequence


TARGET_ABI = 1
TARGET_RECIPE = "dav-3510-rowwise-simd-v1"
REPORT_SCHEMA = "ascify.rowwise-cce-report.v3"
MARKERS = ("TrySoftmaxHybrid", "TryRmsNormHybrid", "TryLayerNormHybrid")
SIMD_HEADER = "ascify/target/dav_c310/rowwise_simd_recipes.hpp"
TRIGRAPH_PATTERN = re.compile(rb"\?\?[=/'()!<>-]")
UTF8_BOM = b"\xef\xbb\xbf"
PHASE2_SPLICE_PATTERN = re.compile(r"\\[ \t\v\f]*(?:\r\n|\n|\r)")
DISPATCH_PREFIX = (
    "::ascify::target::dav_c310::rowwise_simd_v1::"
    "RowwiseHybridFacadeV1::"
)
MODE_OPTIONS = {
    "simt": ("portable", "precise", None),
    "simd-simt": ("dav-c310-vec", "fast", TARGET_RECIPE),
}


class GenerationError(RuntimeError):
    """Raised when generation cannot be completed safely."""


def _split_compiler_tail(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split wrapper arguments from arguments following an explicit ``--``."""

    values = list(argv)
    try:
        separator = values.index("--")
    except ValueError:
        return values, []
    return values[:separator], values[separator + 1 :]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Invoke ascify-clang to generate a SIMT or SIMD+SIMT row-wise CCE "
            "and an integrity report."
        )
    )
    parser.add_argument("source", type=Path, help="CUDA source passed to ascify-clang")
    parser.add_argument("output", type=Path, help="new .cce destination")
    parser.add_argument(
        "--mode",
        choices=tuple(MODE_OPTIONS),
        required=True,
        help="simt uses portable/precise; simd-simt uses the DAV row-wise recipe",
    )
    parser.add_argument(
        "--ascify-clang",
        metavar="PATH",
        help=(
            "ascify-clang executable; otherwise use ASCIFY_CLANG, then PATH"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="new JSON report destination (default: OUTPUT.report.json)",
    )
    parser.add_argument(
        "--cuda-path",
        metavar="PATH",
        help="CUDA toolkit path passed to ascify-clang",
    )
    parser.add_argument(
        "--clang-resource-directory",
        metavar="PATH",
        help="Clang resource directory passed to ascify-clang",
    )
    parser.add_argument(
        "-I",
        "--include",
        action="append",
        default=[],
        metavar="DIR",
        help="compiler include directory; repeat as needed",
    )
    parser.add_argument(
        "--compiler-arg",
        action="append",
        default=[],
        metavar="ARG",
        help=(
            "compiler argument; repeat as needed (use --compiler-arg=-flag for "
            "values beginning with '-')"
        ),
    )
    parser.add_argument(
        "--require-recipe",
        action="store_true",
        help="fail unless generated output contains a supported SIMD recipe marker",
    )
    return parser


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reject_trigraph_spellings(data: bytes, label: str) -> None:
    """Reject phase-1 trigraph spellings before any lexical inspection.

    Trigraph replacement precedes line splicing, comments, string literals,
    and preprocessing directives.  Conservatively rejecting every legal
    spelling keeps this report scanner independent of compiler trigraph flags
    and avoids treating bytes from an inactive or comment-spliced region as
    compiled dispatch evidence.
    """

    match = TRIGRAPH_PATTERN.search(data)
    if match is not None:
        raise GenerationError(
            f"{label} contains a C/C++ trigraph spelling at byte "
            f"{match.start()}; refusing translation-phase-ambiguous input"
        )


def _reject_utf8_bom(data: bytes, label: str) -> None:
    """Reject a leading BOM that Clang ignores but the scanner would retain."""

    if data.startswith(UTF8_BOM):
        raise GenerationError(
            f"{label} starts with a UTF-8 BOM; refusing scanner/compiler "
            "lexical ambiguity"
        )


def _reject_nul_bytes(data: bytes, label: str) -> None:
    """Reject NUL, which Clang may ignore before directive recognition."""

    offset = data.find(b"\0")
    if offset >= 0:
        raise GenerationError(
            f"{label} contains a NUL byte at byte {offset}; refusing "
            "scanner/compiler lexical ambiguity"
        )


def _reject_invalid_utf8(data: bytes, label: str) -> None:
    """Require one unambiguous source encoding for scanner and compiler."""

    try:
        data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise GenerationError(
            f"{label} is not valid UTF-8 at byte {exc.start}; refusing "
            "scanner/compiler encoding ambiguity"
        ) from exc


def _read_regular_file(path: Path, label: str) -> tuple[bytes, tuple[int, int]]:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise GenerationError(f"{label} does not exist: {path}") from exc
    except OSError as exc:
        raise GenerationError(f"cannot inspect {label} {path}: {exc}") from exc

    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise GenerationError(f"{label} must be a regular, non-symlink file: {path}")

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise GenerationError(f"cannot open {label} {path}: {exc}") from exc

    try:
        opened_metadata = os.fstat(descriptor)
        if not stat.S_ISREG(opened_metadata.st_mode):
            raise GenerationError(
                f"{label} must be a regular, non-symlink file: {path}"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    except OSError as exc:
        raise GenerationError(f"cannot read {label} {path}: {exc}") from exc
    finally:
        os.close(descriptor)
    return b"".join(chunks), (opened_metadata.st_dev, opened_metadata.st_ino)


def _resolve_ascify_clang(explicit: str | None) -> Path:
    requested = explicit or os.environ.get("ASCIFY_CLANG") or "ascify-clang"
    candidate: str | None
    if os.sep in requested or (os.altsep is not None and os.altsep in requested):
        candidate = requested
    else:
        candidate = shutil.which(requested)

    if candidate is None:
        raise GenerationError(
            "ascify-clang was not found; pass --ascify-clang, set ASCIFY_CLANG, "
            "or add it to PATH"
        )

    path = Path(candidate)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise GenerationError(f"ascify-clang is not an executable file: {path}")
    return path.resolve()


def _ensure_new_destination(path: Path, label: str) -> None:
    parent = path.parent
    try:
        parent_metadata = parent.stat()
    except FileNotFoundError as exc:
        raise GenerationError(f"{label} parent directory does not exist: {parent}") from exc
    except OSError as exc:
        raise GenerationError(f"cannot inspect {label} parent {parent}: {exc}") from exc
    if not stat.S_ISDIR(parent_metadata.st_mode):
        raise GenerationError(f"{label} parent is not a directory: {parent}")

    try:
        path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise GenerationError(f"cannot inspect {label} {path}: {exc}") from exc
    raise GenerationError(f"refusing to replace existing {label}: {path}")


def _code_only_text(output: bytes) -> str:
    """Apply phase-2 splicing, then blank comments and literals."""

    physical = output.decode("utf-8", errors="strict")
    raw_prefix = re.compile(r'(?:u8|u|U|L)?R"([^\s()\\]{0,16})\(')
    ordinary_literal_prefix = re.compile(r'(?:u8|u|U|L)?(["\'])')

    # Backslash-newline deletion precedes comment and directive recognition in
    # C/C++.  Clang also accepts horizontal whitespace between the backslash
    # and newline as an extension, so model that spelling conservatively too.
    # Scan the resulting logical source so a splice cannot manufacture `/*`,
    # `//`, a control transfer, or a preprocessing directive that this
    # integrity pass misses.
    has_phase2_splice = PHASE2_SPLICE_PATTERN.search(physical) is not None
    text = PHASE2_SPLICE_PATTERN.sub("", physical)
    code = list(text)
    index = 0
    length = len(text)

    def blank(start: int, end: int) -> None:
        for offset in range(start, end):
            if code[offset] not in "\r\n":
                code[offset] = " "

    while index < length:
        if text.startswith("//", index):
            endings = [
                position
                for position in (
                    text.find("\r", index + 2),
                    text.find("\n", index + 2),
                )
                if position >= 0
            ]
            end = min(endings) if endings else length
            blank(index, end)
            index = end
            continue
        if text.startswith("/*", index):
            end = text.find("*/", index + 2)
            end = length if end < 0 else end + 2
            blank(index, end)
            index = end
            continue

        raw_match = raw_prefix.match(text, index)
        if raw_match is not None:
            if has_phase2_splice:
                # Raw-string body spelling is restored after translation
                # phases 1/2.  Without a physical-to-logical source map we
                # cannot prove a file that combines a raw literal and any
                # splice, so reject its structural evidence conservatively.
                return "".join(
                    character if character in "\r\n" else " "
                    for character in text
                )
            terminator = ")" + raw_match.group(1) + '"'
            end = text.find(terminator, raw_match.end())
            end = length if end < 0 else end + len(terminator)
            blank(index, end)
            index = end
            continue

        literal_match = ordinary_literal_prefix.match(text, index)
        if literal_match is not None:
            quote = literal_match.group(1)
            end = literal_match.end()
            while end < length:
                if text[end] == "\\":
                    end = min(end + 2, length)
                    continue
                end += 1
                if text[end - 1] == quote:
                    break
            blank(index, end)
            index = end
            continue

        # Consume a preprocessing-number token before considering apostrophes
        # as character-literal delimiters.  Since C++14, apostrophes inside a
        # pp-number are digit separators; treating them as paired quotes can
        # incorrectly blank arbitrary code between two numeric literals.
        if text[index].isdigit() or (
            text[index] == "."
            and index + 1 < length
            and text[index + 1].isdigit()
        ):
            end = index + 1
            while end < length:
                character = text[end]
                if character.isalnum() or character in "_.":
                    end += 1
                    continue
                if (
                    character == "'"
                    and end + 1 < length
                    and (text[end + 1].isalnum() or text[end + 1] == "_")
                ):
                    end += 1
                    continue
                if character in "+-" and text[end - 1] in "eEpP":
                    end += 1
                    continue
                break
            index = end
            continue

        index += 1
    return "".join(code)


def _c_physical_lines(code: str) -> list[str]:
    """Split only at CR/LF physical line endings used by C/C++."""

    lines: list[str] = []
    start = 0
    index = 0
    while index < len(code):
        if code[index] == "\r":
            end = index + 1
            if end < len(code) and code[end] == "\n":
                end += 1
            lines.append(code[start:end])
            start = end
            index = end
            continue
        if code[index] == "\n":
            end = index + 1
            lines.append(code[start:end])
            start = end
            index = end
            continue
        index += 1
    if start < len(code):
        lines.append(code[start:])
    return lines


def _normalize_digraph_tokens(code: str) -> str:
    """Normalize C/C++ digraph tokens after comments and literals are blanked."""

    # Longest spelling must be handled first because `%:%:` contains two `%:`
    # spellings.  The remaining replacements preserve the token semantics that
    # Clang applies before parsing control scopes.
    replacements = (
        ("%:%:", "##"),
        ("<%", "{"),
        ("%>", "}"),
        ("<:", "["),
        (":>", "]"),
        ("%:", "#"),
    )
    for spelling, replacement in replacements:
        code = code.replace(spelling, replacement)
    return code


def _blank_preprocessor_conditionals(code: str) -> str:
    """Blank conditional-preprocessor regions except a whole-file include guard.

    The integrity report does not run a C preprocessor and therefore cannot
    prove which configurable branch is active.  Recipe evidence inside such a
    branch is ignored.  A conventional outer include guard is retained because
    the reviewed inputs and generated CCEs are headers whose entire contents are
    wrapped by that guard.
    """

    lines = _c_physical_lines(code)
    directives: list[tuple[int, str, str]] = []
    directive_pattern = re.compile(
        r"^\s*(?:#|%:)\s*"
        r"(if|ifdef|ifndef|elifdef|elifndef|elif|else|endif|define)\b(.*)$"
    )
    for line_index, line in enumerate(lines):
        match = directive_pattern.match(line)
        if match is not None:
            directives.append(
                (line_index, match.group(1), match.group(2).strip())
            )

    guard_open: int | None = None
    guard_close: int | None = None
    significant = [
        index for index, line in enumerate(lines) if line.strip()
    ]
    preamble_pattern = re.compile(
        r"^\s*(?:#|%:)\s*(?:include|pragma|undef)\b"
    )
    for guard_position in range(max(0, len(significant) - 2)):
        candidate_open = significant[guard_position]
        candidate_define = significant[guard_position + 1]
        if any(
            preamble_pattern.match(lines[index]) is None
            for index in significant[:guard_position]
        ):
            continue
        first = directive_pattern.match(lines[candidate_open])
        second = directive_pattern.match(lines[candidate_define])
        last = directive_pattern.match(lines[significant[-1]])
        if (
            first is not None
            and first.group(1) == "ifndef"
            and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", first.group(2).strip())
            and second is not None
            and second.group(1) == "define"
            and second.group(2).split(maxsplit=1)[0]
            == first.group(2).strip()
            and last is not None
            and last.group(1) == "endif"
        ):
            depth = 0
            matching_close: int | None = None
            has_guard_alternative = False
            for line_index, directive, _ in directives:
                if line_index < candidate_open:
                    continue
                if directive in ("if", "ifdef", "ifndef"):
                    depth += 1
                elif directive in (
                    "elif",
                    "elifdef",
                    "elifndef",
                    "else",
                ) and depth == 1:
                    has_guard_alternative = True
                elif directive == "endif":
                    depth -= 1
                    if depth == 0:
                        matching_close = line_index
                        break
            if (
                matching_close == significant[-1]
                and not has_guard_alternative
            ):
                guard_open = candidate_open
                guard_close = matching_close
                break

    result = list(code)
    line_offsets: list[int] = []
    offset = 0
    for line in lines:
        line_offsets.append(offset)
        offset += len(line)

    conditional_stack: list[bool] = []
    for line_index, line in enumerate(lines):
        match = directive_pattern.match(line)
        directive = match.group(1) if match is not None else None
        is_guard_directive = (
            (line_index == guard_open and directive == "ifndef")
            or (line_index == guard_close and directive == "endif")
        )

        if directive in ("if", "ifdef", "ifndef"):
            conditional_stack.append(is_guard_directive)

        inside_unproved_conditional = any(
            not is_guard for is_guard in conditional_stack
        )
        if inside_unproved_conditional:
            start = line_offsets[line_index]
            end = start + len(line)
            for position in range(start, end):
                if result[position] not in "\r\n":
                    result[position] = " "

        if directive == "endif" and conditional_stack:
            conditional_stack.pop()

    return "".join(result)


def _blank_preprocessor_directives(code: str) -> str:
    """Blank complete preprocessing logical lines, including continuations.

    The report scanner is intentionally not a preprocessor.  Directives may
    contain arbitrary token sequences (most notably function-like macro
    bodies), so none of their bytes may count as dispatch or launch evidence.
    Conditional regions have already been removed by
    ``_blank_preprocessor_conditionals``; this pass removes the directive lines
    themselves while leaving ordinary code inside the conventional outer
    include guard inspectable.
    """

    lines = _c_physical_lines(code)
    result = list(code)
    offsets: list[int] = []
    offset = 0
    for line in lines:
        offsets.append(offset)
        offset += len(line)

    directive_pattern = re.compile(r"^\s*(?:#|%:)")
    line_index = 0
    while line_index < len(lines):
        if directive_pattern.match(lines[line_index]) is None:
            line_index += 1
            continue

        while line_index < len(lines):
            line = lines[line_index]
            start = offsets[line_index]
            end = start + len(line)
            for position in range(start, end):
                if result[position] not in "\r\n":
                    result[position] = " "

            physical = line.removesuffix("\n").removesuffix("\r")
            line_index += 1
            if not physical.endswith("\\"):
                break

    return "".join(result)


def _dispatch_contract_matches(code: str, marker: str) -> list[re.Match[str]]:
    """Find complete generated dispatch/ownership contracts, not marker text."""

    expression = re.compile(
        r"\bconst\s+auto\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
        + re.escape(DISPATCH_PREFIX + marker)
        + r"\s*\((?P<arguments>[^;]*?)\)\s*;\s*"
        + r"if\s*\(\s*\1\s*\.\s*handled\s*\)\s*"
        + r"\{\s*return\s+\1\s*\.\s*status\s*;\s*\}",
        re.DOTALL,
    )
    return list(expression.finditer(code))


def _simd_header_count(code: str) -> int:
    expression = re.compile(
        r"^[ \t\v\f]*#[ \t\v\f]*include[ \t\v\f]*<"
        + re.escape(SIMD_HEADER)
        + r">[ \t\v\f]*$",
    )
    return sum(
        expression.fullmatch(line.rstrip("\r\n")) is not None
        for line in _c_physical_lines(code)
    )


def _recipe_call_token_count(code: str) -> int:
    return sum(
        len(
            re.findall(
                re.escape(DISPATCH_PREFIX + marker) + r"\s*\(",
                code,
            )
        )
        for marker in MARKERS
    )


def _matching_braces(code: str) -> dict[int, int]:
    stack: list[int] = []
    pairs: dict[int, int] = {}
    for index, character in enumerate(code):
        if character == "{":
            stack.append(index)
        elif character == "}" and stack:
            opening = stack.pop()
            pairs[opening] = index
    return pairs


def _matching_open_paren(code: str, close_paren: int) -> int | None:
    depth = 0
    for index in range(close_paren, -1, -1):
        if code[index] == ")":
            depth += 1
        elif code[index] == "(":
            depth -= 1
            if depth == 0:
                return index
    return None


def _looks_like_function_body(code: str, opening_brace: int) -> bool:
    prefix = code[:opening_brace].rstrip()
    if not prefix.endswith(")"):
        return False
    open_paren = _matching_open_paren(code, len(prefix) - 1)
    if open_paren is None:
        return False
    before = code[:open_paren].rstrip()
    if before.endswith("]"):
        return False
    return re.search(
        r"\b(?:if(?:\s+constexpr)?|for|while|switch|catch)\s*$",
        before,
    ) is None


def _looks_like_lambda_body(code: str, opening_brace: int) -> bool:
    prefix = code[:opening_brace].rstrip()
    if prefix.endswith("]"):
        return True
    if not prefix.endswith(")"):
        return False
    open_paren = _matching_open_paren(code, len(prefix) - 1)
    return (
        open_paren is not None
        and code[:open_paren].rstrip().endswith("]")
    )


IDENTIFIER_PATTERN = r"[A-Za-z_][A-Za-z0-9_]*"


def _has_unapproved_transfer(code: str, start: int, end: int) -> bool:
    """Reject nested exits except the frontend-proved geometry error block.

    RMSNorm performs one error-checked geometry query before the recipe guard;
    Softmax performs the equivalent query between that guard and the fallback
    launch.  That exact block is the only nested transfer accepted in either
    range.  This remains a structural check, not a general C++ CFG proof.
    """

    fragment = list(code[start:end])
    geometry_error = re.compile(
        r"\{\s*(?:const\s+)?(?:"
        + IDENTIFIER_PATTERN
        + r"(?:::\s*"
        + IDENTIFIER_PATTERN
        + r")*\s*[&*]?\s+)(?P<error>"
        + IDENTIFIER_PATTERN
        + r")\s*=\s*[^;{}]+;\s*"
        r"if\s*\(\s*(?P=error)\s*!=\s*(?:ACL_SUCCESS|0)\s*\)\s*"
        r"\{\s*return\s+(?P=error)\s*;\s*\}\s*\}",
        re.DOTALL,
    )
    for match in geometry_error.finditer(code, start, end):
        for index in range(match.start() - start, match.end() - start):
            if fragment[index] not in "\r\n":
                fragment[index] = " "
    return re.search(
        r"\b(?:co_return|return|goto|throw)\b", "".join(fragment)
    ) is not None


def _split_call_arguments(arguments: str) -> list[str] | None:
    """Split the frontend's simple call arguments at top-level commas."""

    parts: list[str] = []
    start = 0
    round_depth = 0
    square_depth = 0
    brace_depth = 0
    for index, character in enumerate(arguments):
        if character == "(":
            round_depth += 1
        elif character == ")":
            round_depth -= 1
        elif character == "[":
            square_depth += 1
        elif character == "]":
            square_depth -= 1
        elif character == "{":
            brace_depth += 1
        elif character == "}":
            brace_depth -= 1
        elif (
            character == ","
            and round_depth == 0
            and square_depth == 0
            and brace_depth == 0
        ):
            parts.append(arguments[start:index].strip())
            start = index + 1
        if round_depth < 0 or square_depth < 0 or brace_depth < 0:
            return None
    if round_depth != 0 or square_depth != 0 or brace_depth != 0:
        return None
    parts.append(arguments[start:].strip())
    return parts


def _generated_call_shape(
    contract: re.Match[str], marker: str
) -> tuple[str, str, str] | None:
    """Return the generated row, column, and compute-type identifiers."""

    arguments = _split_call_arguments(contract.group("arguments"))
    expected_count = {
        MARKERS[0]: 6,
        MARKERS[1]: 8,
        MARKERS[2]: 9,
    }[marker]
    if arguments is None or len(arguments) != expected_count:
        return None

    ordinary = arguments[:-1]
    if any(
        re.fullmatch(IDENTIFIER_PATTERN, argument) is None
        for argument in ordinary
    ):
        return None
    null_type = re.fullmatch(
        r"static_cast<\s*(" + IDENTIFIER_PATTERN + r")\s*\*\s*>"
        r"\s*\(\s*nullptr\s*\)",
        arguments[-1],
    )
    if null_type is None:
        return None
    return ordinary[3], ordinary[4], null_type.group(1)


def _softmax_algorithm_guard_is_valid(condition: str) -> bool:
    qualified_constant = (
        IDENTIFIER_PATTERN + r"(?:::" + IDENTIFIER_PATTERN + r")+"
    )
    return re.fullmatch(
        IDENTIFIER_PATTERN + r"==::" + qualified_constant,
        condition,
    ) is not None


def _shape_guard_is_valid(
    condition: str,
    marker: str,
    rows: str,
    columns: str,
    compute_type: str,
) -> bool:
    """Match one of the three name-independent frontend guard shapes."""

    row = re.escape(rows)
    column = re.escape(columns)
    compute = re.escape(compute_type)
    base = (
        rf"\({row}>0\)&&\({column}>0\)&&"
        rf"\({column}%(?P<pack>{IDENTIFIER_PATTERN})==0\)"
    )
    if re.fullmatch(base, condition) is not None:
        return True

    shared = (
        base
        + rf"&&\((?P<shared>{IDENTIFIER_PATTERN})>=0\)"
        + rf"&&\(static_cast<unsignedlonglong>\({column}\)"
        + rf"<=static_cast<unsignedlonglong>\((?P=shared)\)"
        + rf"/sizeof\({compute}\)\)"
    )
    if re.fullmatch(shared, condition) is not None:
        return True

    if marker == MARKERS[0]:
        warp = (
            base
            + rf"&&\({column}<=\((?P<maximum>{IDENTIFIER_PATTERN})"
            + rf"\*(?P<width>{IDENTIFIER_PATTERN})\)\)"
            + rf"&&\((?P<padding>{IDENTIFIER_PATTERN})\|\|{column}"
            + r"==\((?P=maximum)\*(?P=width)\)\)"
            + rf"&&\({row}%(?P<rows_per_access>{IDENTIFIER_PATTERN})==0\)"
        )
    else:
        warp = (
            base
            + rf"&&\({column}<=\((?P<maximum>{IDENTIFIER_PATTERN})"
            + rf"\*(?P<width>{IDENTIFIER_PATTERN})\)\)"
            + rf"&&\((?P<padding>{IDENTIFIER_PATTERN})\?{column}"
            + rf">\((?P<minimum>{IDENTIFIER_PATTERN})\*(?P=width)\)"
            + rf":{column}==\((?P=maximum)\*(?P=width)\)\)"
        )
    return re.fullmatch(warp, condition) is not None


def _control_scope(code: str, opening_brace: int) -> tuple[str, str] | None:
    """Return the control kind and normalized condition owning a brace."""

    prefix = code[:opening_brace].rstrip()
    if prefix.endswith("else"):
        return "else", ""
    if prefix.endswith("do"):
        return "do", ""
    if not prefix.endswith(")"):
        return None

    open_paren = _matching_open_paren(code, len(prefix) - 1)
    if open_paren is None:
        return None
    before = code[:open_paren].rstrip()
    match = re.search(
        r"\b(if\s+constexpr|if|for|while|switch|catch)\s*$",
        before,
    )
    if match is None:
        return None
    kind = re.sub(r"\s+", " ", match.group(1))
    condition = re.sub(
        r"\s+", "", code[open_paren + 1 : len(prefix) - 1]
    )
    return kind, condition


def _control_header_start(code: str, opening_brace: int) -> int | None:
    """Return the control keyword starting the header owning a brace."""

    prefix = code[:opening_brace].rstrip()
    if not prefix.endswith(")"):
        return None
    open_paren = _matching_open_paren(code, len(prefix) - 1)
    if open_paren is None:
        return None
    before = code[:open_paren].rstrip()
    match = re.search(
        r"\b(?:if\s+constexpr|if|for|while|switch|catch)\s*$",
        before,
    )
    return None if match is None else match.start()


def _has_direct_scope_transfer(code: str, start: int, end: int) -> bool:
    """Whether a return/goto/throw occurs at the range's direct brace depth."""

    depth = 0
    index = start
    transfer = re.compile(r"\b(?:co_return|return|goto|throw)\b")
    while index < end:
        character = code[index]
        if character == "{":
            depth += 1
            index += 1
            continue
        if character == "}":
            if depth > 0:
                depth -= 1
            index += 1
            continue
        if depth == 0:
            match = transfer.match(code, index)
            if match is not None:
                return True
        index += 1
    return False


def _direct_launch_positions(
    code: str, opening_brace: int, closing_brace: int
) -> list[int]:
    """Collect launch tokens at the function body's direct brace depth."""

    depth = 0
    launches: list[int] = []
    index = opening_brace + 1
    while index < closing_brace:
        character = code[index]
        if character == "{":
            depth += 1
        elif character == "}" and depth > 0:
            depth -= 1
        elif depth == 0 and code.startswith("<<<", index):
            launches.append(index)
            index += 2
        index += 1
    return launches


def _direct_statement_prefix(
    code: str, function_open: int, position: int
) -> str:
    """Return the direct-scope statement bytes preceding ``position``."""

    brace_depth = 0
    paren_depth = 0
    bracket_depth = 0
    statement_start = function_open + 1
    index = statement_start
    while index < position:
        character = code[index]
        if character == "{":
            brace_depth += 1
        elif character == "}":
            if brace_depth > 0:
                brace_depth -= 1
                if brace_depth == 0:
                    statement_start = index + 1
        elif brace_depth == 0:
            if character == "(":
                paren_depth += 1
            elif character == ")" and paren_depth > 0:
                paren_depth -= 1
            elif character == "[":
                bracket_depth += 1
            elif character == "]" and bracket_depth > 0:
                bracket_depth -= 1
            elif (
                character == ";"
                and paren_depth == 0
                and bracket_depth == 0
            ):
                statement_start = index + 1
        index += 1

    return code[statement_start:position].strip()


def _launch_is_uncontrolled_direct_statement(
    code: str, function_open: int, launch: int
) -> bool:
    """Require the frontend's direct kernel-callee expression statement."""

    prefix = _direct_statement_prefix(code, function_open, launch)
    qualified_name = (
        r"(?:::)?" + IDENTIFIER_PATTERN
        + r"(?:::(?:" + IDENTIFIER_PATTERN + r"))*"
    )
    # The proved wrappers launch a named kernel, optionally with an explicit
    # template-argument list.  Parentheses or operators before `<<<` would put
    # the launch in a control/comma/unevaluated expression instead of that
    # direct expression statement, so reject them conservatively.
    direct_callee = re.compile(
        qualified_name + r"(?:\s*<[^(){};=?&|!]*>)?\s*$",
        re.DOTALL,
    )
    return direct_callee.fullmatch(prefix) is not None


def _generated_guard_ancestors_are_valid(
    code: str,
    contract: re.Match[str],
    marker: str,
    brace_pairs: dict[int, int],
) -> tuple[int, tuple[int, ...]] | None:
    """Recognize only the fixed guard topology emitted by the frontend."""

    enclosing = sorted(
        (
            opening
            for opening, closing in brace_pairs.items()
            if opening < contract.start() and contract.end() < closing
        ),
        reverse=True,
    )
    function_index = next(
        (
            index
            for index, opening in enumerate(enclosing)
            if _looks_like_function_body(code, opening)
        ),
        None,
    )
    if function_index is None:
        return None
    function_open = enclosing[function_index]
    ancestors = tuple(enclosing[:function_index])
    if any(_looks_like_lambda_body(code, opening) for opening in enclosing):
        return None

    scopes = tuple(_control_scope(code, opening) for opening in ancestors)
    call_shape = _generated_call_shape(contract, marker)
    if call_shape is None:
        return None
    rows, columns, compute_type = call_shape
    if marker == MARKERS[0]:
        if len(scopes) != 2 or any(scope is None for scope in scopes):
            return None
        inner, outer = scopes
        if (
            inner[0] != "if constexpr"
            or not _softmax_algorithm_guard_is_valid(inner[1])
        ):
            return None
        if (
            outer[0] != "if"
            or not _shape_guard_is_valid(
                outer[1], marker, rows, columns, compute_type
            )
        ):
            return None
    else:
        if len(scopes) != 1 or scopes[0] is None:
            return None
        outer = scopes[0]
        if (
            outer[0] != "if"
            or not _shape_guard_is_valid(
                outer[1], marker, rows, columns, compute_type
            )
        ):
            return None

    outermost = ancestors[-1]
    header_start = _control_header_start(code, outermost)
    if header_start is None:
        return None
    # The generated shape guard must itself be a direct function-body
    # statement.  This catches braceless `if`/`for`/`while` parents that do not
    # contribute an enclosing brace to the ancestor list.
    if _direct_statement_prefix(code, function_open, header_start):
        return None
    if _has_unapproved_transfer(
        code, function_open + 1, header_start
    ):
        return None

    # A transfer before the generated declaration at any owning direct scope
    # could make the textual contract unreachable.  Nested transfers are not
    # treated as control-flow proof; this scanner remains deliberately
    # structural and conservative about the topology it recognizes.
    for opening in (function_open, *ancestors):
        if _has_direct_scope_transfer(code, opening + 1, contract.start()):
            return None
    return function_open, ancestors


def _dispatch_fallback_function(
    code: str,
    contract: re.Match[str],
    marker: str,
    brace_pairs: dict[int, int],
) -> int | None:
    """Return the generated wrapper owning one ordered direct fallback launch."""

    context = _generated_guard_ancestors_are_valid(
        code, contract, marker, brace_pairs
    )
    if context is None:
        return None
    function_open, ancestors = context
    function_close = brace_pairs[function_open]
    direct_launches = _direct_launch_positions(
        code, function_open, function_close
    )
    if len(direct_launches) != 1:
        return None
    launch = direct_launches[0]
    if launch <= contract.end():
        return None
    if not _launch_is_uncontrolled_direct_statement(
        code, function_open, launch
    ):
        return None

    outermost = ancestors[-1]
    if _has_unapproved_transfer(
        code, brace_pairs[outermost] + 1, launch
    ):
        return None
    for opening in ancestors:
        if _has_unapproved_transfer(
            code, contract.end(), brace_pairs[opening]
        ):
            return None

    # Check the direct function scope through the fallback, and the remainder
    # of every generated guard scope after the handled-return contract.
    if _has_direct_scope_transfer(code, function_open + 1, launch):
        return None
    for opening in ancestors:
        if _has_direct_scope_transfer(
            code, contract.end(), brace_pairs[opening]
        ):
            return None
    return function_open


def inspect_recipe_output(
    output: bytes, input_kernel_launch_count: int = 1
) -> dict[str, int | bool | str]:
    """Inspect generated code for a complete SIMD dispatch plus SIMT fallback."""

    try:
        output.decode("utf-8", errors="strict")
        valid_utf8 = True
    except UnicodeDecodeError:
        valid_utf8 = False
    if (
        not valid_utf8
        or output.startswith(UTF8_BOM)
        or b"\0" in output
        or TRIGRAPH_PATTERN.search(output) is not None
    ):
        return {
            "dispatch_execution_kind": "none",
            "dispatch_function_count": 0,
            "dispatch_with_simt_fallback_count": 0,
            "kernel_launch_count": 0,
            "recipe_call_token_count": 0,
            "recognized": False,
            "simd_header_count": 0,
            "try_rmsnorm_hybrid_count": 0,
            "try_rmsnorm_simd_count": 0,
            "try_softmax_hybrid_count": 0,
            "try_softmax_simd_count": 0,
            "try_layernorm_hybrid_count": 0,
            "try_layernorm_simd_count": 0,
        }

    code = _normalize_digraph_tokens(_code_only_text(output))
    conditional_inspectable = _blank_preprocessor_conditionals(code)
    header_count = _simd_header_count(conditional_inspectable)
    inspectable = _blank_preprocessor_directives(conditional_inspectable)
    softmax_contracts = _dispatch_contract_matches(inspectable, MARKERS[0])
    rmsnorm_contracts = _dispatch_contract_matches(inspectable, MARKERS[1])
    layernorm_contracts = _dispatch_contract_matches(inspectable, MARKERS[2])
    contracts = [
        *((contract, MARKERS[0]) for contract in softmax_contracts),
        *((contract, MARKERS[1]) for contract in rmsnorm_contracts),
        *((contract, MARKERS[2]) for contract in layernorm_contracts),
    ]
    softmax_count = len(softmax_contracts)
    rmsnorm_count = len(rmsnorm_contracts)
    layernorm_count = len(layernorm_contracts)
    recipe_call_token_count = _recipe_call_token_count(inspectable)
    kernel_launch_count = inspectable.count("<<<")
    brace_pairs = _matching_braces(inspectable)
    fallback_functions = [
        _dispatch_fallback_function(
            inspectable,
            contract,
            marker,
            brace_pairs,
        )
        for contract, marker in contracts
    ]
    fallback_count = sum(function is not None for function in fallback_functions)
    dispatch_function_count = len(
        {function for function in fallback_functions if function is not None}
    )
    dispatch_count = softmax_count + rmsnorm_count + layernorm_count
    recognized = (
        header_count == 1
        and dispatch_count > 0
        and recipe_call_token_count == dispatch_count
        and fallback_count == dispatch_count
        and dispatch_function_count == dispatch_count
        and input_kernel_launch_count > 0
        and kernel_launch_count == input_kernel_launch_count
    )
    return {
        "dispatch_execution_kind": (
            "intra-kernel-simd-simt" if dispatch_count > 0 else "none"
        ),
        "dispatch_function_count": dispatch_function_count,
        "dispatch_with_simt_fallback_count": fallback_count,
        "kernel_launch_count": kernel_launch_count,
        "recipe_call_token_count": recipe_call_token_count,
        "recognized": recognized,
        "simd_header_count": header_count,
        "try_rmsnorm_hybrid_count": rmsnorm_count,
        # Retained as a v3 compatibility alias for existing report readers.
        "try_rmsnorm_simd_count": rmsnorm_count,
        "try_softmax_hybrid_count": softmax_count,
        # Retained as a v3 compatibility alias for existing report readers.
        "try_softmax_simd_count": softmax_count,
        "try_layernorm_hybrid_count": layernorm_count,
        # Retained with the same compatibility convention as the v3 fields.
        "try_layernorm_simd_count": layernorm_count,
    }


def _write_exclusive(path: Path, data: bytes) -> tuple[int, int]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise GenerationError(f"refusing to replace existing destination: {path}") from exc
    except OSError as exc:
        raise GenerationError(f"cannot create destination {path}: {exc}") from exc

    metadata = os.fstat(descriptor)
    identity = (metadata.st_dev, metadata.st_ino)
    try:
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            if written == 0:
                raise GenerationError(f"short write while creating destination: {path}")
            offset += written
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        _remove_if_same_file(path, identity)
        raise
    else:
        os.close(descriptor)
    return identity


def _remove_if_same_file(path: Path, identity: tuple[int, int]) -> None:
    """Roll back only the file created by this process."""

    try:
        metadata = path.lstat()
    except OSError:
        return
    if (metadata.st_dev, metadata.st_ino) == identity:
        try:
            path.unlink()
        except OSError:
            pass


def _publish_pair(
    output_path: Path,
    output_data: bytes,
    report_path: Path,
    report_data: bytes,
) -> None:
    created: list[tuple[Path, tuple[int, int]]] = []
    try:
        created.append((output_path, _write_exclusive(output_path, output_data)))
        created.append((report_path, _write_exclusive(report_path, report_data)))
    except BaseException:
        for path, identity in reversed(created):
            _remove_if_same_file(path, identity)
        raise


def _run_converter(
    executable: Path,
    source: Path,
    candidate: Path,
    mode: str,
    cuda_path: str | None,
    clang_resource_directory: str | None,
    include_dirs: Sequence[str],
    compiler_args: Sequence[str],
) -> list[str]:
    target_policy, simt_math, target_recipe = MODE_OPTIONS[mode]
    command = [
        str(executable),
        str(source),
        f"--target-policy={target_policy}",
        f"--simt-math={simt_math}",
    ]
    if target_recipe is not None:
        command.append(f"--target-recipe={target_recipe}")
    if cuda_path is not None:
        command.append(f"--cuda-path={cuda_path}")
    if clang_resource_directory is not None:
        command.append(
            f"--clang-resource-directory={clang_resource_directory}"
        )
    command.extend(["-o", str(candidate), "--"])
    command.extend(f"-I{directory}" for directory in include_dirs)
    command.extend(compiler_args)

    try:
        result = subprocess.run(command, check=False)
    except OSError as exc:
        raise GenerationError(f"failed to launch ascify-clang: {exc}") from exc
    if result.returncode != 0:
        raise GenerationError(f"ascify-clang failed with exit code {result.returncode}")
    return command


def _normalized_converter_argv(
    command: Sequence[str], executable: Path, source: Path, candidate: Path
) -> list[str]:
    """Replace ephemeral paths while preserving every converter option."""

    replacements = {
        str(executable): "$ASCIFY_CLANG",
        str(source): "$SOURCE",
        str(candidate): "$OUTPUT",
    }
    return [replacements.get(argument, argument) for argument in command]


def generate(args: argparse.Namespace, compiler_tail: Sequence[str]) -> tuple[Path, Path]:
    source = args.source
    output = args.output
    report = args.report or Path(f"{output}.report.json")

    if output.suffix != ".cce":
        raise GenerationError(f"output must use the .cce suffix: {output}")
    if report.suffix != ".json":
        raise GenerationError(f"report must use the .json suffix: {report}")
    if os.path.abspath(output) == os.path.abspath(report):
        raise GenerationError("output and report destinations must be different")
    if args.require_recipe and args.mode != "simd-simt":
        raise GenerationError("--require-recipe is only valid with --mode=simd-simt")

    source_data, source_identity = _read_regular_file(source, "source")
    _reject_invalid_utf8(source_data, "source")
    _reject_utf8_bom(source_data, "source")
    _reject_nul_bytes(source_data, "source")
    _reject_trigraph_spellings(source_data, "source")
    input_sha256 = _sha256(source_data)
    source_code = _normalize_digraph_tokens(_code_only_text(source_data))
    input_kernel_launch_count = source_code.count("<<<")
    if args.mode == "simd-simt" and (
        _simd_header_count(source_code) != 0
        or _recipe_call_token_count(source_code) != 0
    ):
        raise GenerationError(
            "source already contains row-wise SIMD recipe code; refusing to "
            "attribute pre-existing dispatch to this conversion"
        )
    executable = _resolve_ascify_clang(args.ascify_clang)
    executable_data, executable_identity = _read_regular_file(
        executable, "ascify-clang"
    )
    frontend_sha256 = _sha256(executable_data)
    generator_path = Path(__file__).resolve()
    generator_data, generator_identity = _read_regular_file(
        generator_path, "generator"
    )
    generator_sha256 = _sha256(generator_data)

    # Check before the expensive conversion, then use O_EXCL again at publication
    # time to close the check/create race.
    _ensure_new_destination(output, "output")
    _ensure_new_destination(report, "report")

    compiler_args = [*args.compiler_arg, *compiler_tail]
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.ascify-", dir=output.parent
    ) as temporary_directory:
        candidate = Path(temporary_directory) / output.name
        converter_argv = _run_converter(
            executable,
            source,
            candidate,
            args.mode,
            args.cuda_path,
            args.clang_resource_directory,
            args.include,
            compiler_args,
        )
        output_data, _ = _read_regular_file(candidate, "ascify-clang output")
        if not output_data:
            raise GenerationError("ascify-clang produced an empty output file")
        _reject_invalid_utf8(output_data, "ascify-clang output")
        _reject_utf8_bom(output_data, "ascify-clang output")
        _reject_nul_bytes(output_data, "ascify-clang output")
        _reject_trigraph_spellings(output_data, "ascify-clang output")
        normalized_converter_argv = _normalized_converter_argv(
            converter_argv, executable, source, candidate
        )

    # Do not publish a report whose input digest no longer describes the source
    # that was presented to the converter.
    source_after, source_identity_after = _read_regular_file(source, "source")
    if source_identity_after != source_identity or _sha256(source_after) != input_sha256:
        raise GenerationError("source changed while ascify-clang was running")
    executable_after, executable_identity_after = _read_regular_file(
        executable, "ascify-clang"
    )
    if (
        executable_identity_after != executable_identity
        or _sha256(executable_after) != frontend_sha256
    ):
        raise GenerationError("ascify-clang changed while conversion was running")
    generator_after, generator_identity_after = _read_regular_file(
        generator_path, "generator"
    )
    if (
        generator_identity_after != generator_identity
        or _sha256(generator_after) != generator_sha256
    ):
        raise GenerationError("generator changed while conversion was running")

    inspection = inspect_recipe_output(
        output_data, input_kernel_launch_count
    )
    if args.require_recipe and not inspection["recognized"]:
        raise GenerationError(
            "generated output did not contain a structurally valid SIMD dispatch with "
            "handled/status ownership and a later same-wrapper SIMT kernel launch"
        )

    output_sha256 = _sha256(output_data)
    conversion_basis = {
        "compiler_arguments": compiler_args,
        "frontend_sha256": frontend_sha256,
        "generator_sha256": generator_sha256,
        "input_sha256": input_sha256,
        "mode": args.mode,
        "normalized_converter_argv": normalized_converter_argv,
        "output_sha256": output_sha256,
        "schema": REPORT_SCHEMA,
        "target_abi": TARGET_ABI,
        **inspection,
    }
    conversion_id = _sha256(
        json.dumps(
            conversion_basis, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )
    report_payload = {
        "ascify_binary_path": str(executable),
        "ascify_binary_sha256": frontend_sha256,
        "compiler_arguments": compiler_args,
        "conversion_id": f"sha256:{conversion_id}",
        "converter_argv": converter_argv,
        "generator_path": str(generator_path),
        "generator_sha256": generator_sha256,
        "include_directories": list(args.include),
        "input_sha256": input_sha256,
        "input_kernel_launch_count": input_kernel_launch_count,
        "mode": args.mode,
        "normalized_converter_argv": normalized_converter_argv,
        "output_path": str(output.resolve()),
        "output_sha256": output_sha256,
        "schema": REPORT_SCHEMA,
        "source_path": str(source.resolve()),
        "target_abi": TARGET_ABI,
        **inspection,
    }
    report_data = (
        json.dumps(report_payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _publish_pair(output, output_data, report, report_data)
    return output, report


def main(argv: Sequence[str] | None = None) -> int:
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    wrapper_arguments, compiler_tail = _split_compiler_tail(raw_arguments)
    parser = _build_parser()
    args = parser.parse_args(wrapper_arguments)
    try:
        output, report = generate(args, compiler_tail)
    except GenerationError as exc:
        print(f"generate_rowwise_cce.py: error: {exc}", file=sys.stderr)
        return 2
    print(f"generated {output}")
    print(f"report {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
