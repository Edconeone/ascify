#!/usr/bin/env python3
"""Fail-closed mutation gate for the dav-c310 rowwise recipes.

The test intentionally uses the full OneFlow Softmax and RMSNorm headers.  A
canonical header must produce all three direct forward recipes.  Each mutation
preserves enough of the original algebra/topology to catch an overly permissive
matcher, but changes one correctness-critical property.  Such a mutation must
remove exactly the affected recipe(s).

All generated inputs, converter outputs, logs and the result manifest are kept
under one caller-selected work directory.
"""

import argparse
import csv
import dataclasses
import hashlib
import pathlib
import re
import subprocess
import sys
from typing import Callable, Iterable, List, Optional, Sequence, Tuple


SOFTMAX_RECIPE = "::ascify::target::dav_c310::TrySoftmax("
RMS_RECIPE = "::ascify::target::dav_c310::TryRmsNorm("
DEBUG_MARKER = "ASCIFY_RECIPE_DEBUG"


class MutationError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class MutationCase:
    family: str
    name: str
    expected_recipes: int
    contract: str
    mutate: Callable[[str], str]
    validate_output: Optional[Callable[[str], bool]] = None


@dataclasses.dataclass(frozen=True)
class CaseResult:
    family: str
    name: str
    expected_recipes: int
    actual_recipes: int
    return_code: int
    debug_markers: int
    status: str
    contract: str


@dataclasses.dataclass(frozen=True)
class Provenance:
    label: str
    path: pathlib.Path
    sha256: str


def replace_exact(text: str, old: str, new: str, count: int = 1) -> str:
    actual = text.count(old)
    if actual != count:
        raise MutationError(
            "expected {} occurrence(s) of mutation anchor, found {}\n{}".format(
                count, actual, old
            )
        )
    return text.replace(old, new)


def _matching_brace(text: str, opening: int) -> int:
    """Find a C/C++ closing brace while ignoring comments and literals."""
    if opening >= len(text) or text[opening] != "{":
        raise MutationError("function body does not start with an opening brace")
    depth = 0
    index = opening
    state = "code"
    quote = ""
    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""
        if state == "line_comment":
            if char == "\n":
                state = "code"
        elif state == "block_comment":
            if char == "*" and next_char == "/":
                state = "code"
                index += 1
        elif state == "literal":
            if char == "\\":
                index += 1
            elif char == quote:
                state = "code"
        else:
            if char == "/" and next_char == "/":
                state = "line_comment"
                index += 1
            elif char == "/" and next_char == "*":
                state = "block_comment"
                index += 1
            elif char in ("'", '"'):
                state = "literal"
                quote = char
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return index
        index += 1
    raise MutationError("unterminated function body")


def function_span(text: str, signature: str) -> Tuple[int, int]:
    if text.count(signature) != 1:
        raise MutationError(
            "expected one function signature anchor, found {}: {}".format(
                text.count(signature), signature
            )
        )
    start = text.index(signature)
    opening = text.index("{", start + len(signature))
    return start, _matching_brace(text, opening) + 1


def replace_in_function(
    text: str,
    signature: str,
    old: str,
    new: str,
    count: int = 1,
) -> str:
    start, end = function_span(text, signature)
    body = text[start:end]
    replaced = replace_exact(body, old, new, count)
    return text[:start] + replaced + text[end:]


def insert_before_function_template(
    text: str, signature: str, insertion: str
) -> str:
    start, _ = function_span(text, signature)
    template_start = text.rfind("template<", 0, start)
    if template_start < 0 or "{" in text[template_start:start]:
        raise MutationError(
            "cannot locate the template declaration for: " + signature
        )
    return text[:template_start] + insertion + text[template_start:]


def insert_before_function(
    text: str, signature: str, insertion: str
) -> str:
    start, _ = function_span(text, signature)
    return text[:start] + insertion + text[start:]


def insert_at_function_body_start(
    text: str, signature: str, insertion: str
) -> str:
    start, _ = function_span(text, signature)
    opening = text.index("{", start + len(signature))
    return text[: opening + 1] + "\n" + insertion + text[opening + 1 :]


def replace_identifiers_in_function(
    text: str,
    signature: str,
    replacements: Sequence[Tuple[str, str]],
) -> str:
    start, end = function_span(text, signature)
    body = text[start:end]
    for old, new in replacements:
        body, count = re.subn(r"\b" + re.escape(old) + r"\b", new, body)
        if count == 0:
            raise MutationError("rename anchor is absent in function: " + old)
    return text[:start] + body + text[end:]


def unchanged(text: str) -> str:
    return text


def rename_identifiers(text: str, replacements: Sequence[Tuple[str, str]]) -> str:
    """Rename complete C++ identifiers without changing the program topology."""
    for old, new in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
        if old not in text:
            raise MutationError("rename anchor is absent: " + old)
        text = text.replace(old, new)
    return text


SOFTMAX_BLOCK = "__global__ void SoftmaxBlockSMemImpl("
SOFTMAX_WARP_REDUCE = "__inline__ __device__ T WarpAllReduce(T val)"
SOFTMAX_BLOCK_REDUCE = "__inline__ __device__ T BlockAllReduce(T val)"
SOFTMAX_WARP_LAUNCH = "inline cudaError_t LaunchSoftmaxWarpImpl("
SOFTMAX_GEOMETRY = (
    "inline cudaError_t GetNumBlocks(int64_t block_size, int64_t max_blocks, "
    "int64_t waves,"
)

RMS_WARP = "__global__ void RmsNormWarpImpl("
RMS_BLOCK = "__global__ void RmsNormBlockSMemImpl("


def softmax_renamed_spelling(text: str) -> str:
    return rename_identifiers(
        text,
        (
            ("LaunchSoftmaxBlockUncachedImpl", "SubmitProbabilityWideRow"),
            ("LaunchSoftmaxBlockSMemImpl", "SubmitProbabilitySharedRow"),
            ("LaunchSoftmaxWarpImpl", "SubmitProbabilityLaneGroup"),
            ("SoftmaxBlockUncachedImpl", "ProbabilityWideRowKernel"),
            ("SoftmaxBlockSMemImpl", "ProbabilitySharedRowKernel"),
            ("SoftmaxWarpImpl", "ProbabilityLaneGroupKernel"),
        ),
    )


def rms_renamed_spelling(text: str) -> str:
    return rename_identifiers(
        text,
        (
            ("LaunchRmsNormBlockUncachedImpl", "SubmitQuadraticScaleWideRow"),
            ("LaunchRmsNormBlockSMemImpl", "SubmitQuadraticScaleSharedRow"),
            ("LaunchRmsNormWarpImpl", "SubmitQuadraticScaleLaneGroup"),
            ("RmsNormBlockUncachedImpl", "QuadraticScaleWideRowKernel"),
            ("RmsNormBlockSMemImpl", "QuadraticScaleSharedRowKernel"),
            ("RmsNormWarpImpl", "QuadraticScaleLaneGroupKernel"),
        ),
    )


def softmax_miss_first_row(text: str) -> str:
    return replace_in_function(
        text,
        SOFTMAX_BLOCK,
        "for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {",
        "for (int64_t row = blockIdx.x + 1; row < rows; row += gridDim.x) {",
    )


def softmax_miss_final_pack(text: str) -> str:
    return replace_in_function(
        text,
        SOFTMAX_BLOCK,
        "const int num_packs = cols / pack_size;",
        "const int num_packs = cols / pack_size - 1;",
    )


def softmax_direct_local_sum(text: str) -> str:
    return replace_in_function(
        text,
        SOFTMAX_BLOCK,
        "pack[i] = Div(buf[i * num_packs + pack_id], row_sum);",
        "pack[i] = Div(buf[i * num_packs + pack_id], thread_sum);",
    )


def softmax_bad_sum_functor(text: str) -> str:
    return replace_exact(
        text,
        "__device__ __forceinline__ T operator()(const T& a, const T& b) const "
        "{ return a + b; }",
        "__device__ __forceinline__ T operator()(const T& a, const T& b) const "
        "{ return a; }",
    )


def softmax_warp_reducer_identity(text: str) -> str:
    return replace_in_function(
        text,
        SOFTMAX_WARP_REDUCE,
        "val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));",
        "val = val;",
    )


def softmax_block_reducer_identity(text: str) -> str:
    return replace_in_function(
        text,
        SOFTMAX_BLOCK_REDUCE,
        "T result = BlockReduce(temp_storage).Reduce(val, ReductionOp<T>());",
        "T result = val;",
    )


SOFTMAX_SUM_REDUCTION = (
    "const ComputeType row_sum = "
    "BlockAllReduce<SumOp, ComputeType, block_size>(thread_sum);"
)


def softmax_dead_reducer(text: str) -> str:
    replacement = """ComputeType row_sum = thread_sum;
    if (false) {
      row_sum = BlockAllReduce<SumOp, ComputeType, block_size>(thread_sum);
    }"""
    return replace_in_function(
        text, SOFTMAX_BLOCK, SOFTMAX_SUM_REDUCTION, replacement
    )


def softmax_conditional_reducer(text: str) -> str:
    replacement = """ComputeType row_sum = thread_sum;
    if ((threadIdx.x & 1) != 0) {
      row_sum = BlockAllReduce<SumOp, ComputeType, block_size>(thread_sum);
    }"""
    return replace_in_function(
        text, SOFTMAX_BLOCK, SOFTMAX_SUM_REDUCTION, replacement
    )


SOFTMAX_LOAD = (
    "load.template load<pack_size>(pack, row, pack_id * pack_size);"
)
SOFTMAX_STORE = (
    "store.template store<pack_size>(pack, row, pack_id * pack_size);"
)


def softmax_shifted_row(text: str) -> str:
    text = replace_in_function(
        text,
        SOFTMAX_BLOCK,
        SOFTMAX_LOAD,
        "load.template load<pack_size>(pack, row + 1, pack_id * pack_size);",
    )
    return replace_in_function(
        text,
        SOFTMAX_BLOCK,
        SOFTMAX_STORE,
        "store.template store<pack_size>(pack, row + 1, pack_id * pack_size);",
    )


def softmax_shifted_column(text: str) -> str:
    text = replace_in_function(
        text,
        SOFTMAX_BLOCK,
        SOFTMAX_LOAD,
        "load.template load<pack_size>(pack, row, "
        "pack_id * pack_size + pack_size);",
    )
    return replace_in_function(
        text,
        SOFTMAX_BLOCK,
        SOFTMAX_STORE,
        "store.template store<pack_size>(pack, row, "
        "pack_id * pack_size + pack_size);",
    )


def softmax_extra_load(text: str) -> str:
    return replace_in_function(
        text,
        SOFTMAX_BLOCK,
        SOFTMAX_LOAD,
        SOFTMAX_LOAD
        + "\n      load.template load<pack_size>(pack, row, "
        "pack_id * pack_size + pack_size);",
    )


def softmax_extra_store(text: str) -> str:
    return replace_in_function(
        text,
        SOFTMAX_BLOCK,
        SOFTMAX_STORE,
        SOFTMAX_STORE
        + "\n      store.template store<pack_size>(pack, row, "
        "pack_id * pack_size + pack_size);",
    )


def softmax_unconditional_continue(text: str) -> str:
    loop = "for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {"
    return replace_in_function(
        text, SOFTMAX_BLOCK, loop, loop + "\n    continue;"
    )


def softmax_one_sided_store(text: str) -> str:
    return replace_in_function(
        text,
        SOFTMAX_BLOCK,
        SOFTMAX_STORE,
        "if ((threadIdx.x & 1) != 0) {\n"
        "        " + SOFTMAX_STORE + "\n"
        "      }",
    )


def softmax_fake_thread_index(text: str) -> str:
    return replace_in_function(
        text,
        SOFTMAX_BLOCK,
        "const int tid = threadIdx.x;",
        "struct AscifyMutationThreadIndex { unsigned int x; };\n"
        "  const AscifyMutationThreadIndex threadIdx = {0};\n"
        "  const int tid = threadIdx.x;",
    )


def softmax_fake_dim3(text: str) -> str:
    fake_type = """static int ascify_mutation_dim3_effect = 0;
struct dim3 : ::dim3 {
  dim3(unsigned int x, unsigned int y) : ::dim3(x, y, 1) {
    ++ascify_mutation_dim3_effect;
  }
};

"""
    return insert_before_function_template(
        text, SOFTMAX_WARP_LAUNCH, fake_type
    )


def softmax_zero_grid(text: str) -> str:
    assignment = """*num_blocks =
      std::max<int>(1, std::min<int64_t>(max_blocks, sm_count * tpm / block_size * waves));"""
    return replace_in_function(
        text, SOFTMAX_GEOMETRY, assignment, "*num_blocks = 0;"
    )


def softmax_early_geometry_error(text: str) -> str:
    return insert_at_function_body_start(
        text,
        SOFTMAX_GEOMETRY,
        "  return cudaErrorInvalidValue;",
    )


def softmax_mutated_failure_return(text: str) -> str:
    return replace_in_function(
        text,
        SOFTMAX_GEOMETRY,
        "if (err != cudaSuccess) { return err; }",
        "if (err != cudaSuccess) { err = cudaSuccess; return err; }",
        count=3,
    )


def softmax_bare_wrapper_error_check(text: str) -> str:
    return replace_in_function(
        text,
        SOFTMAX_WARP_LAUNCH,
        "if (err != cudaSuccess) { return err; }",
        "if (err != cudaSuccess) return err;",
    )


def softmax_wrapper_atomic_store(text: str) -> str:
    text = insert_at_function_body_start(
        text,
        SOFTMAX_WARP_LAUNCH,
        "  __atomic_store_n(&ascify_mutation_atomic_observable, 1, "
        "__ATOMIC_RELAXED);",
    )
    return insert_before_function_template(
        text,
        SOFTMAX_WARP_LAUNCH,
        "static int ascify_mutation_atomic_observable = 0;\n",
    )


def softmax_geometry_atomic_store(text: str) -> str:
    text = insert_at_function_body_start(
        text,
        SOFTMAX_GEOMETRY,
        "  __atomic_store_n(&ascify_mutation_geometry_atomic, 1, "
        "__ATOMIC_RELAXED);",
    )
    return insert_before_function(
        text,
        SOFTMAX_GEOMETRY,
        "static int ascify_mutation_geometry_atomic = 0;\n",
    )


def softmax_wrapper_volatile_cast_read(text: str) -> str:
    text = insert_at_function_body_start(
        text,
        SOFTMAX_WARP_LAUNCH,
        "  (void)*static_cast<volatile int*>(\n"
        "      &ascify_mutation_volatile_observable);",
    )
    return insert_before_function_template(
        text,
        SOFTMAX_WARP_LAUNCH,
        "static int ascify_mutation_volatile_observable = 0;\n",
    )


def softmax_macro_body_insertion(text: str) -> str:
    start, _ = function_span(text, SOFTMAX_WARP_LAUNCH)
    opening = text.index(
        "{", start + len(SOFTMAX_WARP_LAUNCH)
    )
    text = (
        text[:opening]
        + "ASCIFY_MUTATION_WRAPPER_BODY"
        + text[opening + 1 :]
    )
    return insert_before_function_template(
        text,
        SOFTMAX_WARP_LAUNCH,
        "#define ASCIFY_MUTATION_WRAPPER_BODY {\n",
    )


GENERATED_RESULT_BASE = "ascify_dav_c310_recipe_result"
GENERATED_RESULT_COLLISION = GENERATED_RESULT_BASE + "_generated_4"


def softmax_generated_name_collision(text: str) -> str:
    text = replace_identifiers_in_function(
        text,
        SOFTMAX_WARP_LAUNCH,
        (
            ("stream", GENERATED_RESULT_BASE),
            ("load", GENERATED_RESULT_BASE + "_generated"),
            ("store", GENERATED_RESULT_BASE + "_generated_2"),
        ),
    )
    text = insert_at_function_body_start(
        text,
        SOFTMAX_WARP_LAUNCH,
        "  int {} = 0;".format(
            GENERATED_RESULT_BASE + "_generated_3"
        ),
    )
    return insert_before_function_template(
        text,
        SOFTMAX_WARP_LAUNCH,
        "#define {} ascify_mutation_macro_collision\n".format(
            GENERATED_RESULT_COLLISION
        ),
    )


def softmax_template_parameter_collision(text: str) -> str:
    start, end = function_span(text, SOFTMAX_WARP_LAUNCH)
    template_start = text.rfind("template<", 0, start)
    if template_start < 0 or "{" in text[template_start:start]:
        raise MutationError(
            "cannot locate wrapper template parameter list"
        )
    declaration = text[template_start:end]
    declaration, count = re.subn(
        r"\balgorithm\b",
        GENERATED_RESULT_BASE,
        declaration,
    )
    if count == 0:
        raise MutationError(
            "wrapper template parameter algorithm is absent"
        )
    return text[:template_start] + declaration + text[end:]


def softmax_member_wrapper_duplicate(text: str) -> str:
    start, end = function_span(text, SOFTMAX_WARP_LAUNCH)
    template_start = text.rfind("template<", 0, start)
    if template_start < 0 or "{" in text[template_start:start]:
        raise MutationError(
            "cannot locate wrapper template declaration"
        )
    declaration = text[template_start:end]
    duplicate = (
        "struct AscifyMutationWrapperOwner {\n"
        + declaration
        + "\n};\n\n"
    )
    return text[:template_start] + duplicate + text[template_start:]


def validates_generated_name_collision(converted: str) -> bool:
    return all(
        token in converted
        for token in (
            '#pragma push_macro("{}")'.format(
                GENERATED_RESULT_COLLISION
            ),
            "#undef " + GENERATED_RESULT_COLLISION,
            "const auto " + GENERATED_RESULT_COLLISION + " =",
            '#pragma pop_macro("{}")'.format(
                GENERATED_RESULT_COLLISION
            ),
        )
    )


def validates_template_parameter_collision(converted: str) -> bool:
    result = GENERATED_RESULT_BASE + "_generated"
    return (
        "const auto " + result + " =" in converted
        and '#pragma push_macro("{}")'.format(result) in converted
        and "#undef " + result in converted
    )


def validates_rms_post_geometry(converted: str) -> bool:
    cursor = 0
    for _ in range(3):
        geometry = converted.find(
            "layer_norm::GetNumBlocks(", cursor
        )
        recipe = converted.find(RMS_RECIPE, cursor)
        if geometry < 0 or recipe < 0 or recipe < geometry:
            return False
        error_return = converted.find(
            "return err", geometry, recipe
        )
        geometry_block_close = converted.find(
            "\n  }", error_return, recipe
        )
        launch = converted.find("<<<", recipe)
        if (
            error_return < 0
            or geometry_block_close < 0
            or launch < 0
        ):
            return False
        cursor = recipe + len(RMS_RECIPE)
    return converted.count(RMS_RECIPE) == 3


RMS_REDUCTION = """ComputeType row_square_sum =
        layer_norm::BlockAllReduce<layer_norm::SumOp, ComputeType, block_size>(thread_square_sum);"""


def rms_direct_local_denominator(text: str) -> str:
    return replace_in_function(
        text,
        RMS_BLOCK,
        "layer_norm::Div(row_square_sum, static_cast<ComputeType>(ncol));",
        "layer_norm::Div(thread_square_sum, static_cast<ComputeType>(ncol));",
    )


def rms_dead_reducer(text: str) -> str:
    replacement = """ComputeType row_square_sum = thread_square_sum;
    if (false) {
      row_square_sum =
          layer_norm::BlockAllReduce<layer_norm::SumOp, ComputeType, block_size>(
              thread_square_sum);
    }"""
    return replace_in_function(text, RMS_BLOCK, RMS_REDUCTION, replacement)


def rms_fake_inverse_lane(text: str) -> str:
    leader = "if (threadIdx.x == 0) { inv_rms[row] = row_inv_rms; }"
    replacement = """{
      struct AscifyMutationThreadIndex { unsigned int x; };
      const AscifyMutationThreadIndex threadIdx = {1};
      if (threadIdx.x == 0) { inv_rms[row] = row_inv_rms; }
    }"""
    return replace_in_function(text, RMS_BLOCK, leader, replacement)


def rms_unconditional_row_tail_continue(text: str) -> str:
    return replace_in_function(
        text,
        RMS_WARP,
        "if (row >= nrow) { continue; }",
        "{ continue; }",
        count=2,
    )


RMS_STORE = "store.template store<pack_size>(pack, row, col);"


def rms_affine_one_sided_store(text: str) -> str:
    return replace_in_function(
        text,
        RMS_BLOCK,
        RMS_STORE,
        "if ((threadIdx.x & 1) != 0) {\n"
        "        " + RMS_STORE + "\n"
        "      }",
    )


def rms_miss_first_row(text: str) -> str:
    return replace_in_function(
        text,
        RMS_BLOCK,
        "for (int row = blockIdx.x; row < nrow; row += gridDim.x) {",
        "for (int row = blockIdx.x + 1; row < nrow; row += gridDim.x) {",
    )


def rms_miss_final_pack(text: str) -> str:
    return replace_in_function(
        text,
        RMS_BLOCK,
        "const int num_packs = ncol / pack_size;",
        "const int num_packs = ncol / pack_size - 1;",
    )


def softmax_cases() -> Sequence[MutationCase]:
    return (
        MutationCase(
            "softmax",
            "canonical_full_coverage",
            3,
            "all three real forward kernels cover every routed row and column",
            unchanged,
        ),
        MutationCase(
            "softmax",
            "renamed_kernel_and_wrapper_spelling",
            3,
            "kernel and direct-wrapper recognition is independent of source spelling",
            softmax_renamed_spelling,
        ),
        MutationCase(
            "softmax",
            "miss_first_row",
            2,
            "block-SMEM row domain starts at blockIdx.x, without a hidden offset",
            softmax_miss_first_row,
        ),
        MutationCase(
            "softmax",
            "miss_final_pack",
            2,
            "block-SMEM pack domain covers cols / pack_size exactly",
            softmax_miss_final_pack,
        ),
        MutationCase(
            "softmax",
            "direct_local_sum",
            2,
            "normalization denominator is the cross-thread sum reduction",
            softmax_direct_local_sum,
        ),
        MutationCase(
            "softmax",
            "non_additive_sum_functor",
            0,
            "the reduction functor is structurally addition",
            softmax_bad_sum_functor,
        ),
        MutationCase(
            "softmax",
            "warp_reducer_identity",
            2,
            "warp reduction performs the canonical XOR all-reduce",
            softmax_warp_reducer_identity,
        ),
        MutationCase(
            "softmax",
            "block_reducer_identity",
            1,
            "block reduction performs the canonical CUB reduction and broadcast",
            softmax_block_reducer_identity,
        ),
        MutationCase(
            "softmax",
            "dead_reducer_spoof",
            2,
            "a reducer in a compile-time-dead branch cannot establish data flow",
            softmax_dead_reducer,
        ),
        MutationCase(
            "softmax",
            "conditional_reducer",
            2,
            "the required reduction executes for the full participating domain",
            softmax_conditional_reducer,
        ),
        MutationCase(
            "softmax",
            "shifted_adapter_row",
            2,
            "load and store row coordinates equal the covered logical row",
            softmax_shifted_row,
        ),
        MutationCase(
            "softmax",
            "shifted_adapter_column",
            2,
            "load and store columns equal the canonical pack coordinate",
            softmax_shifted_column,
        ),
        MutationCase(
            "softmax",
            "extra_adapter_load",
            2,
            "the semantic pipeline has exactly one canonical materializing load",
            softmax_extra_load,
        ),
        MutationCase(
            "softmax",
            "extra_adapter_store",
            2,
            "the semantic pipeline has exactly one canonical output store",
            softmax_extra_store,
        ),
        MutationCase(
            "softmax",
            "unconditional_continue",
            2,
            "an unconditional continue cannot make a syntactic pipeline reachable",
            softmax_unconditional_continue,
        ),
        MutationCase(
            "softmax",
            "one_sided_runtime_store",
            2,
            "output stores execute over the complete participating lane domain",
            softmax_one_sided_store,
        ),
        MutationCase(
            "softmax",
            "fake_thread_index",
            2,
            "SIMT coordinates are CUDA builtins with system provenance",
            softmax_fake_thread_index,
        ),
        MutationCase(
            "softmax",
            "fake_dim3_with_host_effect",
            2,
            "launch geometry uses the system dim3 type and has no hidden effect",
            softmax_fake_dim3,
        ),
        MutationCase(
            "softmax",
            "zero_grid_geometry",
            0,
            "the launch helper assigns a uniquely proven positive grid",
            softmax_zero_grid,
        ),
        MutationCase(
            "softmax",
            "early_geometry_error",
            0,
            "an unconditional error cannot leave unreachable grid evidence",
            softmax_early_geometry_error,
        ),
        MutationCase(
            "softmax",
            "mutated_geometry_failure_return",
            0,
            "a guarded error return cannot be rewritten to success before returning",
            softmax_mutated_failure_return,
        ),
        MutationCase(
            "softmax",
            "bare_wrapper_error_check",
            2,
            "the post-geometry insertion boundary requires an exact compound error edge",
            softmax_bare_wrapper_error_check,
        ),
        MutationCase(
            "softmax",
            "wrapper_atomic_store",
            2,
            "an atomic side effect prevents entry fast-path insertion",
            softmax_wrapper_atomic_store,
        ),
        MutationCase(
            "softmax",
            "geometry_atomic_store",
            0,
            "launch geometry containing an atomic effect is not bypassable",
            softmax_geometry_atomic_store,
        ),
        MutationCase(
            "softmax",
            "wrapper_volatile_cast_read",
            2,
            "a volatile read through a cast remains an observable effect",
            softmax_wrapper_volatile_cast_read,
        ),
        MutationCase(
            "softmax",
            "macro_body_insertion",
            2,
            "recipe insertion never targets a macro-produced function brace",
            softmax_macro_body_insertion,
        ),
        MutationCase(
            "softmax",
            "generated_name_collision",
            3,
            "generated names avoid parameters and body declarations and are macro guarded",
            softmax_generated_name_collision,
            validates_generated_name_collision,
        ),
        MutationCase(
            "softmax",
            "template_parameter_name_collision",
            3,
            "generated names avoid enclosing function-template parameters",
            softmax_template_parameter_collision,
            validates_template_parameter_collision,
        ),
        MutationCase(
            "softmax",
            "member_wrapper_duplicate",
            3,
            "direct wrapper recipes apply only to namespace-scope free functions",
            softmax_member_wrapper_duplicate,
        ),
    )


def rms_cases() -> Sequence[MutationCase]:
    return (
        MutationCase(
            "rms_norm",
            "canonical_full_coverage",
            3,
            "all three recipes execute geometry and its error edge before the fast path",
            unchanged,
            validates_rms_post_geometry,
        ),
        MutationCase(
            "rms_norm",
            "renamed_kernel_and_wrapper_spelling",
            3,
            "kernel and direct-wrapper recognition is independent of source spelling",
            rms_renamed_spelling,
        ),
        MutationCase(
            "rms_norm",
            "direct_local_square_denominator",
            2,
            "the square mean consumes the cross-thread sum, not a lane-local sum",
            rms_direct_local_denominator,
        ),
        MutationCase(
            "rms_norm",
            "dead_reducer_spoof",
            2,
            "a reducer in a compile-time-dead branch cannot establish data flow",
            rms_dead_reducer,
        ),
        MutationCase(
            "rms_norm",
            "fake_inverse_lane",
            2,
            "the inverse-RMS side output is written by proven CUDA lane zero",
            rms_fake_inverse_lane,
        ),
        MutationCase(
            "rms_norm",
            "unconditional_row_tail_continue",
            2,
            "only the canonical out-of-range row tail may continue",
            rms_unconditional_row_tail_continue,
        ),
        MutationCase(
            "rms_norm",
            "affine_one_sided_runtime_store",
            2,
            "direct or affine stores execute over the full participating lane domain",
            rms_affine_one_sided_store,
        ),
        MutationCase(
            "rms_norm",
            "miss_first_row",
            2,
            "block-SMEM row coverage starts at blockIdx.x",
            rms_miss_first_row,
        ),
        MutationCase(
            "rms_norm",
            "miss_final_pack",
            2,
            "block-SMEM pack coverage spans ncol / pack_size exactly",
            rms_miss_final_pack,
        ),
    )


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fail-closed real-source mutations for dav-c310 recipes"
    )
    parser.add_argument("--ascify", required=True, type=pathlib.Path)
    parser.add_argument("--softmax", required=True, type=pathlib.Path)
    parser.add_argument("--rms-norm", required=True, type=pathlib.Path)
    parser.add_argument("--layer-norm", required=True, type=pathlib.Path)
    parser.add_argument("--recipe-source", required=True, type=pathlib.Path)
    parser.add_argument("--recipe-header", required=True, type=pathlib.Path)
    parser.add_argument("--cuda-path", required=True, type=pathlib.Path)
    parser.add_argument(
        "--clang-resource-directory", required=True, type=pathlib.Path
    )
    parser.add_argument(
        "--include-dir",
        required=True,
        type=pathlib.Path,
        help="include root containing oneflow/core/cuda/layer_norm.cuh",
    )
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    parser.add_argument("--converter-cwd", type=pathlib.Path)
    parser.add_argument(
        "--converter-arg",
        action="append",
        default=[],
        help="extra ascify argument before -o (repeatable)",
    )
    parser.add_argument(
        "--clang-arg",
        action="append",
        default=[],
        help="extra clang argument after -- (repeatable)",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="run only FAMILY/NAME or NAME (repeatable)",
    )
    return parser.parse_args(argv)


def selected(cases: Iterable[MutationCase], filters: Sequence[str]) -> List[MutationCase]:
    cases = list(cases)
    if not filters:
        return cases
    result = []
    for case in cases:
        full_name = case.family + "/" + case.name
        if case.name in filters or full_name in filters:
            result.append(case)
    missing = sorted(
        filter_name
        for filter_name in filters
        if not any(
            filter_name == case.name
            or filter_name == case.family + "/" + case.name
            for case in cases
        )
    )
    if missing:
        raise MutationError("unknown --case selection(s): " + ", ".join(missing))
    return result


def converter_command(
    args: argparse.Namespace,
    source: pathlib.Path,
    output: pathlib.Path,
) -> List[str]:
    return [
        str(args.ascify),
        str(source),
        "--cuda-path=" + str(args.cuda_path),
        "--clang-resource-directory=" + str(args.clang_resource_directory),
        "--target-policy=dav-c310-vec",
        "--simt-math=fast",
        *args.converter_arg,
        "-o",
        str(output),
        "--",
        "-I" + str(args.include_dir),
        "-include",
        "cuda_fp16.h",
        "-include",
        "cuda_bf16.h",
        "-std=c++17",
        *args.clang_arg,
    ]


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(
    path: pathlib.Path,
    results: Sequence[CaseResult],
    provenance: Sequence[Provenance],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, dialect="excel-tab")
        writer.writerow(("provenance", "path", "sha256"))
        for item in provenance:
            writer.writerow(
                (item.label, str(item.path), item.sha256)
            )
        writer.writerow(())
        writer.writerow(
            (
                "family",
                "case",
                "expected_recipes",
                "actual_recipes",
                "return_code",
                "debug_markers",
                "status",
                "contract",
            )
        )
        for result in results:
            writer.writerow(
                (
                    result.family,
                    result.name,
                    result.expected_recipes,
                    result.actual_recipes,
                    result.return_code,
                    result.debug_markers,
                    result.status,
                    result.contract,
                )
            )


def run_case(
    args: argparse.Namespace,
    case: MutationCase,
    original: str,
) -> CaseResult:
    prefix = case.family + "__" + case.name
    source_path = args.work_dir / (prefix + ".cuh")
    output_path = args.work_dir / (prefix + ".asc.cuh")
    stdout_path = args.work_dir / (prefix + ".stdout")
    stderr_path = args.work_dir / (prefix + ".stderr")
    mutated = case.mutate(original)
    if case.name != "canonical_full_coverage" and mutated == original:
        raise MutationError(prefix + ": mutation did not change the source")
    source_path.write_text(mutated, encoding="utf-8")
    completed = subprocess.run(
        converter_command(args, source_path, output_path),
        cwd=str(args.converter_cwd) if args.converter_cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    converted = (
        output_path.read_text(encoding="utf-8")
        if output_path.is_file()
        else ""
    )
    token = SOFTMAX_RECIPE if case.family == "softmax" else RMS_RECIPE
    actual = converted.count(token)
    debug = (completed.stdout + completed.stderr).count(DEBUG_MARKER)
    output_valid = (
        case.validate_output is None
        or case.validate_output(converted)
    )
    passed = (
        completed.returncode == 0
        and output_path.is_file()
        and actual == case.expected_recipes
        and debug == 0
        and output_valid
    )
    return CaseResult(
        case.family,
        case.name,
        case.expected_recipes,
        actual,
        completed.returncode,
        debug,
        "PASS" if passed else "FAIL",
        case.contract,
    )


def main(argv: Sequence[str]) -> int:
    args = parse_arguments(argv)
    for path, label in (
        (args.ascify, "ascify"),
        (args.softmax, "softmax"),
        (args.rms_norm, "rms_norm"),
        (args.layer_norm, "layer_norm"),
        (args.recipe_source, "recipe_source"),
        (args.recipe_header, "recipe_header"),
        (args.cuda_path, "cuda_path"),
        (args.clang_resource_directory, "clang_resource_directory"),
        (args.include_dir, "include_dir"),
    ):
        if not path.exists():
            raise MutationError("{} does not exist: {}".format(label, path))
    args.work_dir.mkdir(parents=True, exist_ok=True)
    originals = {
        "softmax": args.softmax.read_text(encoding="utf-8"),
        "rms_norm": args.rms_norm.read_text(encoding="utf-8"),
    }
    provenance = tuple(
        Provenance(label, path.resolve(), sha256_file(path))
        for label, path in (
            ("ascify_binary", args.ascify),
            (
                "mutation_harness",
                pathlib.Path(__file__).resolve(),
            ),
            ("softmax_input", args.softmax),
            ("rms_norm_input", args.rms_norm),
            ("layer_norm_input", args.layer_norm),
            ("recipe_source", args.recipe_source),
            ("recipe_header", args.recipe_header),
        )
    )
    cases = selected((*softmax_cases(), *rms_cases()), args.case)
    results: List[CaseResult] = []
    manifest = args.work_dir / "rowwise_mutation_results.tsv"
    for case in cases:
        try:
            result = run_case(args, case, originals[case.family])
        except (MutationError, OSError) as error:
            result = CaseResult(
                case.family,
                case.name,
                case.expected_recipes,
                -1,
                -1,
                -1,
                "HARNESS_ERROR",
                case.contract + " | " + str(error).replace("\n", " "),
            )
        results.append(result)
        write_manifest(manifest, results, provenance)
        print(
            "{}/{}: {} (recipes {}/{}, rc={}, debug={})".format(
                result.family,
                result.name,
                result.status,
                result.actual_recipes,
                result.expected_recipes,
                result.return_code,
                result.debug_markers,
            ),
            flush=True,
        )
    failures = [result for result in results if result.status != "PASS"]
    if failures:
        print(
            "{} mutation gate(s) failed; evidence: {}".format(
                len(failures), manifest
            ),
            file=sys.stderr,
        )
        return 1
    print(
        "{} dav-c310 rowwise mutation gates passed; evidence: {}".format(
            len(results), manifest
        )
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except MutationError as error:
        print("mutation harness error: {}".format(error), file=sys.stderr)
        sys.exit(2)
