import random
import sys
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Optional, Sequence, Tuple, Union

sys.path.append(".")

from src.utils.edit_distance_utils import Edit, compute_edit_distance
from src.utils.program_utils import (
    get_program_diff,
    get_program_masked_token_diff,
    get_program_token_diff,
    parse_python_2,
    parse_python_3,
    tokenize_and_mask_reliables,
)


class FS_Criterion(IntEnum):
    RANDOM = auto()
    ERROR_TYPE = auto()
    MASKED_ERROR_TEXT = auto()
    MASKED_DIFF = auto()
    TOKEN_DIFF = auto()
    MASKED_TOKEN_DIFF = auto()


@dataclass
class ExplanationFewShotExample:
    buggy_program: str
    explanation: str
    fixed_program: str

    @classmethod
    def from_annotation(cls, anno, anno_id):
        return cls(
            buggy_program=anno["buggy_program"],
            explanation=anno["annotation"][anno_id]["feedback"],
            fixed_program=anno["annotation"][anno_id]["fixed_program"],
        )


def get_priority(
    example,
    priority_list: Sequence[FS_Criterion],
    error_type: Optional[str] = None,
    error_text: Optional[str] = None,
    diff: Optional[str] = None,
    token_diff: Optional[Sequence[Tuple[Edit, Union[str, Tuple[str, str]]]]] = None,
) -> Tuple[int]:
    """
    Most-relevant examples will have high returning priority values.
    """
    # Check for required arguments
    for criterion in priority_list:
        if criterion == FS_Criterion.ERROR_TYPE:
            if error_type is None:
                raise ValueError("For FS_Criterion.ERROR_TYPE, the argument `error_type` must be not None.")
        if criterion == FS_Criterion.MASKED_ERROR_TEXT:
            if error_text is None:
                raise ValueError("For FS_Criterion.MASKED_ERROR_TEXT, the argument `error_text` must be not None.")
        if criterion == FS_Criterion.MASKED_DIFF:
            if diff is None:
                raise ValueError("For FS_Criterion.MASKED_DIFF, the argument `diff` must be not None.")
        if criterion == FS_Criterion.TOKEN_DIFF:
            if token_diff is None:
                raise ValueError("For FS_Criterion.TOKEN_DIFF, the argument `token_diff` must be not None.")
            for element in token_diff:
                if element[0] == Edit.Keep:
                    raise ValueError("There should be no Edit.Keep in token_diff")
        if criterion == FS_Criterion.MASKED_TOKEN_DIFF:
            if token_diff is None:
                raise ValueError("For FS_Criterion.MASKED_TOKEN_DIFF, the argument `token_diff` must be not None.")
            for element in token_diff:
                if element[0] == Edit.Keep:
                    raise ValueError("There should be no Edit.Keep in token_diff")

    # Compute priority
    priority = []
    for criterion in priority_list:
        # RANDOM
        if criterion == FS_Criterion.RANDOM:
            priority.append(random.randint(0, 1000000000))
        # ERROR_TYPE
        elif criterion == FS_Criterion.ERROR_TYPE:
            priority.append(int(error_type == example["parser_output"]["error_type"]))
        # MASKED_ERROR_TEXT
        elif criterion == FS_Criterion.MASKED_ERROR_TEXT:
            masked_error_text = tokenize_and_mask_reliables(error_text)
            ed = compute_edit_distance(
                masked_error_text, example["parser_output"]["error_text_reliable_masked_tokenized"]
            )
            priority.append(-ed)
        # MASKED_DIFF
        elif criterion == FS_Criterion.MASKED_DIFF:
            masked_diff = tokenize_and_mask_reliables(diff)
            example_diff = get_program_diff(example["buggy_program"], example["annotation"][0]["fixed_program"])
            example_masked_diff = tokenize_and_mask_reliables(example_diff)
            ed = compute_edit_distance(masked_diff, example_masked_diff)
            priority.append(-ed)
        # TOKEN_DIFF
        elif criterion == FS_Criterion.TOKEN_DIFF:
            example_token_diff = get_program_token_diff(
                buggy_program=example["buggy_program"],
                fixed_program=example["annotation"][0]["fixed_program"],
            )
            ed = compute_edit_distance(token_diff, example_token_diff)
            priority.append(-ed)
        # MASKED_TOKEN_DIFF
        elif criterion == FS_Criterion.MASKED_TOKEN_DIFF:
            masked_token_diff = get_program_masked_token_diff(token_diff=token_diff)
            example_token_diff = get_program_token_diff(
                buggy_program=example["buggy_program"],
                fixed_program=example["annotation"][0]["fixed_program"],
            )
            example_masked_token_diff = get_program_masked_token_diff(token_diff=example_token_diff)
            ed = compute_edit_distance(masked_token_diff, example_masked_token_diff)
            priority.append(-ed)

        else:
            raise ValueError(f"Few-shot Criterion is not valid: `{criterion}`")

    return tuple(priority)


def get_fewshots(
    priority_list: Sequence[FS_Criterion],
    fewshot_pool,
    nshot: int,
    py_major: int,
    buggy_program: Optional[str] = None,
    fixed_program: Optional[str] = None,
) -> Sequence[ExplanationFewShotExample]:
    """
    Select few-shot examples according to `priority_list`.
    """
    if nshot < 0:
        raise ValueError("`nshot` must be a non-negative integer")

    if nshot == 0:
        selected_shots = []
    else:
        # Pre-compute some values for the testing program
        if py_major == 2:
            buggy_parse_error = parse_python_2(buggy_program)
        else:
            assert py_major == 3
            buggy_parse_error = parse_python_3(buggy_program)

        if FS_Criterion.ERROR_TYPE in priority_list:
            error_type = buggy_parse_error["error_type"]
        else:
            error_type = None

        if FS_Criterion.MASKED_ERROR_TEXT in priority_list:
            error_text = buggy_parse_error["error_text"]
        else:
            error_text = None

        if FS_Criterion.MASKED_DIFF in priority_list:
            diff = get_program_diff(buggy_program, fixed_program)
        else:
            diff = None

        if FS_Criterion.TOKEN_DIFF in priority_list or FS_Criterion.MASKED_TOKEN_DIFF in priority_list:
            token_diff = get_program_token_diff(buggy_program=buggy_program, fixed_program=fixed_program)
        else:
            token_diff = None

        # Sort the example pool
        sorted_fewshot_pool = sorted(
            fewshot_pool,
            key=lambda example: get_priority(
                error_type=error_type,
                error_text=error_text,
                diff=diff,
                token_diff=token_diff,
                example=example,
                priority_list=priority_list,
            ),
        )

        # Take the best n shots
        # The best example appear at the end
        selected_shots = [
            ExplanationFewShotExample.from_annotation(example, 0) for example in sorted_fewshot_pool[-nshot:]
        ]

    return selected_shots


if __name__ == "__main__":
    pass
