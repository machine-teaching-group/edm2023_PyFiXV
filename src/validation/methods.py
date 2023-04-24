import sys
from collections import namedtuple
from typing import List, Optional, Sequence

sys.path.append(".")

from src.repair.evaluation.metrics import exact_match, is_parseable
from src.repair.generation.generate_fixes import generate_fixes
from src.validation.utils import split_explanation_to_tokens

Verification_Result = namedtuple("Verification_Result", "ExactMatch Parseable")


def verify_by_back_repair(
    config,
    buggy_program: str,
    fixed_program: Optional[str],
    generated_explanation: str,
    py_major: int,
    n_list: Sequence[int] = (1,),
    temperature_list: Sequence[float] = (0,),
    pass_threshold: int = 1,
) -> Verification_Result:
    back_repairs: List[str] = []
    for n, temperature in zip(n_list, temperature_list):
        back_repairs += generate_fixes(
            program=buggy_program,
            instruction_list=[
                config["Codex_e"]["br_instruction"].format(py_major=py_major, explanation=generated_explanation)
            ],
            n_list=[n],
            temperature_list=[temperature],
        )

    cnt_exact_match, cnt_parseable = 0, 0
    for back_repair in back_repairs:
        cnt_exact_match += exact_match(
            generated_repair=back_repair, groundtruth_repairs=[fixed_program], py_major=py_major
        )
        cnt_parseable += is_parseable(generated_repair=back_repair, py_major=py_major)

    back_repair_verification_result = Verification_Result(
        cnt_exact_match >= pass_threshold,
        cnt_parseable >= pass_threshold,
    )

    return back_repair_verification_result


def verify_by_back_repair_multiple_thresholds(
    buggy_program: str,
    fixed_program: Optional[str],
    generated_explanation: str,
    py_major: int,
    n_list: Sequence[int] = (10,),
    temperature_list: Sequence[float] = (0.5,),
    pass_thresholds: Sequence[int] = (3, 5, 7, 10),
) -> Sequence[Verification_Result]:
    back_repairs: List[str] = []
    for n, temperature in zip(n_list, temperature_list):
        back_repairs += generate_fixes(
            program=buggy_program,
            instruction_list=[generated_explanation],
            n_list=[n],
            temperature_list=[temperature],
        )

    cnt_exact_match, cnt_parseable = 0, 0
    for back_repair in back_repairs:
        cnt_exact_match += exact_match(
            generated_repair=back_repair, groundtruth_repairs=[fixed_program], py_major=py_major
        )
        cnt_parseable += is_parseable(generated_repair=back_repair, py_major=py_major)

    back_repair_verification_results = []
    for pass_threshold in pass_thresholds:
        back_repair_verification_results.append(
            Verification_Result(
                cnt_exact_match >= pass_threshold,
                cnt_parseable >= pass_threshold,
            )
        )

    return back_repair_verification_results


def verify_by_explanation_token_length(generated_explanation: str, threshold_length: int = 60) -> bool:
    explanation_length = len(split_explanation_to_tokens(explanation=generated_explanation))
    return explanation_length < threshold_length
