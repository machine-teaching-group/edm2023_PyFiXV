import sys
from typing import Sequence

sys.path.append(".")

from src.explanation.generation.select_fewshot import ExplanationFewShotExample
from src.utils.program_utils import (
    get_parsing_error_string,
    get_program_diff,
    get_program_inline_diff,
)


def get_fewshot_prompt(
    config,
    target_buggy_program: str,
    target_fixed_program: str,
    fewshot_instances: Sequence[ExplanationFewShotExample],
    prompt_type: str,
    py_major: int,
):
    template = config["Codex_c"]["fewshot"]["template"][prompt_type]
    shot_template = template["shot"]
    target_template = template["target_example"]
    prompt_template = template["prompt"]

    target_program_diff = get_program_diff(target_buggy_program, target_fixed_program)
    target_inline_diff = get_program_inline_diff(target_buggy_program, target_fixed_program)

    target_error_string = get_parsing_error_string(buggy_program=target_buggy_program, py_major=py_major)

    fewshots = []
    for instance in fewshot_instances:
        shot_buggy_program = instance.buggy_program.rstrip()
        shot_fixed_program = instance.fixed_program.rstrip()
        shot_explanation = instance.explanation
        shot_program_diff = get_program_diff(shot_buggy_program, shot_fixed_program)
        shot_program_inline_diff = get_program_inline_diff(shot_buggy_program, shot_fixed_program)
        shot_error_string = get_parsing_error_string(buggy_program=shot_buggy_program, py_major=py_major)

        fewshots.append(
            shot_template.format(
                buggy_program=shot_buggy_program,
                fixed_program=shot_fixed_program,
                diff=shot_program_diff,
                inline_diff=shot_program_inline_diff,
                explanation=shot_explanation,
                py_major=py_major,
                error=shot_error_string,
            )
        )

    target_program_prompt = target_template.format(
        buggy_program=target_buggy_program,
        diff=target_program_diff,
        fixed_program=target_fixed_program,
        inline_diff=target_inline_diff,
        py_major=py_major,
        error=target_error_string,
    )

    prompt = prompt_template.format(
        fewshots_explanation="\n\n\n".join(fewshots), target_program_prompt=target_program_prompt, py_major=py_major
    )

    return prompt


if __name__ == "__main__":
    pass
