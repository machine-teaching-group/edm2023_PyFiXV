import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

sys.path.append(".")

from src.explanation.generation.generate_explanation import generate_explanation
from src.explanation.generation.make_prompt import get_fewshot_prompt
from src.explanation.generation.preprocess import load_annotation_data
from src.explanation.generation.select_fewshot import (
    ExplanationFewShotExample,
    FS_Criterion,
    get_fewshots,
)
from src.utils.IO_utils import load_json_file, write_to_file
from src.utils.program_utils import load_program


def give_explanation(
    prompt: Optional[str] = None,
    prompt_type: Optional[str] = None,
    buggy_program: Optional[str] = None,
    fixed_program: Optional[str] = None,
    py_major: Optional[int] = None,
    fewshot_pool=None,
    fewshot_priority_list: Optional[Sequence[FS_Criterion]] = None,
    nshot: Optional[int] = None,
    config=None,
) -> Tuple[Optional[Sequence[ExplanationFewShotExample]], Optional[str], Optional[str]]:
    """
    Returns: The sequence of few-shot examples, the prompt, and the explanation.
    If the prompt is given in input, the returning sequence of few-shot examples is None.
    """
    # Pre-check some arguments
    if (prompt is not None) + (prompt_type is not None) != 1:
        raise ValueError("either `prompt` or `prompt_type` must be specified, but not both.")
    if prompt_type is not None:
        if buggy_program is None or buggy_program.strip() == "":
            raise ValueError("The `buggy_program` must be specified")
        if fixed_program is None:
            raise ValueError("The `fixed_program` must be specified")
        if py_major is None or py_major not in {2, 3}:
            raise ValueError("The `py_major` must be either 2 or 3")
        if fewshot_pool is None:
            raise ValueError("The `fewshot_pool` must be specified")
        if fewshot_priority_list is None:
            raise ValueError("The `fewshot_priority_list` must be specified")
        if nshot is None or nshot < 0:
            raise ValueError("`nshot` must be a positive integer")
        if config is None:
            raise ValueError("`config` must be specified")

    # Compute the prompt if it is not given
    if prompt_type is not None:
        few_shots = get_fewshots(
            priority_list=fewshot_priority_list,
            fewshot_pool=fewshot_pool,
            nshot=nshot,
            py_major=py_major,
            buggy_program=buggy_program,
            fixed_program=fixed_program,
        )

        # Make the prompt for explanation
        prompt = get_fewshot_prompt(
            config=config,
            target_buggy_program=buggy_program,
            target_fixed_program=fixed_program,
            fewshot_instances=few_shots,
            prompt_type=prompt_type,
            py_major=py_major,
        )
    else:
        few_shots = None

    # Query for explanation
    explanation = generate_explanation(
        prompt=prompt,
        n_list=[1],
        temperature_list=[0],
    )[0]

    return few_shots, prompt, explanation


def main(args):
    fewshot_pool = load_annotation_data(annotation_data_path=args.annotation_data_path, partitions=("training",))
    fewshot_priority_list = eval(args.fewshot_priority_list)

    few_shots, prompt, explanation = give_explanation(
        buggy_program=load_program(args.buggy_program_path),
        fixed_program=load_program(args.fixed_program_path),
        py_major=args.py_major,
        fewshot_pool=fewshot_pool,
        fewshot_priority_list=fewshot_priority_list,
        nshot=args.nshot,
        prompt_type=args.prompt_type,
        config=load_json_file(args.config_path),
    )

    # Write down prompt and explanation
    write_to_file(args.output_folder / "prompt.txt", prompt)
    write_to_file(args.output_folder / "explanation.txt", explanation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=Path, help="Path to the JSON config file.")
    parser.add_argument("--buggy_program_path", type=Path, help="Path to the buggy program.")
    parser.add_argument("--fixed_program_path", type=Path, help="Path to the fixed program.")
    parser.add_argument("--py_major", type=int, help="Python major version of the program (either 2 or 3).")
    parser.add_argument("--nshot", type=int, help="Number of few-shot examples to add to the prompt.")
    parser.add_argument("--prompt_type", type=str, help="Type of prompt format.")
    parser.add_argument(
        "--fewshot_priority_list",
        type=str,
        default="[FS_Criterion.ERROR_TYPE,FS_Criterion.MASKED_ERROR_TEXT]",
        help="Type of prompt format.",
    )
    parser.add_argument("--annotation_data_path", type=Path)
    parser.add_argument("--output_folder", type=Path, help="Path to the output folder")

    args = parser.parse_args()

    main(args)
