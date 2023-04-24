import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

sys.path.append(".")

from src.explanation.generation.generate_explanation import generate_explanation
from src.explanation.generation.make_prompt import get_fewshot_prompt
from src.explanation.generation.select_fewshot import ExplanationFewShotExample
from src.utils.IO_utils import load_json_file, write_to_file
from src.utils.program_utils import load_program


def give_explanation(
    prompt_type: Optional[str],
    buggy_program: Optional[str],
    fixed_program: Optional[str],
    py_major: Optional[int],
    fewshot_examples: Sequence[ExplanationFewShotExample],
    config,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns: The prompt and the explanation.
    """
    # Pre-check some arguments
    if py_major not in {2, 3}:
        raise ValueError("The `py_major` must be either 2 or 3")

    # Make the prompt for explanation
    prompt = get_fewshot_prompt(
        config=config,
        target_buggy_program=buggy_program,
        target_fixed_program=fixed_program,
        fewshot_instances=fewshot_examples,
        prompt_type=prompt_type,
        py_major=py_major,
    )

    # Query for explanation
    explanation = generate_explanation(
        prompt=prompt,
        n_list=[1],
        temperature_list=[0],
    )[0]

    return prompt, explanation


def main(args):
    fewshot_examples = [
        ExplanationFewShotExample.from_annotation(anno, anno_id=0) for anno in load_json_file(args.fewshot_path)
    ]

    prompt, explanation = give_explanation(
        buggy_program=load_program(args.buggy_program_path),
        fixed_program=load_program(args.fixed_program_path),
        fewshot_examples=fewshot_examples,
        py_major=args.py_major,
        prompt_type=args.prompt_type,
        config=load_json_file(args.config_path),
    )

    print(prompt)
    if args.output_path is not None:
        write_to_file(file_path=args.output_path, content=explanation, new_line=False)
    else:
        print(explanation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=Path, help="Path to the JSON config file.")
    parser.add_argument("--buggy_program_path", type=Path, help="Path to the buggy program.")
    parser.add_argument("--fixed_program_path", type=Path, help="Path to the fixed program.")
    parser.add_argument("--fewshot_path", type=Path, help="Path to the few-shot examples.")
    parser.add_argument("--py_major", type=int, help="Python major version of the program (either 2 or 3).")
    parser.add_argument("--prompt_type", type=str, help="Type of prompt format.")
    parser.add_argument("--output_path", type=Path, default=None, help="path to save the generated explanation.")

    args = parser.parse_args()

    main(args)
