import argparse
import sys
from pathlib import Path

sys.path.append(".")

from src.explanation.generation.explain_example import give_explanation
from src.explanation.generation.select_fewshot import ExplanationFewShotExample
from src.repair.generation.repair_pipeline import repair
from src.utils.IO_utils import load_json_file, write_to_file
from src.utils.program_utils import load_program
from src.validation.methods import verify_by_back_repair


def main(args):
    config = load_json_file(args.config_path)
    buggy_program = load_program(args.program_path)

    # Generate the repaired program
    repaired_program = repair(
        buggy_program=buggy_program,
        py_major=args.py_major,
        config=config,
        instruction_type=args.instruction_type,
    )

    # Write down the repaired program
    repaired_program_path = args.output_folder / "repaired_program.py"
    write_to_file(file_path=repaired_program_path, content=repaired_program, new_line=False)

    # If the repaired program is not None, generate the explanation and do validation
    if repaired_program is not None:
        # Generate the explanation
        fewshot_examples = [
            ExplanationFewShotExample.from_annotation(anno, anno_id=0) for anno in load_json_file(args.fewshot_path)
        ]
        _, explanation = give_explanation(
            buggy_program=buggy_program,
            fixed_program=repaired_program,
            fewshot_examples=fewshot_examples,
            py_major=args.py_major,
            prompt_type=args.prompt_type,
            config=config,
        )

        # Write down the explanation
        explanation_path = args.output_folder / "explanation.txt"
        write_to_file(file_path=explanation_path, content=explanation, new_line=False)

        # Do validation
        back_repair_params = load_json_file(args.validation_params_path)
        if verify_by_back_repair(
            config=config,
            buggy_program=buggy_program,
            fixed_program=repaired_program,
            generated_explanation=explanation,
            py_major=args.py_major,
            n_list=back_repair_params["n_list"],
            temperature_list=back_repair_params["temp_list"],
            pass_threshold=back_repair_params["threshold"],
        ).ExactMatch:
            validation_result = "Accepted"
        else:
            validation_result = "Rejected"

        # Write down the validation
        validation_path = args.output_folder / "validation.txt"
        write_to_file(file_path=validation_path, content=validation_result, new_line=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=Path, help="Path to the JSON config file.")
    parser.add_argument("--program_path", type=Path, help="Path to the buggy program.")
    parser.add_argument("--py_major", type=int, help="Python major version of the program (either 2 or 3).")
    parser.add_argument("--instruction_type", type=str, help="Type of instruction for Codex-E (e.g. 'general').")
    parser.add_argument("--fewshot_path", type=Path, help="Path to the few-shot examples.")
    parser.add_argument("--prompt_type", type=str, help="Type of prompt format.")
    parser.add_argument(
        "--verify_by_back_repair",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to use our run-time validation method.",
    )
    parser.add_argument(
        "--validation_params_path",
        type=Path,
        default=None,
        help="Hyper-parameter values for the run-time validation method.",
    )
    parser.add_argument("--output_folder", type=Path, help="path to save the generated feedback.")

    args = parser.parse_args()

    main(args)
